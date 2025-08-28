"""
Scalable RAG pipeline that integrates caching, batching, and progress tracking.
This is an enhanced version of the RAG pipeline with scalability features.
"""

from typing import List, Dict, Any, Optional
from langchain.schema import Document

from RAG.app.interfaces import RAGPipelineInterface
from RAG.app.Performance_Optimization.dependency_injection import RAGServiceLocator
from RAG.app.config import settings
from RAG.app.logger import get_logger
from RAG.app.Performance_Optimization.caching import get_cache_manager, get_cache_stats
from RAG.app.Performance_Optimization.batching import create_document_batcher, create_batch_processor
from RAG.app.Evaluation_Analysis.progress_tracking import track_operation, progress_bar, get_system_info


class IntegratedRAGPipeline(RAGPipelineInterface):
    """Integrated RAG pipeline with caching, batching, and progress tracking."""
    
    def __init__(self, config=None):
        self.config = config or settings
        self.logger = get_logger()
        
        # Get services from dependency injection
        self.document_ingestion = RAGServiceLocator.get_document_ingestion()
        self.query_processing = RAGServiceLocator.get_query_processing()
        self.retrieval = RAGServiceLocator.get_retrieval()
        self.answer_generation = RAGServiceLocator.get_answer_generation()
        self.configuration = RAGServiceLocator.get_configuration()
        self.logging = RAGServiceLocator.get_logging()
        
        # Initialize scalability components
        self.cache_manager = get_cache_manager()
        self.document_batcher = create_document_batcher(batch_size=10)
        self.batch_processor = create_batch_processor(batch_size=10, max_workers=4)
        
        # Pipeline state
        self.documents: List[Document] = []
        self.is_initialized = False
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query through the complete RAG pipeline with caching."""
        with track_operation("Query Processing", 1) as tracker:
            try:
                self.logger.info(f"Processing query: {query}")
                
                # Step 1: Analyze the query (cached)
                query_analysis = self.query_processing.analyze_query(query)
                self.logger.debug(f"Query analysis: {query_analysis}")
                tracker.update(increment=1)
                
                # Step 2: Retrieve relevant documents
                retrieved_docs = self.retrieval.retrieve_documents(
                    query, 
                    top_k=self.config.embedding.CONTEXT_TOP_N
                )
                self.logger.info(f"Retrieved {len(retrieved_docs)} documents")
                tracker.update(increment=1)
                
                # Step 3: Apply filters based on query analysis
                filtered_docs = self.query_processing.apply_filters(retrieved_docs, {})
                self.logger.debug(f"Filtered to {len(filtered_docs)} documents")
                tracker.update(increment=1)
                
                # Step 4: Rerank documents for better relevance
                reranked_docs = self.retrieval.rerank_documents(
                    query, 
                    filtered_docs, 
                    top_k=self.config.embedding.CONTEXT_TOP_N
                )
                self.logger.info(f"Reranked to {len(reranked_docs)} documents")
                tracker.update(increment=1)
                
                # Step 5: Generate answer (cached)
                answer = self.answer_generation.generate_answer(query, reranked_docs)
                tracker.update(increment=1)
                
                # Step 6: Format response
                response = self.answer_generation.format_answer(answer, reranked_docs)
                tracker.update(increment=1)
                
                # Add metadata
                response.update({
                    "query": query,
                    "query_analysis": query_analysis,
                    "pipeline_version": "scalable_v1.0",
                    "cache_stats": get_cache_stats()
                })
                
                self.logger.info("Query processing completed successfully")
                return response
                
            except Exception as e:
                self.logger.error(f"Error processing query: {e}")
                tracker.update(error=str(e))
                return {
                    "answer": "I encountered an error while processing your query. Please try again.",
                    "query": query,
                    "error": str(e),
                    "sources": [],
                    "num_sources": 0,
                    "confidence": "low"
                }
    
    def add_documents(self, file_paths: List[str]) -> bool:
        """Add new documents to the RAG system with batching."""
        with track_operation("Document Addition", len(file_paths)) as tracker:
            try:
                self.logger.info(f"Adding {len(file_paths)} documents to the pipeline")
                
                # Step 1: Ingest documents
                documents = self.document_ingestion.ingest_documents(file_paths)
                self.logger.info(f"Ingested {len(documents)} documents")
                tracker.update(processed_items=len(documents))
                
                # Step 2: Validate documents
                if not self.document_ingestion.validate_documents(documents):
                    self.logger.error("Document validation failed")
                    tracker.update(error="Document validation failed")
                    return False
                
                # Step 3: Preprocess documents in batches
                processed_docs = self.document_ingestion.preprocess_documents(documents)
                self.logger.info(f"Preprocessed {len(processed_docs)} documents")
                tracker.update(processed_items=len(processed_docs))
                
                # Step 4: Add to pipeline state
                self.documents.extend(processed_docs)
                self.is_initialized = True
                
                self.logger.info("Documents added successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Error adding documents: {e}")
                tracker.update(error=str(e))
                return False
    
    def add_documents_batched(self, file_paths: List[str], batch_size: int = 10) -> bool:
        """Add documents using batch processing for large datasets."""
        with track_operation("Batched Document Addition", len(file_paths)) as tracker:
            try:
                self.logger.info(f"Adding {len(file_paths)} documents in batches of {batch_size}")
                
                # Create batches of file paths
                batches = [file_paths[i:i + batch_size] for i in range(0, len(file_paths), batch_size)]
                
                all_documents = []
                
                with progress_bar(len(batches), "Processing document batches") as bar:
                    for i, batch in enumerate(batches):
                        self.logger.info(f"Processing batch {i + 1}/{len(batches)} ({len(batch)} files)")
                        
                        # Process each batch
                        batch_documents = self.document_ingestion.ingest_documents(batch)
                        
                        # Validate and preprocess batch
                        if self.document_ingestion.validate_documents(batch_documents):
                            processed_batch = self.document_ingestion.preprocess_documents(batch_documents)
                            all_documents.extend(processed_batch)
                            tracker.update(processed_items=len(processed_batch))
                        else:
                            tracker.update(warning=f"Batch {i + 1} failed validation")
                        
                        bar.update()
                
                # Add all processed documents to pipeline
                self.documents.extend(all_documents)
                self.is_initialized = True
                
                self.logger.info(f"Successfully added {len(all_documents)} documents in batches")
                return True
                
            except Exception as e:
                self.logger.error(f"Error in batched document addition: {e}")
                tracker.update(error=str(e))
                return False
    
    def initialize_from_data_directory(self) -> bool:
        """Initialize the pipeline by loading documents from the data directory."""
        with track_operation("Pipeline Initialization", 1) as tracker:
            try:
                self.logger.info("Initializing pipeline from data directory")
                
                # Discover input paths
                file_paths = self.document_ingestion.discover_input_paths()
                
                if not file_paths:
                    self.logger.warning("No documents found in data directory")
                    tracker.update(warning="No documents found in data directory")
                    return False
                
                # Use batched addition for better performance
                success = self.add_documents_batched(file_paths)
                
                if success:
                    self.logger.info(f"Pipeline initialized with {len(self.documents)} documents")
                else:
                    self.logger.error("Pipeline initialization failed")
                    tracker.update(error="Pipeline initialization failed")
                
                return success
                
            except Exception as e:
                self.logger.error(f"Error initializing pipeline: {e}")
                tracker.update(error=str(e))
                return False
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get the current status of the pipeline with scalability metrics."""
        system_info = get_system_info()
        cache_stats = get_cache_stats()
        
        return {
            "is_initialized": self.is_initialized,
            "num_documents": len(self.documents),
            "config": {
                "embedding_model": self.config.embedding.EMBEDDING_MODEL_OPENAI,
                "context_top_n": self.config.embedding.CONTEXT_TOP_N,
                "chunk_tok_max": self.config.chunking.CHUNK_TOK_MAX,
                "min_pages": self.config.chunking.MIN_PAGES
            },
            "services": {
                "document_ingestion": "available",
                "query_processing": "available",
                "retrieval": "available",
                "answer_generation": "available"
            },
            "scalability": {
                "cache_stats": cache_stats,
                "system_info": system_info,
                "batch_processing": "enabled",
                "progress_tracking": "enabled"
            }
        }
    
    def update_configuration(self, key: str, value: Any) -> bool:
        """Update a configuration setting."""
        try:
            success = self.configuration.update_setting(key, value)
            if success:
                self.logger.info(f"Updated configuration: {key} = {value}")
            return success
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            return False
    
    def get_document_summary(self) -> Dict[str, Any]:
        """Get a summary of the loaded documents."""
        try:
            if not self.documents:
                return {"message": "No documents loaded"}
            
            # Analyze documents
            file_types = {}
            total_pages = 0
            sections = {}
            
            for doc in self.documents:
                metadata = doc.metadata or {}
                
                # File types
                file_name = metadata.get("file_name", "unknown")
                file_ext = file_name.split(".")[-1] if "." in file_name else "unknown"
                file_types[file_ext] = file_types.get(file_ext, 0) + 1
                
                # Pages
                page = metadata.get("page", 0)
                if isinstance(page, int):
                    total_pages += page
                
                # Sections
                section = metadata.get("section", "unknown")
                sections[section] = sections.get(section, 0) + 1
            
            return {
                "total_documents": len(self.documents),
                "file_types": file_types,
                "total_pages": total_pages,
                "sections": sections,
                "average_content_length": sum(len(doc.page_content or "") for doc in self.documents) / len(self.documents)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating document summary: {e}")
            return {"error": str(e)}
    
    def clear_cache(self) -> bool:
        """Clear all caches."""
        try:
            self.cache_manager.clear()
            self.logger.info("All caches cleared")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing caches: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the pipeline."""
        return {
            "cache_stats": get_cache_stats(),
            "system_info": get_system_info(),
            "pipeline_status": self.get_pipeline_status(),
            "document_summary": self.get_document_summary()
        }


class IntegratedRAGPipelineFactory:
    """Factory for creating integrated RAG pipeline instances."""
    
    @staticmethod
    def create_pipeline(config=None) -> IntegratedRAGPipeline:
        """Create a new integrated RAG pipeline instance."""
        return IntegratedRAGPipeline(config)
    
    @staticmethod
    def create_pipeline_with_documents(file_paths: List[str], config=None) -> IntegratedRAGPipeline:
        """Create an integrated RAG pipeline and initialize it with documents."""
        pipeline = IntegratedRAGPipeline(config)
        if file_paths:
            pipeline.add_documents_batched(file_paths)
        return pipeline


# Convenience functions
def create_integrated_rag_pipeline(config=None) -> IntegratedRAGPipeline:
    """Create an integrated RAG pipeline with the given configuration."""
    return IntegratedRAGPipelineFactory.create_pipeline(config)


def process_query_with_integrated_pipeline(query: str, pipeline: Optional[IntegratedRAGPipeline] = None) -> Dict[str, Any]:
    """Process a query using an integrated RAG pipeline."""
    if pipeline is None:
        pipeline = create_integrated_rag_pipeline()
        # Try to initialize from data directory
        pipeline.initialize_from_data_directory()
    
    return pipeline.process_query(query)
