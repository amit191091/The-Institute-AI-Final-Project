"""
Main pipeline orchestrator for the Hybrid RAG system
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import sys

# Add current directory to path for imports
sys.path.append('.')

from rag_app.config import settings
from rag_app.loaders import get_document_loader
from rag_app.chunking import create_chunker
from rag_app.indexing import create_hybrid_index
from rag_app.retrieve import create_retriever
from rag_app.agents import create_multi_agent_system
from rag_app.validate import create_validator
from langchain.docstore.document import Document

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HybridRAGPipeline:
    """
    Main pipeline for the Metadata-Driven Hybrid RAG system
    """
    
    def __init__(self):
        self.loader = get_document_loader()
        self.chunker = create_chunker()
        self.validator = create_validator()
        self.index = None
        self.retriever = None
        self.agent_system = None
        
        # Ensure directories exist
        settings.DATA_DIR.mkdir(exist_ok=True)
        settings.INDEX_DIR.mkdir(exist_ok=True)
        settings.REPORTS_DIR.mkdir(exist_ok=True)
    
    def initialize(self, load_existing: bool = True) -> Dict[str, Any]:
        """
        Initialize the RAG pipeline
        
        Args:
            load_existing: Whether to load existing index if available
            
        Returns:
            Initialization results
        """
        logger.info("Initializing Hybrid RAG Pipeline...")
        
        try:
            # Create hybrid index
            self.index = create_hybrid_index()
            
            # Try to load existing index
            if load_existing and self.index.load_index():
                logger.info("Loaded existing index")
                loaded = True
            else:
                logger.info("Starting with empty index")
                loaded = False
            
            # Create retriever and agent system
            self.retriever = create_retriever(self.index)
            self.agent_system = create_multi_agent_system(self.retriever)
            
            result = {
                "success": True,
                "message": "Pipeline initialized successfully",
                "index_loaded": loaded,
                "stats": self.index.get_stats()
            }
            
            logger.info("Pipeline initialization completed")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            return {
                "success": False,
                "message": f"Initialization failed: {str(e)}",
                "error": str(e)
            }
    
    def ingest_documents(self, file_paths: List[Path], 
                        client_id: str = None, case_id: str = None) -> Dict[str, Any]:
        """
        Ingest documents into the RAG system
        
        Args:
            file_paths: List of document paths to ingest
            client_id: Optional client identifier
            case_id: Optional case identifier
            
        Returns:
            Ingestion results
        """
        if not self.index:
            return {"success": False, "message": "Pipeline not initialized"}
        
        logger.info(f"Starting document ingestion for {len(file_paths)} files")
        
        results = {
            "success": True,
            "total_files": len(file_paths),
            "processed_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "file_results": [],
            "errors": []
        }
        
        all_documents = []
        
        for file_path in file_paths:
            try:
                logger.info(f"Processing: {file_path}")
                
                # Validate document first
                validation_result = self.validator.validate_document(file_path)
                
                if not validation_result["is_valid"]:
                    error_msg = f"Validation failed for {file_path}: {'; '.join(validation_result['errors'])}"
                    logger.warning(error_msg)
                    results["errors"].append(error_msg)
                    results["failed_files"] += 1
                    
                    results["file_results"].append({
                        "file": str(file_path),
                        "success": False,
                        "error": "Validation failed",
                        "validation": validation_result
                    })
                    continue
                
                # Load document elements
                elements = self.loader.load_elements(file_path)
                
                if not elements:
                    error_msg = f"No content extracted from {file_path}"
                    logger.warning(error_msg)
                    results["errors"].append(error_msg)
                    results["failed_files"] += 1
                    continue
                
                # Extract IDs from filename if not provided
                file_client_id = client_id or self._extract_client_id(file_path.stem)
                file_case_id = case_id or self._extract_case_id(file_path.stem)
                
                # Chunk document
                chunks = self.chunker.chunk_document(
                    elements, 
                    str(file_path),
                    client_id=file_client_id,
                    case_id=file_case_id
                )
                
                # Convert to LangChain documents
                documents = [
                    Document(page_content=chunk["page_content"], metadata=chunk["metadata"])
                    for chunk in chunks
                ]
                
                all_documents.extend(documents)
                
                results["processed_files"] += 1
                results["total_chunks"] += len(chunks)
                
                results["file_results"].append({
                    "file": str(file_path),
                    "success": True,
                    "chunks": len(chunks),
                    "client_id": file_client_id,
                    "case_id": file_case_id,
                    "validation": validation_result
                })
                
                logger.info(f"Successfully processed {file_path}: {len(chunks)} chunks")
                
            except Exception as e:
                error_msg = f"Failed to process {file_path}: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                results["failed_files"] += 1
                
                results["file_results"].append({
                    "file": str(file_path),
                    "success": False,
                    "error": str(e)
                })
        
        # Add all documents to index
        if all_documents:
            try:
                logger.info(f"Adding {len(all_documents)} documents to index")
                self.index.add_documents(all_documents)
                
                # Save index
                self.index.save_index()
                logger.info("Index saved successfully")
                
            except Exception as e:
                error_msg = f"Failed to add documents to index: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                results["success"] = False
        
        # Update results
        if results["failed_files"] > 0:
            results["success"] = results["processed_files"] > 0
        
        logger.info(f"Ingestion completed: {results['processed_files']}/{results['total_files']} files processed")
        
        return results
    
    def query(self, query: str, k_contexts: int = None) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            query: User query
            k_contexts: Number of contexts to retrieve
            
        Returns:
            Query results
        """
        if not self.agent_system:
            return {"success": False, "message": "Pipeline not initialized"}
        
        k_contexts = k_contexts or settings.CONTEXT_TOP_N
        
        logger.info(f"Processing query: {query[:100]}...")
        
        try:
            result = self.agent_system.process_query(query, context_k=k_contexts)
            
            return {
                "success": True,
                "query": query,
                "response": result["response"],
                "metadata": {
                    "agent_type": result.get("agent_type"),
                    "sources_used": result.get("sources_used", 0),
                    "unique_documents": result.get("unique_documents", 0),
                    "routing_info": result.get("routing_info", {}),
                    "context_types": result.get("context_types", [])
                }
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "success": False,
                "message": f"Query failed: {str(e)}",
                "error": str(e)
            }
    
    def integrated_query(self, query: str, include_gear_analysis: bool = True) -> Dict[str, Any]:
        """
        Query with integration to existing gear analysis system
        
        Args:
            query: User query
            include_gear_analysis: Whether to include live gear analysis
            
        Returns:
            Integrated query results
        """
        if not self.agent_system:
            return {"success": False, "message": "Pipeline not initialized"}
        
        logger.info(f"Processing integrated query: {query[:100]}...")
        
        try:
            # Keep it lean: disable heavy gear integrations by default
            result = self.agent_system.process_query(query)
            
            return {
                "success": True,
                "query": query,
                "response": result["response"],
                "metadata": {
                    "agent_type": result.get("agent_type"),
                    "integration_mode": False,
                    "has_picture_data": False,
                    "has_vibration_data": False
                }
            }
            
        except Exception as e:
            logger.error(f"Integrated query processing failed: {e}")
            return {
                "success": False,
                "message": f"Integrated query failed: {str(e)}",
                "error": str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        if not self.index:
            return {"success": False, "message": "Pipeline not initialized"}
        
        try:
            stats = self.index.get_stats()
            return {
                "success": True,
                "stats": stats
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to get stats: {str(e)}"
            }
    
    def _extract_client_id(self, filename: str) -> Optional[str]:
        """Extract client ID from filename"""
        import re
        match = re.search(r'client[-_]?(\\w+)', filename.lower())
        return match.group(1) if match else None
    
    def _extract_case_id(self, filename: str) -> Optional[str]:
        """Extract case ID from filename"""
        import re
        match = re.search(r'case[-_]?(\\w+)', filename.lower())
        return match.group(1) if match else None

def create_pipeline() -> HybridRAGPipeline:
    """Factory function to create pipeline"""
    return HybridRAGPipeline()

def run_full_pipeline():
    """
    Complete pipeline execution with the provided PDF sample
    """
    logger.info("🚀 Starting Full Hybrid RAG Pipeline")
    
    # Initialize pipeline
    pipeline = create_pipeline()
    init_result = pipeline.initialize()
    
    if not init_result["success"]:
        logger.error(f"Pipeline initialization failed: {init_result['message']}")
        return
    
    logger.info("✅ Pipeline initialized successfully")
    
    # Find documents to ingest
    sample_documents = []
    
    # Look for the PDF report
    pdf_path = Path("MG-5025A_Gearbox_Wear_Investigation_Report.pdf")
    if pdf_path.exists():
        sample_documents.append(pdf_path)
        logger.info(f"Found sample PDF: {pdf_path}")
    
    # Look for other documents in data directory
    if settings.DATA_DIR.exists():
        for ext in settings.SUPPORTED_EXTENSIONS:
            sample_documents.extend(settings.DATA_DIR.glob(f"*{ext}"))
    
    if not sample_documents:
        logger.warning("No documents found for ingestion")
        logger.info("Please place documents in the 'data' directory or ensure MG-5025A_Gearbox_Wear_Investigation_Report.pdf exists")
    else:
        # Ingest documents
        logger.info(f"📥 Ingesting {len(sample_documents)} documents")
        ingest_result = pipeline.ingest_documents(sample_documents)
        
        if ingest_result["success"]:
            logger.info(f"✅ Successfully ingested {ingest_result['processed_files']} documents")
            logger.info(f"📊 Total chunks created: {ingest_result['total_chunks']}")
        else:
            logger.error(f"❌ Ingestion failed: {ingest_result.get('message', 'Unknown error')}")
            return
    
    # Test queries
    test_queries = [
        "What are the main findings from the gearbox wear investigation?",
        "What measurements were taken during the analysis?",
        "What are the recommendations for maintenance?",
        "Show me the timeline of the failure investigation",
        "What caused the gear wear in this case?"
    ]
    
    logger.info("🔍 Testing sample queries")
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\\n--- Query {i}: {query} ---")
        
        result = pipeline.query(query)
        
        if result["success"]:
            logger.info("✅ Query successful")
            print(f"\\n**Response:**\\n{result['response'][:300]}...")
            print(f"\\n**Agent Used:** {result['metadata']['agent_type']}")
            print(f"**Sources:** {result['metadata']['sources_used']}")
        else:
            logger.error(f"❌ Query failed: {result['message']}")
    
    # Get final stats
    stats_result = pipeline.get_stats()
    if stats_result["success"]:
        stats = stats_result["stats"]
        logger.info(f"\\n📊 Final System Stats:")
        logger.info(f"Total Documents: {stats['total_documents']}")
        logger.info(f"Index Info: {stats['index_info']}")
    
    logger.info("\\n🎉 Full pipeline execution completed!")
    logger.info("💡 You can now use the Gradio interface by running: python -m rag_app.ui_gradio")

if __name__ == "__main__":
    run_full_pipeline()
