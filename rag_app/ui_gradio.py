"""
Gradio-based user interface for the RAG system
"""
import gradio as gr
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

from .agents import MultiAgentSystem
from .indexing import HybridIndex
from .retrieve import HybridRetriever
from .validate import DocumentValidator
from .config import settings

# Import existing gear analysis components
import sys
sys.path.append('..')
try:
    from vibration_analysis import VibrationAnalysis
    from picture_analysis_menu import analyze_gear_images
    GEAR_ANALYSIS_AVAILABLE = True
except ImportError:
    GEAR_ANALYSIS_AVAILABLE = False
    logging.warning("Gear analysis modules not available")

logger = logging.getLogger(__name__)

class RAGInterface:
    """
    Gradio interface for the hybrid RAG system
    """
    
    def __init__(self):
        self.index: Optional[HybridIndex] = None
        self.retriever: Optional[HybridRetriever] = None
        self.agent_system: Optional[MultiAgentSystem] = None
        self.validator = DocumentValidator()
        
        # Gear analysis components
        self.vibration_analyzer = VibrationAnalysis() if GEAR_ANALYSIS_AVAILABLE else None
        
        # Interface state
        self.indexed_files = []
        self.last_query_result = None
    
    def initialize_system(self):
        """Initialize the RAG system components"""
        try:
            self.index = HybridIndex()
            
            # Try to load existing index
            if self.index.load_index():
                logger.info("Loaded existing index")
                self.indexed_files = self.index.index_info.get("indexed_files", [])
            else:
                logger.info("No existing index found, starting fresh")
            
            self.retriever = HybridRetriever(self.index)
            self.agent_system = MultiAgentSystem(self.retriever)
            
            return "✅ RAG system initialized successfully"
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return f"❌ Failed to initialize system: {str(e)}"
    
    def upload_documents(self, files) -> str:
        """Handle document upload and indexing"""
        if not files:
            return "❌ No files provided"
        
        if not self.index:
            return "❌ System not initialized. Please initialize first."
        
        try:
            results = []
            successful_uploads = 0
            
            for file in files:
                file_path = Path(file.name)
                
                # Validate document
                validation_result = self.validator.validate_document(file_path)
                
                if validation_result["is_valid"]:
                    # Process and index document
                    from .loaders import get_document_loader
                    from .chunking import create_chunker
                    from langchain.docstore.document import Document
                    
                    loader = get_document_loader()
                    chunker = create_chunker()
                    
                    # Load document
                    elements = loader.load_elements(file_path)
                    
                    # Extract client/case IDs from filename or content
                    file_stem = file_path.stem
                    client_id = self._extract_client_id(file_stem)
                    case_id = self._extract_case_id(file_stem)
                    
                    # Chunk document
                    chunks = chunker.chunk_document(
                        elements, 
                        str(file_path), 
                        client_id=client_id, 
                        case_id=case_id
                    )
                    
                    # Convert to LangChain documents
                    documents = [
                        Document(page_content=chunk["page_content"], metadata=chunk["metadata"])
                        for chunk in chunks
                    ]
                    
                    # Add to index
                    self.index.add_documents(documents)
                    
                    self.indexed_files.append(str(file_path))
                    successful_uploads += 1
                    
                    results.append(f"✅ {file_path.name}: Indexed successfully ({len(chunks)} chunks)")
                else:
                    error_msg = "; ".join(validation_result["errors"][:3])
                    results.append(f"❌ {file_path.name}: Validation failed - {error_msg}")
            
            # Save index
            if successful_uploads > 0:
                self.index.save_index()
                
            summary = f"\\n\\n📊 Summary: {successful_uploads}/{len(files)} files indexed successfully"
            
            return "\\n".join(results) + summary
            
        except Exception as e:
            logger.error(f"Document upload failed: {e}")
            return f"❌ Upload failed: {str(e)}"
    
    def query_documents(self, query: str, k_contexts: int = 8) -> tuple[str, str]:
        """Handle document queries"""
        if not query.strip():
            return "❌ Please enter a query", ""
        
        if not self.agent_system:
            return "❌ System not initialized. Please initialize first.", ""
        
        try:
            # Process query through multi-agent system
            result = self.agent_system.process_query(query, context_k=k_contexts)
            
            self.last_query_result = result
            
            # Format response
            response = result["response"]
            
            # Format metadata
            metadata = f"""
**Query Processing Details:**
- Agent Used: {result.get('agent_type', 'unknown').upper()}
- Sources Used: {result.get('sources_used', 0)}
- Unique Documents: {result.get('unique_documents', 0)}
- Routing Reason: {result.get('routing_info', {}).get('reason', 'N/A')}
"""
            
            if result.get('context_types'):
                context_summary = ", ".join(set(result['context_types']))
                metadata += f"- Content Types: {context_summary}\\n"
            
            return response, metadata
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return f"❌ Query failed: {str(e)}", ""
    
    def integrated_analysis(self, query: str, include_vibration: bool = False, 
                          include_pictures: bool = False) -> tuple[str, str]:
        """Perform integrated analysis combining RAG with gear analysis"""
        # For now, just use standard document query to avoid integration issues
        if not self.agent_system:
            return "❌ System not initialized.", ""
        
        try:
            # Use standard document query for reliability
            response, metadata = self.query_documents(query)
            
            # Add integration note
            integration_note = "\\n\\n*Note: Integrated analysis with live vibration/picture data temporarily disabled for stability.*"
            response += integration_note
            
            # Format metadata for integrated analysis
            integrated_metadata = f"""
**Integrated Analysis Details:**
- Integration Mode: Document Analysis Only
- Picture Data Available: {include_pictures}
- Vibration Data Available: {include_vibration}
- Status: Using document-based analysis for reliability
"""
            
            return response, integrated_metadata
            
        except Exception as e:
            logger.error(f"Integrated analysis failed: {e}")
            return f"❌ Analysis failed: {str(e)}", ""
    
    def get_system_stats(self) -> str:
        """Get system statistics"""
        if not self.index:
            return "❌ System not initialized"
        
        try:
            stats = self.index.get_stats()
            
            return f"""
📊 **System Statistics:**

**Index Status:**
- Total Documents: {stats['total_documents']}
- Dense Index Size: {stats['dense_index_size']}
- Sparse Index Size: {stats['sparse_index_size']}

**Indexed Files:**
{chr(10).join(f"• {file}" for file in self.indexed_files[-10:])}
{f"... and {len(self.indexed_files) - 10} more" if len(self.indexed_files) > 10 else ""}

**Metadata Fields:**
{', '.join(stats['metadata_columns']) if stats['metadata_columns'] else 'None available'}

**Last Updated:** {stats['index_info'].get('last_updated', 'Never')}
"""
        except Exception as e:
            return f"❌ Error getting stats: {str(e)}"
    
    def validate_document_upload(self, file) -> str:
        """Validate a document before upload"""
        if not file:
            return "❌ No file provided"
        
        try:
            file_path = Path(file.name)
            validation_result = self.validator.validate_document(file_path)
            
            status = "✅ Valid" if validation_result["is_valid"] else "❌ Invalid"
            
            result_text = f"**{status} - {file_path.name}**\\n\\n"
            
            # Add metadata
            metadata = validation_result["metadata"]
            result_text += f"**Document Info:**\\n"
            result_text += f"• Pages: {metadata.get('page_count', 'Unknown')}\\n"
            result_text += f"• Elements: {metadata.get('element_count', 'Unknown')}\\n"
            result_text += f"• Has Tables: {metadata.get('has_tables', False)}\\n"
            result_text += f"• Has Figures: {metadata.get('has_figures', False)}\\n\\n"
            
            # Add errors
            if validation_result["errors"]:
                result_text += "**❌ Errors:**\\n"
                for error in validation_result["errors"]:
                    result_text += f"• {error}\\n"
                result_text += "\\n"
            
            # Add warnings
            if validation_result["warnings"]:
                result_text += "**⚠️ Warnings:**\\n"
                for warning in validation_result["warnings"]:
                    result_text += f"• {warning}\\n"
                result_text += "\\n"
            
            # Add recommendations
            if validation_result["recommendations"]:
                result_text += "**💡 Recommendations:**\\n"
                for rec in validation_result["recommendations"]:
                    result_text += f"• {rec}\\n"
            
            return result_text
            
        except Exception as e:
            return f"❌ Validation error: {str(e)}"
    
    def _extract_client_id(self, filename: str) -> Optional[str]:
        """Extract client ID from filename"""
        import re
        match = re.search(r'client[-_]?(\w+)', filename.lower())
        return match.group(1) if match else None
    
    def _extract_case_id(self, filename: str) -> Optional[str]:
        """Extract case ID from filename"""
        import re
        match = re.search(r'case[-_]?(\w+)', filename.lower())
        return match.group(1) if match else None
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        with gr.Blocks(title="Hybrid RAG for Failure Analysis", theme=gr.themes.Soft()) as interface:
            
            gr.Markdown("""
            # 🔧 Hybrid RAG for Gear & Bearing Failure Analysis
            
            **Metadata-Driven Document Analysis with Multi-Agent Intelligence**
            
            This system combines advanced document processing with specialized AI agents to analyze
            gear and bearing failure reports, integrating with live analysis capabilities.
            """)
            
            # System Status
            with gr.Row():
                init_btn = gr.Button("🚀 Initialize System", variant="primary")
                status_btn = gr.Button("📊 System Stats")
            
            status_output = gr.Textbox(label="System Status", interactive=False)
            
            # Document Management Tab
            with gr.Tabs():
                
                with gr.TabItem("📄 Document Management"):
                    gr.Markdown("### Upload and Validate Documents")
                    
                    with gr.Row():
                        with gr.Column():
                            file_upload = gr.File(
                                label="Select Documents",
                                file_count="multiple",
                                file_types=[".pdf", ".docx", ".doc", ".txt"]
                            )
                            upload_btn = gr.Button("📤 Upload & Index", variant="primary")
                        
                        with gr.Column():
                            validate_file = gr.File(
                                label="Validate Single Document",
                                file_count="single",
                                file_types=[".pdf", ".docx", ".doc", ".txt"]
                            )
                            validate_btn = gr.Button("🔍 Validate")
                    
                    upload_output = gr.Textbox(label="Upload Results", lines=10)
                    validation_output = gr.Markdown(label="Validation Results")
                
                with gr.TabItem("🤖 Document Query"):
                    gr.Markdown("### Query Documents with Multi-Agent Intelligence")
                    
                    with gr.Row():
                        with gr.Column():
                            query_input = gr.Textbox(
                                label="Enter Your Query",
                                placeholder="E.g., 'What are the main causes of gear wear in case MG-5025A?'",
                                lines=2
                            )
                            
                            with gr.Row():
                                k_contexts = gr.Slider(
                                    minimum=1, maximum=20, value=8, step=1,
                                    label="Number of Contexts to Retrieve"
                                )
                                query_btn = gr.Button("🔍 Query Documents", variant="primary")
                        
                        with gr.Column():
                            gr.Markdown("**Example Queries:**")
                            gr.Markdown("""
                            • *"Summarize the main findings from the gearbox investigation"*
                            • *"What measurements were taken for wear analysis?"*
                            • *"Show me the timeline of events in the failure report"*
                            • *"What are the conclusions about gear tooth wear?"*
                            """)
                    
                    query_response = gr.Textbox(label="Response", lines=10, interactive=False)
                    query_metadata = gr.Markdown(label="Query Details")
                
                with gr.TabItem("🔬 Integrated Analysis"):
                    gr.Markdown("### Combine Document Analysis with Live Gear Analysis")
                    
                    with gr.Row():
                        with gr.Column():
                            integrated_query = gr.Textbox(
                                label="Analysis Query",
                                placeholder="E.g., 'Compare documented failures with current gear condition'",
                                lines=2
                            )
                            
                            with gr.Row():
                                include_vibration = gr.Checkbox(
                                    label="Include Vibration Analysis",
                                    value=GEAR_ANALYSIS_AVAILABLE
                                )
                                include_pictures = gr.Checkbox(
                                    label="Include Picture Analysis",
                                    value=GEAR_ANALYSIS_AVAILABLE
                                )
                            
                            integrated_btn = gr.Button("🔬 Run Integrated Analysis", variant="primary")
                        
                        with gr.Column():
                            gr.Markdown("**Integration Features:**")
                            gr.Markdown("""
                            • Combines document insights with live analysis
                            • Cross-references historical and current data
                            • Provides comprehensive assessment
                            • Identifies patterns and correlations
                            """)
                    
                    integrated_response = gr.Textbox(label="Integrated Analysis Results", lines=12, interactive=False)
                    integrated_metadata = gr.Markdown(label="Analysis Details")
            
            # Event Handlers
            init_btn.click(
                fn=self.initialize_system,
                outputs=status_output
            )
            
            status_btn.click(
                fn=self.get_system_stats,
                outputs=status_output
            )
            
            upload_btn.click(
                fn=self.upload_documents,
                inputs=file_upload,
                outputs=upload_output
            )
            
            validate_btn.click(
                fn=self.validate_document_upload,
                inputs=validate_file,
                outputs=validation_output
            )
            
            query_btn.click(
                fn=self.query_documents,
                inputs=[query_input, k_contexts],
                outputs=[query_response, query_metadata]
            )
            
            integrated_btn.click(
                fn=self.integrated_analysis,
                inputs=[integrated_query, include_vibration, include_pictures],
                outputs=[integrated_response, integrated_metadata]
            )
            
            # Example queries
            examples = gr.Examples(
                examples=[
                    ["What caused the gearbox failure in the MG-5025A case?"],
                    ["Show me all measurement data from the wear investigation"],
                    ["What are the maintenance recommendations?"],
                    ["Compare wear patterns between different cases"],
                    ["What timeline led to the failure?"]
                ],
                inputs=query_input
            )
        
        return interface

def create_gradio_interface():
    """Create and return a Gradio interface without launching"""
    rag_interface = RAGInterface()
    interface = rag_interface.create_interface()
    return interface

def launch_rag_interface():
    """Launch the RAG interface"""
    rag_interface = RAGInterface()
    interface = rag_interface.create_interface()
    
    # Auto-initialize system
    init_result = rag_interface.initialize_system()
    print(f"System initialization: {init_result}")
    
    return interface.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

if __name__ == "__main__":
    launch_rag_interface()
