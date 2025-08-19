"""
Main entry point for the Hybrid Metadata-Driven RAG System
Processes MG-5025A Gearbox Report and launches interactive Gradio interface
"""
from dotenv import load_dotenv
import os
import sys
import logging
from pathlib import Path

# Add rag_app to path
sys.path.append('.')

# Load environment
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import RAG pipeline
from rag_app.pipeline import create_pipeline, run_full_pipeline
from rag_app.ui_gradio import launch_rag_interface

def show_main_menu():
    """Display main menu for Hybrid RAG system"""
    print("\n" + "="*70)
    print("🤖 HYBRID METADATA-DRIVEN RAG SYSTEM")
    print("    Intelligent Document Analysis & Query Interface")
    print("="*70)
    print("\n📋 MENU OPTIONS:")
    print("1. 🚀 Launch Interactive Gradio RAG Interface")
    print("2. 🔍 Process MG-5025A PDF & Initialize System") 
    print("3. 💬 Interactive Query Mode (CLI)")
    print("4. 📊 System Status & Information")
    print("5. 📄 Process Custom Document")
    print("6. ❌ Exit")
    print("="*70)

def initialize_rag_system():
    """Initialize RAG system with MG-5025A PDF processing"""
    print("\n🔍 Initializing Hybrid RAG System...")
    print("📄 Processing MG-5025A Gearbox Investigation Report...")
    
    try:
        # Check if PDF exists
        pdf_path = Path("MG-5025A_Gearbox_Wear_Investigation_Report.pdf")
        if not pdf_path.exists():
            print("❌ MG-5025A PDF not found in current directory")
            return False
        
        # Copy PDF to data directory if needed
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        data_pdf_path = data_dir / pdf_path.name
        if not data_pdf_path.exists():
            import shutil
            shutil.copy2(pdf_path, data_pdf_path)
            print(f"📁 Copied PDF to data directory: {data_pdf_path}")
        
        # Initialize pipeline
        pipeline = create_pipeline()
        init_result = pipeline.initialize()
        
        if not init_result["success"]:
            print(f"❌ RAG initialization failed: {init_result['message']}")
            return False
        
        print("✅ RAG system initialized successfully")
        
        # Process the document
        process_result = pipeline.ingest_documents([data_pdf_path])
        
        if process_result["success"]:
            print("✅ MG-5025A document processed and indexed")
            print(f"📊 Processed {process_result['metadata']['total_chunks']} chunks")
            print(f"🔍 Documents processed: {process_result['metadata']['documents_processed']}")
            return True
        else:
            print(f"❌ Document processing failed: {process_result['message']}")
            return False
    
    except Exception as e:
        print(f"❌ Initialization error: {str(e)}")
        logger.error(f"RAG initialization failed: {e}")
        return False

def run_interactive_queries():
    """Run interactive query mode for RAG system"""
    print("\n💬 Interactive Query Mode")
    print("Ask questions about the MG-5025A Gearbox Investigation Report")
    print("Type 'quit', 'exit', or 'q' to return to main menu")
    print("-" * 60)
    
    try:
        pipeline = create_pipeline()
        init_result = pipeline.initialize()
        
        if not init_result["success"]:
            print(f"❌ System not initialized: {init_result['message']}")
            print("💡 Try option 2 to initialize the system first")
            return
        
        # Provide some example queries
        print("\n💡 Example queries you can try:")
        print("   • What are the main findings of the gearbox investigation?")
        print("   • What caused the gear failure?")
        print("   • What are the recommended maintenance procedures?")
        print("   • Summarize the technical specifications")
        print("   • What testing methods were used?")
        print()
        
        while True:
            query = input("🔍 Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            print("🤖 Processing your query...")
            result = pipeline.query(query)
            
            if result["success"]:
                print("\n📋 RESPONSE:")
                print("=" * 50)
                print(result["response"])
                
                metadata = result.get("metadata", {})
                if metadata:
                    print(f"\n📊 Sources: {metadata.get('sources_used', 'N/A')}")
                    print(f"🤖 Agent: {metadata.get('agent_type', 'N/A')}")
                    print(f"⏱️  Response time: {metadata.get('response_time', 'N/A')}")
                print("-" * 50)
            else:
                print(f"❌ Query failed: {result['message']}")
            
            print()  # Add spacing
    
    except Exception as e:
        print(f"❌ Interactive query error: {str(e)}")
        logger.error(f"Interactive queries failed: {e}")

def process_custom_document():
    """Process a custom document through the RAG pipeline"""
    print("\n📄 Custom Document Processing")
    print("Supported formats: PDF, DOCX, TXT")
    
    try:
        file_path = input("📁 Enter document path (or drag & drop): ").strip().strip('"')
        
        if not file_path:
            print("❌ No file path provided")
            return
        
        doc_path = Path(file_path)
        if not doc_path.exists():
            print(f"❌ File not found: {file_path}")
            return
        
        if doc_path.suffix.lower() not in ['.pdf', '.docx', '.txt']:
            print("❌ Unsupported file format. Please use PDF, DOCX, or TXT")
            return
        
        print(f"📄 Processing: {doc_path.name}")
        
        # Initialize pipeline
        pipeline = create_pipeline()
        init_result = pipeline.initialize()
        
        if not init_result["success"]:
            print(f"❌ System initialization failed: {init_result['message']}")
            return
        
        # Process document
        result = pipeline.ingest_documents([doc_path])
        
        if result["success"]:
            print("✅ Document processed successfully")
            metadata = result.get("metadata", {})
            print(f"📊 Chunks created: {metadata.get('total_chunks', 'N/A')}")
            print(f"📄 Documents processed: {metadata.get('documents_processed', 'N/A')}")
            print(f"🏷️  Processing completed successfully")
        else:
            print(f"❌ Processing failed: {result['message']}")
    
    except Exception as e:
        print(f"❌ Custom document processing error: {str(e)}")
        logger.error(f"Custom document processing failed: {e}")

def show_system_info():
    """Display system information and status"""
    print("\n📊 HYBRID RAG SYSTEM STATUS")
    print("="*50)
    
    # Check API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    print(f"🔑 OpenAI API Key: {'✅ Set' if openai_key else '❌ Not set'}")
    
    # Check for sample documents
    pdf_path = Path("MG-5025A_Gearbox_Wear_Investigation_Report.pdf")
    print(f"📄 Sample PDF: {'✅ Found' if pdf_path.exists() else '❌ Not found'}")
    
    # Check data in data directory
    data_dir = Path("data")
    if data_dir.exists():
        docs = list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.docx")) + list(data_dir.glob("*.txt"))
        print(f"📁 Documents in data/: {len(docs)} files")
        for doc in docs[:5]:  # Show first 5
            print(f"   • {doc.name}")
        if len(docs) > 5:
            print(f"   ... and {len(docs) - 5} more")
    else:
        print("📁 Data directory: ❌ Not found")
    
    # Check directories
    dirs_to_check = [
        Path("data"),
        Path("index"), 
        Path("reports"),
        Path("rag_app")
    ]
    
    print("\n📂 Directory Status:")
    for dir_path in dirs_to_check:
        status = "✅ Exists" if dir_path.exists() else "❌ Missing"
        print(f"   {dir_path}: {status}")
    
    # Check index status
    index_dir = Path("index")
    if index_dir.exists():
        index_files = list(index_dir.iterdir())
        print(f"\n🗂️  Index Status: {len(index_files)} files")
        if index_files:
            print("   Vector database appears to be initialized")
        else:
            print("   Index directory empty - run initialization first")
    
    # System capabilities
    print("\n🚀 System Capabilities:")
    print("   ✅ Hybrid Retrieval (Dense + Sparse + Reranking)")
    print("   ✅ Multi-Agent Query Processing")
    print("   ✅ Metadata-Driven Chunking")
    print("   ✅ Gradio Web Interface")
    print("   ✅ Interactive CLI Queries")
    print("   ✅ Multiple Document Formats (PDF, DOCX, TXT)")

def main():
    """Main function for Hybrid RAG system"""
    print("🤖 Initializing Hybrid Metadata-Driven RAG System...")
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment")
        print("   Please set your OpenAI API key for full functionality")
        print("   You can create a .env file with: OPENAI_API_KEY=your_key_here")
    
    while True:
        try:
            show_main_menu()
            choice = input("\n👉 Select option (1-6): ").strip()
            
            if choice == "1":
                print("\n🚀 Launching Interactive Gradio Interface...")
                print("📱 This will open a web interface in your browser")
                launch_rag_interface()
            
            elif choice == "2":
                initialize_rag_system()
            
            elif choice == "3":
                run_interactive_queries()
            
            elif choice == "4":
                show_system_info()
            
            elif choice == "5":
                process_custom_document()
            
            elif choice == "6":
                print("\n👋 Thank you for using the Hybrid RAG System!")
                break
            
            else:
                print("❌ Invalid choice. Please select 1-6.")
            
            input("\n⏸️  Press Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            logger.error(f"Main menu error: {e}")
            input("⏸️  Press Enter to continue...")

if __name__ == "__main__":
    main()
