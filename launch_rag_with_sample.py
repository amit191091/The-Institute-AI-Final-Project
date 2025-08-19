#!/usr/bin/env python3
"""
Launch RAG system with sample PDF pre-loaded
"""

import os
import sys
from pathlib import Path

# Add the rag_app to Python path
sys.path.append('rag_app')

def setup_environment():
    """Set up the environment for the RAG system"""
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OpenAI API key not found in environment variables")
        print("📝 Please set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("   or set it in your environment")
        print()
        
        # For demo purposes, you can set a placeholder
        # Note: This won't work for actual API calls, but allows testing the interface
        api_key = input("Enter your OpenAI API key (or press Enter to continue with demo mode): ").strip()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            print("🔧 Continuing in demo mode (document processing will work, but queries need API key)")
            os.environ["OPENAI_API_KEY"] = "demo-key-placeholder"
    
    return True

def main():
    """Main function to launch the RAG system"""
    
    print("🚀 Launching Hybrid RAG System with Sample PDF")
    print("=" * 60)
    
    # Setup environment
    if not setup_environment():
        return
    
    # Check if sample PDF exists
    sample_pdf = Path("MG-5025A_Gearbox_Wear_Investigation_Report.pdf")
    if not sample_pdf.exists():
        print(f"❌ Sample PDF not found: {sample_pdf}")
        return
    
    print(f"✅ Sample PDF found: {sample_pdf}")
    
    try:
        # Import and launch the RAG interface
        from rag_app.ui_gradio import launch_rag_interface
        
        print("🎯 Launching Gradio interface...")
        print("📱 This will open a web interface in your browser")
        print("🔗 URL will be: http://localhost:7860")
        print()
        print("📋 To use the sample PDF:")
        print("   1. Click 'Initialize System' button")
        print("   2. Upload the MG-5025A PDF in the Document Management tab")
        print("   3. Ask questions like: 'What caused the gearbox failure?'")
        print()
        
        # Launch the interface
        interface = launch_rag_interface()
        
    except KeyboardInterrupt:
        print("\\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error launching interface: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
