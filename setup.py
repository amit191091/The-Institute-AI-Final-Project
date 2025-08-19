"""
Setup script for the Hybrid RAG system
"""
import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directories...")
    
    directories = [
        "data",
        "index", 
        "reports",
        "rag_app/__pycache__"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ {directory}")

def check_environment():
    """Check environment setup"""
    print("🔍 Checking environment...")
    
    # Check API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    if not openai_key:
        print("⚠️  OPENAI_API_KEY not found in environment")
        print("   Please set it in your .env file or environment variables")
    else:
        print("  ✅ OPENAI_API_KEY found")
    
    if not google_key:
        print("⚠️  GOOGLE_API_KEY not found in environment")
        print("   This is optional for basic functionality")
    else:
        print("  ✅ GOOGLE_API_KEY found")

def create_env_template():
    """Create .env template if it doesn't exist"""
    env_path = Path(".env")
    
    if not env_path.exists():
        print("📝 Creating .env template...")
        
        env_content = """# API Keys for Hybrid RAG System
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
COHERE_API_KEY=your_cohere_api_key_here

# Optional: Weights & Biases for experiment tracking
WANDB_API_KEY=your_wandb_api_key_here
"""
        
        with open(env_path, "w") as f:
            f.write(env_content)
        
        print("  ✅ .env template created")
        print("  🔧 Please edit .env file with your actual API keys")
    else:
        print("  ✅ .env file already exists")

def main():
    """Run complete setup"""
    print("🚀 Setting up Hybrid RAG for Gear Analysis System")
    print("="*60)
    
    # Setup directories
    setup_directories()
    
    # Create env template
    create_env_template()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return False
    
    # Check environment
    check_environment()
    
    print("\n✅ Setup completed!")
    print("\n📚 Next steps:")
    print("1. Edit .env file with your API keys")
    print("2. Place documents in the 'data' directory")
    print("3. Run: python main.py")
    print("4. Or launch Gradio interface: python -m rag_app.ui_gradio")
    
    return True

if __name__ == "__main__":
    main()
