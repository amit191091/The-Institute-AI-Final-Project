"""
Test script to verify RAGAS works with Google API.
"""

import os
import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

def test_google_api_setup():
    """Test if Google API is properly configured for RAGAS."""
    
    print("Testing Google API setup for RAGAS...")
    print("=" * 40)
    
    # Check API key
    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key:
        print("❌ GOOGLE_API_KEY not set")
        print("Please set it with: $env:GOOGLE_API_KEY='your-api-key'")
        return False
    else:
        print(f"✅ GOOGLE_API_KEY found (ends with: ...{google_key[-4:]})")
    
    # Test Google LangChain imports
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        print("✅ langchain-google-genai imported successfully")
    except ImportError as e:
        print(f"❌ langchain-google-genai import failed: {e}")
        print("Install with: pip install langchain-google-genai")
        return False
    
    # Test RAGAS imports
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        print("✅ RAGAS imported successfully")
    except ImportError as e:
        print(f"❌ RAGAS import failed: {e}")
        print("Install with: pip install ragas")
        return False
    
    # Test Google LLM setup
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0
        )
        # Test with a simple prompt
        response = llm.invoke("Say 'test successful'")
        print(f"✅ Google LLM test: {response.content}")
    except Exception as e:
        print(f"❌ Google LLM test failed: {e}")
        return False
    
    # Test Google embeddings
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )
        # Test with simple text
        embed_result = embeddings.embed_query("test embedding")
        print(f"✅ Google embeddings test: {len(embed_result)} dimensions")
    except Exception as e:
        print(f"❌ Google embeddings test failed: {e}")
        return False
    
    print("\n🎉 All Google API tests passed!")
    return True

def test_ragas_with_google():
    """Test RAGAS evaluation with Google API."""
    
    if not test_google_api_setup():
        return False
    
    print("\nTesting RAGAS evaluation with Google API...")
    print("=" * 45)
    
    try:
        from app.eval_ragas import run_eval
        
        # Sample dataset for testing
        test_dataset = {
            "question": ["What is the capital of France?"],
            "answer": ["The capital of France is Paris."],
            "contexts": [["France is a country in Europe. Paris is the capital and largest city of France."]],
            "ground_truth": ["Paris"],
            "ground_truths": [["Paris"]],
        }
        
        print("Running RAGAS evaluation...")
        metrics = run_eval(test_dataset)
        
        print("\n✅ RAGAS Evaluation Results:")
        print(f"   Faithfulness: {metrics.get('faithfulness', 'n/a')}")
        print(f"   Answer Relevancy: {metrics.get('answer_relevancy', 'n/a')}")
        print(f"   Context Precision: {metrics.get('context_precision', 'n/a')}")
        print(f"   Context Recall: {metrics.get('context_recall', 'n/a')}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAGAS evaluation failed: {e}")
        return False

if __name__ == "__main__":
    print("RAGAS + Google API Test")
    print("=" * 25)
    
    if test_ragas_with_google():
        print("\n🎉 SUCCESS: RAGAS is working with Google API!")
        print("\nYou can now use RAGAS evaluation in your RAG system.")
        print("Just make sure GOOGLE_API_KEY is set when running your app.")
    else:
        print("\n❌ SETUP INCOMPLETE")
        print("\nRequired steps:")
        print("1. Set GOOGLE_API_KEY environment variable")
        print("2. Install: pip install langchain-google-genai ragas datasets")
        print("3. Ensure you have Google AI Studio API access")
