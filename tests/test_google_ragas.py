"""
Test script to verify RAGAS works with Google API using actual gear wear data.
"""

import os
import sys
import json
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    print("⚠️  python-dotenv not installed, using system environment variables only")

# Add app to path
sys.path.append(str(Path(__file__).parent.parent / "RAG"))

def test_google_api_setup():
    """Test if Google API is properly configured for RAGAS."""
    
    print("Testing Google API setup for RAGAS...")
    print("=" * 40)
    
    # Check if OpenAI is forced
    force_openai = os.getenv("FORCE_OPENAI_ONLY", "false").lower() == "true"
    if force_openai:
        print("⚠️  FORCE_OPENAI_ONLY is set to true - Google API will be skipped")
        return False
    
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

def load_gear_wear_data():
    """Load actual gear wear QA data for testing."""
    qa_file = Path("RAG/data/gear_wear_qa.jsonl")
    
    if not qa_file.exists():
        print(f"❌ Gear wear QA file not found: {qa_file}")
        return None
    
    questions = []
    answers = []
    ground_truths = []
    
    with open(qa_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                questions.append(data["question"])
                answers.append(data["answer"])
                ground_truths.append([data["answer"]])  # RAGAS expects list format
    
    print(f"✅ Loaded {len(questions)} questions from gear wear data")
    return {
        "question": questions,
        "answer": answers,
        "ground_truths": ground_truths,
        "contexts": [["Gear wear analysis data from MG-5025A gearbox investigation."]] * len(questions)
    }

def test_ragas_with_gear_wear_data():
    """Test RAGAS evaluation with actual gear wear data."""
    
    if not test_google_api_setup():
        return False
    
    print("\nTesting RAGAS evaluation with actual gear wear data...")
    print("=" * 55)
    
    # Load actual gear wear data
    test_dataset = load_gear_wear_data()
    if not test_dataset:
        print("❌ Could not load gear wear data")
        return False
    
    # Test with a subset of questions to avoid API costs
    sample_size = min(5, len(test_dataset["question"]))
    print(f"Testing with {sample_size} sample questions...")
    
    sample_dataset = {
        "question": test_dataset["question"][:sample_size],
        "answer": test_dataset["answer"][:sample_size],
        "ground_truths": test_dataset["ground_truths"][:sample_size],
        "contexts": test_dataset["contexts"][:sample_size]
    }
    
    # Show sample questions
    print("\nSample questions being tested:")
    for i, (q, a) in enumerate(zip(sample_dataset["question"], sample_dataset["answer"])):
        print(f"  {i+1}. Q: {q}")
        print(f"     A: {a}")
        print()
    
    try:
        from app.Evaluation_Analysis.evaluation_utils import run_eval
        
        print("Running RAGAS evaluation...")
        metrics = run_eval(sample_dataset)
        
        print("\n✅ RAGAS Evaluation Results:")
        print(f"   Faithfulness: {metrics.get('faithfulness', 'n/a')}")
        print(f"   Answer Relevancy: {metrics.get('answer_relevancy', 'n/a')}")
        print(f"   Context Precision: {metrics.get('context_precision', 'n/a')}")
        print(f"   Context Recall: {metrics.get('context_recall', 'n/a')}")
        print(f"   Table-QA Accuracy: {metrics.get('table_qa_accuracy', 'n/a')}")
        
        # Print target compliance report
        from app.Evaluation_Analysis.evaluation_utils import print_target_compliance
        print_target_compliance(metrics)
        
        # Calculate average score
        scores = []
        for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'table_qa_accuracy']:
            score = metrics.get(metric)
            if score is not None and not (isinstance(score, float) and score != score):  # Check for NaN
                scores.append(score)
        
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"\n   Average Score: {avg_score:.3f}")
            print(f"   Performance: {'🟢 Excellent' if avg_score > 0.8 else '🟡 Good' if avg_score > 0.6 else '🔴 Needs Improvement'}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAGAS evaluation failed: {e}")
        return False

def test_ragas_with_simple_data():
    """Test RAGAS evaluation with simple test data (fallback)."""
    
    if not test_google_api_setup():
        return False
    
    print("\nTesting RAGAS evaluation with simple test data...")
    print("=" * 45)
    
    try:
        from app.Evaluation_Analysis.evaluation_utils import run_eval
        
        # Simple test dataset
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
    print("RAGAS + Google API Test with Gear Wear Data")
    print("=" * 45)
    
    # Try with actual gear wear data first
    success = test_ragas_with_gear_wear_data()
    
    if not success:
        print("\nFalling back to simple test data...")
        success = test_ragas_with_simple_data()
    
    if success:
        print("\n🎉 SUCCESS: RAGAS is working with Google API!")
        print("\nYou can now use RAGAS evaluation in your RAG system.")
        print("Just make sure GOOGLE_API_KEY is set when running your app.")
    else:
        print("\n❌ SETUP INCOMPLETE")
        print("\nRequired steps:")
        print("1. Set GOOGLE_API_KEY environment variable")
        print("2. Set FORCE_OPENAI_ONLY=false in .env to enable Google API")
        print("3. Install: pip install langchain-google-genai ragas datasets")
        print("4. Ensure you have Google AI Studio API access")
