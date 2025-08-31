#!/usr/bin/env python3
"""
Test script to validate domain filtering improvements for cross-document contamination.
Tests gear-related questions to ensure they don't pull bearing content.
"""
import os
import sys
sys.path.insert(0, ".")

from app.pipeline import ingest_and_upsert, ask, _LLM
from app.logger import get_logger

def test_domain_filtering():
    """Test that gear-related questions are properly scoped to gear documents"""
    print("🔧 Testing Domain Filtering for Cross-Document Contamination")
    print("=" * 60)
    
    # Setup
    log = get_logger()
    llm = _LLM()
    
    print("📁 Ingesting documents and building indices...")
    try:
        # Use the simple test approach - just get a basic pipeline working
        from pathlib import Path
        data_dir = Path("data")
        paths = list(data_dir.glob("*.pdf"))
        print(f"Found {len(paths)} PDF files to ingest")
        
        if not paths:
            print("❌ No PDF files found in data directory")
            return False
            
        docs, hybrid, debug = ingest_and_upsert(paths)
        print(f"✅ Indexed {len(docs)} documents")
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test a single gear question to check for contamination
    test_question = "What are the main causes of gear wear?"
    
    print(f"\n🧪 Testing Question: {test_question}")
    print("-" * 40)
    
    try:
        answer = ask(docs, hybrid, llm, test_question)
        
        # Check for bearing contamination keywords
        bearing_keywords = [
            "sliding bearing", "journal bearing", "thrust bearing",
            "bearing race", "bearing cage", "sliding", "journal"
        ]
        
        contamination_found = []
        answer_lower = answer.lower()
        
        for keyword in bearing_keywords:
            if keyword in answer_lower:
                contamination_found.append(keyword)
        
        print(f"📝 Answer: {answer}")
        print(f"\n🔍 Checking for bearing contamination...")
        
        if contamination_found:
            print(f"   ⚠️  CONTAMINATION DETECTED: {', '.join(contamination_found)}")
            print("\n❌ DOMAIN FILTERING NEEDS IMPROVEMENT")
            return False
        else:
            print(f"   ✅ No bearing-related contamination found")
            print("\n✅ DOMAIN FILTERING WORKING")
            return True
                
    except Exception as e:
        print(f"   ❌ Error processing question: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_domain_filtering()
    sys.exit(0 if success else 1)
