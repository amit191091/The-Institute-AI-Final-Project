#!/usr/bin/env python3
"""
Test the conversation context improvements for fixing cross-document contamination.
Simulates the exact scenario described by Gemini where ambiguous questions get answered 
from the wrong document due to lack of conversational context.
"""

from app.conversation_context import conversation_context
from app.retrieve import query_analyzer

def test_conversation_context():
    print("=== Testing Conversation Context for Cross-Document Contamination Fix ===\n")
    
    # Reset conversation context
    conversation_context.document_history.clear()
    conversation_context.recent_questions.clear()
    conversation_context.topic_keywords.clear()
    
    # Simulate a conversation about gear wear (like user had been asking about)
    print("1. Simulating previous gear wear questions...")
    gear_questions = [
        "What caused the gear wear failure?",
        "What was the operating speed during the gear tests?", 
        "How did the tooth wear progress over time?"
    ]
    
    for q in gear_questions:
        conversation_context.add_interaction("Gear wear Failure.pdf", q, confidence=0.9)
        print(f"   Added: '{q}'")
    
    # Also add some bearing interactions to create ambiguity
    print("\n   Adding some bearing questions to create ambiguity...")
    conversation_context.add_interaction("Sliding_Bearing_Failure_Investigation_Report.pdf", 
                                       "What was the bearing temperature?", confidence=0.8)
    
    print(f"\nConversation summary: {conversation_context.get_context_summary()}")
    
    # Now test the problematic ambiguous question
    ambiguous_question = "how much the rms changed between the tests"
    print(f"\n2. Testing ambiguous question: '{ambiguous_question}'")
    
    # Test query analysis with conversation context
    qa_result = query_analyzer(ambiguous_question)
    print(f"Query analysis result:")
    print(f"   Filters: {qa_result.get('filters', {})}")
    print(f"   Conversation info: {qa_result.get('intent', {}).get('conversation_info', {})}")
    
    # Test ambiguity detection
    ambiguity_info = conversation_context.detect_ambiguous_query(ambiguous_question)
    print(f"\nAmbiguity detection:")
    print(f"   Is ambiguous: {ambiguity_info['is_ambiguous']}")
    print(f"   Preferred document: {ambiguity_info['preferred_document']}")
    print(f"   Needs disambiguation: {ambiguity_info.get('needs_disambiguation', False)}")
    
    # Test bias determination
    bias_doc = conversation_context.should_bias_retrieval(ambiguous_question)
    print(f"   Should bias toward: {bias_doc}")
    
    # Test disambiguation prompt generation
    if ambiguity_info.get('needs_disambiguation'):
        disambiguation = conversation_context.generate_disambiguation_prompt(
            ambiguous_question, 
            ambiguity_info['available_documents']
        )
        print(f"\nDisambiguation prompt:\n   {disambiguation}")
    
    print(f"\n3. Expected behavior:")
    print(f"   ✅ System should bias retrieval toward 'Gear wear Failure.pdf'")
    print(f"   ✅ If multiple docs found, should ask for clarification")
    print(f"   ✅ Should NOT default to bearing document even if it has RMS data")
    
    # Test with a specific question that should NOT be ambiguous
    specific_question = "how much did the RMS change for the gear wear tests"
    print(f"\n4. Testing specific question: '{specific_question}'")
    specific_ambiguity = conversation_context.detect_ambiguous_query(specific_question)
    print(f"   Is ambiguous: {specific_ambiguity['is_ambiguous']} (should be False)")
    print(f"   Has specifics: gear tests mentioned explicitly")
    
    print(f"\n=== Test Results ===")
    if qa_result.get('filters', {}).get('conversation_bias'):
        print("✅ SUCCESS: Conversation bias applied to query analysis")
    else:
        print("❌ ISSUE: No conversation bias detected in filters")
    
    if ambiguity_info['is_ambiguous'] and ambiguity_info['preferred_document']:
        print("✅ SUCCESS: Ambiguous query detected with preferred document")
    else:
        print("❌ ISSUE: Ambiguity detection not working properly")
    
    if not specific_ambiguity['is_ambiguous']:
        print("✅ SUCCESS: Specific question correctly identified as non-ambiguous")
    else:
        print("❌ ISSUE: Specific question incorrectly flagged as ambiguous")

if __name__ == "__main__":
    test_conversation_context()
