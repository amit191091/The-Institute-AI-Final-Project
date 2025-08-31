# Conversation Context Fixes - Implementation Summary

## Problem Identified by Gemini
The original system suffered from **cross-document contamination** where ambiguous questions like "how much the rms changed between the tests" would retrieve content from the wrong document (Sliding_Bearing_Failure_Investigation_Report.pdf instead of Gear wear Failure.pdf) because:

1. **No Conversational Context**: System forgot previous questions were about gear wear
2. **No Ambiguity Detection**: Failed to recognize vague queries that could apply to multiple documents  
3. **No Disambiguation Prompting**: Didn't ask for clarification when unsure

## Solution Implemented

### 1. Conversation Context Manager (`app/conversation_context.py`)
- **ConversationContext class** tracks document interactions with timestamps
- **Document preference** based on recency and frequency of interactions
- **Topic keyword extraction** from recent questions
- **Ambiguity detection** using regex patterns for vague queries
- **Bias determination** for retrieval scoping

### 2. Enhanced Query Analysis (`app/retrieve.py`)
- **Conversation context integration** in `query_analyzer()` function
- **Conversation bias filters** added to query analysis
- **Document preference** injection into filter metadata
- **Conversation info** included in query analysis trace

### 3. Improved Retrieval Reranking (`app/retrieve.py`)
- **Conversation bias scoring** in `rerank_candidates()` function
- **Document preference boost** (+0.15 score) for preferred documents
- **Conversation-aware filtering** that respects chat history

### 4. UI Integration (`app/ui_gradio.py`)
- **Disambiguation prompting** in `on_ask()` function
- **Conversation tracking** after each answer is generated
- **Document confidence scoring** based on retrieval results
- **Automatic context management** with interaction logging

## Test Results

✅ **Conversation Bias Applied**: System now biases retrieval toward documents from recent conversation  
✅ **Ambiguity Detection Working**: Recognizes vague questions like "how much the rms changed between the tests"  
✅ **Disambiguation Prompting**: Asks for clarification when multiple documents are possible  
✅ **Specific Questions Handled**: Non-ambiguous questions like "gear wear tests" work normally  

## Example Behavior

**Before Fix:**
```
Q: "how much the rms changed between the tests"
→ Returns results from Sliding_Bearing_Failure_Investigation_Report.pdf (wrong!)
```

**After Fix:**
```
Previous context: User asking about gear wear document
Q: "how much the rms changed between the tests"
→ System detects ambiguity + has context
→ Biases retrieval toward "Gear wear Failure.pdf" 
→ OR asks: "Are you asking about Sliding_Bearing or Gear wear document?"
```

## Technical Details

### Conversation Context Tracking
- **10-minute context window** for recent interactions
- **Document scoring** by recency and confidence
- **Topic keyword** extraction and storage
- **Automatic cleanup** of old interactions

### Ambiguity Detection Patterns
```python
ambiguous_patterns = [
    r'\b(how much|what was|what is|how did|when did)\b.*\b(change|differ|vary)\b',
    r'\b(the|this|that)\s+(test|experiment|measurement|value|result)\b',
    r'\b(between|across|during)\s+(test|experiment|trial)s?\b',
    r'\b(rms|vibration|frequency|temperature)\b.*\b(change|increase|decrease)\b'
]
```

### Conversation Bias Scoring
```python
if conversation_bias == "gear_wear" and "gear" in file_name:
    conversation_boost += 0.15  # Strong bias toward gear documents
```

## Integration Points

1. **Query Analysis**: `conversation_bias` filter added to metadata
2. **Rerank Scoring**: Conversation boost applied during document scoring  
3. **UI Handling**: Disambiguation prompts returned instead of wrong answers
4. **Context Tracking**: Document interactions logged after each Q&A session

This implementation directly addresses Gemini's analysis and provides a robust solution for maintaining topic coherence across multi-document conversations.
