# Using Google API with RAGAS - Simplified Setup

## Why Google API is Better for RAGAS

1. **FREE**: Google AI Studio provides generous free tier
2. **FAST**: Gemini models are optimized for evaluation tasks  
3. **COST-EFFECTIVE**: Much cheaper than OpenAI for evaluation
4. **RELIABLE**: Good API stability and rate limits

## Simple Setup Steps

### 1. Get Google API Key
- Visit: https://makersuite.google.com/
- Sign in with Google account
- Click "Create API Key"
- Copy the key (starts with "AIza...")

### 2. Install Dependencies
```bash
pip install langchain-google-genai ragas datasets
```

### 3. Set Environment Variable
```bash
# PowerShell (recommended - works immediately)
$env:GOOGLE_API_KEY="AIzaSyC-your-actual-key-here"

# Or add to your .env file for persistence
echo "GOOGLE_API_KEY=AIzaSyC-your-actual-key-here" >> .env
```

### 4. Run Evaluation
RAGAS will automatically detect and use Google API:

```bash
# Run your RAG system with evaluation enabled
$env:GOOGLE_API_KEY="your-key" ; python main.py
```

## What Happens Automatically

- ✅ RAGAS detects `GOOGLE_API_KEY` environment variable
- ✅ Uses Gemini Pro for evaluation LLM tasks
- ✅ Uses Google's text-embedding-004 for similarity calculations
- ✅ All costs go to Google (much cheaper than OpenAI)
- ✅ Your main RAG system works exactly the same

## Expected Results

When working, you'll see:
```
Using RAGAS with default configuration (from environment variables)
Faithfulness: 0.85
Answer relevancy: 0.92
Context precision: 0.78
Context recall: 0.81
```

## Cost Comparison

| Provider | Model | Cost per 1K tokens |
|----------|-------|-------------------|
| OpenAI | GPT-3.5-turbo | $0.0015 |
| Google | Gemini Pro | $0.00025 |
| **Savings** | | **~83% cheaper** |

For 100 evaluation queries: Google ~$0.50 vs OpenAI ~$3.00

## Troubleshooting

### Error: "No LLM available for RAGAS"
```bash
# Make sure the key is set correctly
echo $env:GOOGLE_API_KEY
# Should show: AIzaSyC-your-key...
```

### Error: "Invalid API key"
- Check your API key is correct (starts with "AIza")
- Verify it's enabled in Google AI Studio
- Make sure you have API access enabled

### Error: "Quota exceeded"
- Google has generous free limits for development
- Check usage in Google AI Studio console
- Consider upgrading if needed (still much cheaper than OpenAI)

## Alternative: OpenAI Fallback

If Google API doesn't work, you can use OpenAI as fallback:
```bash
$env:OPENAI_API_KEY="sk-your-openai-key"
```

RAGAS will automatically try Google first, then fallback to OpenAI.
