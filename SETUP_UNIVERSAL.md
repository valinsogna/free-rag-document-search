# 🌍 Universal RAG System - Setup Guide

Complete guide for using FREE local models and/or premium API models (OpenAI, Anthropic, Google).

## 📊 Quick Cost Comparison

| Model | Provider | Input $/1K | Output $/1K | Monthly Cost* | Quality |
|-------|----------|------------|-------------|--------------|---------|
| **Llama 3.2** | Ollama | $0 | $0 | **$0** | ⭐⭐⭐ |
| **Phi-2** | HuggingFace | $0 | $0 | **$0** | ⭐⭐⭐ |
| **Gemini Flash** | Google | $0.000075 | $0.0003 | **~$0.11** | ⭐⭐⭐⭐ |
| **Claude Haiku** | Anthropic | $0.00025 | $0.00125 | **~$0.42** | ⭐⭐⭐⭐ |
| **GPT-3.5** | OpenAI | $0.0005 | $0.0015 | **~$0.60** | ⭐⭐⭐⭐ |
| **Claude Sonnet** | Anthropic | $0.003 | $0.015 | **~$5.40** | ⭐⭐⭐⭐⭐ |
| **GPT-4o** | OpenAI | $0.005 | $0.015 | **~$6.00** | ⭐⭐⭐⭐⭐ |
| **GPT-4 Turbo** | OpenAI | $0.01 | $0.03 | **~$10.20** | ⭐⭐⭐⭐⭐ |

*Monthly estimates based on 10 queries/day

## 🚀 Quick Start

### Option 1: FREE Local Models (Recommended for Development)

```bash
# Install base dependencies
pip install -r requirements_universal.txt

# For HuggingFace models (no Ollama needed!)
pip install transformers sentence-transformers torch

# Run with HuggingFace
python rag_universal_complete.py /path/to/documents
# Select "1" for FREE models, then "2" for HuggingFace
```

### Option 2: With Ollama (Alternative Free Option)

```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.2
ollama pull nomic-embed-text

# Run
python rag_universal_complete.py /path/to/documents
# Select "1" for FREE models, then "1" for Ollama
```

### Option 3: Premium API Models

```bash
# Install API client for your provider
pip install langchain-openai      # For OpenAI
pip install langchain-anthropic   # For Anthropic  
pip install langchain-google-genai # For Google

# Run
python rag_universal_complete.py /path/to/documents
# Select provider (2-5) and enter API key when prompted
```

## 🎯 Detailed Setup by Provider

### 🆓 FREE Models

#### HuggingFace (No Installation Required!)
```bash
pip install transformers sentence-transformers torch

# That's it! Models download automatically on first use
```

**Available Models:**
- Microsoft Phi-2 (2.7B) - Fast
- Mistral-7B - Balanced
- Llama-2-7B - Quality (needs HF token)

#### Ollama (Requires Installation)
```bash
# Install from https://ollama.ai
# Then:
ollama pull llama3.2
ollama pull mistral
ollama pull phi3
```

### 💰 Commercial Models

#### OpenAI Setup
1. Get API key from: https://platform.openai.com/api-keys
2. Install: `pip install langchain-openai openai`
3. Usage:
```python
config = ModelConfigurations.OPENAI_GPT35
config.api_key = "your-api-key"
```

#### Anthropic Claude Setup
1. Get API key from: https://console.anthropic.com/
2. Install: `pip install langchain-anthropic anthropic`
3. Usage:
```python
config = ModelConfigurations.ANTHROPIC_HAIKU  # Cheapest
# or
config = ModelConfigurations.ANTHROPIC_SONNET  # Best value
config.api_key = "your-api-key"
```

#### Google Gemini Setup
1. Get API key from: https://makersuite.google.com/app/apikey
2. Install: `pip install langchain-google-genai google-generativeai`
3. Usage:
```python
config = ModelConfigurations.GOOGLE_GEMINI_FLASH  # Ultra-cheap!
config.api_key = "your-api-key"
```

## 📱 Streamlit Web UI

### Basic Usage
```bash
# Install Streamlit
pip install streamlit

# Run the universal app
streamlit run app_universal.py
```

### Features:
- ✅ Switch between models without restarting
- ✅ Real-time cost tracking
- ✅ API key management
- ✅ Side-by-side model comparison
- ✅ Export conversation history

## 💡 Usage Patterns & Best Practices

### Development Workflow
```python
# 1. Start with FREE models for development
rag = UniversalRAG(
    documents_path="./docs",
    model_config=ModelConfigurations.HUGGINGFACE_PHI
)

# 2. Test with cheap API for quality check
rag.switch_model(ModelConfigurations.GOOGLE_GEMINI_FLASH)  # $0.000075/1K

# 3. Use premium only when needed
rag.switch_model(ModelConfigurations.OPENAI_GPT4)  # $0.01/1K
```

### Cost Optimization Strategy

1. **Development**: Use FREE local models
2. **Testing**: Use Gemini Flash ($0.000075/1K) or Claude Haiku ($0.00025/1K)
3. **Production**: 
   - Standard: GPT-3.5 ($0.0005/1K)
   - Premium: Claude Sonnet ($0.003/1K) or GPT-4o ($0.005/1K)
   - Ultimate: Claude Opus ($0.015/1K) or GPT-4 Turbo ($0.01/1K)

### Hybrid Approach (Recommended)
```python
# Use free model for simple queries
simple_rag = UniversalRAG(
    documents_path="./docs",
    model_config=ModelConfigurations.HUGGINGFACE_PHI
)

# Use premium for complex queries only
complex_rag = UniversalRAG(
    documents_path="./docs",
    model_config=ModelConfigurations.ANTHROPIC_SONNET
)

# Route based on query complexity
if is_simple_question(query):
    result = simple_rag.query(query)  # FREE
else:
    result = complex_rag.query(query)  # PAID
```

## 🔧 Environment Variables

Create a `.env` file for API keys:
```bash
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
COHERE_API_KEY=...
MISTRAL_API_KEY=...
VOYAGE_API_KEY=...  # For Anthropic embeddings
```

Load in Python:
```python
from dotenv import load_dotenv
import os

load_dotenv()

config = ModelConfigurations.OPENAI_GPT35
config.api_key = os.getenv("OPENAI_API_KEY")
```

## 📈 Performance Tips

### For FREE Models
- Use GPU if available (10x faster)
- Enable 8-bit quantization to reduce memory
- Use smaller models for simple tasks

### For API Models
- Batch similar queries
- Cache common responses
- Use cheaper models first, escalate only if needed

### Vector Store Optimization
- Pre-process documents once
- Reuse vector store across sessions
- Adjust chunk_size based on document type

## 🐛 Troubleshooting

### Issue: "Out of Memory" with Free Models
```python
# Solution 1: Use smaller model
config = ModelConfigurations.HUGGINGFACE_PHI  # 2.7B instead of 7B

# Solution 2: Enable quantization
rag = UniversalRAG(
    documents_path="./docs",
    model_config=config,
    use_quantization=True  # Reduces memory by 50%
)
```

### Issue: "API Key Invalid"
```python
# Verify key format
print(f"Key starts with: {config.api_key[:10]}...")
print(f"Key length: {len(config.api_key)}")

# Test with simple call
from openai import OpenAI
client = OpenAI(api_key=config.api_key)
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Issue: "Slow Performance"
```python
# Reduce chunk size for faster processing
rag = UniversalRAG(
    documents_path="./docs",
    model_config=config,
    chunk_size=500,  # Smaller chunks
    chunk_overlap=50  # Less overlap
)

# Use fewer retrieved chunks
result = rag.query(question, k=2)  # Only 2 chunks instead of default 3
```

## 📊 Cost Monitoring

### Track costs in code:
```python
rag = UniversalRAG(documents_path="./docs", model_config=config)
rag.initialize()

# After queries
total_cost = rag.get_total_cost()
print(f"Session cost: ${total_cost:.6f}")

# Per query cost
result = rag.query("What is the main topic?")
print(f"Query cost: ${result['estimated_cost']:.6f}")
```

### Set cost limits:
```python
MAX_COST = 1.00  # $1 limit

if rag.get_total_cost() > MAX_COST:
    print("Cost limit reached! Switching to free model...")
    rag.switch_model(ModelConfigurations.HUGGINGFACE_PHI)
```

## 🎯 Model Selection Guide

### By Use Case

**Customer Support Bot:**
- Dev: HuggingFace Phi-2 (FREE)
- Prod: Claude Haiku ($0.00025/1K)

**Legal Document Analysis:**
- Dev: Ollama Mistral (FREE)
- Prod: Claude Sonnet ($0.003/1K) or GPT-4 ($0.01/1K)

**Research Assistant:**
- Dev: Ollama Llama 3.2 (FREE)
- Prod: GPT-4o ($0.005/1K)

**Simple Q&A:**
- All stages: Gemini Flash ($0.000075/1K) - So cheap it's almost free!

### By Budget

**$0/month:** Use only free models
**<$1/month:** Gemini Flash for everything
**<$5/month:** Mix of Gemini Flash + Claude Haiku
**<$20/month:** GPT-3.5 for most, Claude Sonnet for complex
**<$50/month:** Claude Sonnet as primary
**Unlimited:** GPT-4 Turbo or Claude Opus

## 🚀 Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements_universal.txt .
RUN pip install -r requirements_universal.txt

# Install specific providers
RUN pip install langchain-openai langchain-anthropic langchain-google-genai

COPY . .

CMD ["streamlit", "run", "app_universal.py", "--server.port=8501"]
```

### Environment-based Configuration
```python
import os

def get_model_config():
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "development":
        return ModelConfigurations.HUGGINGFACE_PHI  # FREE
    elif env == "staging":
        return ModelConfigurations.GOOGLE_GEMINI_FLASH  # Ultra-cheap
    elif env == "production":
        config = ModelConfigurations.ANTHROPIC_SONNET
        config.api_key = os.getenv("ANTHROPIC_API_KEY")
        return config
```

## 📝 License & Support

- Free models: No license required
- API models: Check provider's terms
- Code: MIT License

For issues or questions, check the provider-specific documentation:
- OpenAI: https://platform.openai.com/docs
- Anthropic: https://docs.anthropic.com
- Google: https://ai.google.dev/docs
- HuggingFace: https://huggingface.co/docs

---

**Remember:** Start FREE, upgrade only when needed! 🎉
