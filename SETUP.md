# 🛠️ Setup Guide - Universal RAG System

Complete installation and configuration guide for the Universal RAG Document Search System.

## 📋 Table of Contents

- [Requirements](#-requirements)
- [Quick Setup](#-quick-setup)
- [Detailed Installation](#-detailed-installation)
- [Model Setup by Provider](#-model-setup-by-provider)
- [Running the Application](#-running-the-application)
- [Configuration Options](#-configuration-options)
- [Troubleshooting](#-troubleshooting)

---

## 📋 Requirements

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9 | 3.11 |
| RAM | 8GB | 16GB |
| Disk Space | 5GB | 15GB |
| OS | Windows/Mac/Linux | Any |

### Software Dependencies

- **Python 3.9-3.11** (3.11 recommended)
- **Conda** (recommended) or pip
- **Ollama** (for free local models) - optional
- **Git** (for cloning repository)

---

## 🚀 Quick Setup

### One-liner Setup (Conda)

```bash
# Clone, create environment, and activate
git clone https://github.com/yourusername/universal-rag-system.git && \
cd universal-rag-system && \
conda env create -f environment.yml && \
conda activate rag
```

### Minimal Setup (pip)

```bash
git clone https://github.com/yourusername/universal-rag-system.git
cd universal-rag-system
pip install langchain langchain-community chromadb streamlit pypdf python-docx
```

---

## 📦 Detailed Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/universal-rag-system.git
cd universal-rag-system
```

### Step 2: Create Python Environment

#### Option A: Using Conda (Recommended)

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate rag
```

#### Option B: Using pip + venv

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
# Check LangChain versions
pip list | grep langchain

# Expected output:
# langchain              0.3.x
# langchain-community    0.3.x
# langchain-core         0.3.x
# langchain-text-splitters 0.3.x

# Test imports
python -c "
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
print('✅ All imports successful!')
"
```

---

## 🤖 Model Setup by Provider

### 🆓 FREE: Ollama (Recommended for Local)

#### 1. Install Ollama

**macOS:**
```bash
brew install ollama
# OR download from https://ollama.ai/download/mac
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
- Download from https://ollama.ai/download/windows
- Run the installer

#### 2. Download Models

```bash
# Required: Embedding model
ollama pull nomic-embed-text

# Choose ONE LLM model:
ollama pull llama3.2      # Fast, lightweight (2GB)
ollama pull mistral       # Balanced (4GB)
ollama pull llama3.1:8b   # Best quality (5GB)
ollama pull phi3          # Efficient (2.3GB)
```

#### 3. Verify Ollama

```bash
# Check installed models
ollama list

# Expected output:
# NAME                ID          SIZE
# nomic-embed-text    ...         274 MB
# llama3.2            ...         2.0 GB
```

---

### 🆓 FREE: HuggingFace (No Installation Required)

```bash
# Install HuggingFace dependencies
pip install transformers torch sentence-transformers langchain-huggingface

# Models download automatically on first use!
# No additional setup needed.
```

**Available Models:**
- `microsoft/phi-2` - Fast, efficient (2.7B parameters)
- `mistralai/Mistral-7B-v0.1` - Balanced quality
- `meta-llama/Llama-2-7b-hf` - Requires HF token

---

### 💰 PREMIUM: OpenAI

#### 1. Get API Key
- Go to https://platform.openai.com/api-keys
- Create new secret key
- Copy and save securely

#### 2. Install Package
```bash
pip install langchain-openai openai
```

#### 3. Configure
```bash
# Option A: Environment variable
export OPENAI_API_KEY="sk-..."

# Option B: .env file
echo 'OPENAI_API_KEY=sk-...' >> .env

# Option C: In code (when prompted)
```

**Available Models:**
| Model | Cost/1K Input | Cost/1K Output | Quality |
|-------|---------------|----------------|---------|
| gpt-3.5-turbo | $0.0005 | $0.0015 | ⭐⭐⭐⭐ |
| gpt-4-turbo | $0.01 | $0.03 | ⭐⭐⭐⭐⭐ |
| gpt-4o | $0.005 | $0.015 | ⭐⭐⭐⭐⭐ |

---

### 💰 PREMIUM: Anthropic Claude

#### 1. Get API Key
- Go to https://console.anthropic.com/
- Create API key
- Copy and save securely

#### 2. Install Package
```bash
pip install langchain-anthropic anthropic
```

#### 3. Configure
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Available Models:**
| Model | Cost/1K Input | Cost/1K Output | Quality |
|-------|---------------|----------------|---------|
| claude-3-haiku | $0.00025 | $0.00125 | ⭐⭐⭐⭐ |
| claude-3.5-sonnet | $0.003 | $0.015 | ⭐⭐⭐⭐⭐ |
| claude-3-opus | $0.015 | $0.075 | ⭐⭐⭐⭐⭐ |

---

### 💰 PREMIUM: Google Gemini

#### 1. Get API Key
- Go to https://makersuite.google.com/app/apikey
- Create API key
- Copy and save securely

#### 2. Install Package
```bash
pip install langchain-google-genai google-generativeai
```

#### 3. Configure
```bash
export GOOGLE_API_KEY="AIza..."
```

**Available Models:**
| Model | Cost/1K Input | Cost/1K Output | Quality |
|-------|---------------|----------------|---------|
| gemini-1.5-flash | $0.000075 | $0.0003 | ⭐⭐⭐⭐ |
| gemini-1.5-pro | $0.00125 | $0.005 | ⭐⭐⭐⭐⭐ |

---

## ▶️ Running the Application

### Web UI (Streamlit)

```bash
# Start the web interface
streamlit run rag_app.py

# Opens automatically at http://localhost:8501
```

**Web UI Features:**
- Model selection dropdown
- Real-time parameter adjustment
- Cost tracking display
- Document folder browser
- Chat history

### Command Line Interface

```bash
# Basic usage
python rag.py /path/to/your/documents
```

**CLI Commands:**
```
temp <value>    → Change temperature (0.0-1.0)
tokens <value>  → Change max tokens (1-8192)
k <value>       → Change sources/chunks (1-20)
params          → Show current parameters
cost            → Show total session cost
help            → Show available commands
exit            → Quit application
```

---

## ⚙️ Configuration Options

### Dynamic Parameters (Runtime Adjustable)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `temperature` | 0.5 | 0.0-1.0 | Response creativity |
| `max_tokens` | 2000 | 256-8192 | Max response length |
| `k` | 3 | 1-20 | Number of source chunks |

### Fixed Parameters (Set at Initialization)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `chunk_size` | 1000 | 200-2000 | Text chunk size |
| `chunk_overlap` | 200 | 0-500 | Overlap between chunks |

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys (only needed for premium models)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...

# Optional settings
CHROMA_TELEMETRY=false
ANONYMIZED_TELEMETRY=false
```

---

## 🐛 Troubleshooting

### Common Issues

#### "Ollama not found"
```bash
# Verify Ollama is installed
ollama --version

# Start Ollama service
ollama serve

# If not installed, download from https://ollama.ai
```

#### "Model not found"
```bash
# List available models
ollama list

# Pull missing model
ollama pull llama3.2
ollama pull nomic-embed-text
```

#### "Out of memory"
```python
# Use smaller model
config = ModelConfigurations.OLLAMA_LLAMA  # 2GB instead of 5GB

# Reduce chunk size
rag = UniversalRAG(
    documents_path="./docs",
    model_config=config,
    chunk_size=500,  # Smaller chunks
    chunk_overlap=50
)

# Use fewer sources
rag.update_k(2)  # Instead of default 3
```

#### "API key invalid"
```bash
# Check key format
echo $OPENAI_API_KEY | head -c 10  # Should start with "sk-"

# Verify in Python
python -c "
import os
key = os.getenv('OPENAI_API_KEY', '')
print(f'Key length: {len(key)}')
print(f'Starts with: {key[:7]}...')
"
```

#### "Import errors"
```bash
# Reinstall core packages
pip install --upgrade langchain langchain-core langchain-community

# Check versions
pip list | grep langchain
```

#### "ChromaDB errors"
```bash
# Clear existing database
rm -rf ./chroma_db

# Reinstall ChromaDB
pip install --upgrade chromadb
```

### Performance Issues

#### Slow Responses
1. Use faster model (`llama3.2` instead of `llama3.1:8b`)
2. Reduce `k` value (fewer source chunks)
3. Reduce `chunk_size` (smaller context)
4. Use GPU if available

#### High Memory Usage
1. Use quantized models
2. Reduce batch size
3. Close other applications
4. Use smaller model

---

## 📞 Getting Help

If you encounter issues not covered here:

1. Review provider documentation:
   - [LangChain Docs](https://python.langchain.com/docs/)
   - [Ollama Docs](https://ollama.ai/docs)
   - [OpenAI Docs](https://platform.openai.com/docs)
   - [Anthropic Docs](https://docs.anthropic.com)
   - [Google AI Docs](https://ai.google.dev/docs)

---

## ✅ Setup Checklist

- [ ] Python 3.11 installed
- [ ] Environment created and activated
- [ ] Core packages installed
- [ ] Model provider configured (Ollama/HuggingFace/API)
- [ ] Models downloaded (for Ollama)
- [ ] API keys set (for premium models)
- [ ] Test import successful
- [ ] Application runs without errors

---

**🎉 Setup complete! Start with FREE local models and upgrade only when needed.**
