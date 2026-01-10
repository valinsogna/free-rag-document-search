# 🌐 Universal RAG Document Search System

A production-ready Retrieval-Augmented Generation (RAG) system that enables intelligent document search using **free local LLMs** or **premium API models** (OpenAI, Anthropic, Google).

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![LangChain 0.3.x](https://img.shields.io/badge/LangChain-0.3.x-green.svg)](https://python.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Key Features

- **🆓 100% Free Option** - Run entirely locally with Ollama or HuggingFace models
- **💰 Premium API Support** - OpenAI, Anthropic Claude, Google Gemini
- **🔒 Complete Privacy** - Your data never leaves your machine (with local models)
- **📄 Multi-format Support** - PDF, DOCX, DOC, TXT documents
- **🎛️ Dynamic Parameters** - Adjust Temperature, Max Tokens, and Sources (k) in real-time
- **💵 Cost Tracking** - Real-time cost estimation for paid models
- **🌐 Web UI** - Streamlit interface
- **⌨️ CLI Mode** - Full-featured command line interface
- **Offline-capable** - Works without internet connection

## 🏗️ Architecture

```
Documents → Text Extraction → Chunking → Embeddings → ChromaDB Vector Store
                                              ↓
User Query → Embedding → Similarity Search → Context → LLM → Answer
```

## 📊 Model Comparison

| Model | Provider | Cost/1K tokens | Quality | Speed |
|-------|----------|----------------|---------|-------|
| **Llama 3.2** | Ollama | $0 (FREE) | ⭐⭐⭐ | ⚡⚡⚡ |
| **Phi-2** | HuggingFace | $0 (FREE) | ⭐⭐⭐ | ⚡⚡⚡ |
| **Gemini Flash** | Google | $0.000075 | ⭐⭐⭐⭐ | ⚡⚡⚡ |
| **Claude Haiku** | Anthropic | $0.00025 | ⭐⭐⭐⭐ | ⚡⚡⚡ |
| **GPT-3.5 Turbo** | OpenAI | $0.0005 | ⭐⭐⭐⭐ | ⚡⚡ |
| **Claude 3.5 Sonnet** | Anthropic | $0.003 | ⭐⭐⭐⭐⭐ | ⚡⚡ |
| **GPT-4o** | OpenAI | $0.005 | ⭐⭐⭐⭐⭐ | ⚡⚡ |
| **GPT-4 Turbo** | OpenAI | $0.01 | ⭐⭐⭐⭐⭐ | ⚡ |

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/universal-rag-system.git
cd universal-rag-system
```

### 2. Create Environment
```bash
# Using conda (recommended)
conda env create -f environment.yml
conda activate rag

# OR using pip
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Setup Models

#### Option A: FREE Local Models (Ollama)
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.2
ollama pull nomic-embed-text
```

#### Option B: FREE Local Models (HuggingFace)
```bash
# No additional setup needed - models download automatically
pip install transformers torch sentence-transformers
```

#### Option C: Premium API Models
```bash
# Install provider-specific packages
pip install langchain-openai       # For OpenAI
pip install langchain-anthropic    # For Anthropic
pip install langchain-google-genai # For Google
```

### 4. Run the Application

#### Web UI (Recommended)
```bash
streamlit run rag_app.py
```

#### Command Line
```bash
python rag.py /path/to/your/documents
```

## 💻 Usage

### Web UI Features

1. **Select Provider & Model** - Choose from free or premium models
2. **Configure Parameters** - Adjust settings in real-time:
   - 🌡️ **Temperature** (0.0-1.0): Controls response creativity
   - 📝 **Max Tokens** (256-4096): Maximum response length
   - 📚 **Sources (k)** (1-10): Number of document chunks to retrieve
3. **Enter Documents Path** - Point to your documents folder
4. **Initialize & Query** - Start asking questions!

### CLI Commands

```bash
❓ Question: What are the main topics?
💡 Answer: Based on the documents...

# Special commands:
temp 0.7      # Change temperature
tokens 2000   # Change max tokens
k 5           # Change number of sources
params        # Show current parameters
cost          # Show session cost
help          # Show all commands
exit          # Quit
```

### Dynamic Parameters

Parameters can be modified **without restarting** the application:

| Parameter | Range | Dynamic | Description |
|-----------|-------|---------|-------------|
| Temperature | 0.0-1.0 | ✅ Yes | Response creativity |
| Max Tokens | 256-4096 | ✅ Yes | Response length limit |
| Sources (k) | 1-10 | ✅ Yes | Document chunks to retrieve |
| Chunk Size | 200-2000 | ❌ No | Requires re-initialization |
| Chunk Overlap | 0-500 | ❌ No | Requires re-initialization |

## 📁 Project Structure

```
universal-rag-system/
├── rag.py                 # Core RAG implementation with CLI
├── rag_app.py             # Streamlit Web UI
├── environment.yml        # Conda environment specification
├── requirements.txt       # Pip requirements
├── SETUP.md              # Detailed setup guide
├── README.md             # This file
├── LICENSE               # MIT License
└── .gitignore            # Git ignore rules
```

## 🔧 Configuration

### Environment Variables (for API models)

Create a `.env` file:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
```

### Programmatic Usage

```python
from rag import UniversalRAG, ModelConfigurations

# Initialize with free model
rag = UniversalRAG(
    documents_path="/path/to/docs",
    model_config=ModelConfigurations.OLLAMA_LLAMA,
    chunk_size=1000,
    chunk_overlap=200
)

# Initialize system
rag.initialize()

# Query documents
result = rag.query("What is the main topic?")
print(result["answer"])
print(f"Sources: {[s['filename'] for s in result['sources']]}")

# Update parameters dynamically
rag.update_temperature(0.7)
rag.update_max_tokens(2000)
rag.update_k(5)

# Check current parameters
params = rag.get_current_params()
print(params)

# Switch to premium model
from rag import ModelConfigurations
config = ModelConfigurations.OPENAI_GPT35.copy()
config.api_key = "your-api-key"
rag.switch_model(config)
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Ollama not found" | Install from https://ollama.ai and run `ollama serve` |
| "Model not found" | Run `ollama pull llama3.2` and `ollama pull nomic-embed-text` |
| "Out of memory" | Use smaller model (llama3.2) or reduce chunk_size |
| "API key invalid" | Verify key format and check provider dashboard |
| "Slow responses" | Reduce k value or use faster model |

## 📈 Performance Tips

### For Free Models
- Use GPU if available (10x faster)
- Start with `llama3.2` for speed, upgrade for quality
- Reduce `k` for faster responses

### For API Models
- Use Gemini Flash for ultra-cheap testing ($0.000075/1K)
- Cache common queries
- Monitor costs with built-in tracking

### Vector Store Optimization
- Process documents once, reuse vector store
- Adjust chunk_size based on document type:
  - Legal/technical: 1000-1500
  - General: 500-1000
  - Short texts: 300-500

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Valeria Insogna** - AI Engineering Portfolio Project

---

⭐ **Star this repo** if you find it useful!

**Remember:** Start FREE with local models, upgrade only when needed! 🎉
