# 🚀 Local RAG Document Search System

A production-ready Retrieval-Augmented Generation (RAG) system for intelligent document search using free, open-source LLMs.

## ✨ Key Features

- **100% Free** - No API costs, everything runs locally
- **Complete Privacy** - Your data never leaves your machine
- **Multi-format Support** - PDF, DOCX, TXT documents
- **Multilingual** - Italian-focused with English support
- **Modern Stack** - LangChain, ChromaDB, Streamlit UI
- **Offline-capable** - Works without internet connection

## 🏗️ Architecture

```
Documents → Text Extraction → Chunking → Embeddings → Vector DB
                                              ↓
User Query → Embedding → Similarity Search → Context → LLM → Answer
```

## 📋 Requirements

- Python 3.8-3.11 (3.11 recommended)
- 8GB+ RAM (16GB for larger models)
- 10GB disk space for models
- Windows/Mac/Linux

## 🚀 Quick Start

### Option 1: Using Ollama (Recommended)

1. **Install Ollama**
   ```bash
   # Mac
   brew install ollama
   
   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Windows: Download from https://ollama.ai/download/windows
   ```

2. **Download Models**
   ```bash
   ollama pull nomic-embed-text  # For embeddings
   ollama pull llama3.2          # For generation (2GB)
   ```

3. **Setup Python Environment**
   ```bash
   # Using conda (recommended)
   conda env create -f environment.yml
   conda activate rag
   
   # OR using pip
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   # CLI version
   python rag_free_ollama.py /path/to/your/documents
   
   # Web UI version
   streamlit run app_streamlit_free.py
   ```

### Option 2: Using HuggingFace Models (Alternative - No Ollama needed)

See the [Alternative Setup](#alternative-setup-without-ollama) section below.

## 💻 Usage Examples

### CLI Mode
```bash
python rag_free_ollama.py /Users/Documents/MyPDFs

🤔 Domanda: What are the main topics in these documents?
💡 Risposta: Based on the documents...
📚 Fonti: document1.pdf, document2.docx
```

### Web UI Mode
1. Open browser to `http://localhost:8501`
2. Select documents folder
3. Choose model
4. Start asking questions!

## 🎮 Available Models

| Model | Size | RAM | Speed | Quality | Best For |
|-------|------|-----|-------|---------|----------|
| `llama3.2` | 2GB | 8GB | ⚡⚡⚡ | ⭐⭐⭐ | Quick responses |
| `mistral` | 4GB | 8GB | ⚡⚡ | ⭐⭐⭐⭐ | Balanced |
| `llama3.1:8b` | 5GB | 16GB | ⚡⚡ | ⭐⭐⭐⭐⭐ | Best quality |
| `phi3` | 2.3GB | 8GB | ⚡⚡⚡ | ⭐⭐⭐ | Efficient |

## 🔧 Configuration

Edit `rag_free_ollama.py` to customize:

```python
# Change default model
MODEL_NAME = "mistral"  # or "llama3.2", "phi3", etc.

# Adjust chunk size for performance
chunk_size=500  # Smaller = faster but less context
chunk_overlap=100

# Number of relevant chunks to retrieve
search_kwargs={"k": 3}  # Lower = faster but less context
```

## Alternative Setup Without Ollama

You can use HuggingFace models directly without installing Ollama:

```python
# See alternative_rag_huggingface.py (to be created)
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Uses models directly from HuggingFace Hub
# No Ollama installation required!
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Ollama not found" | Ensure Ollama is installed and run `ollama serve` |
| "Model not found" | Run `ollama pull llama3.2` |
| "Out of memory" | Use smaller model or reduce chunk_size |
| "Too slow" | Use llama3.2 instead of larger models |

## 📁 Project Structure

```
.
├── rag_free_ollama.py      # Core RAG implementation
├── app_streamlit_free.py   # Web UI
├── environment.yml         # Conda environment
├── requirements.txt        # Pip requirements
├── documents/              # Your documents here
└── chroma_db/             # Vector database storage
```

## 🚀 Future Improvements

- [ ] Add HuggingFace transformers support (no Ollama needed)
- [ ] Implement conversation memory
- [ ] Add more document formats (Excel, CSV, JSON)
- [ ] Create Docker container
- [ ] Add authentication for web UI
- [ ] Implement document upload via UI

## 📜 License

MIT License - See LICENSE.txt

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📧 Contact

AI Engineering Portfolio Project by Valeria

---

**Note**: This is a portfolio project demonstrating RAG implementation with completely free, local LLMs. No API keys or costs required!
