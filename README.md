# free-rag-document-search

A production-ready Retrieval-Augmented Generation (RAG) system that enables natural language search across your local documents using completely free, open-source LLMs via Ollama.

✨ Features:
- 100% free - no API costs ever
- Complete privacy - everything runs locally  
- Multi-format support (PDF, DOCX, TXT)
- Languages: it's supposed to interacte in italian, but can understand english.
- Vector embeddings with ChromaDB
- Web UI with Streamlit
- Offline-capable

🛠️ Built With: Python | LangChain | Ollama | ChromaDB

🎯 AI Engineering portfolio project


## 1. Installa Ollama da https://ollama.ai

## 2. Scarica modelli (GRATIS!)
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## 3. Installa dipendenze Python
Con pip:
```bash
pip install -r requirements.txt
```

Oppure runna il file environment.yml se usi conda+Pip:
```bash
conda env create -f environment.yml
conda activate rag

## 4. Esegui!
```bash
python rag_free_ollama.py
```
## OPPURE con interfaccia web:
```bash
streamlit run app_streamlit_free.py
```
