"""
Streamlit Web Interface for Local RAG System
Clean, modern UI with both Ollama and HuggingFace support
"""

import streamlit as st
import subprocess
from pathlib import Path
import json
import os
from datetime import datetime

# Try to import both RAG implementations
try:
    from rag_local import FreeLocalRAG, check_ollama_installed
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from rag_huggingface import HuggingFaceRAG
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Local RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    
    .stats-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .success-box {
        background-color: #d4edda;
        border-color: #c3e6cb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border-color: #ffeeba;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


class RAGSystemUI:
    """Main UI class for the RAG system"""
    
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state variables"""
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = None
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'system_type' not in st.session_state:
            st.session_state.system_type = None
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">🔍 Local RAG Document Search</h1>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 Cost", "$0", "Free Forever")
        with col2:
            st.metric("🔒 Privacy", "100%", "Local Only")
        with col3:
            st.metric("🌐 Internet", "Optional", "Works Offline")
        with col4:
            gpu_status = "✅ Available" if torch.cuda.is_available() else "❌ Not Found" if HUGGINGFACE_AVAILABLE else "N/A"
            st.metric("🖥️ GPU", gpu_status)
    
    def render_sidebar(self):
        """Render the sidebar configuration"""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # System selection
            st.subheader("🤖 Select System")
            
            systems = []
            if OLLAMA_AVAILABLE:
                systems.append("Ollama (Recommended)")
            if HUGGINGFACE_AVAILABLE:
                systems.append("HuggingFace (No Ollama)")
            
            if not systems:
                st.error("No RAG systems available! Please install dependencies.")
                st.code("pip install -r requirements.txt", language="bash")
                return None
            
            system_choice = st.selectbox(
                "Choose RAG System",
                options=systems,
                help="Ollama requires separate installation, HuggingFace downloads models automatically"
            )
            
            # System-specific configuration
            if "Ollama" in system_choice and OLLAMA_AVAILABLE:
                return self.configure_ollama()
            elif "HuggingFace" in system_choice and HUGGINGFACE_AVAILABLE:
                return self.configure_huggingface()
    
    def configure_ollama(self):
        """Configure Ollama-based system"""
        config = {"type": "ollama"}
        
        # Check Ollama status
        is_installed, message = check_ollama_installed()
        
        if not is_installed:
            st.error("❌ Ollama not installed!")
            st.markdown("""
            ### 📥 Installation Steps:
            1. Visit [ollama.ai](https://ollama.ai)
            2. Download and install
            3. Run these commands:
            ```bash
            ollama pull llama3.2
            ollama pull nomic-embed-text
            ```
            """)
            return None
        
        st.success("✅ Ollama is ready!")
        
        # Show available models
        with st.expander("📋 Available Models", expanded=False):
            st.code(message, language="text")
        
        # Model selection
        st.subheader("🎯 Model Selection")
        
        models = {
            "llama3.2": "Fast & Light (2GB)",
            "mistral": "Balanced (4GB)",
            "llama3.1:8b": "Powerful (5GB)",
            "phi3": "Efficient (2.3GB)"
        }
        
        config["model"] = st.selectbox(
            "LLM Model",
            options=list(models.keys()),
            format_func=lambda x: f"{x} - {models[x]}"
        )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            config["chunk_size"] = st.slider("Chunk Size", 200, 2000, 1000, 100)
            config["chunk_overlap"] = st.slider("Chunk Overlap", 0, 500, 200, 50)
            config["num_chunks"] = st.slider("Retrieved Chunks", 1, 10, 3)
            config["temperature"] = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        
        return config
    
    def configure_huggingface(self):
        """Configure HuggingFace-based system"""
        config = {"type": "huggingface"}
        
        st.info("🤗 HuggingFace models will be downloaded on first use")
        
        # Model selection
        st.subheader("🎯 Model Selection")
        
        models = {
            "fast": "Microsoft Phi-2 (2.7B)",
            "balanced": "Mistral-7B (7B)",
            "tiny": "DialoGPT (345M)",
        }
        
        config["model_type"] = st.selectbox(
            "Model Type",
            options=list(models.keys()),
            format_func=lambda x: models[x]
        )
        
        # Embedding model
        embeddings = {
            "default": "English (80MB)",
            "multilingual": "Multilingual (420MB)",
            "large": "Large English (420MB)"
        }
        
        config["embedding_model"] = st.selectbox(
            "Embedding Model",
            options=list(embeddings.keys()),
            format_func=lambda x: embeddings[x]
        )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            config["chunk_size"] = st.slider("Chunk Size", 200, 2000, 1000, 100)
            config["chunk_overlap"] = st.slider("Chunk Overlap", 0, 500, 200, 50)
            config["num_chunks"] = st.slider("Retrieved Chunks", 1, 10, 3)
            config["use_quantization"] = st.checkbox(
                "Use 8-bit Quantization (GPU only)",
                value=torch.cuda.is_available() if HUGGINGFACE_AVAILABLE else False
            )
        
        return config
    
    def get_documents_path(self):
        """Get and validate documents path"""
        st.subheader("📁 Documents Folder")
        
        # Path input
        docs_path = st.text_input(
            "Enter full path to documents folder",
            placeholder="/Users/name/Documents/PDFs",
            help="Folder containing PDF, DOCX, and TXT files"
        )
        
        # Quick access buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📂 Desktop"):
                docs_path = os.path.expanduser("~/Desktop")
        with col2:
            if st.button("📄 Documents"):
                docs_path = os.path.expanduser("~/Documents")
        with col3:
            if st.button("⬇️ Downloads"):
                docs_path = os.path.expanduser("~/Downloads")
        
        # Validate path
        if docs_path:
            path = Path(docs_path)
            if path.exists() and path.is_dir():
                # Count files
                file_stats = self.analyze_directory(path)
                
                st.success(f"✅ Found: {file_stats['total']} files")
                
                # Show file breakdown
                cols = st.columns(4)
                with cols[0]:
                    st.metric("PDF", file_stats.get('pdf', 0))
                with cols[1]:
                    st.metric("DOCX", file_stats.get('docx', 0))
                with cols[2]:
                    st.metric("TXT", file_stats.get('txt', 0))
                with cols[3]:
                    st.metric("Other", file_stats.get('other', 0))
                
                return docs_path
            else:
                st.error(f"❌ Path not found or not a directory: {docs_path}")
        
        return None
    
    def analyze_directory(self, path: Path) -> dict:
        """Analyze directory for document files"""
        stats = {'total': 0, 'pdf': 0, 'docx': 0, 'txt': 0, 'other': 0}
        
        for file in path.rglob('*'):
            if file.is_file():
                stats['total'] += 1
                ext = file.suffix.lower()
                if ext == '.pdf':
                    stats['pdf'] += 1
                elif ext in ['.docx', '.doc']:
                    stats['docx'] += 1
                elif ext == '.txt':
                    stats['txt'] += 1
                else:
                    stats['other'] += 1
        
        return stats
    
    def initialize_rag_system(self, config: dict, docs_path: str):
        """Initialize the selected RAG system"""
        try:
            with st.spinner("🔄 Initializing RAG system... This may take a few minutes on first run."):
                if config['type'] == 'ollama':
                    rag = FreeLocalRAG(
                        documents_path=docs_path,
                        model_name=config['model'],
                        chunk_size=config.get('chunk_size', 1000),
                        chunk_overlap=config.get('chunk_overlap', 200),
                        temperature=config.get('temperature', 0.0),
                        verbose=True
                    )
                elif config['type'] == 'huggingface':
                    rag = HuggingFaceRAG(
                        documents_path=docs_path,
                        model_type=config['model_type'],
                        embedding_model=config['embedding_model'],
                        chunk_size=config.get('chunk_size', 1000),
                        chunk_overlap=config.get('chunk_overlap', 200),
                        use_quantization=config.get('use_quantization', False)
                    )
                else:
                    raise ValueError(f"Unknown system type: {config['type']}")
                
                if rag.initialize():
                    st.session_state.rag_system = rag
                    st.session_state.system_type = config['type']
                    st.session_state.initialized = True
                    st.session_state.config = config
                    st.success("✅ RAG System initialized successfully!")
                    st.balloons()
                    return True
                else:
                    st.error("❌ Failed to initialize RAG system")
                    return False
                    
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return False
    
    def render_chat_interface(self):
        """Render the chat interface"""
        st.subheader("💬 Ask Questions About Your Documents")
        
        # Display system info
        if st.session_state.system_type:
            system_name = "Ollama" if st.session_state.system_type == "ollama" else "HuggingFace"
            model_name = st.session_state.config.get('model', st.session_state.config.get('model_type', 'Unknown'))
            st.caption(f"🤖 Using: {system_name} - {model_name}")
        
        # Query input
        query = st.text_area(
            "Your question:",
            placeholder="What are the main topics discussed in the documents?",
            height=100
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            num_chunks = st.slider("Number of sources", 1, 10, 3)
        
        with col2:
            search_button = st.button("🔍 Search", type="primary")
        
        with col3:
            if st.button("🗑️ Clear History"):
                st.session_state.messages = []
                st.rerun()
        
        # Process query
        if search_button and query:
            with st.spinner("🤔 Thinking..."):
                try:
                    # Query the system
                    if st.session_state.system_type == "ollama":
                        result = st.session_state.rag_system.query(query, k=num_chunks)
                    else:
                        result = st.session_state.rag_system.query(query)
                    
                    # Save to history
                    st.session_state.messages.append({
                        "question": query,
                        "answer": result["answer"],
                        "sources": result["sources"],
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    # Display result
                    with st.container():
                        st.markdown("### 💡 Answer")
                        st.success(result["answer"])
                        
                        if result["sources"]:
                            st.markdown("### 📚 Sources")
                            for i, source in enumerate(result["sources"], 1):
                                with st.expander(f"Source {i}: {source['filename']}"):
                                    st.text(source.get('preview', source.get('content', 'No preview')))
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Display history
        if st.session_state.messages:
            st.divider()
            st.subheader("📜 Conversation History")
            
            for i, msg in enumerate(reversed(st.session_state.messages[-5:]), 1):
                with st.expander(f"[{msg['timestamp']}] {msg['question'][:50]}..."):
                    st.markdown(f"**Question:** {msg['question']}")
                    st.markdown(f"**Answer:** {msg['answer']}")
                    if msg['sources']:
                        sources = [s['filename'] for s in msg['sources']]
                        st.caption(f"Sources: {', '.join(sources)}")
    
    def run(self):
        """Main application loop"""
        self.render_header()
        
        # Sidebar configuration
        config = self.render_sidebar()
        
        if config is None:
            st.info("👈 Please configure the system in the sidebar")
            return
        
        # Get documents path
        docs_path = self.get_documents_path()
        
        if docs_path:
            # Initialize button
            if not st.session_state.initialized:
                if st.button("🚀 Initialize RAG System", type="primary"):
                    self.initialize_rag_system(config, docs_path)
            else:
                # Check if configuration changed
                if st.button("🔄 Reinitialize with New Settings"):
                    st.session_state.initialized = False
                    st.session_state.rag_system = None
                    st.rerun()
        
        # Chat interface
        if st.session_state.initialized:
            st.divider()
            self.render_chat_interface()
        elif docs_path:
            st.info("👆 Click 'Initialize RAG System' to start")
        
        # Footer
        st.divider()
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>🔍 Local RAG System | 100% Free & Private</p>
            <p>Built with LangChain, ChromaDB, and ❤️</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main entry point"""
    app = RAGSystemUI()
    app.run()


if __name__ == "__main__":
    main()
