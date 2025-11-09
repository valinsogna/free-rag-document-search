"""
Streamlit Web Interface - 100% GRATUITA con Ollama
"""

import streamlit as st
import subprocess
from pathlib import Path
from rag_free_ollama import FreeLocalRAG

st.set_page_config(
    page_title="RAG Gratuito con Ollama",
    page_icon="🦙",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .free-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
        margin: 1rem 0;
    }
    .model-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def check_ollama():
    """Check if Ollama is installed and running"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return True, result.stdout
        return False, "Ollama non risponde"
    except FileNotFoundError:
        return False, "Ollama non installato"
    except subprocess.TimeoutExpired:
        return False, "Ollama timeout"


@st.cache_resource
def init_rag(docs_path, model_name):
    """Initialize RAG system"""
    rag = FreeLocalRAG(
        documents_path=docs_path,
        model_name=model_name
    )
    if rag.initialize():
        return rag
    return None


def main():
    # Header
    st.markdown('<div class="main-header">🦙 RAG 100% Gratuito</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align: center;"><span class="free-badge">🆓 NESSUN COSTO - TUTTO LOCALE</span></div>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurazione")
        
        # Check Ollama
        ollama_ok, ollama_msg = check_ollama()
        
        if not ollama_ok:
            st.error("❌ Ollama non trovato!")
            st.markdown("""
            ### 📥 Installa Ollama:
            1. Vai su [ollama.ai](https://ollama.ai)
            2. Scarica per il tuo OS
            3. Installa i modelli:
            ```bash
            ollama pull llama3.2
            ollama pull nomic-embed-text
            ```
            """)
            return
        
        st.success("✅ Ollama attivo!")
        
        # Model selection
        st.subheader("🤖 Seleziona Modello")
        
        with st.expander("📋 Modelli disponibili", expanded=True):
            st.text(ollama_msg)
        
        model_options = {
            "Llama 3.2 (Veloce)": "llama3.2",
            "Mistral (Bilanciato)": "mistral",
            "Llama 3.1 8B (Potente)": "llama3.1:8b",
            "Phi3 (Efficiente)": "phi3",
            "Qwen 2.5 (Multilingua)": "qwen2.5"
        }
        
        selected_model = st.selectbox(
            "Modello LLM",
            options=list(model_options.keys()),
            help="Scegli il modello da usare"
        )
        
        model_name = model_options[selected_model]
        
        # Model info cards
        st.markdown("### 💡 Info Modello")
        
        model_info = {
            "llama3.2": {"size": "2GB", "ram": "8GB", "speed": "⚡⚡⚡"},
            "mistral": {"size": "4GB", "ram": "8GB", "speed": "⚡⚡"},
            "llama3.1:8b": {"size": "5GB", "ram": "16GB", "speed": "⚡⚡"},
            "phi3": {"size": "2.3GB", "ram": "8GB", "speed": "⚡⚡⚡"},
            "qwen2.5": {"size": "varia", "ram": "8GB+", "speed": "⚡⚡"}
        }
        
        info = model_info.get(model_name, {})
        st.markdown(f"""
        <div class="model-card">
        📦 Dimensione: {info.get('size', 'N/A')}<br>
        🧠 RAM: {info.get('ram', 'N/A')}<br>
        ⚡ Velocità: {info.get('speed', 'N/A')}
        </div>
        """, unsafe_allow_html=True)
        
        # Documents path
        docs_path = st.text_input(
            "📁 Cartella documenti",
            value="./documents"
        )
        
        st.divider()
        
        # Info
        st.subheader("ℹ️ Vantaggi")
        st.success("""
        ✅ 100% Gratuito
        ✅ Privacy totale
        ✅ Funziona offline
        ✅ Nessun limite d'uso
        ✅ Open source
        """)
        
        st.divider()
        
        st.subheader("👤 Portfolio")
        st.markdown("""
        **AI Engineering Project**
        
        Sistema RAG completamente locale
        
        [GitHub](#) | [LinkedIn](#)
        """)
    
    # Main content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💰 Costo", "€0", "Gratis!")
    
    with col2:
        st.metric("🔒 Privacy", "100%", "Locale")
    
    with col3:
        st.metric("📡 Internet", "No", "Offline OK")
    
    st.divider()
    
    # Initialize button
    if st.button("🚀 Inizializza Sistema RAG", type="primary", use_container_width=True):
        if not Path(docs_path).exists():
            st.error(f"⚠️ Cartella '{docs_path}' non trovata!")
            return
        
        with st.spinner(f"🔄 Inizializzazione con {selected_model}..."):
            try:
                rag = init_rag(docs_path, model_name)
                
                if rag:
                    st.session_state.rag = rag
                    st.session_state.model_name = model_name
                    st.success(f"✅ Sistema pronto con {selected_model}!")
                    st.balloons()
                else:
                    st.error("❌ Errore nell'inizializzazione")
            except Exception as e:
                st.error(f"❌ Errore: {str(e)}")
    
    st.divider()
    
    # Query interface
    if "rag" in st.session_state:
        st.subheader(f"💬 Chatta con i tuoi documenti")
        st.caption(f"🤖 Usando: {st.session_state.model_name}")
        
        # Query input
        query = st.text_area(
            "La tua domanda:",
            placeholder="Es: Quali sono i progetti menzionati nei documenti?",
            height=100
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_button = st.button("🔍 Cerca Risposta", type="primary", use_container_width=True)
        
        with col2:
            if st.button("🗑️ Reset Chat", use_container_width=True):
                if "messages" in st.session_state:
                    st.session_state.messages = []
                st.rerun()
        
        if search_button and query:
            with st.spinner("🤔 Il modello sta pensando..."):
                try:
                    result = st.session_state.rag.query(query)
                    
                    # Save to history
                    if "messages" not in st.session_state:
                        st.session_state.messages = []
                    
                    st.session_state.messages.append({
                        "question": query,
                        "answer": result["answer"],
                        "sources": result["sources"]
                    })
                    
                    # Display latest answer
                    st.subheader("💡 Risposta")
                    st.success(result["answer"])
                    
                    # Display sources
                    st.subheader("📚 Fonti")
                    for i, source in enumerate(result["sources"], 1):
                        with st.expander(f"📄 {i}. {source['filename']}"):
                            st.text(source['content'])
                
                except Exception as e:
                    st.error(f"❌ Errore: {str(e)}")
        
        # Conversation history
        if "messages" in st.session_state and st.session_state.messages:
            st.divider()
            st.subheader("📜 Cronologia")
            
            for i, msg in enumerate(reversed(st.session_state.messages), 1):
                with st.expander(f"💬 Domanda {i}: {msg['question'][:50]}..."):
                    st.markdown(f"**Domanda:** {msg['question']}")
                    st.markdown(f"**Risposta:** {msg['answer']}")
                    st.caption(f"Fonti: {', '.join([s['filename'] for s in msg['sources']])}")
    
    else:
        # Instructions
        st.info("👆 Clicca 'Inizializza Sistema RAG' per iniziare!")
        
        # Setup guide
        with st.expander("📖 Guida Setup Rapida"):
            st.markdown("""
            ### Setup in 4 passi:
            
            1. **Installa Ollama:**
               ```bash
               # Vai su https://ollama.ai
               ```
            
            2. **Scarica modelli:**
               ```bash
               ollama pull llama3.2
               ollama pull nomic-embed-text
               ```
            
            3. **Prepara documenti:**
               ```bash
               mkdir documents
               # Copia i tuoi PDF, DOCX, TXT
               ```
            
            4. **Inizializza e usa!**
            """)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🦙 RAG Gratuito con Ollama | Nessun costo, massima privacy</p>
        <p>💻 Progetto Portfolio per AI Engineering</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()