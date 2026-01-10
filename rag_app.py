"""
Enhanced Streamlit Interface for Universal RAG System
Compatible with LangChain 1.x (uses LCEL approach)

UPDATED: Dynamic parameter updates for Temperature, Max Tokens, and K
         Chunk Size and Chunk Overlap disabled after initialization

Supports Free (Ollama/HuggingFace) and Commercial (OpenAI/Anthropic/Google) Models
"""

import streamlit as st
from pathlib import Path
import os
from datetime import datetime

# Import the RAG system
from rag import (
    UniversalRAG, 
    ModelConfig, 
    ModelProvider,
    ModelConfigurations    
)

# Page configuration - MUST be first Streamlit call
st.set_page_config(
    page_title="Universal RAG System",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    
    .free-badge {
        background: linear-gradient(135deg, #00d084 0%, #00a046 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .paid-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .cost-display {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        color: #333;
    }
    
    .params-box {
        background: #1a1a2e;
        border: 1px solid #4a4a6a;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .params-title {
        color: #00d084;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .param-item {
        color: #e0e0e0;
        font-size: 0.85rem;
        padding: 0.2rem 0;
    }
    
    .param-fixed {
        color: #888;
        font-style: italic;
    }
    
    .provider-openai { border-left: 4px solid #10a37f; padding-left: 1rem; margin: 0.5rem 0; }
    .provider-anthropic { border-left: 4px solid #d97757; padding-left: 1rem; margin: 0.5rem 0; }
    .provider-google { border-left: 4px solid #4285f4; padding-left: 1rem; margin: 0.5rem 0; }
    .provider-free { border-left: 4px solid #00d084; padding-left: 1rem; margin: 0.5rem 0; }
    </style>
""", unsafe_allow_html=True)


def get_config_copy(config_name: str) -> ModelConfig:
    """Get a COPY of a config by name to avoid mutation"""
    config = getattr(ModelConfigurations, config_name, None)
    if config is None:
        raise ValueError(f"Unknown config: {config_name}")
    return config.copy()


class UniversalRAGUI:
    """Enhanced UI for Universal RAG System"""
    
    # Model catalog - store config NAMES, not references
    MODELS_CATALOG = {
        "🆓 Free Local Models": {
            "Ollama Llama 3.2": {
                "config_name": "OLLAMA_LLAMA",
                "description": "Fast & lightweight (2GB)",
                "requirements": "Requires Ollama installation"
            },
            "HuggingFace Phi-2": {
                "config_name": "HUGGINGFACE_PHI",
                "description": "Microsoft's efficient model (2.7B)",
                "requirements": "Auto-downloads on first use"
            }
        },
        "🟢 OpenAI": {
            "GPT-3.5 Turbo": {
                "config_name": "OPENAI_GPT35",
                "description": "Fast & affordable",
                "cost": "$0.0005/1K input, $0.0015/1K output"
            },
            "GPT-4 Turbo": {
                "config_name": "OPENAI_GPT4",
                "description": "Most capable GPT model",
                "cost": "$0.01/1K input, $0.03/1K output"
            },
            "GPT-4o": {
                "config_name": "OPENAI_GPT4O",
                "description": "Optimized GPT-4",
                "cost": "$0.005/1K input, $0.015/1K output"
            }
        },
        "🟠 Anthropic Claude": {
            "Claude 3 Haiku": {
                "config_name": "ANTHROPIC_HAIKU",
                "description": "Fastest & cheapest Claude",
                "cost": "$0.00025/1K input, $0.00125/1K output"
            },
            "Claude 3.5 Sonnet": {
                "config_name": "ANTHROPIC_SONNET",
                "description": "Best balance of speed & capability",
                "cost": "$0.003/1K input, $0.015/1K output"
            },
            "Claude 3 Opus": {
                "config_name": "ANTHROPIC_OPUS",
                "description": "Most capable Claude model",
                "cost": "$0.015/1K input, $0.075/1K output"
            }
        },
        "🔵 Google Gemini": {
            "Gemini 1.5 Flash": {
                "config_name": "GOOGLE_GEMINI_FLASH",
                "description": "Ultra-fast & cheap",
                "cost": "$0.000075/1K input, $0.0003/1K output"
            },
            "Gemini 1.5 Pro": {
                "config_name": "GOOGLE_GEMINI_PRO",
                "description": "Advanced reasoning",
                "cost": "$0.00125/1K input, $0.005/1K output"
            }
        }
    }
    
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state variables"""
        defaults = {
            'rag_system': None,
            'messages': [],
            'total_cost': 0.0,
            'initialized': False,
            'api_keys': {},
            'current_model': None,
            'docs_path': "",
            'selected_model_name': None,
            # Dynamic parameters with defaults
            'temperature': 0.5,
            'max_tokens': 2000,
            'k': 3,
            # Fixed parameters (set at init)
            'chunk_size': 1000,
            'chunk_overlap': 200
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">🌐 Universal RAG System</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center;">Free Local Models + Premium APIs in One Interface</p>', 
                   unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.session_state.current_model:
                if st.session_state.current_model.cost_per_1k_input == 0:
                    st.markdown('<span class="free-badge">FREE MODEL</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="paid-badge">PAID MODEL</span>', unsafe_allow_html=True)
        
        with col2:
            st.metric("💰 Session Cost", f"${st.session_state.total_cost:.6f}")
        
        with col3:
            if st.session_state.current_model:
                st.metric("🤖 Current Model", st.session_state.current_model.model_name.split('/')[-1])
        
        with col4:
            if st.session_state.initialized:
                st.metric("📚 Status", "Ready", "✅")
            else:
                st.metric("📚 Status", "Not Initialized", "⚠️")
    
    def render_params_display(self):
        """Render current parameters display box"""
        if st.session_state.initialized and st.session_state.rag_system:
            params = st.session_state.rag_system.get_current_params()
            
            st.markdown("""
            <div class="params-box">
                <div class="params-title">📊 Parametri Attivi</div>
                <div class="param-item">🌡️ Temperature: <b>{temp}</b></div>
                <div class="param-item">📝 Max Tokens: <b>{tokens}</b></div>
                <div class="param-item">📚 Sources (k): <b>{k}</b></div>
                <div class="param-item param-fixed">📦 Chunk Size: {chunk_size} (fisso)</div>
                <div class="param-item param-fixed">🔗 Chunk Overlap: {chunk_overlap} (fisso)</div>
            </div>
            """.format(
                temp=params['temperature'],
                tokens=params['max_tokens'],
                k=params['k'],
                chunk_size=params['chunk_size'],
                chunk_overlap=params['chunk_overlap']
            ), unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with model selection and parameters"""
        with st.sidebar:
            st.header("⚙️ Configuration")
            st.subheader("🤖 Select Model")
            
            provider_category = st.selectbox(
                "Provider Category",
                options=list(self.MODELS_CATALOG.keys())
            )
            
            models_in_category = self.MODELS_CATALOG[provider_category]
            model_display_name = st.selectbox(
                "Model",
                options=list(models_in_category.keys())
            )
            
            model_info = models_in_category[model_display_name]
            config = get_config_copy(model_info["config_name"])
            
            with st.expander("📊 Model Details", expanded=True):
                st.write(f"**Description:** {model_info['description']}")
                if 'cost' in model_info:
                    st.write(f"**Pricing:** {model_info['cost']}")
                else:
                    st.write("**Pricing:** FREE - Runs locally")
                if 'requirements' in model_info:
                    st.info(model_info['requirements'])
                
                if "Free" in provider_category:
                    st.markdown('<div class="provider-free">✅ No API costs</div>', unsafe_allow_html=True)
                elif "OpenAI" in provider_category:
                    st.markdown('<div class="provider-openai">🟢 OpenAI API</div>', unsafe_allow_html=True)
                elif "Anthropic" in provider_category:
                    st.markdown('<div class="provider-anthropic">🟠 Anthropic API</div>', unsafe_allow_html=True)
                elif "Google" in provider_category:
                    st.markdown('<div class="provider-google">🔵 Google API</div>', unsafe_allow_html=True)
            
            # API Key management
            if config.cost_per_1k_input > 0:
                st.divider()
                st.subheader("🔑 API Configuration")
                
                if config.provider == ModelProvider.OPENAI:
                    api_key = st.text_input("OpenAI API Key", type="password",
                                           value=st.session_state.api_keys.get('openai', ''))
                    if api_key:
                        st.session_state.api_keys['openai'] = api_key
                        config.api_key = api_key
                
                elif config.provider == ModelProvider.ANTHROPIC:
                    api_key = st.text_input("Anthropic API Key", type="password",
                                           value=st.session_state.api_keys.get('anthropic', ''))
                    if api_key:
                        st.session_state.api_keys['anthropic'] = api_key
                        config.api_key = api_key
                
                elif config.provider == ModelProvider.GOOGLE:
                    api_key = st.text_input("Google API Key", type="password",
                                           value=st.session_state.api_keys.get('google', ''))
                    if api_key:
                        st.session_state.api_keys['google'] = api_key
                        config.api_key = api_key
            
            # ============================================================
            # PARAMETERS SECTION
            # ============================================================
            st.divider()
            st.subheader("🎛️ Parameters")
            
            # Check if system is initialized
            is_initialized = st.session_state.initialized
            
            # ----- DYNAMIC PARAMETERS (always editable) -----
            st.markdown("**Parametri Dinamici** *(modificabili in tempo reale)*")
            
            # Temperature
            new_temperature = st.slider(
                "🌡️ Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.temperature,
                step=0.1,
                help="Controlla la creatività delle risposte (0=deterministico, 1=creativo)"
            )
            
            # Update temperature dynamically
            if new_temperature != st.session_state.temperature:
                st.session_state.temperature = new_temperature
                if is_initialized and st.session_state.rag_system:
                    st.session_state.rag_system.update_temperature(new_temperature)
            
            # Max Tokens
            new_max_tokens = st.slider(
                "📝 Max Tokens",
                min_value=256,
                max_value=4096,
                value=st.session_state.max_tokens,
                step=256,
                help="Lunghezza massima della risposta"
            )
            
            # Update max_tokens dynamically
            if new_max_tokens != st.session_state.max_tokens:
                st.session_state.max_tokens = new_max_tokens
                if is_initialized and st.session_state.rag_system:
                    st.session_state.rag_system.update_max_tokens(new_max_tokens)
            
            # K (Sources)
            new_k = st.slider(
                "📚 Sources (k)",
                min_value=1,
                max_value=10,
                value=st.session_state.k,
                step=1,
                help="Numero di chunk di documenti da recuperare per ogni query"
            )
            
            # Update k dynamically
            if new_k != st.session_state.k:
                st.session_state.k = new_k
                if is_initialized and st.session_state.rag_system:
                    st.session_state.rag_system.update_k(new_k)
            
            st.markdown("---")
            
            # ----- FIXED PARAMETERS (disabled after init) -----
            st.markdown("**Parametri Fissi** *(solo prima dell'inizializzazione)*")
            
            # Chunk Size - disabled after initialization
            chunk_size = st.slider(
                "📦 Chunk Size",
                min_value=200,
                max_value=2000,
                value=st.session_state.chunk_size,
                step=100,
                disabled=is_initialized,
                help="Dimensione dei chunk di testo (richiede re-inizializzazione)"
            )
            if not is_initialized:
                st.session_state.chunk_size = chunk_size
            
            # Chunk Overlap - disabled after initialization
            chunk_overlap = st.slider(
                "🔗 Chunk Overlap",
                min_value=0,
                max_value=500,
                value=st.session_state.chunk_overlap,
                step=50,
                disabled=is_initialized,
                help="Sovrapposizione tra chunk (richiede re-inizializzazione)"
            )
            if not is_initialized:
                st.session_state.chunk_overlap = chunk_overlap
            
            if is_initialized:
                st.caption("ℹ️ Chunk Size e Overlap sono fissi dopo l'inizializzazione. Per modificarli, clicca 'Switch Model'.")
            
            # Apply dynamic params to config
            config.temperature = st.session_state.temperature
            config.max_tokens = st.session_state.max_tokens
            
            # Display current params
            st.divider()
            self.render_params_display()
            
            # Cost estimation
            if st.button("💰 Estimate Monthly Costs"):
                self.show_cost_comparison()
            
            st.session_state.selected_model_name = config.model_name
            
            return config, st.session_state.chunk_size, st.session_state.chunk_overlap
    
    def show_cost_comparison(self):
        """Display cost comparison"""
        with st.expander("💰 Cost Comparison", expanded=True):
            queries_per_day = st.number_input("Queries per day", 1, 1000, 10)
            
            comparison_data = []
            for category, models in self.MODELS_CATALOG.items():
                for model_name, model_info in models.items():
                    config = get_config_copy(model_info["config_name"])
                    daily_cost = (500 * queries_per_day / 1000) * (config.cost_per_1k_input + config.cost_per_1k_output)
                    monthly_cost = 30 * daily_cost
                    comparison_data.append({
                        "Provider": category.split()[1] if len(category.split()) > 1 else category,
                        "Model": model_name,
                        "Monthly Cost": f"${monthly_cost:.2f}"
                    })
            
            import pandas as pd
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)
    
    def get_documents_path(self):
        """Get and validate documents path"""
        st.subheader("📂 Documents Folder")
        
        docs_path = st.text_input(
            "Enter full path to documents folder",
            value=st.session_state.docs_path,
            placeholder="/Users/name/Documents/PDFs"
        )
        
        if docs_path != st.session_state.docs_path:
            st.session_state.docs_path = docs_path
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📂 Desktop"):
                st.session_state.docs_path = os.path.expanduser("~/Desktop")
                st.rerun()
        with col2:
            if st.button("📄 Documents"):
                st.session_state.docs_path = os.path.expanduser("~/Documents")
                st.rerun()
        with col3:
            if st.button("⬇️ Downloads"):
                st.session_state.docs_path = os.path.expanduser("~/Downloads")
                st.rerun()
        
        if st.session_state.docs_path:
            path = Path(st.session_state.docs_path)
            if path.exists() and path.is_dir():
                file_stats = self.analyze_directory(path)
                st.success(f"✅ Found: {file_stats['total']} files")
                
                cols = st.columns(4)
                cols[0].metric("PDF", file_stats.get('pdf', 0))
                cols[1].metric("DOCX/DOC", file_stats.get('docx/doc', 0))
                cols[2].metric("TXT", file_stats.get('txt', 0))
                cols[3].metric("Other", file_stats.get('other', 0))
                
                supported = file_stats.get('pdf', 0) + file_stats.get('docx/doc', 0) + file_stats.get('txt', 0)
                if supported == 0:
                    st.warning("⚠️ No supported files (PDF, DOCX/DOC, TXT) found!")
                    return None
                
                return st.session_state.docs_path
            else:
                st.error(f"❌ Path not found: {st.session_state.docs_path}")
        
        return None
    
    def analyze_directory(self, path: Path) -> dict:
        """Analyze directory for document files"""
        stats = {'total': 0, 'pdf': 0, 'docx/doc': 0, 'txt': 0, 'other': 0}
        try:
            for file in path.rglob('*'):
                if file.is_file():
                    stats['total'] += 1
                    ext = file.suffix.lower()
                    if ext == '.pdf':
                        stats['pdf'] += 1
                    elif ext in ['.docx', '.doc']:
                        stats['docx/doc'] += 1
                    elif ext == '.txt':
                        stats['txt'] += 1
                    else:
                        stats['other'] += 1
        except PermissionError:
            pass
        return stats
    
    def initialize_rag_system(self, config, docs_path, chunk_size, chunk_overlap):
        """Initialize the RAG system"""
        try:
            if config.cost_per_1k_input > 0 and not config.api_key:
                st.error("🔑 API key required for this model!")
                return False
            
            with st.spinner(f"🔄 Initializing {config.model_name}..."):
                rag = UniversalRAG(
                    documents_path=docs_path,
                    model_config=config,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    verbose=True
                )
                
                if rag.initialize():
                    st.session_state.rag_system = rag
                    st.session_state.initialized = True
                    st.session_state.current_model = config
                    
                    # Sync dynamic params from rag system
                    params = rag.get_current_params()
                    st.session_state.temperature = params['temperature']
                    st.session_state.max_tokens = params['max_tokens']
                    st.session_state.k = params['k']
                    
                    st.success(f"✅ RAG System initialized with {config.model_name}!")
                    
                    if config.cost_per_1k_input > 0:
                        st.warning("💰 Using paid model - costs will be tracked")
                    else:
                        st.success("🆓 Using free model - no costs!")
                    
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
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.session_state.current_model:
                st.caption(f"🤖 Using: {st.session_state.current_model.provider.value} - {st.session_state.current_model.model_name}")
        with col2:
            if st.session_state.total_cost > 0:
                st.markdown(f'<div class="cost-display">💰 ${st.session_state.total_cost:.6f}</div>', unsafe_allow_html=True)
        
        query = st.text_area("Your question:", placeholder="What are the main topics?", height=100)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        with col2:
            if st.button("🔄 Switch Model", use_container_width=True):
                st.session_state.initialized = False
                st.session_state.rag_system = None
                st.rerun()
        with col3:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = []
                st.session_state.total_cost = 0.0
                if st.session_state.rag_system:
                    st.session_state.rag_system.reset_cost_tracking()
                st.rerun()
        
        if search_button and query:
            with st.spinner("🤔 Processing..."):
                try:
                    # Query uses current k from session state (already synced)
                    result = st.session_state.rag_system.query(query)
                    
                    if result['estimated_cost'] > 0:
                        st.session_state.total_cost += result['estimated_cost']
                    
                    st.session_state.messages.append({
                        "question": query,
                        "answer": result["answer"],
                        "sources": result["sources"],
                        "cost": result['estimated_cost'],
                        "model": result['model'],
                        "params": result.get('params_used', {}),
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    st.markdown("### 💡 Answer")
                    st.success(result["answer"])
                    
                    if result["sources"]:
                        st.markdown("### 📚 Sources")
                        for i, source in enumerate(result["sources"], 1):
                            with st.expander(f"Source {i}: {source['filename']}"):
                                st.text(source.get('preview', 'No preview available'))
                    
                    if result['estimated_cost'] > 0:
                        st.info(f"💰 Query cost: ${result['estimated_cost']:.6f}")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        if st.session_state.messages:
            st.divider()
            st.subheader("📜 History")
            for i, msg in enumerate(reversed(st.session_state.messages[-5:]), 1):
                with st.expander(f"[{msg['timestamp']}] {msg['question'][:50]}..."):
                    st.markdown(f"**Q:** {msg['question']}")
                    st.markdown(f"**A:** {msg['answer']}")
                    if msg.get('params'):
                        st.caption(f"Params: temp={msg['params'].get('temperature', 'N/A')}, k={msg['params'].get('k', 'N/A')}")
    
    def run(self):
        """Main application loop"""
        self.render_header()
        
        config_result = self.render_sidebar()
        if config_result is None:
            st.info("👈 Configure in sidebar")
            return
        
        config, chunk_size, chunk_overlap = config_result
        docs_path = self.get_documents_path()
        
        if docs_path:
            if not st.session_state.initialized:
                if st.button("🚀 Initialize RAG System", type="primary", use_container_width=True):
                    self.initialize_rag_system(config, docs_path, chunk_size, chunk_overlap)
            else:
                if (st.session_state.current_model and 
                    config.model_name != st.session_state.current_model.model_name):
                    st.info(f"Model changed to {config.model_name}")
                    if st.button("🔄 Reinitialize", type="primary"):
                        st.session_state.initialized = False
                        st.session_state.rag_system = None
                        st.rerun()
        
        if st.session_state.initialized:
            st.divider()
            self.render_chat_interface()
        elif docs_path:
            st.info("👆 Click 'Initialize RAG System' to start")
        
        st.divider()
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>🌐 Universal RAG System | LangChain 1.x | Free & Premium Models</p>
            <p>Dynamic Parameters: Temperature, Max Tokens, Sources (k)</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    app = UniversalRAGUI()
    app.run()


if __name__ == "__main__":
    main()
