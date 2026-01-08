"""
Enhanced Streamlit Interface for Universal RAG System
Supports Free (Ollama/HuggingFace) and Commercial (OpenAI/Anthropic/Google) Models
"""

import streamlit as st
from pathlib import Path
import os
from datetime import datetime
from typing import Optional
import json

# Import the universal RAG system
from rag_universal_complete import (
    UniversalRAG, 
    ModelConfig, 
    ModelProvider,
    ModelConfigurations,
    estimate_costs
)

# Page configuration
st.set_page_config(
    page_title="Universal RAG System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with provider-specific colors
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
    }
    
    .provider-openai {
        border-left: 4px solid #10a37f;
        padding-left: 1rem;
        margin: 0.5rem 0;
    }
    
    .provider-anthropic {
        border-left: 4px solid #d97757;
        padding-left: 1rem;
        margin: 0.5rem 0;
    }
    
    .provider-google {
        border-left: 4px solid #4285f4;
        padding-left: 1rem;
        margin: 0.5rem 0;
    }
    
    .provider-free {
        border-left: 4px solid #00d084;
        padding-left: 1rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


class UniversalRAGUI:
    """Enhanced UI for Universal RAG System"""
    
    # Organize models by provider
    MODELS_CATALOG = {
        "🆓 Free Local Models": {
            "Ollama Llama 3.2": {
                "config": ModelConfigurations.OLLAMA_LLAMA,
                "description": "Fast & lightweight (2GB)",
                "requirements": "Requires Ollama installation"
            },
            "HuggingFace Phi-2": {
                "config": ModelConfigurations.HUGGINGFACE_PHI,
                "description": "Microsoft's efficient model (2.7B)",
                "requirements": "Auto-downloads on first use"
            }
        },
        "🟢 OpenAI": {
            "GPT-3.5 Turbo": {
                "config": ModelConfigurations.OPENAI_GPT35,
                "description": "Fast & affordable",
                "cost": "$0.0005/1K input, $0.0015/1K output"
            },
            "GPT-4 Turbo": {
                "config": ModelConfigurations.OPENAI_GPT4,
                "description": "Most capable GPT model",
                "cost": "$0.01/1K input, $0.03/1K output"
            },
            "GPT-4o": {
                "config": ModelConfigurations.OPENAI_GPT4O,
                "description": "Optimized GPT-4",
                "cost": "$0.005/1K input, $0.015/1K output"
            }
        },
        "🟠 Anthropic Claude": {
            "Claude 3 Haiku": {
                "config": ModelConfigurations.ANTHROPIC_HAIKU,
                "description": "Fastest & cheapest Claude",
                "cost": "$0.00025/1K input, $0.00125/1K output"
            },
            "Claude 3.5 Sonnet": {
                "config": ModelConfigurations.ANTHROPIC_SONNET,
                "description": "Best balance of speed & capability",
                "cost": "$0.003/1K input, $0.015/1K output"
            },
            "Claude 3 Opus": {
                "config": ModelConfigurations.ANTHROPIC_OPUS,
                "description": "Most capable Claude model",
                "cost": "$0.015/1K input, $0.075/1K output"
            }
        },
        "🔵 Google Gemini": {
            "Gemini 1.5 Flash": {
                "config": ModelConfigurations.GOOGLE_GEMINI_FLASH,
                "description": "Ultra-fast & cheap",
                "cost": "$0.000075/1K input, $0.0003/1K output"
            },
            "Gemini 1.5 Pro": {
                "config": ModelConfigurations.GOOGLE_GEMINI_PRO,
                "description": "Advanced reasoning",
                "cost": "$0.00125/1K input, $0.005/1K output"
            }
        }
    }
    
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state variables"""
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = None
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'total_cost' not in st.session_state:
            st.session_state.total_cost = 0.0
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
        if 'api_keys' not in st.session_state:
            st.session_state.api_keys = {}
        if 'current_model' not in st.session_state:
            st.session_state.current_model = None
    
    def render_header(self):
        """Render the main header with cost tracking"""
        st.markdown('<h1 class="main-header">🌍 Universal RAG System</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center;">Free Local Models + Premium APIs in One Interface</p>', 
                   unsafe_allow_html=True)
        
        # Metrics row
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
    
    def render_sidebar(self):
        """Render sidebar with model selection and API key management"""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Model selection
            st.subheader("🤖 Select Model")
            
            # Provider selection
            provider_category = st.selectbox(
                "Provider Category",
                options=list(self.MODELS_CATALOG.keys()),
                help="Choose between free local models or premium API models"
            )
            
            # Model selection within provider
            models_in_category = self.MODELS_CATALOG[provider_category]
            model_name = st.selectbox(
                "Model",
                options=list(models_in_category.keys()),
                help="Select specific model"
            )
            
            model_info = models_in_category[model_name]
            config = model_info["config"]
            
            # Display model information
            with st.expander("📊 Model Details", expanded=True):
                st.write(f"**Description:** {model_info['description']}")
                
                if 'cost' in model_info:
                    st.write(f"**Pricing:** {model_info['cost']}")
                else:
                    st.write("**Pricing:** FREE - Runs locally")
                
                if 'requirements' in model_info:
                    st.info(model_info['requirements'])
                
                # Show provider-specific styling
                if "Free" in provider_category:
                    st.markdown('<div class="provider-free">✅ No API costs</div>', unsafe_allow_html=True)
                elif "OpenAI" in provider_category:
                    st.markdown('<div class="provider-openai">🟢 OpenAI API</div>', unsafe_allow_html=True)
                elif "Anthropic" in provider_category:
                    st.markdown('<div class="provider-anthropic">🟠 Anthropic API</div>', unsafe_allow_html=True)
                elif "Google" in provider_category:
                    st.markdown('<div class="provider-google">🔵 Google API</div>', unsafe_allow_html=True)
            
            # API Key management for paid models
            if config.cost_per_1k_input > 0:
                st.divider()
                st.subheader("🔑 API Configuration")
                
                if config.provider == ModelProvider.OPENAI:
                    api_key = st.text_input(
                        "OpenAI API Key",
                        type="password",
                        value=st.session_state.api_keys.get('openai', ''),
                        help="Get from: https://platform.openai.com/api-keys"
                    )
                    if api_key:
                        st.session_state.api_keys['openai'] = api_key
                        config.api_key = api_key
                
                elif config.provider == ModelProvider.ANTHROPIC:
                    api_key = st.text_input(
                        "Anthropic API Key",
                        type="password",
                        value=st.session_state.api_keys.get('anthropic', ''),
                        help="Get from: https://console.anthropic.com/"
                    )
                    if api_key:
                        st.session_state.api_keys['anthropic'] = api_key
                        config.api_key = api_key
                
                elif config.provider == ModelProvider.GOOGLE:
                    api_key = st.text_input(
                        "Google API Key",
                        type="password",
                        value=st.session_state.api_keys.get('google', ''),
                        help="Get from: https://makersuite.google.com/app/apikey"
                    )
                    if api_key:
                        st.session_state.api_keys['google'] = api_key
                        config.api_key = api_key
            
            # Advanced settings
            st.divider()
            with st.expander("🔧 Advanced Settings"):
                chunk_size = st.slider("Chunk Size", 200, 2000, 1000, 100)
                chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50)
                temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
                max_tokens = st.slider("Max Tokens", 256, 4096, 1024, 256)
                
                config.temperature = temperature
                config.max_tokens = max_tokens
            
            # Cost estimation
            if st.button("💰 Estimate Monthly Costs"):
                self.show_cost_comparison()
            
            return config, chunk_size, chunk_overlap
    
    def show_cost_comparison(self):
        """Display cost comparison modal"""
        with st.expander("💰 Cost Comparison", expanded=True):
            queries_per_day = st.number_input("Queries per day", 1, 1000, 10)
            
            # Create comparison table
            comparison_data = []
            
            for category, models in self.MODELS_CATALOG.items():
                for model_name, model_info in models.items():
                    config = model_info["config"]
                    
                    # Calculate monthly cost
                    daily_input_cost = (500 * queries_per_day / 1000) * config.cost_per_1k_input
                    daily_output_cost = (500 * queries_per_day / 1000) * config.cost_per_1k_output
                    monthly_cost = 30 * (daily_input_cost + daily_output_cost)
                    
                    comparison_data.append({
                        "Provider": category.split()[1] if len(category.split()) > 1 else category,
                        "Model": model_name,
                        "Input $/1K": f"${config.cost_per_1k_input:.5f}",
                        "Output $/1K": f"${config.cost_per_1k_output:.5f}",
                        "Monthly Cost": f"${monthly_cost:.2f}"
                    })
            
            import pandas as pd
            df = pd.DataFrame(comparison_data)
            
            # Style the dataframe
            def highlight_free(val):
                if "$0.00000" in str(val) or "$0.00" in str(val):
                    return 'background-color: #d4edda'
                return ''
            
            styled_df = df.style.applymap(highlight_free)
            st.dataframe(styled_df, use_container_width=True)
            
            st.info(f"📊 Estimates based on {queries_per_day} queries/day with ~500 tokens per query")
    
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
                st.rerun()
        with col2:
            if st.button("📄 Documents"):
                docs_path = os.path.expanduser("~/Documents")
                st.rerun()
        with col3:
            if st.button("⬇️ Downloads"):
                docs_path = os.path.expanduser("~/Downloads")
                st.rerun()
        
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
    
    def initialize_rag_system(self, config: ModelConfig, docs_path: str, chunk_size: int, chunk_overlap: int):
        """Initialize the RAG system with selected configuration"""
        try:
            # Check API key for paid models
            if config.cost_per_1k_input > 0 and not config.api_key:
                st.error("🔑 API key required for this model!")
                return False
            
            with st.spinner(f"🔄 Initializing {config.model_name}... This may take a few minutes."):
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
                    st.success(f"✅ RAG System initialized with {config.model_name}!")
                    
                    if config.cost_per_1k_input > 0:
                        st.warning(f"💰 Using paid model - costs will be tracked")
                    else:
                        st.success(f"🆓 Using free model - no costs!")
                    
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
        
        # Display current model and cost
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.session_state.current_model:
                model_name = st.session_state.current_model.model_name
                provider = st.session_state.current_model.provider.value
                st.caption(f"🤖 Using: {provider} - {model_name}")
        
        with col2:
            if st.session_state.total_cost > 0:
                st.markdown(f'<div class="cost-display">💰 ${st.session_state.total_cost:.6f}</div>', 
                          unsafe_allow_html=True)
        
        # Query input
        query = st.text_area(
            "Your question:",
            placeholder="What are the main topics discussed in the documents?",
            height=100
        )
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            num_chunks = st.slider("Number of sources", 1, 10, 3)
        
        with col2:
            search_button = st.button("🔍 Search", type="primary")
        
        with col3:
            if st.button("🔄 Switch Model"):
                st.session_state.initialized = False
                st.rerun()
        
        with col4:
            if st.button("🗑️ Clear History"):
                st.session_state.messages = []
                st.session_state.total_cost = 0.0
                st.rerun()
        
        # Process query
        if search_button and query:
            with st.spinner("🤔 Processing..."):
                try:
                    # Query the system
                    result = st.session_state.rag_system.query(query, k=num_chunks)
                    
                    # Update total cost
                    if result['estimated_cost'] > 0:
                        st.session_state.total_cost += result['estimated_cost']
                    
                    # Save to history
                    st.session_state.messages.append({
                        "question": query,
                        "answer": result["answer"],
                        "sources": result["sources"],
                        "cost": result['estimated_cost'],
                        "model": result['model'],
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
                                    st.text(source.get('preview', 'No preview'))
                        
                        if result['estimated_cost'] > 0:
                            st.info(f"💰 Query cost: ${result['estimated_cost']:.6f}")
                
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
                    st.caption(f"Model: {msg['model']}")
                    if msg['sources']:
                        sources = [s['filename'] for s in msg['sources']]
                        st.caption(f"Sources: {', '.join(sources)}")
                    if msg['cost'] > 0:
                        st.caption(f"Cost: ${msg['cost']:.6f}")
    
    def run(self):
        """Main application loop"""
        self.render_header()
        
        # Sidebar configuration
        config_result = self.render_sidebar()
        
        if config_result is None:
            st.info("👈 Please configure the system in the sidebar")
            return
        
        config, chunk_size, chunk_overlap = config_result
        
        # Get documents path
        docs_path = self.get_documents_path()
        
        if docs_path:
            # Initialize button
            if not st.session_state.initialized:
                if st.button("🚀 Initialize RAG System", type="primary", use_container_width=True):
                    self.initialize_rag_system(config, docs_path, chunk_size, chunk_overlap)
            else:
                # Check if model changed
                if st.session_state.current_model and config.model_name != st.session_state.current_model.model_name:
                    st.info(f"Model changed from {st.session_state.current_model.model_name} to {config.model_name}")
                    if st.button("🔄 Reinitialize with New Model", type="primary"):
                        st.session_state.initialized = False
                        st.rerun()
        
        # Chat interface
        if st.session_state.initialized:
            st.divider()
            self.render_chat_interface()
        elif docs_path:
            st.info("👆 Click 'Initialize RAG System' to start")
        
        # Footer with tips
        st.divider()
        
        with st.expander("💡 Tips & Best Practices"):
            st.markdown("""
            ### 🎯 Model Selection Guide
            
            **For Testing & Development:**
            - Use **Free Models** (Ollama/HuggingFace) - No costs!
            - Start with Llama 3.2 or Phi-2
            
            **For Better Quality:**
            - **GPT-3.5 Turbo**: Good balance of cost/quality ($0.0005/1K input)
            - **Claude 3 Haiku**: Cheapest premium option ($0.00025/1K input)
            - **Gemini 1.5 Flash**: Ultra-cheap Google option ($0.000075/1K input)
            
            **For Best Quality:**
            - **GPT-4 Turbo**: OpenAI's best ($0.01/1K input)
            - **Claude 3.5 Sonnet**: Anthropic's balanced best ($0.003/1K input)
            - **Claude 3 Opus**: Ultimate quality ($0.015/1K input)
            
            ### 💰 Cost Optimization
            - Use free models for development
            - Switch to paid models only for production
            - Start with cheaper models (Haiku/Flash)
            - Monitor costs in real-time (shown above)
            """)
        
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>🌍 Universal RAG System | Free & Premium Models</p>
            <p>Built with LangChain, ChromaDB, and ❤️</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main entry point"""
    app = UniversalRAGUI()
    app.run()


if __name__ == "__main__":
    main()
