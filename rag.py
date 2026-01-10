"""
Universal RAG System - Complete Implementation
Compatible with LangChain 1.x (uses LCEL approach)

Support for Free (Ollama/HuggingFace) and Commercial Models (OpenAI/Google/Anthropic)

UPDATED: Dynamic parameter updates for temperature, max_tokens, and k
"""

import os
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Optional
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# IMPORTS - LangChain 1.x compatible (LCEL approach)
# ============================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ============================================================
# MODEL CONFIGURATION
# ============================================================

class ModelProvider(Enum):
    """Available model providers"""
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass
class ModelConfig:
    """Configuration for different models"""
    provider: ModelProvider
    model_name: str
    embedding_model: str
    api_key: Optional[str] = None
    temperature: float = 0.5
    max_tokens: int = 2000
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    
    def copy(self) -> 'ModelConfig':
        """Create a copy to avoid mutation issues"""
        return ModelConfig(
            provider=self.provider,
            model_name=self.model_name,
            embedding_model=self.embedding_model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            cost_per_1k_input=self.cost_per_1k_input,
            cost_per_1k_output=self.cost_per_1k_output
        )


class ModelConfigurations:
    """Pre-configured model settings"""
    
    # ===== FREE MODELS =====
    OLLAMA_LLAMA = ModelConfig(
        provider=ModelProvider.OLLAMA,
        model_name="llama3.2",
        embedding_model="nomic-embed-text"
    )
    
    HUGGINGFACE_PHI = ModelConfig(
        provider=ModelProvider.HUGGINGFACE,
        model_name="microsoft/phi-2",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # ===== OPENAI MODELS =====
    OPENAI_GPT35 = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        embedding_model="text-embedding-3-small",
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015
    )
    
    OPENAI_GPT4 = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4-turbo",
        embedding_model="text-embedding-3-large",
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03
    )
    
    OPENAI_GPT4O = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o",
        embedding_model="text-embedding-3-large",
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015
    )
    
    # ===== ANTHROPIC MODELS =====
    ANTHROPIC_HAIKU = ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-haiku-20240307",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.00125
    )
    
    ANTHROPIC_SONNET = ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-5-sonnet-20241022",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015
    )
    
    ANTHROPIC_OPUS = ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-opus-20240229",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075
    )
    
    # ===== GOOGLE MODELS =====
    GOOGLE_GEMINI_FLASH = ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_name="gemini-1.5-flash",
        embedding_model="models/text-embedding-004",
        cost_per_1k_input=0.000075,
        cost_per_1k_output=0.0003
    )
    
    GOOGLE_GEMINI_PRO = ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_name="gemini-1.5-pro",
        embedding_model="models/text-embedding-004",
        cost_per_1k_input=0.00125,
        cost_per_1k_output=0.005
    )


# ============================================================
# PROVIDERS
# ============================================================

class BaseRAGProvider(ABC):
    """Abstract base class for RAG providers"""
    
    @abstractmethod
    def get_llm(self, config: ModelConfig):
        pass
    
    @abstractmethod
    def get_embeddings(self, config: ModelConfig):
        pass


class OllamaProvider(BaseRAGProvider):
    """Provider for Ollama models"""
    
    def get_llm(self, config: ModelConfig):
        from langchain_community.llms import Ollama
        return Ollama(
            model=config.model_name,
            temperature=config.temperature,
            num_ctx=4096
        )
    
    def get_embeddings(self, config: ModelConfig):
        from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(model=config.embedding_model)


class HuggingFaceProvider(BaseRAGProvider):
    """Provider for HuggingFace models"""
    
    def get_llm(self, config: ModelConfig):
        try:
            from langchain_huggingface import HuggingFacePipeline
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            do_sample = config.temperature > 0
            
            pipe_kwargs = {
                "task": "text-generation",
                "model": model,
                "tokenizer": tokenizer,
                "max_new_tokens": config.max_tokens,
                "do_sample": do_sample,
            }
            
            if do_sample:
                pipe_kwargs["temperature"] = config.temperature
                pipe_kwargs["top_p"] = 0.95
            
            pipe = pipeline(**pipe_kwargs)
            return HuggingFacePipeline(pipeline=pipe)
            
        except ImportError:
            raise ImportError("Install: pip install transformers torch langchain-huggingface")
    
    def get_embeddings(self, config: ModelConfig):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=config.embedding_model)
        except ImportError:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(model_name=config.embedding_model)
            except ImportError:
                raise ImportError("Install: pip install sentence-transformers langchain-huggingface")


class OpenAIProvider(BaseRAGProvider):
    """Provider for OpenAI models"""
    
    def get_llm(self, config: ModelConfig):
        try:
            from langchain_openai import ChatOpenAI
            
            api_key = config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key.")
            
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                api_key=api_key
            )
        except ImportError:
            raise ImportError("Install: pip install langchain-openai")
    
    def get_embeddings(self, config: ModelConfig):
        try:
            from langchain_openai import OpenAIEmbeddings
            
            api_key = config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key.")
            
            return OpenAIEmbeddings(
                model=config.embedding_model,
                api_key=api_key
            )
        except ImportError:
            raise ImportError("Install: pip install langchain-openai")


class AnthropicProvider(BaseRAGProvider):
    """Provider for Anthropic Claude models"""
    
    def get_llm(self, config: ModelConfig):
        try:
            from langchain_anthropic import ChatAnthropic
            
            api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key.")
            
            return ChatAnthropic(
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                anthropic_api_key=api_key
            )
        except ImportError:
            raise ImportError("Install: pip install langchain-anthropic")
    
    def get_embeddings(self, config: ModelConfig):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            logger.info("Using HuggingFace embeddings (Anthropic has no embeddings API)")
            return HuggingFaceEmbeddings(model_name=config.embedding_model)
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=config.embedding_model)


class GoogleProvider(BaseRAGProvider):
    """Provider for Google Gemini models"""
    
    def get_llm(self, config: ModelConfig):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            api_key = config.api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Google API key required. Set GOOGLE_API_KEY or pass api_key.")
            
            return ChatGoogleGenerativeAI(
                model=config.model_name,
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
                google_api_key=api_key
            )
        except ImportError:
            raise ImportError("Install: pip install langchain-google-genai")
    
    def get_embeddings(self, config: ModelConfig):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            
            api_key = config.api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Google API key required. Set GOOGLE_API_KEY or pass api_key.")
            
            return GoogleGenerativeAIEmbeddings(
                model=config.embedding_model,
                google_api_key=api_key
            )
        except ImportError:
            raise ImportError("Install: pip install langchain-google-genai")


# ============================================================
# TOKEN ESTIMATION
# ============================================================

def estimate_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """Estimate token count using tiktoken if available"""
    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        return len(text) // 4


# ============================================================
# UNIVERSAL RAG SYSTEM (LCEL approach for LangChain 1.x)
# ============================================================

class UniversalRAG:
    """Universal RAG system supporting multiple providers - uses LCEL"""
    
    PROVIDERS = {
        ModelProvider.OLLAMA: OllamaProvider(),
        ModelProvider.HUGGINGFACE: HuggingFaceProvider(),
        ModelProvider.OPENAI: OpenAIProvider(),
        ModelProvider.ANTHROPIC: AnthropicProvider(),
        ModelProvider.GOOGLE: GoogleProvider()
    }
    
    # Default values for dynamic parameters
    DEFAULT_TEMPERATURE = 0.5
    DEFAULT_MAX_TOKENS = 4000
    DEFAULT_K = 5
    
    def __init__(
        self,
        documents_path: str,
        model_config: ModelConfig,
        persist_directory: str = "./chroma_db",
        chunk_size: int = 600,
        chunk_overlap: int = 150,
        verbose: bool = True
    ):
        """Initialize Universal RAG System"""
        self.documents_path = Path(documents_path)
        self.model_config = model_config.copy()
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.verbose = verbose
        
        # Dynamic parameters (can be changed at runtime)
        self._current_k = self.DEFAULT_K
        
        # Cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # Validate path
        if not self.documents_path.exists():
            raise ValueError(f"Path not found: {documents_path}")
        
        if not self.documents_path.is_dir():
            raise ValueError(f"Path is not a directory: {documents_path}")
        
        if model_config.provider not in self.PROVIDERS:
            raise ValueError(f"Unsupported provider: {model_config.provider}")
        
        self.provider = self.PROVIDERS[model_config.provider]
        
        # Components
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        
        # Store last retrieved docs for sources
        self._last_docs = []
        
        if verbose:
            logger.info(f"Using {model_config.provider.value} - {model_config.model_name}")
            if model_config.cost_per_1k_input > 0:
                logger.info(f"💰 Costs: ${model_config.cost_per_1k_input:.4f}/1K in, ${model_config.cost_per_1k_output:.4f}/1K out")
    
    # ============================================================
    # DYNAMIC PARAMETER UPDATE METHODS
    # ============================================================
    
    def update_temperature(self, temperature: float) -> bool:
        """
        Update LLM temperature dynamically.
        
        Args:
            temperature: New temperature value (0.0 to 1.0)
            
        Returns:
            True if successful, False otherwise
        """
        if not 0.0 <= temperature <= 1.0:
            logger.warning(f"Temperature must be between 0.0 and 1.0, got {temperature}")
            return False
        
        try:
            self.model_config.temperature = temperature
            
            # Update LLM if already initialized
            if self.llm is not None:
                # Different providers have different attribute names
                if hasattr(self.llm, 'temperature'):
                    self.llm.temperature = temperature
                elif hasattr(self.llm, 'model_kwargs'):
                    self.llm.model_kwargs['temperature'] = temperature
                
                # Rebuild the chain with new LLM settings
                self._rebuild_chain()
            
            if self.verbose:
                logger.info(f"✅ Temperature updated: {temperature}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update temperature: {str(e)}")
            return False
    
    def update_max_tokens(self, max_tokens: int) -> bool:
        """
        Update LLM max_tokens dynamically.
        
        Args:
            max_tokens: New max tokens value (1 to 8192)
            
        Returns:
            True if successful, False otherwise
        """
        if not 1 <= max_tokens <= 8192:
            logger.warning(f"Max tokens must be between 1 and 8192, got {max_tokens}")
            return False
        
        try:
            self.model_config.max_tokens = max_tokens
            
            # Update LLM if already initialized
            if self.llm is not None:
                if hasattr(self.llm, 'max_tokens'):
                    self.llm.max_tokens = max_tokens
                elif hasattr(self.llm, 'max_output_tokens'):
                    self.llm.max_output_tokens = max_tokens
                elif hasattr(self.llm, 'model_kwargs'):
                    self.llm.model_kwargs['max_tokens'] = max_tokens
                
                # Rebuild the chain
                self._rebuild_chain()
            
            if self.verbose:
                logger.info(f"✅ Max tokens updated: {max_tokens}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update max_tokens: {str(e)}")
            return False
    
    def update_k(self, k: int) -> bool:
        """
        Update the number of chunks to retrieve (k) dynamically.
        
        Args:
            k: Number of chunks to retrieve (1 to 20)
            
        Returns:
            True if successful, False otherwise
        """
        if not 1 <= k <= 20:
            logger.warning(f"k must be between 1 and 20, got {k}")
            return False
        
        try:
            self._current_k = k
            
            # Update retriever if initialized
            if self.retriever is not None:
                self.retriever.search_kwargs["k"] = k
            
            if self.verbose:
                logger.info(f"✅ Sources (k) updated: {k}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update k: {str(e)}")
            return False
    
    def get_current_params(self) -> Dict:
        """
        Get current parameter values for display.
        
        Returns:
            Dictionary with all current parameters
        """
        return {
            "temperature": self.model_config.temperature,
            "max_tokens": self.model_config.max_tokens,
            "k": self._current_k,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "model_name": self.model_config.model_name,
            "provider": self.model_config.provider.value
        }
    
    def _rebuild_chain(self):
        """Rebuild the RAG chain after parameter changes"""
        if self.vectorstore is not None and self.llm is not None:
            # Recreate LLM with new parameters
            self.llm = self.provider.get_llm(self.model_config)
            self.setup_rag_chain(k=self._current_k)
    
    # ============================================================
    # CORE METHODS
    # ============================================================
    
    def _initialize_models(self):
        """Initialize LLM and embeddings"""
        try:
            self.llm = self.provider.get_llm(self.model_config)
            self.embeddings = self.provider.get_embeddings(self.model_config)
            
            if self.verbose:
                logger.info("✅ Models initialized")
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            raise
    
    def load_documents(self) -> List:
        """Load all supported documents"""
        documents = []
        
        loaders_map = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader
        }
        
        if self.verbose:
            logger.info(f"📂 Scanning: {self.documents_path}")
        
        file_count = 0
        for file_path in self.documents_path.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                
                if ext in loaders_map:
                    file_count += 1
                    try:
                        loader = loaders_map[ext](str(file_path))
                        docs = loader.load()
                        
                        for doc in docs:
                            doc.metadata['source'] = str(file_path)
                            doc.metadata['filename'] = file_path.name
                            doc.metadata['file_type'] = ext[1:]
                        
                        documents.extend(docs)
                        
                        if self.verbose:
                            logger.info(f"✅ Loaded: {file_path.name}")
                    
                    except Exception as e:
                        logger.error(f"❌ Error loading {file_path.name}: {str(e)}")
        
        if self.verbose:
            logger.info(f"📊 Total files: {file_count}, documents (pages): {len(documents)}")
        
        return documents
    
    def create_vector_store(self, documents: List):
        """Create vector store from documents"""
        if self.verbose:
            logger.info("🔄 Splitting documents...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
        )
        
        texts = text_splitter.split_documents(documents)
        
        if self.verbose:
            logger.info(f"📊 Created {len(texts)} chunks")
            logger.info("🧠 Creating embeddings...")
        
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        if self.verbose:
            logger.info("✅ Vector store created!")
    
    def _format_docs_and_store(self, docs):
        """Format docs and store them for later access"""
        self._last_docs = docs
        return "\n\n".join(doc.page_content for doc in docs)
    
    def setup_rag_chain(self, k: int = None):
        """Setup the RAG chain using LCEL (LangChain 1.x approach)"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        # Use provided k or current k
        if k is not None:
            self._current_k = k
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self._current_k})
        
        # Define prompt
        prompt = ChatPromptTemplate.from_template(
            """Use the following context to answer the question.
If you don't know the answer, say you don't know.
Answer in the same language as the question.

Context:
{context}

Question: {question}

Answer:"""
        )
        
        # Build LCEL chain
        self.rag_chain = (
            {
                "context": self.retriever | self._format_docs_and_store,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        if self.verbose:
            logger.info(f"🔗 RAG Chain ready (k={self._current_k})")

    def query(self, question: str, k: int = None) -> Dict:
        """Query the RAG system"""
        if not self.rag_chain:
            raise ValueError("Not initialized. Run initialize() first")
        
        if self.verbose:
            logger.info(f"🤔 Query: {question[:50]}...")
        
        # Update k if provided and different
        if k is not None and k != self._current_k:
            self.update_k(k)

        # Clear last docs
        self._last_docs = []
        
        # Execute query
        answer = self.rag_chain.invoke(question)
        
        # Get the docs that were used
        docs = self._last_docs
        
        # Estimate tokens
        input_tokens = estimate_tokens(question, self.model_config.model_name)
        for doc in docs:
            input_tokens += estimate_tokens(doc.page_content, self.model_config.model_name)
        output_tokens = estimate_tokens(answer, self.model_config.model_name)
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        # Calculate cost
        cost = (
            (input_tokens / 1000) * self.model_config.cost_per_1k_input +
            (output_tokens / 1000) * self.model_config.cost_per_1k_output
        )
        
        # Format response
        response = {
            "answer": answer,
            "relevant_chunks": len(docs),
            "sources": [],
            "estimated_cost": cost,
            "total_cost": self.get_total_cost(),
            "model": self.model_config.model_name,
            "provider": self.model_config.provider.value,
            "params_used": self.get_current_params()
        }
        
        # Process sources
        seen = set()
        for doc in docs:
            filename = doc.metadata.get("filename", "Unknown")
            if filename not in seen:
                seen.add(filename)
                response["sources"].append({
                    "filename": filename,
                    "file_type": doc.metadata.get("file_type", "unknown"),
                    "preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
        
        if self.verbose and cost > 0:
            logger.info(f"💰 Query cost: ${cost:.6f}")
        
        return response
    
    def get_total_cost(self) -> float:
        """Get total accumulated cost"""
        return (
            (self.total_input_tokens / 1000) * self.model_config.cost_per_1k_input +
            (self.total_output_tokens / 1000) * self.model_config.cost_per_1k_output
        )
    
    def reset_cost_tracking(self):
        """Reset cost tracking counters"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    def initialize(self) -> bool:
        """Complete initialization"""
        try:
            if self.verbose:
                logger.info("🚀 Initializing RAG System...")
            
            self._initialize_models()
            
            documents = self.load_documents()
            if not documents:
                logger.error("No documents found!")
                return False
            
            self.create_vector_store(documents)
            self.setup_rag_chain()
            
            if self.verbose:
                logger.info("✅ RAG System Ready!")
                if self.model_config.cost_per_1k_input > 0:
                    logger.info("💰 Using paid model - costs will be tracked")
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def switch_model(self, new_config: ModelConfig):
        """Switch to different model"""
        if self.verbose:
            logger.info(f"Switching to {new_config.model_name}")
        
        self.model_config = new_config.copy()
        
        if new_config.provider not in self.PROVIDERS:
            raise ValueError(f"Unsupported provider: {new_config.provider}")
        self.provider = self.PROVIDERS[new_config.provider]
        
        self._initialize_models()
        
        if self.vectorstore:
            self.setup_rag_chain()
        
        if self.verbose:
            logger.info(f"✅ Switched to {new_config.model_name}")


# ============================================================
# CLI INTERFACE
# ============================================================

def print_help():
    """Print available commands"""
    print("\n" + "=" * 60)
    print("📋 COMANDI DISPONIBILI:")
    print("-" * 60)
    print("  temp <valore>    → Cambia temperature (0.0-1.0)")
    print("  tokens <valore>  → Cambia max tokens (1-8192)")
    print("  k <valore>       → Cambia sources/chunks (1-20)")
    print("  params           → Mostra parametri attuali")
    print("  cost             → Mostra costo totale sessione")
    print("  help             → Mostra questo messaggio")
    print("  exit/quit/q      → Esci")
    print("=" * 60 + "\n")


def main():
    """Interactive CLI"""
    from getpass import getpass
    
    print("=" * 60)
    print("🌐 UNIVERSAL RAG SYSTEM (LangChain 1.x)")
    print("=" * 60 + "\n")
    
    # Get documents path
    if len(sys.argv) > 1:
        docs_path = sys.argv[1]
    else:
        docs_path = input("📂 Documents path: ").strip()
    
    if not Path(docs_path).exists():
        print(f"❌ Path not found: {docs_path}")
        return
    
    # Select provider
    print("\n🎯 Select Provider:")
    print("1. FREE Local (Ollama/HuggingFace)")
    print("2. OpenAI")
    print("3. Anthropic")
    print("4. Google")
    
    choice = input("\nChoice (1-4): ").strip()
    
    config = None
    
    if choice == "1":
        print("\n🆓 FREE Models:")
        print("1. Ollama Llama 3.2")
        print("2. HuggingFace Phi-2")
        
        model_choice = input("Choice: ").strip()
        
        if model_choice == "1":
            config = ModelConfigurations.OLLAMA_LLAMA.copy()
        else:
            config = ModelConfigurations.HUGGINGFACE_PHI.copy()
    
    elif choice == "2":
        api_key = getpass("🔑 OpenAI API key: ").strip()
        
        print("\n1. GPT-3.5 ($0.0005/1K)")
        print("2. GPT-4 ($0.01/1K)")
        print("3. GPT-4o ($0.005/1K)")
        
        model_choice = input("Choice: ").strip()
        
        if model_choice == "1":
            config = ModelConfigurations.OPENAI_GPT35.copy()
        elif model_choice == "2":
            config = ModelConfigurations.OPENAI_GPT4.copy()
        else:
            config = ModelConfigurations.OPENAI_GPT4O.copy()
        
        config.api_key = api_key
    
    elif choice == "3":
        api_key = getpass("🔑 Anthropic API key: ").strip()
        
        print("\n1. Haiku ($0.00025/1K)")
        print("2. Sonnet ($0.003/1K)")
        print("3. Opus ($0.015/1K)")
        
        model_choice = input("Choice: ").strip()
        
        if model_choice == "1":
            config = ModelConfigurations.ANTHROPIC_HAIKU.copy()
        elif model_choice == "2":
            config = ModelConfigurations.ANTHROPIC_SONNET.copy()
        else:
            config = ModelConfigurations.ANTHROPIC_OPUS.copy()
        
        config.api_key = api_key
    
    elif choice == "4":
        api_key = getpass("🔑 Google API key: ").strip()
        
        print("\n1. Gemini Flash ($0.000075/1K)")
        print("2. Gemini Pro ($0.00125/1K)")
        
        model_choice = input("Choice: ").strip()
        
        if model_choice == "1":
            config = ModelConfigurations.GOOGLE_GEMINI_FLASH.copy()
        else:
            config = ModelConfigurations.GOOGLE_GEMINI_PRO.copy()
        
        config.api_key = api_key
    
    if not config:
        print("❌ No model selected")
        return
    
    # Initialize
    print(f"\n🔄 Initializing {config.model_name}...")
    
    rag = UniversalRAG(
        documents_path=docs_path,
        model_config=config,
        verbose=True
    )
    
    if not rag.initialize():
        return
    
    # Show help at startup
    print_help()
    
    # Query loop
    while True:
        question = input("❓ Question: ").strip()
        
        # Exit commands
        if question.lower() in ['exit', 'quit', 'q']:
            if rag.get_total_cost() > 0:
                print(f"\n💰 Total cost: ${rag.get_total_cost():.6f}")
            print("👋 Ciao!")
            break
        
        # Help command
        if question.lower() == 'help':
            print_help()
            continue
        
        # Cost command
        if question.lower() == 'cost':
            print(f"💰 Current cost: ${rag.get_total_cost():.6f}\n")
            continue
        
        # Params command
        if question.lower() == 'params':
            params = rag.get_current_params()
            print("\n📊 Parametri attuali:")
            print(f"   - Temperature: {params['temperature']}")
            print(f"   - Max Tokens: {params['max_tokens']}")
            print(f"   - Sources (k): {params['k']}")
            print(f"   - Chunk Size: {params['chunk_size']} (fisso)")
            print(f"   - Chunk Overlap: {params['chunk_overlap']} (fisso)")
            print(f"   - Model: {params['model_name']}")
            print(f"   - Provider: {params['provider']}\n")
            continue
        
        # Temperature command
        if question.lower().startswith('temp '):
            try:
                new_temp = float(question.split()[1])
                rag.update_temperature(new_temp)
            except (ValueError, IndexError):
                print("❌ Uso: temp <valore> (es: temp 0.7)\n")
            continue
        
        # Max tokens command
        if question.lower().startswith('tokens '):
            try:
                new_tokens = int(question.split()[1])
                rag.update_max_tokens(new_tokens)
            except (ValueError, IndexError):
                print("❌ Uso: tokens <valore> (es: tokens 2000)\n")
            continue
        
        # K command
        if question.lower().startswith('k '):
            try:
                new_k = int(question.split()[1])
                rag.update_k(new_k)
            except (ValueError, IndexError):
                print("❌ Uso: k <valore> (es: k 5)\n")
            continue
        
        # Empty question
        if not question:
            continue
        
        # Regular query
        try:
            result = rag.query(question)
            
            print(f"\n💡 Answer: {result['answer']}\n")
            
            if result['sources']:
                print(f"📚 Sources: {', '.join(s['filename'] for s in result['sources'])}")
            
            if result['estimated_cost'] > 0:
                print(f"💰 Cost: ${result['estimated_cost']:.6f} (Total: ${result['total_cost']:.6f})")
            
            print("-" * 60 + "\n")
        
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")


if __name__ == "__main__":
    main()
