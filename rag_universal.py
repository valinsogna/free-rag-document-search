"""
Universal RAG System - Support for Free (Ollama/HuggingFace) and Commercial Models (OpenAI/Google/Anthropic)
Seamlessly switch between free local models and premium cloud APIs
"""

import os
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Literal
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Core imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class ModelProvider(Enum):
    """Available model providers"""
    # Free/Local
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    
    # Commercial
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    MISTRAL_API = "mistral_api"


@dataclass
class ModelConfig:
    """Configuration for different models"""
    provider: ModelProvider
    model_name: str
    embedding_model: str
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1024
    cost_per_1k_input: float = 0.0  # in USD
    cost_per_1k_output: float = 0.0  # in USD


class ModelConfigurations:
    """Pre-configured model settings"""
    
    # Free Models
    OLLAMA_LLAMA = ModelConfig(
        provider=ModelProvider.OLLAMA,
        model_name="llama3.2",
        embedding_model="nomic-embed-text",
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0
    )
    
    HUGGINGFACE_PHI = ModelConfig(
        provider=ModelProvider.HUGGINGFACE,
        model_name="microsoft/phi-2",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0
    )
    
    # OpenAI Models
    OPENAI_GPT35 = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        embedding_model="text-embedding-3-small",
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015
    )
    
    OPENAI_GPT4 = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4-turbo-preview",
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
    
    # Anthropic Models
    ANTHROPIC_HAIKU = ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-haiku-20240307",
        embedding_model="voyage-2",  # Anthropic recommends Voyage AI
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.00125
    )
    
    ANTHROPIC_SONNET = ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-5-sonnet-20241022",
        embedding_model="voyage-2",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015
    )
    
    ANTHROPIC_OPUS = ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-opus-20240229",
        embedding_model="voyage-2",
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075
    )
    
    # Google Models
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
    
    # Cohere Models
    COHERE_COMMAND = ModelConfig(
        provider=ModelProvider.COHERE,
        model_name="command-r",
        embedding_model="embed-english-v3.0",
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015
    )
    
    # Mistral API Models
    MISTRAL_SMALL = ModelConfig(
        provider=ModelProvider.MISTRAL_API,
        model_name="mistral-small-latest",
        embedding_model="mistral-embed",
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.003
    )
    
    MISTRAL_LARGE = ModelConfig(
        provider=ModelProvider.MISTRAL_API,
        model_name="mistral-large-latest",
        embedding_model="mistral-embed",
        cost_per_1k_input=0.004,
        cost_per_1k_output=0.012
    )


class BaseRAGProvider(ABC):
    """Abstract base class for RAG providers"""
    
    @abstractmethod
    def get_llm(self, config: ModelConfig):
        """Get the LLM instance"""
        pass
    
    @abstractmethod
    def get_embeddings(self, config: ModelConfig):
        """Get the embeddings instance"""
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
        from langchain_community.llms import HuggingFacePipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature
        )
        
        return HuggingFacePipeline(pipeline=pipe)
    
    def get_embeddings(self, config: ModelConfig):
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=config.embedding_model)


class OpenAIProvider(BaseRAGProvider):
    """Provider for OpenAI models"""
    
    def get_llm(self, config: ModelConfig):
        from langchain_openai import ChatOpenAI
        
        if not config.api_key:
            raise ValueError("OpenAI API key is required")
        
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.api_key
        )
    
    def get_embeddings(self, config: ModelConfig):
        from langchain_openai import OpenAIEmbeddings
        
        if not config.api_key:
            raise ValueError("OpenAI API key is required")
        
        return OpenAIEmbeddings(
            model=config.embedding_model,
            api_key=config.api_key
        )


class AnthropicProvider(BaseRAGProvider):
    """Provider for Anthropic Claude models"""
    
    def get_llm(self, config: ModelConfig):
        from langchain_anthropic import ChatAnthropic
        
        if not config.api_key:
            raise ValueError("Anthropic API key is required")
        
        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            anthropic_api_key=config.api_key
        )
    
    def get_embeddings(self, config: ModelConfig):
        # Anthropic doesn't provide embeddings, use Voyage AI or OpenAI
        try:
            from langchain_voyageai import VoyageAIEmbeddings
            return VoyageAIEmbeddings(
                model=config.embedding_model,
                voyage_api_key=os.getenv("VOYAGE_API_KEY")
            )
        except:
            # Fallback to HuggingFace embeddings
            from langchain_community.embeddings import HuggingFaceEmbeddings
            logger.warning("Using HuggingFace embeddings as fallback for Anthropic")
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


class GoogleProvider(BaseRAGProvider):
    """Provider for Google Gemini models"""
    
    def get_llm(self, config: ModelConfig):
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        if not config.api_key:
            raise ValueError("Google API key is required")
        
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            google_api_key=config.api_key
        )
    
    def get_embeddings(self, config: ModelConfig):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        
        if not config.api_key:
            raise ValueError("Google API key is required")
        
        return GoogleGenerativeAIEmbeddings(
            model=config.embedding_model,
            google_api_key=config.api_key
        )


class CohereProvider(BaseRAGProvider):
    """Provider for Cohere models"""
    
    def get_llm(self, config: ModelConfig):
        from langchain_cohere import ChatCohere
        
        if not config.api_key:
            raise ValueError("Cohere API key is required")
        
        return ChatCohere(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            cohere_api_key=config.api_key
        )
    
    def get_embeddings(self, config: ModelConfig):
        from langchain_cohere import CohereEmbeddings
        
        if not config.api_key:
            raise ValueError("Cohere API key is required")
        
        return CohereEmbeddings(
            model=config.embedding_model,
            cohere_api_key=config.api_key
        )


class MistralAPIProvider(BaseRAGProvider):
    """Provider for Mistral API models"""
    
    def get_llm(self, config: ModelConfig):
        from langchain_mistralai import ChatMistralAI
        
        if not config.api_key:
            raise ValueError("Mistral API key is required")
        
        return ChatMistralAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            mistral_api_key=config.api_key
        )
    
    def get_embeddings(self, config: ModelConfig):
        from langchain_mistralai import MistralAIEmbeddings
        
        if not config.api_key:
            raise ValueError("Mistral API key is required")
        
        return MistralAIEmbeddings(
            model=config.embedding_model,
            mistral_api_key=config.api_key
        )


class UniversalRAG:
    """Universal RAG system supporting multiple providers"""
    
    PROVIDERS = {
        ModelProvider.OLLAMA: OllamaProvider(),
        ModelProvider.HUGGINGFACE: HuggingFaceProvider(),
        ModelProvider.OPENAI: OpenAIProvider(),
        ModelProvider.ANTHROPIC: AnthropicProvider(),
        ModelProvider.GOOGLE: GoogleProvider(),
        ModelProvider.COHERE: CohereProvider(),
        ModelProvider.MISTRAL_API: MistralAPIProvider()
    }
    
    def __init__(
        self,
        documents_path: str,
        model_config: ModelConfig,
        persist_directory: str = "./chroma_db_universal",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        verbose: bool = True
    ):
        """
        Initialize Universal RAG System
        
        Args:
            documents_path: Path to documents folder
            model_config: Model configuration object
            persist_directory: Where to store vector database
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            verbose: Enable verbose logging
        """
        self.documents_path = Path(documents_path)
        self.model_config = model_config
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.verbose = verbose
        
        # Token usage tracking for cost calculation
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # Validate documents path
        if not self.documents_path.exists():
            raise ValueError(f"Documents path does not exist: {documents_path}")
        
        # Get provider
        if model_config.provider not in self.PROVIDERS:
            raise ValueError(f"Unsupported provider: {model_config.provider}")
        
        self.provider = self.PROVIDERS[model_config.provider]
        
        # Initialize components
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        
        if verbose:
            logger.info(f"Initializing {model_config.provider.value} with model: {model_config.model_name}")
            if model_config.cost_per_1k_input > 0:
                logger.info(f"💰 Cost: ${model_config.cost_per_1k_input:.4f}/1K input, ${model_config.cost_per_1k_output:.4f}/1K output")
    
    def _initialize_models(self):
        """Initialize LLM and embeddings"""
        try:
            self.llm = self.provider.get_llm(self.model_config)
            self.embeddings = self.provider.get_embeddings(self.model_config)
            
            if self.verbose:
                logger.info("✅ Models initialized successfully")
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
            logger.info(f"📁 Scanning directory: {self.documents_path}")
        
        file_count = 0
        for file_path in self.documents_path.rglob('*'):
            if file_path.is_file():
                extension = file_path.suffix.lower()
                
                if extension in loaders_map:
                    file_count += 1
                    try:
                        loader_class = loaders_map[extension]
                        loader = loader_class(str(file_path))
                        docs = loader.load()
                        
                        for doc in docs:
                            doc.metadata['source'] = str(file_path)
                            doc.metadata['filename'] = file_path.name
                            doc.metadata['file_type'] = extension[1:]
                        
                        documents.extend(docs)
                        
                        if self.verbose:
                            logger.info(f"✅ Loaded: {file_path.name} ({len(docs)} pages)")
                    
                    except Exception as e:
                        logger.error(f"❌ Error loading {file_path.name}: {str(e)}")
        
        if self.verbose:
            logger.info(f"📊 Total files: {file_count}, Document chunks: {len(documents)}")
        
        return documents
    
    def create_vector_store(self, documents: List):
        """Create vector store from documents"""
        if self.verbose:
            logger.info("📄 Splitting documents into chunks...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        texts = text_splitter.split_documents(documents)
        
        if self.verbose:
            logger.info(f"📊 Created {len(texts)} text chunks")
            logger.info("🧠 Creating embeddings...")
            if self.model_config.cost_per_1k_input > 0:
                # Estimate embedding costs
                total_chars = sum(len(t.page_content) for t in texts)
                estimated_tokens = total_chars / 4  # Rough estimate
                cost = (estimated_tokens / 1000) * self.model_config.cost_per_1k_input * 0.1  # Embeddings are cheaper
                logger.info(f"💰 Estimated embedding cost: ${cost:.4f}")
        
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        self.vectorstore.persist()
        
        if self.verbose:
            logger.info("✅ Vector store created and persisted!")
    
    def setup_qa_chain(self, k: int = 3):
        """Setup the QA chain"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        # Provider-specific prompt templates
        if self.model_config.provider == ModelProvider.ANTHROPIC:
            template = """Human: You are a helpful assistant analyzing documents. 
Use the following context to answer the question.
If you don't know the answer based on the context, say "I don't have enough information in the documents to answer this question."

Context:
{context}

Question: {question}