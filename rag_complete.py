"""
Universal RAG System - Complete Implementation
Support for Free (Ollama/HuggingFace) and Commercial Models (OpenAI/Google/Anthropic)
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Core imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain



class ModelProvider(Enum):
    """Available model providers"""
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
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
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


class ModelConfigurations:
    """Pre-configured model settings"""
    
    # Free Models
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
            from langchain_community.llms import HuggingFacePipeline
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
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                do_sample=True,
                top_p=0.95
            )
            
            return HuggingFacePipeline(pipeline=pipe)
        except ImportError:
            raise ImportError("Install transformers: pip install transformers torch")
    
    def get_embeddings(self, config: ModelConfig):
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=config.embedding_model)
        except ImportError:
            raise ImportError("Install: pip install sentence-transformers")


class OpenAIProvider(BaseRAGProvider):
    """Provider for OpenAI models"""
    
    def get_llm(self, config: ModelConfig):
        try:
            from langchain_openai import ChatOpenAI
            
            if not config.api_key:
                config.api_key = os.getenv("OPENAI_API_KEY")
                if not config.api_key:
                    raise ValueError("OpenAI API key required")
            
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                api_key=config.api_key
            )
        except ImportError:
            raise ImportError("Install: pip install langchain-openai")
    
    def get_embeddings(self, config: ModelConfig):
        try:
            from langchain_openai import OpenAIEmbeddings
            
            if not config.api_key:
                config.api_key = os.getenv("OPENAI_API_KEY")
                if not config.api_key:
                    raise ValueError("OpenAI API key required")
            
            return OpenAIEmbeddings(
                model=config.embedding_model,
                api_key=config.api_key
            )
        except ImportError:
            raise ImportError("Install: pip install langchain-openai")


class AnthropicProvider(BaseRAGProvider):
    """Provider for Anthropic Claude models"""
    
    def get_llm(self, config: ModelConfig):
        try:
            from langchain_anthropic import ChatAnthropic
            
            if not config.api_key:
                config.api_key = os.getenv("ANTHROPIC_API_KEY")
                if not config.api_key:
                    raise ValueError("Anthropic API key required")
            
            return ChatAnthropic(
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                anthropic_api_key=config.api_key
            )
        except ImportError:
            raise ImportError("Install: pip install langchain-anthropic")
    
    def get_embeddings(self, config: ModelConfig):
        # Anthropic doesn't provide embeddings, use HuggingFace
        from langchain_community.embeddings import HuggingFaceEmbeddings
        logger.warning("Using HuggingFace embeddings for Anthropic")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


class GoogleProvider(BaseRAGProvider):
    """Provider for Google Gemini models"""
    
    def get_llm(self, config: ModelConfig):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            if not config.api_key:
                config.api_key = os.getenv("GOOGLE_API_KEY")
                if not config.api_key:
                    raise ValueError("Google API key required")
            
            return ChatGoogleGenerativeAI(
                model=config.model_name,
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
                google_api_key=config.api_key
            )
        except ImportError:
            raise ImportError("Install: pip install langchain-google-genai")
    
    def get_embeddings(self, config: ModelConfig):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            
            if not config.api_key:
                config.api_key = os.getenv("GOOGLE_API_KEY")
                if not config.api_key:
                    raise ValueError("Google API key required")
            
            return GoogleGenerativeAIEmbeddings(
                model=config.embedding_model,
                google_api_key=config.api_key
            )
        except ImportError:
            raise ImportError("Install: pip install langchain-google-genai")


class UniversalRAG:
    """Universal RAG system supporting multiple providers"""
    
    PROVIDERS = {
        ModelProvider.OLLAMA: OllamaProvider(),
        ModelProvider.HUGGINGFACE: HuggingFaceProvider(),
        ModelProvider.OPENAI: OpenAIProvider(),
        ModelProvider.ANTHROPIC: AnthropicProvider(),
        ModelProvider.GOOGLE: GoogleProvider()
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
        """Initialize Universal RAG System"""
        self.documents_path = Path(documents_path)
        self.model_config = model_config
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.verbose = verbose
        
        # Cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # Validate
        if not self.documents_path.exists():
            raise ValueError(f"Path not found: {documents_path}")
        
        if model_config.provider not in self.PROVIDERS:
            raise ValueError(f"Unsupported provider: {model_config.provider}")
        
        self.provider = self.PROVIDERS[model_config.provider]
        
        # Components
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        
        if verbose:
            logger.info(f"Using {model_config.provider.value} - {model_config.model_name}")
            if model_config.cost_per_1k_input > 0:
                logger.info(f"💰 Costs: ${model_config.cost_per_1k_input:.4f}/1K in, ${model_config.cost_per_1k_output:.4f}/1K out")
    
    def _initialize_models(self):
        """Initialize LLM and embeddings"""
        try:
            self.llm = self.provider.get_llm(self.model_config)
            self.embeddings = self.provider.get_embeddings(self.model_config)
            
            if self.verbose:
                logger.info("✅ Models initialized")
        except Exception as e:
            logger.error(f"Failed to initialize: {str(e)}")
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
            logger.info(f"📁 Scanning: {self.documents_path}")
        
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
                        logger.error(f"❌ Error: {file_path.name}: {str(e)}")
        
        if self.verbose:
            logger.info(f"📊 Total files: {file_count}, chunks: {len(documents)}")
        
        return documents
    
    def create_vector_store(self, documents: List):
        """Create vector store from documents"""
        if self.verbose:
            logger.info("📄 Splitting documents...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
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
        
        self.vectorstore.persist()
        
        if self.verbose:
            logger.info("✅ Vector store created!")
    
    def setup_qa_chain(self, k: int = 3):
        """Setup the QA chain (LangChain 0.2+)"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")

        # Prompt base (chat-style, richiesto dai nuovi LLM)
        prompt = ChatPromptTemplate.from_template(
            """Use the following context to answer the question.
            If you don't know the answer, say you don't know.

            Context:
            {context}

            Question:
            {input}

            Answer:"""
        )

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt
        )

        self.qa_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=document_chain
        )

        if self.verbose:
            logger.info(f"📗 QA Chain ready (k={k})")

    
    def query(self, question: str, k: int = 3) -> Dict:
        """Query the RAG system"""
        if not self.qa_chain:
            raise ValueError("Not initialized. Run initialize() first")
        
        if self.verbose:
            logger.info(f"🤔 Query: {question[:50]}...")
        
        # Update k if different
        if k != self.qa_chain.retriever.search_kwargs.get("k", 3):
            self.qa_chain.retriever.search_kwargs["k"] = k
        
        # Execute
        result = self.qa_chain.invoke({"input": question})
        answer = result["answer"]
        docs = result.get("context", [])

        
        # Estimate tokens (rough)
        input_tokens = len(question) / 4
        for doc in docs:
            input_tokens += len(doc.page_content) / 4
        output_tokens = len(answer) / 4
        
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
            "provider": self.model_config.provider.value
        }
        
        # Process sources
        seen = set()
        for doc in result.get("source_documents", []):
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
            self.setup_qa_chain()
            
            if self.verbose:
                logger.info("✅ Ready!")
                if self.model_config.cost_per_1k_input > 0:
                    logger.info("💰 Using paid model - tracking costs")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed: {str(e)}")
            return False
    
    def switch_model(self, new_config: ModelConfig):
        """Switch to different model without recreating vector store"""
        if self.verbose:
            logger.info(f"Switching to {new_config.model_name}")
        
        self.model_config = new_config
        self._initialize_models()
        
        if self.vectorstore:
            self.setup_qa_chain()
        
        if self.verbose:
            logger.info(f"✅ Switched to {new_config.model_name}")


def main():
    """Interactive CLI"""
    import sys
    from getpass import getpass
    
    print("=" * 60)
    print("🌍 UNIVERSAL RAG SYSTEM")
    print("=" * 60 + "\n")
    
    # Get documents path
    if len(sys.argv) > 1:
        docs_path = sys.argv[1]
    else:
        docs_path = input("📁 Documents path: ").strip()
    
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
            config = ModelConfigurations.OLLAMA_LLAMA
        else:
            config = ModelConfigurations.HUGGINGFACE_PHI
    
    elif choice == "2":
        api_key = getpass("🔑 OpenAI API key: ").strip()
        
        print("\n1. GPT-3.5 ($0.0005/1K)")
        print("2. GPT-4 ($0.01/1K)")
        print("3. GPT-4o ($0.005/1K)")
        
        model_choice = input("Choice: ").strip()
        
        if model_choice == "1":
            config = ModelConfigurations.OPENAI_GPT35
        elif model_choice == "2":
            config = ModelConfigurations.OPENAI_GPT4
        else:
            config = ModelConfigurations.OPENAI_GPT4O
        
        config.api_key = api_key
    
    elif choice == "3":
        api_key = getpass("🔑 Anthropic API key: ").strip()
        
        print("\n1. Haiku ($0.00025/1K)")
        print("2. Sonnet ($0.003/1K)")
        print("3. Opus ($0.015/1K)")
        
        model_choice = input("Choice: ").strip()
        
        if model_choice == "1":
            config = ModelConfigurations.ANTHROPIC_HAIKU
        elif model_choice == "2":
            config = ModelConfigurations.ANTHROPIC_SONNET
        else:
            config = ModelConfigurations.ANTHROPIC_OPUS
        
        config.api_key = api_key
    
    elif choice == "4":
        api_key = getpass("🔑 Google API key: ").strip()
        
        print("\n1. Gemini Flash ($0.000075/1K)")
        print("2. Gemini Pro ($0.00125/1K)")
        
        model_choice = input("Choice: ").strip()
        
        if model_choice == "1":
            config = ModelConfigurations.GOOGLE_GEMINI_FLASH
        else:
            config = ModelConfigurations.GOOGLE_GEMINI_PRO
        
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
    
    # Query loop
    print("\n" + "=" * 60)
    print("💬 Ready! ('exit' to quit, 'cost' for total)")
    print("=" * 60 + "\n")
    
    while True:
        question = input("❓ Question: ").strip()
        
        if question.lower() in ['exit', 'quit', 'q']:
            if rag.get_total_cost() > 0:
                print(f"\n💰 Total cost: ${rag.get_total_cost():.6f}")
            break
        
        if question.lower() == 'cost':
            print(f"💰 Current cost: ${rag.get_total_cost():.6f}\n")
            continue
        
        if not question:
            continue
        
        try:
            result = rag.query(question)
            
            print(f"\n💡 Answer: {result['answer']}\n")
            print(f"📚 Sources: {', '.join(s['filename'] for s in result['sources'])}")
            
            if result['estimated_cost'] > 0:
                print(f"💰 Cost: ${result['estimated_cost']:.6f} (Total: ${result['total_cost']:.6f})")
            
            print("-" * 60 + "\n")
        
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")


if __name__ == "__main__":
    main()
