"""
Alternative RAG implementation using HuggingFace models
No Ollama installation required - uses transformers library directly
"""

import os
import warnings
from pathlib import Path
from typing import List, Dict, Optional
import logging
import torch

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig


class HuggingFaceRAG:
    """
    RAG system using HuggingFace models directly.
    No Ollama required - downloads models from HuggingFace Hub.
    """
    
    # Recommended free models
    MODELS = {
        "fast": {
            "name": "microsoft/phi-2",
            "size": "2.7B",
            "description": "Fast and efficient"
        },
        "balanced": {
            "name": "mistralai/Mistral-7B-Instruct-v0.2",
            "size": "7B", 
            "description": "Good balance of speed and quality"
        },
        "quality": {
            "name": "meta-llama/Llama-2-7b-chat-hf",
            "size": "7B",
            "description": "High quality (requires HF token)"
        },
        "tiny": {
            "name": "microsoft/DialoGPT-medium",
            "size": "345M",
            "description": "Very small, for limited resources"
        }
    }
    
    EMBEDDING_MODELS = {
        "default": "sentence-transformers/all-MiniLM-L6-v2",  # 384 dimensions, 80MB
        "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 384 dim, 420MB
        "large": "sentence-transformers/all-mpnet-base-v2"  # 768 dimensions, 420MB
    }
    
    def __init__(
        self,
        documents_path: str,
        model_type: str = "fast",
        embedding_model: str = "default",
        persist_directory: str = "./chroma_db_hf",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_quantization: bool = True,
        device: str = None
    ):
        """
        Initialize HuggingFace RAG system.
        
        Args:
            documents_path: Path to documents
            model_type: One of "fast", "balanced", "quality", "tiny"
            embedding_model: One of "default", "multilingual", "large"
            persist_directory: Where to store vector DB
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            use_quantization: Use 8-bit quantization to reduce memory usage
            device: "cuda", "cpu", or None (auto-detect)
        """
        self.documents_path = Path(documents_path)
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Check for GPU
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Validate paths
        if not self.documents_path.exists():
            raise ValueError(f"Documents path does not exist: {documents_path}")
        
        # Get model configuration
        if model_type not in self.MODELS:
            raise ValueError(f"Unknown model type: {model_type}. Choose from {list(self.MODELS.keys())}")
        
        self.model_config = self.MODELS[model_type]
        self.model_name = self.model_config["name"]
        
        logger.info(f"Loading model: {self.model_name} ({self.model_config['description']})")
        
        # Initialize embeddings
        embedding_model_name = self.EMBEDDING_MODELS.get(embedding_model, self.EMBEDDING_MODELS["default"])
        logger.info(f"Loading embeddings: {embedding_model_name}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize LLM
        self.llm = self._initialize_llm(use_quantization)
        
        self.vectorstore = None
        self.qa_chain = None
    
    def _initialize_llm(self, use_quantization: bool):
        """
        Initialize the language model.
        
        Args:
            use_quantization: Whether to use 8-bit quantization
            
        Returns:
            HuggingFacePipeline instance
        """
        logger.info("Loading language model (this may take a few minutes on first run)...")
        
        # Quantization config for reduced memory usage
        quantization_config = None
        if use_quantization and self.device == "cuda":
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
                logger.info("Using 8-bit quantization to reduce memory usage")
            except ImportError:
                logger.warning("bitsandbytes not installed, skipping quantization")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with appropriate settings
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if self.device == "cuda" else None,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15,
            do_sample=True,
            device=self.device if self.device == "cpu" else -1  # -1 for GPU with device_map="auto"
        )
        
        # Wrap in LangChain
        llm = HuggingFacePipeline(pipeline=pipe)
        
        logger.info("✅ Model loaded successfully!")
        return llm
    
    def load_documents(self) -> List:
        """Load documents from the specified path."""
        documents = []
        
        loaders_map = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader
        }
        
        logger.info(f"Scanning directory: {self.documents_path}")
        
        for file_path in self.documents_path.rglob('*'):
            if file_path.is_file():
                extension = file_path.suffix.lower()
                
                if extension in loaders_map:
                    try:
                        loader_class = loaders_map[extension]
                        loader = loader_class(str(file_path))
                        docs = loader.load()
                        
                        for doc in docs:
                            doc.metadata['source'] = str(file_path)
                            doc.metadata['filename'] = file_path.name
                        
                        documents.extend(docs)
                        logger.info(f"✅ Loaded: {file_path.name}")
                    
                    except Exception as e:
                        logger.error(f"❌ Error loading {file_path.name}: {str(e)}")
        
        logger.info(f"📊 Total documents loaded: {len(documents)}")
        return documents
    
    def create_vector_store(self, documents: List):
        """Create vector store from documents."""
        logger.info("Splitting documents into chunks...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        
        texts = text_splitter.split_documents(documents)
        logger.info(f"Created {len(texts)} text chunks")
        
        logger.info("Creating embeddings (this may take a few minutes)...")
        
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        self.vectorstore.persist()
        logger.info("✅ Vector store created!")
    
    def setup_qa_chain(self, k: int = 3):
        """Setup the QA chain."""
        template = """Use the following context to answer the question.
If you don't know the answer, say you don't know.

Context: {context}

Question: {question}

Answer:"""
        
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": k}),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            return_source_documents=True
        )
        
        logger.info("✅ QA chain ready!")
    
    def query(self, question: str) -> Dict:
        """Query the RAG system."""
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized")
        
        logger.info(f"Processing query: {question[:50]}...")
        
        result = self.qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "sources": [
                {
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "content": doc.page_content[:200] + "..."
                }
                for doc in result.get("source_documents", [])
            ]
        }
    
    def initialize(self) -> bool:
        """Initialize the complete system."""
        try:
            documents = self.load_documents()
            
            if not documents:
                logger.error("No documents found!")
                return False
            
            self.create_vector_store(documents)
            self.setup_qa_chain()
            
            logger.info("✅ System ready!")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            return False


def main():
    """Main function for testing."""
    import sys
    
    print("=" * 60)
    print("🤗 HuggingFace RAG System (No Ollama Required)")
    print("=" * 60 + "\n")
    
    # Check for GPU
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ No GPU detected, using CPU (will be slower)")
    
    print()
    
    # Get documents path
    if len(sys.argv) > 1:
        docs_path = sys.argv[1]
    else:
        docs_path = input("📁 Enter documents folder path: ").strip()
    
    if not Path(docs_path).exists():
        print(f"❌ Path not found: {docs_path}")
        return
    
    # Select model
    print("\n🤖 Available models:")
    for key, config in HuggingFaceRAG.MODELS.items():
        print(f"   {key:<10} - {config['name']} ({config['size']}) - {config['description']}")
    
    model_type = input("\nSelect model type [fast]: ").strip() or "fast"
    
    # Initialize
    print("\n🔄 Initializing (first run will download models)...")
    
    rag = HuggingFaceRAG(
        documents_path=docs_path,
        model_type=model_type,
        use_quantization=True  # Reduce memory usage
    )
    
    if not rag.initialize():
        return
    
    # Query loop
    print("\n💬 Ready! Ask questions (type 'exit' to quit)\n")
    
    while True:
        question = input("❓ Question: ").strip()
        
        if question.lower() in ['exit', 'quit', 'q']:
            break
        
        if not question:
            continue
        
        try:
            result = rag.query(question)
            print(f"\n💡 Answer: {result['answer']}\n")
            print("📚 Sources:", ', '.join(s['filename'] for s in result['sources']))
            print("-" * 60 + "\n")
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")


if __name__ == "__main__":
    main()
