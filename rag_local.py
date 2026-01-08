"""
Local RAG System - 100% Free with Ollama
Clean, production-ready implementation
"""

import os
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Suppress warnings and telemetry
warnings.filterwarnings('ignore')
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY'] = 'False'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress ChromaDB stderr messages
import io
_stderr = sys.stderr
sys.stderr = io.StringIO()

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        UnstructuredWordDocumentLoader
    )
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.llms import Ollama
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
finally:
    sys.stderr = _stderr


class FreeLocalRAG:
    """
    A completely free RAG system using Ollama for local LLM inference.
    No API costs, complete privacy, works offline.
    """
    
    def __init__(
        self, 
        documents_path: str,
        model_name: str = "llama3.2",
        embedding_model: str = "nomic-embed-text",
        persist_directory: str = "./chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        temperature: float = 0.0,
        verbose: bool = True
    ):
        """
        Initialize the RAG system.
        
        Args:
            documents_path: Path to documents folder
            model_name: Ollama model for generation (llama3.2, mistral, phi3, etc.)
            embedding_model: Ollama model for embeddings
            persist_directory: Where to store vector database
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
            temperature: LLM temperature (0 = deterministic)
            verbose: Whether to print progress messages
        """
        self.documents_path = Path(documents_path)
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.verbose = verbose
        
        # Validate documents path
        if not self.documents_path.exists():
            raise ValueError(f"Documents path does not exist: {documents_path}")
        
        self.vectorstore = None
        self.qa_chain = None
        self.model_name = model_name
        
        if verbose:
            logger.info(f"Initializing RAG with model: {model_name}")
        
        # Initialize embeddings with Ollama
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            show_progress=verbose
        )
        
        # Initialize LLM with Ollama
        callback_manager = None
        if verbose:
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        self.llm = Ollama(
            model=model_name,
            callback_manager=callback_manager,
            temperature=temperature,
            num_ctx=4096  # Context window
        )
    
    def load_documents(self) -> List:
        """
        Load all supported documents from the specified path.
        
        Returns:
            List of loaded documents
        """
        documents = []
        
        # Map file extensions to their respective loaders
        loaders_map = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader
        }
        
        if self.verbose:
            logger.info(f"Scanning directory: {self.documents_path}")
        
        # Recursively find all files
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
                        
                        # Add metadata
                        for doc in docs:
                            doc.metadata['source'] = str(file_path)
                            doc.metadata['filename'] = file_path.name
                            doc.metadata['file_type'] = extension[1:]  # Remove dot
                        
                        documents.extend(docs)
                        
                        if self.verbose:
                            logger.info(f"✅ Loaded: {file_path.name} ({len(docs)} pages)")
                    
                    except Exception as e:
                        logger.error(f"❌ Error loading {file_path.name}: {str(e)}")
        
        if self.verbose:
            logger.info(f"📊 Total files processed: {file_count}")
            logger.info(f"📄 Total document chunks: {len(documents)}")
        
        if not documents:
            logger.warning("⚠️ No documents found in the specified path!")
        
        return documents
    
    def create_vector_store(self, documents: List) -> None:
        """
        Create vector store from documents.
        
        Args:
            documents: List of documents to process
        """
        if self.verbose:
            logger.info("📄 Splitting documents into chunks...")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        texts = text_splitter.split_documents(documents)
        
        if self.verbose:
            logger.info(f"📊 Created {len(texts)} text chunks")
            logger.info("🧠 Creating vector embeddings (this may take a few minutes)...")
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Persist to disk
        self.vectorstore.persist()
        
        if self.verbose:
            logger.info("✅ Vector store created and persisted!")
    
    def setup_qa_chain(self, k: int = 3) -> None:
        """
        Setup the QA chain for querying.
        
        Args:
            k: Number of relevant chunks to retrieve
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Run create_vector_store first.")
        
        # Create a custom prompt template
        template = """You are a helpful assistant analyzing documents. 
Use the following context to answer the question.
If you don't know the answer based on the context, say "I don't have enough information in the documents to answer this question."

Context:
{context}

Question: {question}

Answer (in the same language as the question):"""
        
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
        
        # Setup QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": k}
            ),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            return_source_documents=True,
            verbose=False  # Set to True for debugging
        )
        
        if self.verbose:
            logger.info(f"📗 QA Chain configured (retrieving top {k} chunks)")
    
    def query(self, question: str, k: int = 3) -> Dict:
        """
        Query the RAG system.
        
        Args:
            question: The question to ask
            k: Number of relevant chunks to retrieve
            
        Returns:
            Dict containing answer and sources
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Run setup_qa_chain first.")
        
        if self.verbose:
            logger.info(f"🤔 Processing query: {question[:50]}...")
        
        # Update retriever k if different
        if k != self.qa_chain.retriever.search_kwargs.get("k", 3):
            self.qa_chain.retriever.search_kwargs["k"] = k
        
        # Execute query
        result = self.qa_chain({"query": question})
        
        # Format response
        response = {
            "answer": result["result"],
            "sources": [],
            "relevant_chunks": len(result.get("source_documents", []))
        }
        
        # Process source documents
        seen_sources = set()
        for doc in result.get("source_documents", []):
            filename = doc.metadata.get("filename", "Unknown")
            if filename not in seen_sources:
                seen_sources.add(filename)
                response["sources"].append({
                    "filename": filename,
                    "file_type": doc.metadata.get("file_type", "unknown"),
                    "preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
        
        return response
    
    def initialize(self) -> bool:
        """
        Complete initialization process.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.verbose:
                logger.info("🚀 Initializing RAG System...")
            
            # Load documents
            documents = self.load_documents()
            
            if not documents:
                logger.error("No documents found to process!")
                return False
            
            # Create vector store
            self.create_vector_store(documents)
            
            # Setup QA chain
            self.setup_qa_chain()
            
            if self.verbose:
                logger.info("✅ RAG System ready!")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize: {str(e)}")
            return False
    
    def update_model(self, model_name: str) -> None:
        """
        Change the LLM model.
        
        Args:
            model_name: New model name (must be available in Ollama)
        """
        self.model_name = model_name
        self.llm.model = model_name
        
        # Recreate QA chain with new model
        if self.vectorstore:
            self.setup_qa_chain()
        
        if self.verbose:
            logger.info(f"✅ Switched to model: {model_name}")


def check_ollama_installed() -> tuple[bool, str]:
    """
    Check if Ollama is installed and running.
    
    Returns:
        Tuple of (is_installed, message)
    """
    import subprocess
    
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, "Ollama is installed but not running. Run: ollama serve"
            
    except FileNotFoundError:
        return False, "Ollama is not installed. Visit: https://ollama.ai"
    except subprocess.TimeoutExpired:
        return False, "Ollama is not responding. Try: ollama serve"


def setup_nltk() -> None:
    """Download required NLTK data (for some document processors)."""
    try:
        import nltk
        import ssl
        
        # Handle SSL certificates
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Download required packages silently
        required_packages = ['punkt', 'averaged_perceptron_tagger']
        
        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}')
            except LookupError:
                nltk.download(package, quiet=True)
    except ImportError:
        pass  # NLTK is optional


def main():
    """Interactive CLI for the RAG system."""
    
    # Header
    print("=" * 60)
    print("🚀 Local RAG System - 100% Free with Ollama")
    print("=" * 60 + "\n")
    
    # Setup NLTK (optional)
    setup_nltk()
    
    # Check Ollama
    is_installed, message = check_ollama_installed()
    
    if not is_installed:
        print(f"⚠️ {message}")
        print("\n📥 To install Ollama:")
        print("   1. Visit: https://ollama.ai")
        print("   2. Download and install for your OS")
        print("   3. Run: ollama pull llama3.2")
        print("   4. Run: ollama pull nomic-embed-text")
        return
    
    print("✅ Ollama is installed and running!\n")
    print("Available models:")
    print(message)
    print()
    
    # Get documents path
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        print("📁 Enter the path to your documents folder:")
        print("   (Example: /Users/name/Documents)")
        folder_path = input("\nPath: ").strip()
    
    if not folder_path or not Path(folder_path).exists():
        print(f"\n❌ Invalid path: {folder_path}")
        return
    
    # Select model
    print("\n🤖 Available models:")
    models = {
        "1": ("llama3.2", "Fast and lightweight (2GB)"),
        "2": ("mistral", "Balanced performance (4GB)"),
        "3": ("llama3.1:8b", "Best quality (5GB, needs 16GB RAM)"),
        "4": ("phi3", "Microsoft's efficient model (2.3GB)")
    }
    
    for key, (name, desc) in models.items():
        print(f"   {key}. {name:<15} - {desc}")
    
    choice = input("\nSelect model (1-4) [default: 1]: ").strip() or "1"
    model_name = models.get(choice, models["1"])[0]
    
    print(f"\n✅ Using model: {model_name}")
    
    # Initialize RAG
    print("\n🔄 Initializing RAG system...")
    
    try:
        rag = FreeLocalRAG(
            documents_path=folder_path,
            model_name=model_name,
            verbose=True
        )
        
        if not rag.initialize():
            return
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return
    
    # Interactive query loop
    print("\n" + "=" * 60)
    print("💬 Ready! Ask questions about your documents")
    print("   Commands: 'exit' to quit, 'model' to change model")
    print("=" * 60 + "\n")
    
    while True:
        question = input("🤔 Question: ").strip()
        
        if question.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if question.lower() == 'model':
            print("\n🔄 Available models:")
            for key, (name, desc) in models.items():
                print(f"   {key}. {name}")
            
            choice = input("Select model: ").strip()
            if choice in models:
                new_model = models[choice][0]
                rag.update_model(new_model)
            continue
        
        if not question:
            continue
        
        try:
            # Query the system
            result = rag.query(question)
            
            # Display results
            print(f"\n💡 Answer:\n{result['answer']}\n")
            
            if result['sources']:
                print("📚 Sources:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"   {i}. {source['filename']} ({source['file_type']})")
            
            print("\n" + "-" * 60 + "\n")
        
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")


if __name__ == "__main__":
    main()
