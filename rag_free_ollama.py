"""
RAG System 100% GRATUITO con Ollama
Nessun costo, tutto locale sul tuo PC!
"""

import os
from pathlib import Path
from typing import List
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
import sys

class FreeLocalRAG:
    """RAG completamente gratuito usando Ollama"""
    
    def __init__(
        self, 
        documents_path: str,
        model_name: str = "llama3.2",  # Modello gratis da Ollama
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize FREE RAG system
        
        Args:
            documents_path: Path to documents folder
            model_name: Ollama model to use (default: llama3.2)
                       Altri modelli: mistral, phi3, llama3.1, qwen2.5
            persist_directory: Vector DB storage path
        """
        self.documents_path = Path(documents_path)
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.qa_chain = None
        self.model_name = model_name
        
        print(f"🦙 Using Ollama model: {model_name}")
        
        # GRATIS: Embeddings locali con Ollama
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text"  # Modello embedding gratuito
        )
        
        # GRATIS: LLM locale con Ollama
        self.llm = Ollama(
            model=model_name,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            temperature=0
        )
    
    def load_documents(self) -> List:
        """Load all supported documents from the specified path"""
        documents = []
        
        loaders_map = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader
        }
        
        print(f"📁 Scanning directory: {self.documents_path}")
        
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
                        print(f"✅ Loaded: {file_path.name}")
                    
                    except Exception as e:
                        print(f"❌ Error loading {file_path.name}: {str(e)}")
        
        print(f"\n📊 Total documents loaded: {len(documents)}")
        return documents
    
    def create_vector_store(self, documents: List):
        """Create vector store from documents"""
        print("\n🔄 Splitting documents into chunks...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        texts = text_splitter.split_documents(documents)
        print(f"📝 Created {len(texts)} text chunks")
        
        print("\n🧠 Creating vector embeddings with Ollama (FREE)...")
        print("   ⏳ This may take a few minutes on first run...")
        
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        print("✅ Vector store created successfully!")
    
    def setup_qa_chain(self):
        """Setup the QA chain for querying"""
        
        template = """Usa i seguenti pezzi di contesto per rispondere alla domanda.
            Se non conosci la risposta, di' semplicemente che non lo sai.
            Rispondi in italiano in modo chiaro e conciso.

            Contesto:
            {context}
            
            Domanda: {question}
            
            Risposta:"""
        
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            return_source_documents=True
        )
        
        print("🔗 QA Chain configured!")
    
    def query(self, question: str) -> dict:
        """Query the RAG system"""
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized.")
        
        print(f"\n💭 Thinking...")
        result = self.qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "sources": [
                {
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "content": doc.page_content[:200] + "..."
                }
                for doc in result["source_documents"]
            ]
        }
    
    def initialize(self):
        """Complete initialization process"""
        print("🚀 Initializing FREE RAG System with Ollama...\n")
        
        documents = self.load_documents()
        
        if not documents:
            print("⚠️  No documents found!")
            return False
        
        self.create_vector_store(documents)
        self.setup_qa_chain()
        
        print("\n✅ FREE RAG System ready!")
        return True


def check_ollama_installed():
    """Check if Ollama is installed and running"""
    import subprocess
    
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def main():
    """Main function"""
    
    print("="*60)
    print("🆓 RAG System 100% GRATUITO con Ollama")
    print("="*60 + "\n")
    
    # Check Ollama
    if not check_ollama_installed():
        print("⚠️  Ollama non trovato!")
        print("\n📥 Installa Ollama:")
        print("   1. Vai su: https://ollama.ai")
        print("   2. Scarica e installa per il tuo OS")
        print("   3. Esegui: ollama pull llama3.2")
        print("   4. Esegui: ollama pull nomic-embed-text")
        print("\n💡 Modelli consigliati:")
        print("   - llama3.2 (3B) - Veloce, leggero")
        print("   - mistral (7B) - Bilanciato")
        print("   - llama3.1 (8B) - Più potente")
        return
    
    print("✅ Ollama trovato!\n")
    
    # Configuration of docume ts path from input:
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        print("📁 Inserisci il percorso della cartella da analizzare:")
        print("   (Es: /Users/nome/Desktop/Scrivania)")
        print()
        folder_path = input("Percorso: ").strip()
    
    if not folder_path:
        folder_path = "./documents"
        print(f"\n→ Uso cartella di default: {folder_path}\n")
        
    DOCUMENTS_PATH = folder_path
    MODEL_NAME = "llama3.2"  # Cambia: mistral, phi3, etc.
    
    # Initialize RAG
    rag = FreeLocalRAG(
        documents_path=DOCUMENTS_PATH,
        model_name=MODEL_NAME
    )
    
    if not rag.initialize():
        return
    
    # Interactive query loop
    print("\n" + "="*60)
    print("💬 Sistema Pronto! Fai domande sui tuoi documenti")
    print("   Scrivi 'exit' per uscire")
    print("   Scrivi 'model' per cambiare modello")
    print("="*60 + "\n")
    
    while True:
        question = input("🤔 Domanda: ").strip()
        
        if question.lower() in ['exit', 'quit', 'q']:
            print("👋 Ciao!")
            break
        
        if question.lower() == 'model':
            print("\n📋 Modelli disponibili:")
            print("   - llama3.2 (veloce)")
            print("   - mistral (bilanciato)")
            print("   - llama3.1 (potente)")
            print("   - phi3 (efficiente)")
            new_model = input("Scegli modello: ").strip()
            rag.llm.model = new_model
            print(f"✅ Cambiato a: {new_model}\n")
            continue
        
        if not question:
            continue
        
        try:
            result = rag.query(question)
            
            print(f"\n💡 Risposta: {result['answer']}\n")
            print("📚 Fonti:")
            for i, source in enumerate(result['sources'], 1):
                print(f"   {i}. {source['filename']}")
            
            print("\n" + "-"*60 + "\n")
        
        except Exception as e:
            print(f"❌ Errore: {str(e)}\n")


if __name__ == "__main__":
    main()