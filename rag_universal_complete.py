
Assistant: Human!

Answer:"""
        elif self.model_config.provider == ModelProvider.GOOGLE:
            template = """You are a helpful assistant analyzing documents.
Use the following context to answer the question.
If the context doesn't contain the answer, say "I don't have enough information in the documents to answer this question."

Context:
{context}

Question: {question}

Answer:"""
        else:
            # Default template for OpenAI, HuggingFace, etc.
            template = """You are a helpful assistant analyzing documents.
Use the following context pieces to answer the question.
If you don't know the answer based on the context, say "I don't have enough information in the documents to answer this question."

Context:
{context}

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
            return_source_documents=True,
            verbose=False
        )
        
        if self.verbose:
            logger.info(f"📗 QA Chain configured (retrieving top {k} chunks)")
    
    def query(self, question: str, k: int = 3) -> Dict:
        """
        Query the RAG system
        
        Args:
            question: The question to ask
            k: Number of relevant chunks to retrieve
            
        Returns:
            Dict with answer, sources, and cost estimate
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Run initialize() first.")
        
        if self.verbose:
            logger.info(f"🤔 Processing query: {question[:50]}...")
        
        # Update retriever k if different
        if k != self.qa_chain.retriever.search_kwargs.get("k", 3):
            self.qa_chain.retriever.search_kwargs["k"] = k
        
        # Execute query
        result = self.qa_chain({"query": question})
        
        # Estimate tokens (rough approximation)
        input_tokens = len(question) / 4 + sum(len(doc.page_content) / 4 for doc in result.get("source_documents", []))
        output_tokens = len(result["result"]) / 4
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        # Calculate cost
        cost = (
            (input_tokens / 1000) * self.model_config.cost_per_1k_input +
            (output_tokens / 1000) * self.model_config.cost_per_1k_output
        )
        
        # Format response
        response = {
            "answer": result["result"],
            "sources": [],
            "relevant_chunks": len(result.get("source_documents", [])),
            "estimated_cost": cost,
            "total_cost": self.get_total_cost(),
            "model": self.model_config.model_name,
            "provider": self.model_config.provider.value
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
        """Complete initialization process"""
        try:
            if self.verbose:
                logger.info("🚀 Initializing Universal RAG System...")
                logger.info(f"Provider: {self.model_config.provider.value}")
                logger.info(f"Model: {self.model_config.model_name}")
            
            # Initialize models
            self._initialize_models()
            
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
                if self.model_config.cost_per_1k_input > 0:
                    logger.info(f"💰 Using paid model - costs will be tracked")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize: {str(e)}")
            return False
    
    def switch_model(self, new_config: ModelConfig):
        """Switch to a different model without recreating vector store"""
        if self.verbose:
            logger.info(f"Switching from {self.model_config.model_name} to {new_config.model_name}")
        
        self.model_config = new_config
        
        # Re-initialize models
        self._initialize_models()
        
        # Recreate QA chain with new model
        if self.vectorstore:
            self.setup_qa_chain()
        
        if self.verbose:
            logger.info(f"✅ Switched to {new_config.model_name}")


def estimate_costs(documents_path: str, queries_per_day: int = 10):
    """Estimate costs for different models"""
    from pathlib import Path
    
    # Count approximate tokens in documents
    total_chars = 0
    for file_path in Path(documents_path).rglob('*.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                total_chars += len(f.read())
        except:
            pass
    
    for file_path in Path(documents_path).rglob('*.pdf'):
        total_chars += 3000  # Rough estimate per PDF page
    
    estimated_tokens = total_chars / 4
    
    print("\n💰 COST COMPARISON (Monthly estimates for {queries_per_day} queries/day)\n")
    print("-" * 80)
    print(f"{'Model':<25} {'Provider':<12} {'Input $/1K':<12} {'Output $/1K':<12} {'Monthly $':<12}")
    print("-" * 80)
    
    models_to_compare = [
        ("FREE - Llama 3.2", ModelConfigurations.OLLAMA_LLAMA),
        ("FREE - HF Phi-2", ModelConfigurations.HUGGINGFACE_PHI),
        ("GPT-3.5 Turbo", ModelConfigurations.OPENAI_GPT35),
        ("GPT-4 Turbo", ModelConfigurations.OPENAI_GPT4),
        ("GPT-4o", ModelConfigurations.OPENAI_GPT4O),
        ("Claude 3 Haiku", ModelConfigurations.ANTHROPIC_HAIKU),
        ("Claude 3.5 Sonnet", ModelConfigurations.ANTHROPIC_SONNET),
        ("Claude 3 Opus", ModelConfigurations.ANTHROPIC_OPUS),
        ("Gemini 1.5 Flash", ModelConfigurations.GOOGLE_GEMINI_FLASH),
        ("Gemini 1.5 Pro", ModelConfigurations.GOOGLE_GEMINI_PRO),
    ]
    
    for name, config in models_to_compare:
        # Estimate monthly cost
        daily_input = (estimated_tokens * 0.1 * queries_per_day) / 1000  # Assume 10% of docs per query
        daily_output = (500 * queries_per_day) / 1000  # Assume 500 tokens per response
        
        monthly_cost = 30 * (
            daily_input * config.cost_per_1k_input +
            daily_output * config.cost_per_1k_output
        )
        
        print(f"{name:<25} {config.provider.value:<12} ${config.cost_per_1k_input:<11.5f} ${config.cost_per_1k_output:<11.5f} ${monthly_cost:<11.2f}")
    
    print("-" * 80)
    print(f"\n📊 Based on ~{estimated_tokens:,.0f} tokens in documents and {queries_per_day} queries/day")


def main():
    """Interactive CLI for Universal RAG"""
    import sys
    from getpass import getpass
    
    print("=" * 80)
    print("🌍 UNIVERSAL RAG SYSTEM - Free & Premium Models")
    print("=" * 80 + "\n")
    
    # Get documents path
    if len(sys.argv) > 1:
        docs_path = sys.argv[1]
    else:
        docs_path = input("📁 Enter documents folder path: ").strip()
    
    if not Path(docs_path).exists():
        print(f"❌ Path not found: {docs_path}")
        return
    
    # Show cost comparison
    estimate_costs(docs_path)
    
    # Select provider category
    print("\n🎯 Select Provider Category:")
    print("1. FREE Local Models (Ollama/HuggingFace)")
    print("2. OpenAI (GPT-3.5/GPT-4)")
    print("3. Anthropic (Claude)")
    print("4. Google (Gemini)")
    print("5. Other (Cohere/Mistral)")
    
    choice = input("\nChoice (1-5): ").strip()
    
    config = None
    
    if choice == "1":
        # Free models
        print("\n🆓 FREE Models:")
        print("1. Ollama Llama 3.2")
        print("2. HuggingFace Phi-2")
        
        model_choice = input("Choice: ").strip()
        
        if model_choice == "1":
            # Check if Ollama is installed
            try:
                import subprocess
                subprocess.run(["ollama", "list"], capture_output=True, check=True)
                config = ModelConfigurations.OLLAMA_LLAMA
            except:
                print("⚠️ Ollama not installed. Install from: https://ollama.ai")
                return
        else:
            config = ModelConfigurations.HUGGINGFACE_PHI
    
    elif choice == "2":
        # OpenAI
        api_key = getpass("🔑 Enter OpenAI API key: ").strip()
        
        print("\nOpenAI Models:")
        print("1. GPT-3.5 Turbo ($0.0005/1K input)")
        print("2. GPT-4 Turbo ($0.01/1K input)")
        print("3. GPT-4o ($0.005/1K input)")
        
        model_choice = input("Choice: ").strip()
        
        if model_choice == "1":
            config = ModelConfigurations.OPENAI_GPT35
        elif model_choice == "2":
            config = ModelConfigurations.OPENAI_GPT4
        else:
            config = ModelConfigurations.OPENAI_GPT4O
        
        config.api_key = api_key
    
    elif choice == "3":
        # Anthropic
        api_key = getpass("🔑 Enter Anthropic API key: ").strip()
        
        print("\nAnthropic Models:")
        print("1. Claude 3 Haiku ($0.00025/1K input)")
        print("2. Claude 3.5 Sonnet ($0.003/1K input)")
        print("3. Claude 3 Opus ($0.015/1K input)")
        
        model_choice = input("Choice: ").strip()
        
        if model_choice == "1":
            config = ModelConfigurations.ANTHROPIC_HAIKU
        elif model_choice == "2":
            config = ModelConfigurations.ANTHROPIC_SONNET
        else:
            config = ModelConfigurations.ANTHROPIC_OPUS
        
        config.api_key = api_key
    
    elif choice == "4":
        # Google
        api_key = getpass("🔑 Enter Google API key: ").strip()
        
        print("\nGoogle Models:")
        print("1. Gemini 1.5 Flash ($0.000075/1K input)")
        print("2. Gemini 1.5 Pro ($0.00125/1K input)")
        
        model_choice = input("Choice: ").strip()
        
        if model_choice == "1":
            config = ModelConfigurations.GOOGLE_GEMINI_FLASH
        else:
            config = ModelConfigurations.GOOGLE_GEMINI_PRO
        
        config.api_key = api_key
    
    else:
        print("Not implemented yet")
        return
    
    if not config:
        print("❌ No model selected")
        return
    
    # Initialize RAG
    print(f"\n🔄 Initializing with {config.model_name}...")
    
    rag = UniversalRAG(
        documents_path=docs_path,
        model_config=config,
        verbose=True
    )
    
    if not rag.initialize():
        return
    
    # Query loop
    print("\n" + "=" * 80)
    print("💬 Ready! Ask questions (type 'exit' to quit, 'switch' to change model)")
    print("=" * 80 + "\n")
    
    while True:
        question = input("❓ Question: ").strip()
        
        if question.lower() in ['exit', 'quit', 'q']:
            if rag.get_total_cost() > 0:
                print(f"\n💰 Total session cost: ${rag.get_total_cost():.6f}")
            break
        
        if question.lower() == 'switch':
            print("\n🔄 Switch to which model?")
            print("1. GPT-3.5 (cheap)")
            print("2. GPT-4 (quality)")
            print("3. Claude Haiku (cheapest)")
            print("4. Gemini Flash (ultra-cheap)")
            
            switch_choice = input("Choice: ").strip()
            
            # Would implement model switching here
            print("Model switching implemented in full version")
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
