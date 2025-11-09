#!/bin/bash
# Script Demo RAG per Video

clear

echo "╔════════════════════════════════════════════════════════╗"
echo "║     🤖 RAG DOCUMENT SEARCH - DEMO                      ║"
echo "║     Free, Local, Privacy-First                         ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
sleep 2

echo "📋 Step 1: Verifico Environment"
echo "================================"
conda activate rag
python --version
echo ""
sleep 2

echo "📋 Step 2: Verifico Ollama"
echo "================================"
ollama --version
echo ""
echo "Modelli installati:"
ollama list
echo ""
sleep 3

echo "📋 Step 3: Analizzo Documenti"
echo "================================"
echo "Numero di file da indicizzare:"
find ~/Desktop -name "*.pdf" | wc -l
echo ""
sleep 2

echo "📋 Step 4: Avvio RAG System"
echo "================================"
echo "Premi INVIO per continuare..."
read

# Qui partirà il tuo RAG
python rag_free_ollama.py 

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║     ✅ DEMO COMPLETATA!                                ║"
echo "╚════════════════════════════════════════════════════════╝"