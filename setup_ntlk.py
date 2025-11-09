#!/usr/bin/env python3
"""
Setup NLTK - Esegui UNA VOLTA per installare tutto
"""

import nltk
import ssl

print("🔧 Installing all NLTK packages for RAG...")
print("=" * 60)

# Fix SSL certificate issues on Mac
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Lista completa di pacchetti necessari
packages = [
    'punkt',
    'punkt_tab',
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng',
    'wordnet',
    'omw-1.4',
    'stopwords',
    'brown'
]

print("\n📥 Downloading packages...\n")

for package in packages:
    try:
        print(f"   → {package}...", end=" ")
        nltk.download(package, quiet=True)
        print("✅")
    except Exception as e:
        print(f"⚠️  (errore: {e})")

print("\n" + "=" * 60)
print("✅ Setup NLTK completato!")
print("=" * 60)
print("\nOra puoi eseguire il RAG senza errori:")
print("   python rag_free_ollama.py")
print()