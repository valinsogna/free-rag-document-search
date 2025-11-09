# 🆓 Setup Completo - Versione GRATUITA con Ollama

## 🎯 Questa versione è 100% GRATIS!

Nessun costo API, tutto gira localmente sul tuo PC.

---

## 📋 Prerequisiti

- Python 3.8+
- 8GB+ RAM (16GB consigliato per modelli grandi)
- 10GB spazio disco per modelli

---

## 🚀 Installazione Passo-Passo

### 1️⃣ Installa Ollama

**Windows:**
```bash
# Scarica da: https://ollama.ai/download/windows
# Esegui l'installer
```

**Mac:**
```bash
# Scarica da: https://ollama.ai/download/mac
# Oppure con Homebrew:
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2️⃣ Verifica installazione

```bash
ollama --version
```

### 3️⃣ Scarica i modelli (GRATIS!)

```bash
# Modello per embeddings (necessario)
ollama pull nomic-embed-text

# Scegli UN modello LLM:

# Opzione 1: Leggero e veloce (CONSIGLIATO per iniziare)
ollama pull llama3.2

# Opzione 2: Bilanciato qualità/velocità
ollama pull mistral

# Opzione 3: Più potente (richiede più RAM)
ollama pull llama3.1:8b

# Opzione 4: Efficiente di Microsoft
ollama pull phi3
```

**Verifica modelli scaricati:**
```bash
ollama list
```

Dovresti vedere:
```
NAME                    ID              SIZE
nomic-embed-text        latest          274 MB
llama3.2                latest          2.0 GB
```

### 4️⃣ Installa dipendenze Python

```bash
# Crea virtual environment
python -m venv venv

# Attiva (Linux/Mac)
source venv/bin/activate

# Attiva (Windows)
venv\Scripts\activate

# Installa dipendenze
pip install -r requirements_free.txt
```

### 5️⃣ Prepara documenti

```bash
mkdir documents
# Copia i tuoi PDF, DOCX, TXT qui
```

### 6️⃣ Esegui il sistema

```bash
python rag_free_ollama.py
```

---

## 🎮 Come Usare

```
🤔 Domanda: Quali sono i progetti principali nel mio CV?

💭 Thinking...
💡 Risposta: I progetti principali sono...

📚 Fonti:
   1. curriculum.pdf
```

**Comandi speciali:**
- `exit` - Esci
- `model` - Cambia modello LLM
- `q` - Esci

---

## 🔧 Configurazione Avanzata

### Cambiare modello nel codice

```python
rag = FreeLocalRAG(
    documents_path="./documents",
    model_name="mistral"  # Cambia qui!
)
```

### Modelli disponibili

| Modello | Dimensione | RAM | Velocità | Qualità | Uso |
|---------|------------|-----|----------|---------|-----|
| `llama3.2` | 2GB | 8GB | ⚡⚡⚡ | ⭐⭐⭐ | Generale |
| `mistral` | 4GB | 8GB | ⚡⚡ | ⭐⭐⭐⭐ | Bilanciato |
| `llama3.1:8b` | 5GB | 16GB | ⚡⚡ | ⭐⭐⭐⭐⭐ | Qualità |
| `phi3` | 2.3GB | 8GB | ⚡⚡⚡ | ⭐⭐⭐ | Efficiente |
| `qwen2.5` | varia | 8GB+ | ⚡⚡ | ⭐⭐⭐⭐ | Multilingua |

### Ottimizzare le prestazioni

```python
# Nel file rag_free_ollama.py modifica:

# 1. Riduci chunk size per risposte più veloci
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # era 1000
    chunk_overlap=100
)

# 2. Riduci numero di chunks recuperati
retriever=self.vectorstore.as_retriever(
    search_kwargs={"k": 2}  # era 3
)

# 3. Usa modello più leggero
model_name="llama3.2"  # invece di llama3.1:8b
```
---

## Usa la modalità Web UI con Streamlit
   ```bash
   pip install streamlit
   streamlit run app_streamlit_free.py
   ```

---

## 🐛 Troubleshooting

### Problema: "Ollama non trovato"
```bash
# Verifica che Ollama sia installato
ollama --version

# Avvia il servizio
ollama serve
```

### Problema: "Modello non trovato"
```bash
# Scarica il modello
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Problema: "Out of memory"
- Chiudi altre applicazioni
- Usa un modello più piccolo (`llama3.2` invece di `llama3.1:8b`)
- Riduci `chunk_size` nel codice

### Problema: "Troppo lento"
- Usa un modello più piccolo
- Riduci il numero di documenti
- Riduci `k` nel retriever (meno chunks)

---


## 🔄 Prossimi Passi

2. **Deploy locale:**
   - Crea Docker container
   - Condividi sulla rete locale

3. **Espandi funzionalità:**
   - Aggiungi più formati file
   - Implementa conversazione multi-turn
   - Aggiungi analytics

---

## 📞 Supporto

Se hai problemi:
1. Verifica che Ollama sia in esecuzione: `ollama list`
2. Verifica che i modelli siano scaricati
3. Controlla i log per errori

---

## ⚡ Quick Reference

```bash
# Setup completo in 4 comandi
ollama pull nomic-embed-text
ollama pull llama3.2
pip install -r requirements.txt
python rag_free_ollama.py
```

🎉 **Tutto GRATIS!** Nessun costo nascosto, nessuna API key richiesta!