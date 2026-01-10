"""
Script di Analisi: Stima Tempi per la TUA Cartella
Analizza una cartella e ti dice esattamente quanto tempo ci vorrà
"""

import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
import humanize

# Configura humanize per italiano
humanize.i18n.activate("it_IT")


def analyze_folder(folder_path: str):
    """Analizza una cartella e stima i tempi di processing"""
    
    base = Path(folder_path)
    
    if not base.exists():
        print(f"❌ Cartella '{folder_path}' non trovata!")
        return
    
    print("="*80)
    print(f"📊 ANALISI CARTELLA: {folder_path}")
    print("="*80 + "\n")
    print("⏳ Scansione in corso...\n")
    
    # Statistiche
    stats = {
        'total_files': 0,
        'total_size': 0,
        'supported_files': 0,
        'supported_size': 0,
        'by_extension': defaultdict(lambda: {'count': 0, 'size': 0}),
        'by_age': defaultdict(lambda: {'count': 0, 'size': 0}),
        'large_files': [],  # File > 10 MB
    }
    
    supported_extensions = {'.pdf', '.txt', '.docx', '.doc'}
    
    # Calcola età dei file
    now = datetime.now()
    one_month = now - timedelta(days=30)
    six_months = now - timedelta(days=180)
    one_year = now - timedelta(days=365)
    
    # Scansiona ricorsivamente
    for file_path in base.rglob('*'):
        if file_path.is_file():
            try:
                stats['total_files'] += 1
                file_size = file_path.stat().st_size
                stats['total_size'] += file_size
                
                extension = file_path.suffix.lower()
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                # Per estensione
                stats['by_extension'][extension]['count'] += 1
                stats['by_extension'][extension]['size'] += file_size
                
                # Per età
                if file_time > one_month:
                    age = 'ultimo_mese'
                elif file_time > six_months:
                    age = 'ultimi_6_mesi'
                elif file_time > one_year:
                    age = 'ultimo_anno'
                else:
                    age = 'piu_vecchio'
                
                stats['by_age'][age]['count'] += 1
                stats['by_age'][age]['size'] += file_size
                
                # File supportati
                if extension in supported_extensions:
                    stats['supported_files'] += 1
                    stats['supported_size'] += file_size
                    
                    # File grandi
                    if file_size > 10 * 1024 * 1024:  # > 10 MB
                        stats['large_files'].append({
                            'path': file_path,
                            'size': file_size,
                            'name': file_path.name
                        })
            
            except (PermissionError, OSError) as e:
                # Ignora file inaccessibili
                pass
    
    # ==================== RISULTATI ====================
    
    print("="*80)
    print("📈 RISULTATI ANALISI")
    print("="*80 + "\n")
    
    # Statistiche generali
    print("📁 STATISTICHE GENERALI")
    print("-" * 80)
    print(f"File totali trovati:        {stats['total_files']:,}")
    print(f"Dimensione totale:          {humanize.naturalsize(stats['total_size'], binary=True)}")
    print(f"Dimensione media per file:  {humanize.naturalsize(stats['total_size'] // stats['total_files'] if stats['total_files'] > 0 else 0, binary=True)}")
    
    print(f"\n✅ File supportati (PDF/DOCX/TXT/DOC):")
    print(f"   Numero:     {stats['supported_files']:,} ({stats['supported_files']/stats['total_files']*100:.1f}% del totale)")
    print(f"   Dimensione: {humanize.naturalsize(stats['supported_size'], binary=True)}")
    
    unsupported = stats['total_files'] - stats['supported_files']
    print(f"\n⏭️  File non supportati (verranno ignorati):")
    print(f"   Numero:     {unsupported:,} ({unsupported/stats['total_files']*100:.1f}% del totale)")
    
    # Per tipo
    print(f"\n📊 DISTRIBUZIONE PER TIPO")
    print("-" * 80)
    
    sorted_ext = sorted(stats['by_extension'].items(), 
                       key=lambda x: x[1]['count'], 
                       reverse=True)[:10]
    
    for ext, data in sorted_ext:
        ext_name = ext if ext else '(no extension)'
        supported = "✅" if ext in supported_extensions else "⏭️ "
        print(f"{supported} {ext_name:15} {data['count']:6,} file(s)  "
              f"{humanize.naturalsize(data['size'], binary=True):>10}")
    
    # Per età
    print(f"\n📅 DISTRIBUZIONE PER ETÀ")
    print("-" * 80)
    age_labels = {
        'ultimo_mese': 'Ultimo mese',
        'ultimi_6_mesi': '1-6 mesi fa',
        'ultimo_anno': '6-12 mesi fa',
        'piu_vecchio': 'Più di 1 anno'
    }
    
    for age_key in ['ultimo_mese', 'ultimi_6_mesi', 'ultimo_anno', 'piu_vecchio']:
        if age_key in stats['by_age']:
            data = stats['by_age'][age_key]
            print(f"{age_labels[age_key]:15} {data['count']:6,} file(s)  "
                  f"{humanize.naturalsize(data['size'], binary=True):>10}")
    
    # File grandi
    if stats['large_files']:
        print(f"\n⚠️  FILE GRANDI (> 10 MB) - Rallenteranno il processing")
        print("-" * 80)
        sorted_large = sorted(stats['large_files'], key=lambda x: x['size'], reverse=True)[:10]
        for f in sorted_large:
            print(f"   {humanize.naturalsize(f['size'], binary=True):>10}  {f['name'][:60]}")
    
    # ==================== STIME TEMPORALI ====================
    
    print("\n" + "="*80)
    print("⏱️  STIMA TEMPI DI PROCESSING")
    print("="*80 + "\n")
    
    # Calcoli
    num_files = stats['supported_files']
    total_size_mb = stats['supported_size'] / (1024 * 1024)
    
    # Stime basate su benchmark reali
    # Ollama: ~15-20 file/minuto, ~5-8 MB/minuto
    # OpenAI: ~40-60 file/minuto, ~15-25 MB/minuto
    
    ollama_time_files = num_files / 17  # minuti
    ollama_time_size = total_size_mb / 6.5  # minuti
    ollama_time = max(ollama_time_files, ollama_time_size)
    
    openai_time_files = num_files / 50  # minuti
    openai_time_size = total_size_mb / 20  # minuti
    openai_time = max(openai_time_files, openai_time_size)
    
    # Costi OpenAI (approssimativi)
    # Embeddings: ~$0.0001 per 1K tokens
    # Stima: ~500 tokens per pagina, ~10 pagine per documento
    estimated_tokens = num_files * 5000  # tokens
    embedding_cost = (estimated_tokens / 1000) * 0.0001
    
    print("🦙 OLLAMA (Locale - GRATIS)")
    print("-" * 80)
    print(f"⏰ Tempo prima indicizzazione:  {int(ollama_time)} - {int(ollama_time * 1.3)} minuti")
    print(f"                                ({humanize.naturaldelta(timedelta(minutes=ollama_time))})")
    print(f"💰 Costo:                        €0 (Completamente gratis!)")
    print(f"🔍 Tempo per query:              5-8 secondi")
    print(f"📊 Qualità risultati:            ⭐⭐⭐⭐")
    
    print("\n💻 OPENAI (Cloud - A PAGAMENTO)")
    print("-" * 80)
    print(f"⏰ Tempo prima indicizzazione:  {int(openai_time)} - {int(openai_time * 1.2)} minuti")
    print(f"                                ({humanize.naturaldelta(timedelta(minutes=openai_time))})")
    print(f"💰 Costo stimato:                €{embedding_cost * 1.1:.2f} - €{embedding_cost * 1.5:.2f}")
    print(f"🔍 Tempo per query:              2-4 secondi")
    print(f"📊 Qualità risultati:            ⭐⭐⭐⭐⭐")
    
    # ==================== RACCOMANDAZIONI ====================
    
    print("\n" + "="*80)
    print("💡 RACCOMANDAZIONI")
    print("="*80 + "\n")
    
    if num_files == 0:
        print("⚠️  NESSUN FILE SUPPORTATO TROVATO!")
        print("   Aggiungi file PDF, DOCX, TXT o DOC alla cartella.")
    
    elif num_files < 100:
        print("✅ OTTIMO! Numero di file gestibile")
        print(f"   Con {num_files} file, l'indicizzazione sarà veloce.")
        print(f"   → Usa Ollama (gratis): ~{int(ollama_time)} minuti")
        print(f"   → Oppure OpenAI (veloce): ~{int(openai_time)} minuti, ~€{embedding_cost * 1.2:.2f}")
    
    elif num_files < 500:
        print("✅ BUONO! Numero medio di file")
        print(f"   Con {num_files} file, ci vorrà un po' di tempo.")
        print(f"   → Raccomandazione: Usa Ollama (gratis) e lascia girare")
        print(f"   → Tempo stimato: ~{int(ollama_time)} minuti")
        print(f"   → Alternative: OpenAI se hai fretta (~{int(openai_time)} min)")
    
    elif num_files < 1500:
        print("⚠️  ATTENZIONE! Molti file da processare")
        print(f"   Con {num_files} file, l'indicizzazione richiederà tempo.")
        print(f"")
        print(f"   📋 OPZIONI:")
        print(f"   1. FILTRA PER ETÀ: Indicizza solo file recenti")
        
        recent = stats['by_age'].get('ultimo_mese', {'count': 0})['count']
        six_m = stats['by_age'].get('ultimi_6_mesi', {'count': 0})['count']
        
        print(f"      - Ultimo mese: {recent} file → ~{int(recent/17)} min con Ollama")
        print(f"      - Ultimi 6 mesi: {recent + six_m} file → ~{int((recent + six_m)/17)} min")
        print(f"")
        print(f"   2. USA OPENAI: Più veloce ma costa ~€{embedding_cost * 1.2:.2f}")
        print(f"")
        print(f"   3. INDICIZZA TUTTO: Avvia di notte/weekend")
        print(f"      → Tempo: ~{int(ollama_time)} minuti ({humanize.naturaldelta(timedelta(minutes=ollama_time))})")
    
    else:
        print("🔴 TROPPI FILE! Forte raccomandazione di filtrare")
        print(f"   Con {num_files} file, il processing sarà MOLTO lungo.")
        print(f"")
        print(f"   ⚡ STRATEGIE CONSIGLIATE:")
        print(f"")
        print(f"   1. FILTRA PER TIPO: Rimuovi formati non necessari")
        print(f"   2. FILTRA PER ETÀ: Solo file recenti (ultimo anno)")
        recent_year = (stats['by_age'].get('ultimo_mese', {'count': 0})['count'] + 
                      stats['by_age'].get('ultimi_6_mesi', {'count': 0})['count'] + 
                      stats['by_age'].get('ultimo_anno', {'count': 0})['count'])
        print(f"      → File ultimo anno: {recent_year} → ~{int(recent_year/17)} min")
        print(f"")
        print(f"   3. FILTRA PER CARTELLA: Solo documenti importanti")
        print(f"")
        print(f"   4. SE PROPRIO VUOI TUTTO:")
        print(f"      - Usa OpenAI: ~{int(openai_time)} min (~€{embedding_cost * 1.3:.2f})")
        print(f"      - Oppure Ollama: ~{int(ollama_time)} min (gratis ma lungo)")
    
    # Dimensione DB
    estimated_db_size = stats['supported_size'] * 0.2  # Vector DB ~20% dimensione originale
    print(f"\n💾 SPAZIO DISCO NECESSARIO")
    print(f"   Vector Database stimato: {humanize.naturalsize(estimated_db_size, binary=True)}")
    
    if estimated_db_size > 1024 * 1024 * 1024:  # > 1 GB
        print(f"   ⚠️  Database grande! Assicurati di avere spazio sufficiente.")
    
    print("\n" + "="*80)
    print("✅ Analisi completata!")
    print("="*80)
    
    return stats


if __name__ == "__main__":
    import sys
    
    # Path da analizzare
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
    
    analyze_folder(folder_path)