import os
import re
import sys

# Redirect caches to D: drive
os.environ["HF_HOME"] = r"D:\novamind_data\cache\huggingface"
os.environ["TORCH_HOME"] = r"D:\novamind_data\cache\torch"

from core.mind import NovaMind
from brain_persistence import load_brain, save_brain
from knowledge.manager import KnowledgeManager

def extract_relevant_sentences(filepath, keywords, max_sentences=60):
    sentences = []
    if not os.path.exists(filepath):
        return sentences
        
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
        
    raw_sentences = KnowledgeManager.split_into_sentences(text, max_words=30)
    
    # Filter sentences that contain at least one of the keywords
    for s in raw_sentences:
        s_lower = s.lower()
        if any(k in s_lower for k in keywords):
            sentences.append(s)
            if len(sentences) >= max_sentences:
                break
                
    return sentences

def main():
    print("====================================================")
    # AGI Text Training Phase
    print("       NOVAMIND AGI TEXT TRAINING PHASE             ")
    print("====================================================\n")
    
    # Disable semantic recall (heavy HuggingFace transformers) to run AGI in native, high-performance HDC mode.
    # This prevents RAM and VRAM blowup and makes training extremely fast.
    mind = NovaMind(use_semantic_recall=False)
    load_brain(mind)
    
    dataset_dir = r"D:\novamind_data\text_datasets"
    
    # 1. Load declarative facts ( Rio is a city, Bentinho is a character, etc. )
    facts_path = os.path.join(dataset_dir, "declarative_facts.txt")
    if os.path.exists(facts_path):
        print("[TRAIN] Feeding initial declarative facts...")
        with open(facts_path, "r", encoding="utf-8") as f:
            facts = f.readlines()
        for fact in facts:
            fact = fact.strip()
            if fact:
                print(f"  Feed: '{fact}'")
                try:
                    mind.think(fact)
                except Exception as e:
                    print(f"  Error feeding fact '{fact}': {e}")
                
    # 2. Ingest relevant book sentences containing characters
    keywords = [
        "bentinho", "capitu", "ezequiel", "sherlock", "watson", 
        "alice", "frankenstein", "monster", "vampire", "dracula",
        "london", "rio"
    ]
    
    books_files = [
        "dom_casmurro_pt.txt",
        "bras_cubas_pt.txt",
        "alice_en.txt",
        "sherlock_en.txt",
        "frenstein_en.txt"
    ]
    
    all_sentences = []
    for f in books_files:
        path = os.path.join(dataset_dir, f)
        book_sents = extract_relevant_sentences(path, keywords, max_sentences=40)
        print(f"[TRAIN] Extracted {len(book_sents)} relevant sentences from {f}")
        all_sentences.extend(book_sents)
        
    print(f"\n[TRAIN] Total corpus size: {len(all_sentences)} sentences.")
    
    # Feed to mind.think
    print("\n[TRAIN] Training NovaMind cognitive memory system...")
    import gc
    import torch
    import time
    
    start_train_time = time.time()
    
    for idx, sentence in enumerate(all_sentences):
        # Every 20 sentences, print progress
        if idx % 20 == 0:
            print(f"  Processed {idx}/{len(all_sentences)} sentences...")
            # Clean memory periodically
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        try:
            mind.think(sentence)
        except Exception as e:
            print(f"  Error on sentence '{sentence[:50]}': {e}")
            
    # Final cleanup before saving
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    elapsed_train_time = time.time() - start_train_time
    print(f"\n[TRAIN] Saving updated brain state... (Training took {elapsed_train_time:.2f} seconds)")
    save_brain(mind)
    print("====================================================")
    print("            TRAINING COMPLETED SUCCESSFULLY         ")
    print("====================================================")

if __name__ == "__main__":
    main()
