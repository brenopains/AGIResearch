import os
import sentencepiece as spm
from datasets import load_dataset

def train_tokenizer():
    # paper: Kudo & Richardson 2018, "SentencePiece: A simple and language independent subword tokenizer", arXiv:1808.06226
    print("Loading Wikitext-103...")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
    
    os.makedirs("novamind/v10/tokenizer", exist_ok=True)
    os.makedirs("tmp", exist_ok=True)
    
    corpus_file = "tmp/wikitext_corpus.txt"
    with open(corpus_file, "w", encoding="utf-8") as f:
        count = 0
        for item in ds:
            text = item["text"].strip()
            if text:
                f.write(text + "\n")
                count += 1
            if count >= 300000: # Sufficient for a 16k vocab training
                break
                
    print(f"Collected {count} lines. Training SentencePiece...")
    
    # Train SPM
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix='novamind/v10/tokenizer/spm_16k',
        vocab_size=16384,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='<PAD>',
        unk_piece='<UNK>',
        bos_piece='<BOS>',
        eos_piece='<EOS>',
        model_type='bpe',
        character_coverage=1.0 # Ensures all standard characters are covered
    )
    print("Training complete! Model saved to novamind/v10/tokenizer/spm_16k.model")

if __name__ == "__main__":
    train_tokenizer()
