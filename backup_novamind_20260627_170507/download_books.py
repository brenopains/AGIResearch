import os
import urllib.request
import re

BOOKS = {
    "dom_casmurro_pt.txt": "https://www.gutenberg.org/cache/epub/55752/pg55752.txt",
    "bras_cubas_pt.txt": "https://www.gutenberg.org/cache/epub/54829/pg54829.txt",
    "alice_en.txt": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    "sherlock_en.txt": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    "frenstein_en.txt": "https://www.gutenberg.org/cache/epub/84/pg84.txt"
}

# Declarative facts to pre-feed the AGI's perception and build initial concepts
DECLARATIVE_FACTS = """
Bentinho is a character.
Capitu is a character.
Ezequiel is a child.
Sherlock is a detective.
Watson is a doctor.
Alice is a girl.
Frankenstein is a scientist.
Dracula is a vampire.
London is a city.
Rio is a city.
"""

def clean_gutenberg_text(text):
    # Remove licensing information from top and bottom
    lines = text.splitlines()
    if len(lines) > 400:
        lines = lines[200:-200]
    return "\n".join(lines)

def main():
    target_dir = r"D:\novamind_data\text_datasets"
    os.makedirs(target_dir, exist_ok=True)
    
    print("--- Project Gutenberg Dataset Downloader ---")
    
    for filename, url in BOOKS.items():
        filepath = os.path.join(target_dir, filename)
        if os.path.exists(filepath):
            print(f"File {filename} already exists. Skipping download.")
            continue
            
        print(f"Downloading {filename} from {url}...")
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response:
                content = response.read().decode('utf-8', errors='ignore')
                
            clean_content = clean_gutenberg_text(content)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(clean_content)
            print(f"Saved cleaned {filename} successfully.")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            
    # Write declarative facts to D:\novamind_data\text_datasets\declarative_facts.txt
    facts_path = os.path.join(target_dir, "declarative_facts.txt")
    with open(facts_path, "w", encoding="utf-8") as f:
        f.write(DECLARATIVE_FACTS.strip())
    print("Saved declarative_facts.txt successfully.")
    
    print("\nDataset preparation finished successfully at:", target_dir)

if __name__ == "__main__":
    main()
