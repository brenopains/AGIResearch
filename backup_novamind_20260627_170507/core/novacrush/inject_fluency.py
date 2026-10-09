"""
NovaCrush -- Fluency Injection (DeepSeek's Option B)
======================================================
Downloads a nanoscopic language model (TinyStories-1M) and distills
its pre-trained grammatical structure directly into our Ternary Substrate.

Instead of training for days to learn that "the" is followed by a noun,
we inject this statistical scaffolding natively. The NovaCrush engine 
will then use this "grammar skeleton" to reason causally.

Requirements: pip install transformers huggingface-hub
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.novacrush.crushed_substrate import CrushedSubstrate
from core.novacrush.language_gen import NativeLanguageGenerator

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("❌ Transformers library not found.")
    print("Please run: pip install transformers huggingface-hub")
    sys.exit(1)

def inject_fluency(target_dim: int = 64, max_vocab: int = 4000):
    print("=" * 60)
    print("  NovaCrush -- Fluency Injector")
    print("=" * 60)
    
    model_name = "distilgpt2"
    print(f">> Downloading/Loading '{model_name}'...")
    print("   (This is a small pre-trained model to bootstrap grammar)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print("\n>> Extracting semantic scaffolding...")
    
    # Extract the vocabulary
    vocab = tokenizer.get_vocab()
    
    # Filter vocabulary to alphanumeric, readable, lowercase words
    filtered_tokens = []
    for token, idx in vocab.items():
        clean_token = token.strip('Ġ').lower() # GPT-Neo style tokenization
        if clean_token.isalpha() and len(clean_token) >= 2:
            filtered_tokens.append((clean_token, idx))
            
    # Sort by index and take the most common words
    filtered_tokens.sort(key=lambda x: x[1])
    selected_tokens = []
    seen = set()
    
    for token, original_idx in filtered_tokens:
        if token not in seen and len(selected_tokens) < max_vocab:
            seen.add(token)
            selected_tokens.append((token, original_idx))
            
    print(f"   Selected {len(selected_tokens)} high-quality core concepts.")
    
    # Get the embedding weights and LM head weights
    original_embeddings = model.transformer.wte.weight.data
    # distilgpt2 has embedding dim = 768. 
    model_dim = original_embeddings.shape[1]
    
    print(f"   Original embedding dimension: {model_dim}")
    
    # Initialize our CrushedSubstrate
    print("\n>> Initializing CrushedSubstrate...")
    substrate = CrushedSubstrate(
        initial_concepts=len(selected_tokens),
        embedding_dim=model_dim,
        hdc_dim=4096
    )
    
    # Map the tokens and embeddings
    print(">> Injecting DNA (Embeddings & Vocabulary)...")
    new_embeddings = torch.zeros((len(selected_tokens), model_dim))
    
    for i, (token, orig_idx) in enumerate(selected_tokens):
        # Register in our substrate
        substrate.token_to_id[token] = i
        substrate.id_to_token[i] = token
        
        # Copy the embedding
        new_embeddings[i] = original_embeddings[orig_idx]
        
    # Replace the substrate's random embeddings with the distilled ones
    substrate.embeddings = torch.nn.Parameter(new_embeddings)
    substrate.vocab_size = len(selected_tokens)
    
    print(">> Injecting Real Grammar Transition (from LM Head)...")
    # The LM head in GPT2 maps: hidden_state -> logits over vocab.
    # For our filtered vocab, we compute:
    #   transition(embedding_A) ≈ embedding_B  where B is the most likely next token.
    #
    # We approximate this by: T = E_filtered^+ * LM_head^T * E_filtered
    # where E_filtered^+ is the pseudo-inverse of our filtered embedding matrix.
    # This gives us a transition in embedding space that encodes real English grammar.
    
    with torch.no_grad():
        # The lm_head in distilgpt2 shares weights with wte (tied embeddings)
        # So lm_head(x) = x @ wte.T, meaning P(next_token) = softmax(h @ E.T)
        # For our transition: we want T such that T @ e_A ≈ e_B
        # We use the first transformer layer's attention output projection as a proxy
        
        # Actually, the simplest and most effective approach:
        # Run a few hundred real sentences through the model and collect
        # (current_embedding, next_embedding) pairs to fit the transition via least squares.
        
        print("   Computing transition matrix from real text sequences...")
        
        # Generate training pairs from the model itself
        sample_texts = [
            "The cat sat on the mat and looked at the door",
            "She went to the store and bought some food",
            "He said that he would come back later today",
            "They have been working on this project for years",
            "The weather was very cold and it started to rain",
            "I think we should go home now before it gets dark",
            "The children played in the park until the sun went down",
            "She opened the book and started to read the first chapter",
            "The old man walked slowly down the street",
            "We need to find a better way to solve this problem",
            "The dog ran across the field and jumped over the fence",
            "He was very happy when he heard the good news",
            "The teacher asked the students to open their books",
            "She told him that she loved him very much",
            "The music was so beautiful that everyone stopped to listen",
            "They decided to move to a new city next year",
            "The baby started to cry when the lights went out",
            "He picked up the phone and called his mother",
            "The flowers in the garden were blooming beautifully",
            "She could not believe what she was seeing",
            "The train arrived at the station right on time",
            "He put on his coat and went outside into the cold",
            "The birds were singing in the trees early in the morning",
            "She smiled and said hello to her old friend",
            "The river flowed quietly through the green valley",
            "He worked hard every day to support his family",
            "The stars were shining brightly in the clear night sky",
            "She took a deep breath and opened the door",
            "The king sat upon his throne and ruled the land",
            "Alice was beginning to get very tired of sitting",
        ]
        
        # Collect (input_emb, target_emb) pairs
        all_inputs = []
        all_targets = []
        
        for text in sample_texts:
            tokens_enc = tokenizer.encode(text, add_special_tokens=False)
            for i in range(len(tokens_enc) - 1):
                tok_a = tokenizer.decode([tokens_enc[i]]).strip().lower()
                tok_b = tokenizer.decode([tokens_enc[i+1]]).strip().lower()
                
                if tok_a in substrate.token_to_id and tok_b in substrate.token_to_id:
                    id_a = substrate.token_to_id[tok_a]
                    id_b = substrate.token_to_id[tok_b]
                    all_inputs.append(new_embeddings[id_a])
                    all_targets.append(new_embeddings[id_b])
        
        print(f"   Collected {len(all_inputs)} transition pairs from sample sentences.")
        
        if len(all_inputs) > 10:
            X = torch.stack(all_inputs)  # [N, 768]
            Y = torch.stack(all_targets)  # [N, 768]
            
            # Least squares: find T such that T @ X.T ≈ Y.T
            # => T = Y.T @ X @ (X.T @ X)^-1
            # Using torch.linalg.lstsq for numerical stability
            solution = torch.linalg.lstsq(X, Y)
            T = solution.solution  # [768, 768]
            
            # Mix with small identity for stability
            T = 0.8 * T + 0.2 * torch.eye(model_dim)
            
            substrate.transition.weight.data = T.to(substrate.transition.weight.device)
            print("   Transition matrix injected with REAL grammatical patterns!")
        else:
            print("   WARNING: Not enough transition pairs, using identity fallback.")
            substrate.transition.weight.data = torch.eye(model_dim) * 0.5 + torch.randn(model_dim, model_dim) * 0.05

    # Save the injected checkpoint
    checkpoint_dir = os.path.join(PROJECT_ROOT, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    save_path = os.path.join(checkpoint_dir, "substrate_distilled.pt")
    
    # We save exactly how train_loop expects it
    state = {
        'embeddings': substrate.embeddings.data.cpu(),
        'transition_weight': substrate.transition.weight.data.cpu(),
        'token_to_id': substrate.token_to_id,
        'id_to_token': substrate.id_to_token,
        'vocab_size': substrate.vocab_size,
        'step': 0,
        'surprise': 1.0, # Reset surprise
        'compression_stats': substrate.get_compression_stats(),
    }
    torch.save(state, save_path)
    print(f"\n>> Distillation complete! Checkpoint saved to:\n   {save_path}")
    
    # Test generation with the injected knowledge
    print("\n>> Initial Broca's Area Test (No training yet):")
    substrate = substrate.cpu()  # Ensure all on same device for test
    substrate.device = torch.device('cpu')
    generator = NativeLanguageGenerator(substrate)
    for seed in ["the", "he", "it"]:
        if seed in substrate.token_to_id:
            res = generator.generate(seed=seed, max_tokens=10, temperature=0.2)
            print(f"   [{seed}] -> {res['text']}")

    print("\n>> Next Step: Run the train_loop using this checkpoint to solidify the grammar!")
    print("   python -m core.novacrush.train_loop --resume checkpoints/substrate_distilled.pt")

if __name__ == "__main__":
    inject_fluency()
