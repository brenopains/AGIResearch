"""
NovaCrush -- Native Language Generator (Broca's Area)
=======================================================
THIS is the missing piece.

The NovaMind substrate already learns token->token transitions via
predictive coding. It already knows "cat" is followed by "sat" more
often than "quantum". That IS language understanding at a primitive level.

What was missing: a DECODER that walks those transitions to generate
actual sentences. This module adds that capability WITHOUT any LLM.

How it works:
    1. Given a seed word/concept, look up its embedding
    2. Pass through the transition matrix to predict "what comes next"
    3. Find the closest known token to that prediction
    4. Repeat until sentence is complete
    
This is EXACTLY how the brain generates speech:
    Concept activation -> Motor planning -> Word selection -> Output

The quality of generation depends entirely on how much text the
substrate has been trained on. More training = better transitions =
more coherent output. No LLM involved.

Why this makes it MORE AGI, not less:
    - LLM = pre-trained frozen model doing pattern matching
    - This = learned transitions from continuous experience
    - A child learns to speak the same way: hear patterns, learn
      transitions, generate by walking the learned graph
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
import re
import time


class NativeLanguageGenerator:
    """
    Generates text directly from the CrushedSubstrate's learned transitions.
    
    No LLM. No API. Pure learned behavior.
    
    The substrate's transition matrix encodes P(next_token | current_token)
    as geometric similarity in embedding space. We exploit this to generate
    coherent sequences by:
    
    1. Start with seed embedding
    2. Apply transition -> get predicted embedding
    3. Find nearest known token to prediction
    4. Use that token's embedding as next input
    5. Repeat
    
    Temperature controls creativity:
        T=0.1: very deterministic, repetitive but safe
        T=0.5: balanced
        T=1.0: creative but potentially incoherent
        T>1.0: chaotic
    """
    
    def __init__(self, substrate):
        """
        Args:
            substrate: CrushedSubstrate or DynamicPredictiveNetwork instance
        """
        self.substrate = substrate
        self._generation_count = 0
        
    def generate(self, seed: str = "", max_tokens: int = 20,
                 temperature: float = 0.5, 
                 stop_on_repeat: bool = True,
                 guiding_context: List[str] = None) -> Dict:
        """
        Generate a sequence of tokens from the substrate's learned transitions.
        
        Args:
            seed: starting word/phrase (empty = random start)
            max_tokens: maximum tokens to generate
            temperature: randomness (0.1=safe, 1.0=creative)
            stop_on_repeat: stop if we enter a loop
            guiding_context: tokens to boost during generation (from WorldModel)
            
        Returns:
            Dict with generated text, tokens, and generation stats
        """
        t0 = time.perf_counter()
        
        guiding_context = guiding_context or []
        
        # Get seed token
        if seed:
            words = [w.strip() for w in re.split(r'\W+', seed.lower()) if w.strip()]
            if not words:
                words = [self._random_token()]
        else:
            words = [self._random_token()]
        
        # Initialize with seed
        generated_tokens = list(words)
        current_token = words[-1]
        
        # Track for repeat detection
        recent_window = []
        
        for step in range(max_tokens):
            # Get current token's embedding
            idx = self.substrate.get_token_id(current_token)
            inp = torch.tensor([idx], dtype=torch.long, 
                             device=self._get_device())
            
            # Predict next via transition matrix
            with torch.no_grad():
                prediction = self.substrate.forward(inp)  # [1, embed_dim]
            
            # Find nearest known token to prediction, boosting guiding context
            next_token, similarity = self._find_nearest_token(
                prediction, temperature, 
                exclude=[current_token] if stop_on_repeat else [],
                guiding_context=guiding_context
            )
            
            if next_token is None:
                break
                
            generated_tokens.append(next_token)
            
            # Remove from context if used
            if next_token in guiding_context:
                guiding_context.remove(next_token)
                
            # Repeat detection
            recent_window.append(next_token)
            if len(recent_window) > 6:
                recent_window.pop(0)
            if stop_on_repeat and len(recent_window) >= 4:
                # Check for 2-gram loops (A B A B)
                if (recent_window[-1] == recent_window[-3] and 
                    recent_window[-2] == recent_window[-4]):
                    generated_tokens = generated_tokens[:-2]  # Remove loop
                    break
            
            current_token = next_token
        
        # Build text
        text = ' '.join(generated_tokens)
        gen_time = (time.perf_counter() - t0) * 1000
        self._generation_count += 1
        
        return {
            'text': text,
            'tokens': generated_tokens,
            'num_tokens': len(generated_tokens),
            'seed': seed,
            'temperature': temperature,
            'generation_time_ms': gen_time,
            'vocab_size': self.substrate.vocab_size,
            'generation_id': self._generation_count,
        }
    
    def _find_nearest_token(self, prediction: torch.Tensor, 
                            temperature: float = 0.5,
                            exclude: List[str] = None,
                            guiding_context: List[str] = None) -> Tuple[Optional[str], float]:
        """
        Find the token whose embedding is closest to the prediction.
        
        Uses softmax with temperature over cosine similarities to
        allow probabilistic selection (not always the greedy best).
        """
        exclude = exclude or []
        guiding_context = guiding_context or []
        
        with torch.no_grad():
            # Compute similarity to all known token embeddings
            all_embeddings = self.substrate.embeddings[:len(self.substrate.token_to_id)]
            
            if len(all_embeddings) == 0:
                return None, 0.0
            
            # Cosine similarity
            pred_norm = F.normalize(prediction, dim=-1)
            emb_norm = F.normalize(all_embeddings, dim=-1)
            similarities = torch.matmul(pred_norm, emb_norm.T).squeeze(0)
            
            # Mask excluded tokens
            for token in exclude:
                if token in self.substrate.token_to_id:
                    exc_idx = self.substrate.token_to_id[token]
                    if exc_idx < len(similarities):
                        similarities[exc_idx] = -1.0
                        
            # Boost guiding context tokens
            for token in guiding_context:
                if token in self.substrate.token_to_id:
                    ctx_idx = self.substrate.token_to_id[token]
                    if ctx_idx < len(similarities):
                        similarities[ctx_idx] += 0.5  # Significant boost
            
            # Temperature-scaled softmax for probabilistic selection
            if temperature < 0.01:
                # Greedy
                best_idx = similarities.argmax().item()
            else:
                probs = F.softmax(similarities / temperature, dim=-1)
                best_idx = torch.multinomial(probs, 1).item()
            
            best_sim = similarities[best_idx].item()
            
            # Look up token
            if best_idx in self.substrate.id_to_token:
                return self.substrate.id_to_token[best_idx], best_sim
            
        return None, 0.0
    
    def _random_token(self) -> str:
        """Pick a random known token as seed."""
        if self.substrate.token_to_id:
            tokens = list(self.substrate.token_to_id.keys())
            return tokens[np.random.randint(0, len(tokens))]
        return "start"
    
    def _get_device(self):
        return self.substrate.embeddings.device
    
    def generate_multiple(self, seed: str = "", count: int = 5, 
                         **kwargs) -> List[Dict]:
        """Generate multiple outputs for comparison."""
        results = []
        for _ in range(count):
            results.append(self.generate(seed, **kwargs))
        return results
