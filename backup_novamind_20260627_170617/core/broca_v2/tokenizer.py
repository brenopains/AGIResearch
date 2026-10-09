"""
BPETokenizer — Tokenizador Subword Rápido (HuggingFace tokenizers)
==================================================================
Usa a implementação Rust BPE do HuggingFace para treino rápido.
Vocabulário de ~32K tokens. Suporta encode/decode.

Treinamento:
    tok = BPETokenizer(vocab_size=32768)
    tok.train(corpus_path)
    tok.save(path)
"""

import os
import json
from typing import List, Optional


class BPETokenizer:
    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    def __init__(self, vocab_size: int = 32768):
        self.vocab_size = vocab_size
        self._tokenizer = None
        self._trained = False
        self._token_to_id = {}
        self._id_to_token = {}
        self._real_vocab_size = len(self.SPECIAL_TOKENS)

    def train(self, text_path: str, max_lines: int = 500000):
        """Treina BPE usando HuggingFace tokenizers (Rust, rápido)."""
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

        print(f"[BPE] Treinando BPE (vocab={self.vocab_size}, max_lines={max_lines})")

        self._tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

        # Pré-tokenizador: split por whitespace
        self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self._tokenizer.decoder = decoders.ByteLevel()

        # Trainer BPE
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.SPECIAL_TOKENS,
            min_frequency=2,
            show_progress=True,
        )

        # Alimenta o trainer linha por linha (limitado)
        def line_iterator():
            count = 0
            with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if count >= max_lines:
                        break
                    line = line.strip()
                    if len(line) > 10:
                        yield line
                        count += 1
                        if count % 100000 == 0:
                            print(f"  [BPE] Lidas {count} linhas...")

        self._tokenizer.train_from_iterator(line_iterator(), trainer=trainer)

        # Mapear tokens
        self._token_to_id = self._tokenizer.get_vocab()
        self._id_to_token = {v: k for k, v in self._token_to_id.items()}
        self._real_vocab_size = len(self._token_to_id)
        self._trained = True

        print(f"[BPE] Treino completo: {self._real_vocab_size} tokens")

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Converte texto em IDs de tokens."""
        if not self._trained and self._tokenizer is None:
            raise RuntimeError("Tokenizer nao treinado")
        output = self._tokenizer.encode(text)
        ids = output.ids
        if add_special:
            return [self.BOS_ID] + ids + [self.EOS_ID]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Converte IDs de tokens em texto."""
        if not self._trained and self._tokenizer is None:
            raise RuntimeError("Tokenizer nao treinado")
        if skip_special:
            ids = [i for i in ids if i >= len(self.SPECIAL_TOKENS)]
        return self._tokenizer.decode(ids)

    def save(self, path: str):
        """Salva tokenizer no formato HuggingFace."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if self._tokenizer:
            self._tokenizer.save(os.path.splitext(path)[0] + "_hf.json")
        data = {
            "vocab_size": self.vocab_size,
            "trained": self._trained,
            "token_to_id": self._token_to_id,
            "id_to_token": {str(k): v for k, v in self._id_to_token.items()},
            "special": self.SPECIAL_TOKENS,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[BPE] Salvo em {path} ({self._real_vocab_size} tokens)")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        from tokenizers import Tokenizer
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(data["vocab_size"])
        tok._trained = data["trained"]
        tok._token_to_id = data["token_to_id"]
        tok._id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        tok._real_vocab_size = len(tok._token_to_id)

        # Tenta carregar tokenizer HF
        hf_path = os.path.splitext(path)[0] + "_hf.json"
        if os.path.exists(hf_path):
            tok._tokenizer = Tokenizer.from_file(hf_path)
        return tok

    def __len__(self) -> int:
        return self._real_vocab_size

    @property
    def vocab_size_real(self) -> int:
        return self._real_vocab_size
