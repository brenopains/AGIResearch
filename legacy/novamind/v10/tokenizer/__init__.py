import os
import sentencepiece as spm

class Tokenizer:
    # paper: Kudo & Richardson 2018, "SentencePiece: A simple and language independent subword tokenizer", arXiv:1808.06226
    def __init__(self, model_path=None):
        if model_path is None:
            # Default to the expected relative path
            # Assume we are run from the project root or adjust path
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "spm_16k.model")
            
        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor()
        if os.path.exists(model_path):
            self.sp.load(model_path)
            
    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()
        
    @property
    def pad_id(self) -> int:
        return self.sp.pad_id()

    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()
        
    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()
        
    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        return self.sp.encode_as_ids(text)

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        return self.sp.decode_ids(ids)

    # For making Tokenizer picklable
    def __getstate__(self):
        return {'model_path': self.model_path}

    def __setstate__(self, state):
        self.model_path = state['model_path']
        self.sp = spm.SentencePieceProcessor()
        if os.path.exists(self.model_path):
            self.sp.load(self.model_path)
