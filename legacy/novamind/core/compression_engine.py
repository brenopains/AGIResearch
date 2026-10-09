"""
CompressionEngine — Compressao de Pensamento em Atomos Cognitivos

Diferente de compressao de memoria (L0-L4).
E compressao de RACIOCINIO: sequencias longas de operadores
sao comprimidas em atomos cognitivos reutilizaveis.

Exemplo matematico:
  1+1+1+1+1+1+1+1+1+1 = 10 (compressao)
  10*10*10 = 1000 (segunda compressao)
  1000 = 10^3 (terceira compressao)
  log10(1000) = 3 (quarta compressao)

Cada nivel elimina operacoes e vira um novo "atom cognitivo".

Diferenca do SymbolLibrary:
  SymbolLibrary: descobre macros de traces (reconhecimento de padrao)
  CompressionEngine: ATIVAMENTE comprime sequencias em novos operadores
"""
import time, hashlib, math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CognitiveAtom:
    """Um atomo cognitivo — raciocinio comprimido em operador unico."""
    id: str = ""
    name: str = ""
    original_sequence: List[str] = field(default_factory=list)
    compressed_to: str = ""           # o operador que substitui a sequencia
    compression_ratio: float = 1.0    # len(original) / 1
    n_applications: int = 0
    confidence: float = 0.3
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {"id": self.id[:8], "name": self.name,
                "ratio": round(self.compression_ratio, 1),
                "seq": " → ".join(self.original_sequence[:4])}


class CompressionEngine:
    """Comprime raciocinio em atomos cognitivos.

    Ciclo:
      1. Monitora o CognitiveDAG e CognitiveTraces
      2. Detecta sequencias que aparecem juntas > N vezes
      3. Propoe um novo atomo que substitui a sequencia
      4. O atomo pode ser usado como operador unico
      5. Pode ser comprimido NOVAMENTE em niveis superiores
    """

    def __init__(self, cge=None, loop_evaluator=None):
        self.cge = cge
        self.loop_evaluator = loop_evaluator
        self.atoms: Dict[str, CognitiveAtom] = {}
        self._compression_history: List[Dict] = []
        self._min_occurrences: int = 3

    def compress(self, sequence: List[str], name: str = "") -> Optional[CognitiveAtom]:
        """Comprime uma sequencia em um atomo cognitivo."""
        if len(sequence) < 2:
            return None
        key = "+".join(sequence)
        existing = [a for a in self.atoms.values() if a.original_sequence == sequence]
        if existing:
            existing[0].n_applications += 1
            existing[0].confidence = min(1.0, existing[0].confidence + 0.05)
            return existing[0]

        compressed_name = name or f"atom_{len(self.atoms)+1}"
        atom = CognitiveAtom(
            id=f"atom_{hashlib.md5(key.encode()).hexdigest()[:8]}",
            name=compressed_name,
            original_sequence=sequence,
            compressed_to=compressed_name,
            compression_ratio=len(sequence),
            confidence=0.3,
            n_applications=1,
        )
        self.atoms[atom.id] = atom
        self._compression_history.append({
            "time": time.time(), "atom": compressed_name,
            "ratio": len(sequence), "seq": sequence[:4],
        })
        return atom

    def mine_from_traces(self, trace_data: List[Dict]) -> List[CognitiveAtom]:
        """Minera traces para comprimir sequencias recorrentes."""
        new_atoms = []

        # Extrair n-grams de modo dos traces
        ngram_counts = defaultdict(list)
        for td in trace_data:
            steps = td.get("steps", []) if isinstance(td, dict) else []
            if not steps:
                continue
            modes = [s.get("mode", "") for s in steps if isinstance(s, dict)]
            rewards = [s.get("reward", 0) for s in steps if isinstance(s, dict)]

            for n in [3, 4, 5]:
                for i in range(len(modes) - n + 1):
                    gram = tuple(modes[i:i+n])
                    avg_r = sum(rewards[i:i+n]) / max(1, n) if len(rewards) >= i+n else 0
                    ngram_counts[gram].append(avg_r)

        for gram, rewards in ngram_counts.items():
            if len(rewards) < self._min_occurrences:
                continue
            avg_r = sum(rewards) / len(rewards)
            if avg_r < 0.4:  # so comprime se a sequencia tem boa recompensa
                continue

            atom = self.compress(list(gram),
                                 name=f"Ω{len(self.atoms)+1}")
            if atom:
                new_atoms.append(atom)

        # Compressao de segundo nivel: atoms que aparecem juntos
        atom_names = list(self.atoms.keys())
        if len(atom_names) >= 3:
            # Verificar se 2+ atoms formam uma sequencia maior
            for i in range(len(atom_names)):
                for j in range(i+1, len(atom_names)):
                    combined = (self.atoms[atom_names[i]].original_sequence +
                               self.atoms[atom_names[j]].original_sequence)
                    if len(combined) <= 8:  # limite razoavel
                        super_atom = self.compress(combined,
                            name=f"Σ{len(self.atoms)+1}")
                        if super_atom and super_atom.compression_ratio > 3:
                            new_atoms.append(super_atom)

        return new_atoms

    def decompress(self, atom_id: str) -> Optional[List[str]]:
        """Descomprime um atomo de volta para sua sequencia original."""
        atom = self.atoms.get(atom_id)
        return list(atom.original_sequence) if atom else None

    def get_state(self) -> Dict:
        return {
            "n_atoms": len(self.atoms),
            "total_compression_ratio": sum(a.compression_ratio for a in self.atoms.values())
                                       / max(1, len(self.atoms)),
            "best": [a.to_dict() for a in sorted(
                self.atoms.values(), key=lambda a: -a.confidence)[:3]],
            "recent": self._compression_history[-3:] if self._compression_history else [],
        }
