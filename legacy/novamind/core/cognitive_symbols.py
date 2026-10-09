"""
CognitiveSymbols — Linguagem Interna Emergente + Macros

A Forja descobre padroes de raciocinio recorrentes e os comprime
em SIMBOLOS internos (ex: Ω17, Φ91).

Um simbolo encapsula uma sequencia completa de operadores/programas
em uma unica operacao cognitiva.

Diferenca de programas cognitivos:
  - Programa: sequencia executavel com gatilhos
  - Simbolo: compressao mental que o sistema usa como operador unico
  - Macro: simbolo especifico para compressao de raciocinio repetitivo

Exemplo:
  "Ω17 = Observe(entrada) → Compare(contexto) → Predict(futuro)"
  Depois: "Ω17(X)" substitui 3 operacoes por 1.
"""
import time, hashlib, random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CognitiveSymbol:
    """Um simbolo interno descoberto pela Forja.

    Ex: Ω17 encapsula Observe+Compare+Predict em um unico passo.
    """
    id: str = ""               # "Ω17", "Φ91", etc.
    name: str = ""
    symbol: str = ""            # caracter ou codigo do simbolo
    sequence: List[str] = field(default_factory=list)  # operadores/programas que encapsula
    description: str = ""

    # Metricas
    compression_ratio: float = 1.0  # (tamanho original) / 1. Maior = melhor
    confidence: float = 0.3
    n_applications: int = 0
    success_rate: float = 0.0
    created_at: float = field(default_factory=time.time)

    @property
    def length_original(self) -> int:
        return len(self.sequence)

    def to_dict(self) -> Dict:
        return {"id": self.id, "symbol": self.symbol, "name": self.name[:20],
                "compression": round(self.compression_ratio, 1),
                "confidence": round(self.confidence, 2),
                "seq_preview": " → ".join(self.sequence[:4])}


# Simbolos gregos disponiveis para a Forja usar
SYMBOL_GLYPH_POOL = [
    "Ω17", "Φ91", "Ψ42", "Δ38", "Θ75", "Λ23", "Ξ56", "Π89",
    "Σ12", "Υ34", "Ω55", "Φ23", "Ψ99", "Δ64", "Θ01", "Λ77",
    "Ξ88", "Π44", "Σ90", "Υ12", "Ω03", "Φ77", "Ψ11", "Δ92",
]
_symbol_idx = [0]  # mutable counter


def _next_symbol() -> str:
    idx = _symbol_idx[0] % len(SYMBOL_GLYPH_POOL)
    _symbol_idx[0] += 1
    return SYMBOL_GLYPH_POOL[idx]


class SymbolLibrary:
    """Biblioteca de simbolos internos.

    A Forja chama discover_macros() para minerar CognitiveTraces
    e comprimir padroes recorrentes em simbolos.
    """

    def __init__(self, cge=None):
        self.cge = cge
        self.symbols: Dict[str, CognitiveSymbol] = {}
        self._discovery_log: List[Dict] = []

    def register(self, sequence: List[str], name: str = "",
                 confidence: float = 0.3) -> CognitiveSymbol:
        """Cria um novo simbolo para uma sequencia."""
        key = "+".join(sequence)
        existing = [s for s in self.symbols.values() if s.sequence == sequence]
        if existing:
            existing[0].n_applications += 1
            existing[0].confidence = min(1.0, existing[0].confidence + 0.05)
            return existing[0]

        glyph = _next_symbol()
        sym = CognitiveSymbol(
            id=f"sym_{hashlib.md5(key.encode()).hexdigest()[:8]}",
            name=name or glyph,
            symbol=glyph,
            sequence=sequence,
            description=f"Macro: {' → '.join(sequence[:6])}",
            compression_ratio=len(sequence) / max(1, 1),
            confidence=confidence,
            n_applications=1,
        )
        self.symbols[sym.id] = sym
        self._discovery_log.append({
            "time": time.time(), "symbol": glyph,
            "sequence": sequence[:5], "confidence": confidence,
        })
        return sym

    def discover_macros(self, traces_data: List[Dict]) -> List[CognitiveSymbol]:
        """Minera traces para descobrir macros (sequencias compativeis)."""
        new_symbols = []

        # Extrair todas as sequencias de modo de traces
        seq_counts = defaultdict(list)
        for td in traces_data:
            steps = td.get("steps", []) if isinstance(td, dict) else []
            if not steps:
                continue
            modes = [s.get("mode", "") for s in steps if isinstance(s, dict)]
            rewards = [s.get("reward", 0) for s in steps if isinstance(s, dict)]

            # Bigramas e trigramas
            for n in [2, 3, 4]:
                for i in range(len(modes) - n + 1):
                    gram = tuple(modes[i:i+n])
                    avg_r = sum(rewards[i:i+n]) / max(1, n) if len(rewards) >= i+n else 0
                    seq_counts[gram].append(avg_r)

        # Filtrar sequencias com alta recompensa consistente
        for gram, rewards in seq_counts.items():
            if len(rewards) < 3:
                continue
            avg_r = sum(rewards) / len(rewards)
            if avg_r < 0.5:
                continue

            sym = self.register(list(gram), confidence=min(0.7, avg_r * 0.6 + len(rewards) * 0.02))
            if sym.id in [s.id for s in new_symbols]:
                continue
            new_symbols.append(sym)

        return new_symbols

    def lookup(self, operator_name: str) -> Optional[CognitiveSymbol]:
        """Busca se um nome de operador corresponde a um simbolo."""
        for sym in self.symbols.values():
            if sym.name == operator_name or sym.symbol == operator_name:
                return sym
        return None

    def decompress(self, symbol_id: str) -> Optional[List[str]]:
        """Descomprime um simbolo de volta para sua sequencia original."""
        sym = self.symbols.get(symbol_id)
        return list(sym.sequence) if sym else None

    def get_state(self) -> Dict:
        return {
            "n_symbols": len(self.symbols),
            "recent": self._discovery_log[-5:] if self._discovery_log else [],
            "best": [s.to_dict() for s in sorted(
                self.symbols.values(), key=lambda s: -s.confidence)[:3]],
        }
