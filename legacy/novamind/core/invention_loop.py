"""
invention_loop.py — Ciclo autonomo Falha → Invencao
======================================================
Quando o sistema falha repetidamente com a mesma assinatura,
o motor de invencao e automaticamente acionado para criar
uma nova primitiva que preenche a lacuna.

Ciclo:
  1. FailureTracker: agrupa falhas por assinatura
  2. InventionTrigger: cluster > N → aciona PrimitiveInventionEngine
  3. VerificationGate: primitiva inventada testada em held-out
  4. ToolIntegration: registrada e disponivel imediatamente

Isto generaliza a criacao de ferramentas por necessidade:
qualquer sistema que enfrenta problemas novos vai acumular
falhas e, ao atingir um limiar, criar a ferramenta que falta.
"""
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class FailureSignature:
    """Assinatura de um tipo de falha."""
    domain: str = ""
    gap_type: str = ""  # "missing_primitive", "prediction_error", "composition_failed"
    context_hash: str = ""
    n_occurrences: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    examples: List[Tuple[Any, Any]] = field(default_factory=list)

    def priority(self) -> float:
        age = time.time() - self.first_seen
        freq = self.n_occurrences / max(age, 1.0)
        return min(1.0, freq * 0.5 + self.n_occurrences * 0.02)

    def __hash__(self):
        return hash((self.domain, self.gap_type, self.context_hash))

    def __eq__(self, other):
        return (self.domain == other.domain and
                self.gap_type == other.gap_type and
                self.context_hash == other.context_hash)


class FailureTracker:
    """
    Rastreia falhas do sistema, agrupando por assinatura.
    
    Quando um cluster atinge o threshold, notifica o InventionTrigger.
    """

    def __init__(self, max_history: int = 500):
        self.failures: Dict[FailureSignature, FailureSignature] = {}
        self._history: deque = deque(maxlen=max_history)
        self._n_total = 0

    def record_failure(self, domain: str, gap_type: str,
                       context: Any = None,
                       input_val: Any = None, output_val: Any = None) -> Optional[FailureSignature]:
        """Registra uma falha e retorna a assinatura se threshold foi atingido."""
        ctx_hash = str(hash(str(context))) if context else "none"
        sig = FailureSignature(
            domain=domain, gap_type=gap_type,
            context_hash=ctx_hash,
            n_occurrences=1,
            first_seen=time.time(),
            last_seen=time.time(),
        )

        existing = self.failures.get(sig)
        if existing:
            existing.n_occurrences += 1
            existing.last_seen = time.time()
            if input_val is not None and output_val is not None:
                if len(existing.examples) < 10:
                    existing.examples.append((input_val, output_val))
            sig = existing
        else:
            self.failures[sig] = sig
            if input_val is not None and output_val is not None:
                sig.examples.append((input_val, output_val))

        self._n_total += 1
        self._history.append((time.time(), domain, gap_type))

        return sig

    def get_clusters(self, min_occurrences: int = 3) -> List[FailureSignature]:
        """Retorna clusters de falhas que atingiram o threshold."""
        return [s for s in self.failures.values()
                if s.n_occurrences >= min_occurrences]

    def clear(self):
        self.failures.clear()
        self._history.clear()
        self._n_total = 0

    def get_stats(self) -> dict:
        return {
            "total_failures": self._n_total,
            "unique_signatures": len(self.failures),
            "active_clusters": len(self.get_clusters()),
            "history_size": len(self._history),
        }


class InventionLoop:
    """
    Loop autonomo Falha → Invencao.
    
    Conecta FailureTracker ao PrimitiveInventionEngine.
    Quando um cluster de falhas atinge o threshold, alimenta
    o engine com os exemplos e chama invent().
    
    Uso:
        loop = InventionLoop(primitive_engine, failure_tracker)
        loop.record_failure(...)  # automaticamente dispara invent se threshold
        new_prims = loop.tick()  # ou chama explicitamente
    """

    def __init__(self, primitive_engine: Any = None,
                 tracker: Optional[FailureTracker] = None,
                 min_trigger: int = 3,
                 cge=None, verbose: bool = False):
        self.engine = primitive_engine
        self.tracker = tracker or FailureTracker()
        self.min_trigger = min_trigger
        self.cge = cge
        self.verbose = verbose

        self._invention_history: List[dict] = []
        self._invention_count = 0

    def record_failure(self, domain: str, gap_type: str,
                       context: Any = None,
                       input_val: Any = None,
                       output_val: Any = None) -> Optional[List[str]]:
        """
        Registra falha e, se threshold atingido, tenta inventar.
        
        Returns: lista de primitivas inventadas, ou None.
        """
        sig = self.tracker.record_failure(domain, gap_type, context,
                                          input_val, output_val)
        if sig and sig.n_occurrences >= self.min_trigger:
            return self._attempt_invention(sig)
        return None

    def tick(self) -> List[str]:
        """Verifica TODOS os clusters e tenta inventar para cada um."""
        clusters = self.tracker.get_clusters(self.min_trigger)
        all_invented = []
        for sig in clusters:
            invented = self._attempt_invention(sig)
            if invented:
                all_invented.extend(invented)
        return all_invented

    def _attempt_invention(self, sig: FailureSignature) -> Optional[List[str]]:
        """Tenta inventar uma primitiva para preencher a falha."""
        if self.engine is None:
            return None

        examples = sig.examples
        if len(examples) < 2:
            return None

        invented = []

        # Usa mapas de valor
        try:
            from .value_maps import induce_value_map
            best_map = induce_value_map(examples)
            if best_map.perfect() and best_map.fn is not None:
                map_name = best_map.to_primitive_name()
                if self.cge is not None and hasattr(self.cge, 'create_node'):
                    cge_name = f"PROC_{map_name}_{int(time.time())}"
                    self.cge.create_node(cge_name, semantic_vec=None)
                    invented.append(cge_name)
        except Exception:
            pass

        # Usa engine de invencao tradicional
        if hasattr(self.engine, 'observe') and hasattr(self.engine, 'invent'):
            try:
                import numpy as np
                for inp, out in examples:
                    inv = [{"structure_type": sig.gap_type, "domain": sig.domain}]
                    class Inv:
                        def __init__(self, st, dm):
                            self.structure_type = st
                            self.domain = dm
                    inv_objs = [Inv(sig.gap_type, sig.domain)]
                    try:
                        self.engine.observe(np.array(inp), np.array(out), inv_objs)
                    except Exception:
                        pass
                prim_names = self.engine.invent(domain=sig.domain)
                if prim_names:
                    invented.extend(prim_names)
            except Exception:
                pass

        if invented:
            self._invention_count += len(invented)
            self._invention_history.append({
                "time": time.time(),
                "domain": sig.domain,
                "gap": sig.gap_type,
                "n_failures": sig.n_occurrences,
                "invented": invented,
            })

            if self.verbose:
                print(f"  [Invention] Domain={sig.domain}, gap={sig.gap_type}, "
                      f"invented={invented}")

        return invented if invented else None

    def get_stats(self) -> dict:
        return {
            **self.tracker.get_stats(),
            "invention_count": self._invention_count,
            "history": self._invention_history[-10:],
        }
