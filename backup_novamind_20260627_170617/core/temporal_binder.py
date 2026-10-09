"""
TemporalBinder — Binding Temporal Genuíno para Memória Episódica
================================================================
Implementa binding de conceitos com tempo, contexto e estado emocional
num único vetor HDC — habilitando memória episódica real e raciocínio
contrafactual ("o que eu sabia sobre X antes do incidente?").

Operações HDC:
  bind(a, b)    = a * b  (XOR bipolar — preserva distância)
  bundle(vecs)  = sign(sum(vecs))  (OR bipolar — não preserva distância)
  unbind(ab, b) = ab * b = a  (inverso do bind, pois b*b = 1 nos bipolares)

Propriedades:
  - Tempos próximos → vetores similares (via hash suave por bucket)
  - Retrieval por qualquer eixo (conceito, tempo, contexto, emoção)
  - Contrafactual: troca eixo temporal ou de contexto num vetor já bound

Referências:
  Howard & Kahana (2002) - A distributed representation of temporal context
  Kanerva (2009) - Hyperdimensional computing
  Friston (2019) - A free energy principle for a particular physics
"""

import time
import math
import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple

DIM = 10_000


def _stable_random(seed_str: str, global_seed: int, dim: int = DIM) -> np.ndarray:
    """
    Gera vetor bipolar deterministico a partir de uma string.
    Mesma string → sempre mesmo vetor. Diferente string → vetores ortogonais.
    """
    h = hashlib.md5(f"{global_seed}:{seed_str}".encode()).hexdigest()
    rng = np.random.default_rng(int(h[:16], 16) % (2**31))
    return np.sign(rng.standard_normal(dim)).astype(np.float32)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Binding bipolar: a ⊗ b = a * b (elemento a elemento)."""
    return a * b


def unbind(ab: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unbinding: recupera 'a' dado 'ab' e 'b'. Para bipolares: ab * b = a."""
    return ab * b


def bundle(vecs: List[np.ndarray]) -> np.ndarray:
    """Bundle: superposição de múltiplos vetores."""
    if not vecs:
        return np.zeros(DIM, dtype=np.float32)
    return np.sign(np.sum(np.stack(vecs), axis=0)).astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class TemporalBinder:
    """
    Codificador e binder temporal para memória episódica HDC.

    Cada memória episódica é um vetor que une:
      conceito ⊗ tempo ⊗ contexto ⊗ estado_emocional

    Recuperação por qualquer combinação de eixos.
    Raciocínio contrafactual via troca de eixo.
    """

    # Resoluções temporais disponíveis (tamanho do bucket em segundos)
    RESOLUTIONS = {
        "second":    1.0,
        "minute":    60.0,
        "hour":      3600.0,
        "day":       86400.0,
        "circadian": 86400.0 / (2 * math.pi),  # por radiano de fase
    }

    def __init__(self, seed: int = 77):
        self.seed = seed
        self.dim  = DIM

        # Cache de vetores de tempo já gerados (evita recalcular)
        self._time_cache: Dict[str, np.ndarray] = {}

        # Vetores fixos para contextos conhecidos
        self._context_cache: Dict[str, np.ndarray] = {}

        # Vetores para buckets de NT
        self._nt_cache: Dict[str, np.ndarray] = {}

        # Banco episódico em memória
        self._episodes: List[Dict] = []
        self._max_episodes = 50_000

    # ─── Codificação de tempo ──────────────────────────────────────────────────

    def encode_time_bin(self, t: float, resolution: str = "minute") -> np.ndarray:
        """
        Codifica um instante de tempo t (timestamp Unix) em vetor HDC.

        Tempos no mesmo bucket → mesmo vetor.
        Tempos em buckets adjacentes → vetores similares (via blend).

        Args:
            t:          timestamp Unix (segundos)
            resolution: granularidade ('second', 'minute', 'hour', 'day', 'circadian')

        Returns:
            Vetor HDC bipolar de DIM dimensões.
        """
        bucket_size = self.RESOLUTIONS.get(resolution, 60.0)

        if resolution == "circadian":
            # Fase circadiana: 0 → 2π (baseada na hora do dia)
            phase = (t % 86400.0) / 86400.0 * 2 * math.pi
            bucket = int(phase / (math.pi / 12))  # 24 buckets (um por hora)
            key = f"circ:{bucket}"
        else:
            bucket = int(t // bucket_size)
            key = f"{resolution}:{bucket}"

        if key in self._time_cache:
            return self._time_cache[key].copy()

        # Vetor principal do bucket
        vec_main = _stable_random(key, self.seed, self.dim)

        # Blend suave com bucket anterior (similaridade entre tempos próximos)
        prev_key = f"{resolution}:{bucket - 1}" if resolution != "circadian" else f"circ:{(bucket-1)%24}"
        if prev_key in self._time_cache:
            vec_prev = self._time_cache[prev_key]
            # Posição dentro do bucket (0.0 = início, 1.0 = fim)
            if resolution != "circadian":
                pos = (t % bucket_size) / bucket_size
            else:
                pos = 0.5
            # blend: começo do bucket → similar ao anterior; fim → diferente
            blend_factor = 0.3 * (1.0 - pos)
            mixed = vec_main + blend_factor * vec_prev
            vec_main = np.sign(mixed).astype(np.float32)

        self._time_cache[key] = vec_main
        return vec_main.copy()

    # ─── Codificação de NT state ───────────────────────────────────────────────

    def encode_nt_state(self, nt_dict: Dict[str, float]) -> np.ndarray:
        """
        Converte estado NT em vetor HDC bipolar.
        NTs similares → vetores similares.
        """
        if not nt_dict:
            return _stable_random("nt_neutral", self.seed, self.dim)

        # Discretiza cada NT em 10 buckets
        parts = []
        for nt_name in ["DA", "NA", "ACh", "5HT", "CRH", "OXT"]:
            val = nt_dict.get(nt_name, 0.5)
            bucket = int(val * 10)
            key = f"nt:{nt_name}:{bucket}"
            if key not in self._nt_cache:
                self._nt_cache[key] = _stable_random(key, self.seed, self.dim)
            parts.append(self._nt_cache[key])

        return bundle(parts)

    # ─── Codificação de contexto ───────────────────────────────────────────────

    def encode_context(self, context_keys: List[str]) -> np.ndarray:
        """
        Codifica lista de contextos em vetor HDC (bundle dos contextos ativos).
        """
        if not context_keys:
            return _stable_random("ctx_empty", self.seed, self.dim)

        vecs = []
        for ctx in context_keys:
            if ctx not in self._context_cache:
                self._context_cache[ctx] = _stable_random(f"ctx:{ctx}", self.seed, self.dim)
            vecs.append(self._context_cache[ctx])
        return bundle(vecs)

    # ─── Binding completo (escrita episódica) ──────────────────────────────────

    def write_episodic(self,
                       concept_vec: np.ndarray,
                       context_keys: List[str],
                       nt_state: Dict[str, float],
                       t: Optional[float] = None,
                       metadata: Optional[Dict] = None,
                       resolution: str = "minute") -> np.ndarray:
        """
        Cria e armazena vetor episódico completo:
            bound = concept ⊗ time ⊗ context ⊗ nt_state

        Args:
            concept_vec:  vetor HDC do conceito/conteúdo
            context_keys: contextos ativos (ex: ['kitchen', 'gas_leak'])
            nt_state:     estado emocional no momento
            t:            timestamp (default: agora)
            metadata:     informações extras (label, source, etc.)
            resolution:   granularidade temporal

        Returns:
            Vetor bound completo (pode ser guardado e recuperado depois).
        """
        if t is None:
            t = time.time()

        time_vec    = self.encode_time_bin(t, resolution)
        context_vec = self.encode_context(context_keys)
        nt_vec      = self.encode_nt_state(nt_state)

        # Binding encadeado (XOR preserva a distância no hiperespaço)
        # bound = concept ⊗ time ⊗ context ⊗ nt
        # Para recuperar concept de um bound: concept = bound ⊗ time ⊗ context ⊗ nt
        #   (pois para bipolares: bind(a,b)*b = a*b*b = a*1 = a)
        bound = bind(bind(bind(concept_vec, time_vec), context_vec), nt_vec)

        episode = {
            "bound":   bound,
            "concept": concept_vec.copy(),
            "t":       t,
            "contexts": list(context_keys),
            "nt_state": dict(nt_state),
            "metadata": metadata or {},
        }
        self._episodes.append(episode)
        if len(self._episodes) > self._max_episodes:
            self._episodes.pop(0)

        return bound

    # ─── Query episódica ───────────────────────────────────────────────────────

    def query_episodic(self,
                       concept_vec: np.ndarray,
                       context_filter: Optional[List[str]] = None,
                       time_filter: Optional[Dict] = None,
                       nt_filter: Optional[Dict] = None,
                       resolution: str = "minute",
                       top_k: int = 5) -> List[Dict]:
        """
        Recupera memórias episódicas por qualquer combinação de eixos.

        Args:
            concept_vec:    vetor do conceito a buscar
            context_filter: filtra por contextos (optional)
            time_filter:    {'before': t}, {'after': t}, {'around': t, 'window': 60}
            nt_filter:      filtra por estado NT (optional)
            top_k:          quantos resultados retornar

        Returns:
            Lista de episódios ordenados por similaridade.
        """
        # Constrói vetor de query com os filtros disponíveis
        query = concept_vec.copy()

        if context_filter:
            ctx_vec = self.encode_context(context_filter)
            query   = bind(query, ctx_vec)

        if nt_filter:
            nt_vec = self.encode_nt_state(nt_filter)
            query  = bind(query, nt_vec)

        if time_filter:
            if "around" in time_filter:
                t_ref = time_filter["around"]
            elif "before" in time_filter or "after" in time_filter:
                t_ref = time_filter.get("before") or time_filter.get("after")
            else:
                t_ref = None

            if t_ref:
                t_vec  = self.encode_time_bin(t_ref, resolution)
                query  = bind(query, t_vec)

        # Busca por similaridade: compara concept_vec com ep["concept"] armazenado
        # Filtros de contexto/tempo/NT adicionam peso ao score mas nao mudam a busca base.
        # Isso é correto: o conceito é o "tema" do episódio. Contexto/tempo/NT refinam o ranking.
        results = []
        for ep in self._episodes:
            # 1. Filtro temporal estrito (elimina antes de calcular sim)
            if time_filter:
                if "before" in time_filter and ep["t"] >= time_filter["before"]:
                    continue
                if "after" in time_filter and ep["t"] <= time_filter["after"]:
                    continue
                if "around" in time_filter:
                    window = time_filter.get("window", 300.0)
                    if abs(ep["t"] - time_filter["around"]) > window:
                        continue

            # 2. Similaridade do conceito (busca principal)
            sim_concept = cosine_sim(concept_vec, ep["concept"])

            # 3. Boost por filtro de contexto (se especificado)
            sim_ctx = 0.0
            if context_filter and ep["contexts"]:
                overlap = len(set(context_filter) & set(ep["contexts"]))
                sim_ctx = overlap / max(len(context_filter), len(ep["contexts"]))

            # 4. Boost por filtro NT (se especificado)
            sim_nt = 0.0
            if nt_filter and ep["nt_state"]:
                nt_diffs = [abs(nt_filter.get(k, 0.5) - ep["nt_state"].get(k, 0.5))
                            for k in nt_filter]
                sim_nt = 1.0 - (sum(nt_diffs) / len(nt_diffs) if nt_diffs else 0.0)

            # 5. Score combinado
            score = sim_concept * 0.7 + sim_ctx * 0.2 + sim_nt * 0.1

            if score > 0.01:
                results.append({
                    "similarity": float(score),
                    "sim_concept": float(sim_concept),
                    "t":          ep["t"],
                    "contexts":   ep["contexts"],
                    "nt_state":   ep["nt_state"],
                    "metadata":   ep["metadata"],
                    "concept":    ep["concept"],
                })

        results.sort(key=lambda x: -x["similarity"])
        return results[:top_k]


    # ─── Contrafactual ─────────────────────────────────────────────────────────

    def counterfactual(self,
                       bound_vec: np.ndarray,
                       original_context: List[str],
                       original_t: float,
                       original_nt: Dict,
                       swap_time: Optional[float] = None,
                       swap_context: Optional[List[str]] = None,
                       swap_nt: Optional[Dict] = None,
                       resolution: str = "minute") -> np.ndarray:
        """
        Gera vetor contrafactual substituindo eixos no vetor bound.

        Permite raciocinar: "o que eu sabia sobre X ANTES do incidente?"
        ou "como eu teria avaliado X num contexto DIFERENTE?"

        Args:
            bound_vec:       vetor episódico original
            original_*:      valores originais dos eixos a remover
            swap_*:          novos valores dos eixos (None = não troca)

        Returns:
            Novo vetor com os eixos trocados.
        """
        result = bound_vec.copy()

        # Remove eixo original e adiciona novo (unbind + bind)
        if swap_time is not None:
            orig_t_vec = self.encode_time_bin(original_t, resolution)
            new_t_vec  = self.encode_time_bin(swap_time, resolution)
            result = bind(unbind(result, orig_t_vec), new_t_vec)

        if swap_context is not None:
            orig_ctx = self.encode_context(original_context)
            new_ctx  = self.encode_context(swap_context)
            result = bind(unbind(result, orig_ctx), new_ctx)

        if swap_nt is not None:
            orig_nt = self.encode_nt_state(original_nt)
            new_nt  = self.encode_nt_state(swap_nt)
            result = bind(unbind(result, orig_nt), new_nt)

        return result

    # ─── Estado ────────────────────────────────────────────────────────────────

    def get_state(self) -> Dict:
        return {
            "episodes_stored":    len(self._episodes),
            "time_cache_size":    len(self._time_cache),
            "context_cache_size": len(self._context_cache),
            "nt_cache_size":      len(self._nt_cache),
        }


# ─── Teste rápido ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    tb = TemporalBinder(seed=77)
    rng = np.random.default_rng(99)

    # Dois conceitos
    fire_vec  = np.sign(rng.standard_normal(DIM)).astype(np.float32)
    water_vec = np.sign(rng.standard_normal(DIM)).astype(np.float32)

    now = time.time()

    # Escreve memória: fogo + gás (perigo, agora)
    ep1 = tb.write_episodic(
        fire_vec, ["gas_leak", "kitchen"],
        {"DA": 0.2, "CRH": 0.9, "NA": 0.8}, t=now,
        metadata={"label": "fogo_gas"}
    )

    # Escreve memória: fogo + lareira (conforto, 1h atrás)
    ep2 = tb.write_episodic(
        fire_vec, ["fireplace", "livingroom"],
        {"DA": 0.7, "CRH": 0.1, "NA": 0.1}, t=now - 3600,
        metadata={"label": "fogo_lareira"}
    )

    print("=== Query: fogo em geral ===")
    results = tb.query_episodic(fire_vec, top_k=3)
    for r in results:
        print(f"  sim={r['similarity']:.3f} ctx={r['contexts']} label={r['metadata'].get('label')}")

    print("\n=== Query: fogo ANTES de agora (contrafactual) ===")
    results2 = tb.query_episodic(
        fire_vec,
        time_filter={"before": now - 1800}  # memórias de mais de 30min atrás
    )
    for r in results2:
        print(f"  sim={r['similarity']:.3f} ctx={r['contexts']} label={r['metadata'].get('label')}")

    print("\n=== Contrafactual: como seria fogo se contexto fosse fireplace? ===")
    cf_vec = tb.counterfactual(
        ep1,
        original_context=["gas_leak", "kitchen"], original_t=now,
        original_nt={"DA": 0.2, "CRH": 0.9, "NA": 0.8},
        swap_context=["fireplace", "livingroom"]
    )
    sim_ep2 = cosine_sim(cf_vec, ep2)
    print(f"  Similaridade do contrafactual com memória de lareira: {sim_ep2:.3f}")

    print("\nEstado:", tb.get_state())
