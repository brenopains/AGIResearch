"""
Archetypes — consolidação de padrões cognitivos em arquétipos reutilizáveis
============================================================================

Adapta os mecanismos validados no relatório para o domínio do NovaMind:
  - InductiveCore (H45): descobre regras por compressão/MDL
  - Archetype auto-generation (H189): auto-gera arquétipos de padrões recorrentes
  - Spelke Cognitive (H125-H150): primitivas cognitivas como vocabulário de base
  - Curiosidade por progresso (H22b): sinal intrínseco = taxa de compressão

Em vez de grids, opera sobre SEQUÊNCIAS DE CONCEITOS (o que o NovaMind
produz ao assistir YouTube, ler Wikipedia, brincar no playground).
"""

import time
import random
from collections import Counter, defaultdict, deque
from typing import Dict, List, Optional, Tuple


class ArchetypeEngine:
    """
    Motor de arquétipos: descobre, consolida e aplica padrões cognitivos.

    Arquétipo = composição de primitivas que se repete.
    Ex: SE[chuva] → MOLHADO vira o arquétipo "chuva-molha"

    Funciona em 3 modos:
      1. OBSERVAR: registra padrões de conceitos que co-ocorrem
      2. DORMIR: comprime padrões frequentes em arquétipos (InductiveCore)
      3. ACORDAR: arquétipos viram primitivas disponíveis pra pensar mais rápido
    """

    def __init__(self):
        # Primitivas cognitivas de base (do spelke_cognitive)
        self.base_primitives = self._load_base_primitives()

        # Arquétipos consolidados
        self.archetypes = {}       # nome → descrição do padrão
        self.archetype_usage = Counter()  # frequência de uso

        # Buffer de observações recentes (pares de conceitos)
        self.observation_buffer = deque(maxlen=200)

        # Estatísticas de compressão (pra curiosidade)
        self.raw_concept_count = 0
        self.compressed_count = 0
        self.compression_history = deque(maxlen=50)
        self._last_progress = 0.0

    def _load_base_primitives(self) -> Dict:
        """Carrega as primitivas cognitivas do spelke_cognitive como vocabulário base."""
        primitives = {}
        # Famílias com mapeamento semântico
        families = {
            "LINGUAGEM": ["SE", "E_SE", "SE_NAO", "TALVEZ", "ENQUANTO", "QUANDO", "PORQUE", "PARA"],
            "LOGICA": ["TODO", "ALGUM", "NENHUM", "EXATAMENTE_UM", "IMPLICA", "NEGA"],
            "ONTOLOGIA": ["OBJETO", "PARTE_DE", "MEMBRO_DE", "IGUAL_A", "DIFERENTE_DE", "TIPO_DE", "TEM_PROPRIEDADE"],
            "CAUSALIDADE": ["CAUSA", "PERMITE", "IMPEDE", "CORRELACIONA", "SEMPRE_QUE", "ANTES_DE", "DEPOIS_DE"],
            "MODALIDADE": ["POSSIVEL", "NECESSARIO", "DEVE", "PODE", "QUER", "SABE", "DUVIDA"],
            "METAFORA": ["LIQUIDO", "SOLIDO", "FOGOSO", "FRIO", "TRANSPARENTE", "LEVE", "PESADO"],
            "TEMPORAL": ["ANTES", "DEPOIS", "DURANTE", "ATE", "SEMPRE", "NUNCA", "AGORA", "EM_PARALELO"],
            "PERSPECTIVA": ["EU_VEJO", "VOCE_VE", "TODOS_VEEM", "NINGUEM_VE", "NO_FUTURO", "NO_PASSADO"],
            "METACOGNICAO": ["SEI", "NAO_SEI", "ACREDITO", "DUVIDO", "ISSO_PARECE_ESTRANHO"],
        }
        for family, preds in families.items():
            for p in preds:
                primitives[p] = {"family": family, "type": "cognitive"}
        return primitives

    def observe(self, concepts: List[str]):
        """
        Registra uma observação de conceitos.
        Extrai pares e padrões que podem virar arquétipos.
        """
        if not concepts:
            return

        self.raw_concept_count += len(concepts)

        # Extrai pares de conceitos co-ocorrentes (bigramas)
        for i in range(len(concepts) - 1):
            pair = (concepts[i], concepts[i + 1])
            self.observation_buffer.append(pair)

        # Tenta casar com primitivas cognitivas
        for c in concepts:
            c_up = c.upper().replace(" ", "_")
            for prim_name, prim_info in self.base_primitives.items():
                if prim_name in c_up or c_up in prim_name:
                    self.archetype_usage[prim_name] += 1

    def consolidate(self) -> List[str]:
        """
        CICLO DE SONO: comprime padrões frequentes em arquétipos.
        Retorna lista de novos arquétipos criados.

        Usa InductiveCore-lite:
          1. Conta pares frequentes no buffer
          2. Se um par aparece > threshold, vira arquétipo candidato
          3. MDL: prioriza o padrão que mais comprime (maior freq / menor tamanho)
        """
        if len(self.observation_buffer) < 10:
            return []

        freq = Counter(self.observation_buffer)
        total = len(self.observation_buffer)
        threshold = max(3, total * 0.05)

        candidates = []
        for (a, b), count in freq.most_common(20):
            if count < threshold:
                break
            # MDL score: quanto esse padrão economizaria se fosse um arquétipo
            saving = count * 2 - 1  # 2 tokens → 1, repetido count vezes
            candidates.append((saving, a, b))

        if not candidates:
            return []

        # Cria arquétipos dos melhores candidatos
        new_archetypes = []
        for saving, a, b in sorted(candidates, key=lambda x: -x[0])[:5]:
            name = self._name_archetype(a, b)
            if name not in self.archetypes:
                self.archetypes[name] = {
                    "pattern": (a, b),
                    "compression_saving": saving,
                    "created_at": time.time(),
                    "times_used": 0,
                }
                new_archetypes.append(name)

        # Limpa buffer após consolidação
        self.observation_buffer.clear()
        self.compressed_count += len(new_archetypes)
        self.compression_history.append(len(new_archetypes))

        return new_archetypes

    def _name_archetype(self, a: str, b: str) -> str:
        """Gera nome legível pro arquétipo."""
        # Tenta mapear primitiva cognitiva
        a_up = a.upper().replace(" ", "_")
        b_up = b.upper().replace(" ", "_")
        for prim in self.base_primitives:
            if prim in a_up:
                a = prim
            if prim in b_up:
                b = prim
        return f"ARQ_{a}_{b}"[:40]

    def apply_archetypes(self, concepts: List[str]) -> List[str]:
        """
        Aplica arquétipos conhecidos pra comprimir o pensamento.
        Se um padrão de arquétipo aparece, substitui pelos nomes.
        """
        if not self.archetypes or not concepts:
            return concepts

        result = list(concepts)
        # Aplica do mais específico pro mais geral
        for name, info in sorted(
            self.archetypes.items(),
            key=lambda x: -x[1]["compression_saving"],
        ):
            a, b = info["pattern"]
            i = 0
            while i < len(result) - 1:
                a_norm = result[i].lower()[:8]
                b_norm = result[i + 1].lower()[:8]
                if a_norm in result[i].lower() and b_norm in result[i + 1].lower():
                    result[i] = name
                    del result[i + 1]
                    self.archetypes[name]["times_used"] += 1
                i += 1

        return result

    def get_curiosity_score(self) -> float:
        """
        Sinal de curiosidade por progresso (H22b adaptado):
        Quanto maior a taxa de compressão recente, mais a AGI quer
        continuar nessa direção (está aprendendo).

        Retorna 0.0 (nada curioso) a 1.0 (muito curioso).
        """
        if self.raw_concept_count == 0:
            return 0.0

        # Compression ratio: quanto dos conceitos brutos viram arquétipos
        ratio = self.compressed_count / max(1, self.raw_concept_count)
        progress = ratio - self._last_progress
        self._last_progress = ratio

        # Progresso recente (últimos 50 ciclos)
        recent = list(self.compression_history)
        if not recent:
            return min(1.0, ratio * 3)

        # Se a compressão está acelerando → curioso
        recent_progress = sum(recent[-10:]) / max(1, len(recent[-10:]))
        score = min(1.0, recent_progress * 2 + ratio)
        return score

    def get_state(self) -> Dict:
        return {
            "total_archetypes": len(self.archetypes),
            "archetype_list": list(self.archetypes.keys())[-20:],
            "base_primitives": len(self.base_primitives),
            "compression_ratio": self.compressed_count / max(1, self.raw_concept_count),
            "curiosity": self.get_curiosity_score(),
            "observation_buffer_size": len(self.observation_buffer),
            "observation_buffer": len(self.observation_buffer),
        }
