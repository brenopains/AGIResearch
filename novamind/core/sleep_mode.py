"""
SleepMode — Replay Offline + Consolidação de Archetypes
========================================================
Quando o neuromodulador indica sono (5HT baixo + ACh alto):
  1. Busca N chunks aleatórios da memória HDV
  2. Recombina pares no Sandbox (bind aleatório)
  3. Mede Energia Livre de cada combinação
  4. Se FE cai significativamente, consolida como novo Archetype
"""

import random, time
from typing import List, Optional

from .coglang import Manifold, calculate_free_energy, bind_roles


class SleepMode:
    """
    Ciclo de sono: recombinação offline de memórias.
    O cérebro "sonha" combinando experiências passadas para
    descobrir analogias e consolidar intuições.

    Uso:
        sm = SleepMode(mind)
        novos = sm.replay(num_pairs=100)
        # novos = lista de archetypes consolidados
    """

    def __init__(self, mind):
        self.mind = mind
        self.sleep_count = 0
        self.total_discoveries = 0

    def replay(self, num_pairs: int = 100) -> List[str]:
        """
        Ciclo principal de replay offline.
        Pega chunks aleatórios da memória, recombina no Sandbox,
        e consolida descobertas como novos archetypes.

        Args:
            num_pairs: número de combinações aleatórias a testar

        Returns:
            Lista de nomes de novos archetypes criados
        """
        engine = self.mind.hdc_memory.engine
        sandbox = getattr(self.mind, 'sandbox', None)
        if sandbox is None:
            return []

        novos = []

        # 1. Coleta textos da memória HDV
        textos = self._sample_texts(min(max(50, num_pairs // 2), 200))
        if len(textos) < 4:
            return []

        # 2. Para cada par aleatório, tenta bind no Sandbox
        for _ in range(num_pairs):
            a, b = random.sample(textos, 2)
            try:
                # Compõe cada texto como Manifold
                mf_a = sandbox.compose_concept(a)
                mf_b = sandbox.compose_concept(b)
                if mf_a is None or mf_b is None:
                    continue

                # Mede Energia Livre ANTES do bind
                fe_antes = calculate_free_energy(mf_a, mf_b)

                # Tenta bind criativo: a * b (bind direto no HDV)
                mf_combined = mf_a * mf_b  # binding

                # Mede Energia Livre DEPOIS do bind
                # Se a combinação é "natural", a FE cai
                # porque os conceitos se complementam
                identidade = Manifold(torch.ones(10000, device=engine.device))
                fe_depois = calculate_free_energy(mf_combined, identidade)

                # 3. Se a FE caiu significativamente → descoberta analógica
                if fe_depois < fe_antes * 0.6:
                    nome = self._nomear(a, b, fe_antes, fe_depois)
                    self._consolidar(nome, a, b)
                    novos.append(nome)
                    self.total_discoveries += 1

            except Exception:
                continue

        self.sleep_count += 1
        if novos:
            print(f"[SLEEP] Ciclo {self.sleep_count}: +{len(novos)} descobertas ({self.total_discoveries} total)")

        return novos

    def _sample_texts(self, n: int) -> List[str]:
        """Amostra textos da memória HDV."""
        textos = []
        for indices in self.mind.hdc_memory.indices.values():
            for meta in indices:
                t = meta.get('text', '') if isinstance(meta, dict) else ''
                if t and len(t) > 10:
                    textos.append(t)
        random.shuffle(textos)
        return textos[:n]

    def _nomear(self, a: str, b: str, fe_antes: float, fe_depois: float) -> str:
        """Gera nome para o novo archetype."""
        a_short = a[:20].replace(' ', '_')
        b_short = b[:20].replace(' ', '_')
        saving = int((fe_antes - fe_depois) * 100)
        return f"ARQ_{a_short}+{b_short}_S{saving}"

    def _consolidar(self, nome: str, a: str, b: str):
        """Registra o novo archetype no motor de arquétipos."""
        try:
            if hasattr(self.mind, 'archetypes'):
                self.mind.archetypes.observe([a[:30], b[:30]])
                # Força consolidação imediata
                self.mind.archetypes.consolidate()
        except Exception as e:
            print(f"[SLEEP] Erro ao consolidar: {e}")
