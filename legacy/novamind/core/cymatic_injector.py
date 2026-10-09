"""
cymatic_injector.py — Injeção de Estrutura Cimática com Gate de Adequação
==========================================================================
"Ressonância encontra ESTABILIDADE. Adequação encontra COGNIÇÃO.
Precisamos das duas juntas."

O CymaticMedium gera padrões estáveis por interferência. Mas estabilidade
NÃO implica valor cognitivo (um cristal é estável e não pensa). Este
módulo é o PORTÃO: só deixa um padrão cimático entrar no sistema cognitivo
se passar por DOIS testes:

  1. ESTABILIDADE: ressoou (attractor) — vem do CymaticMedium
  2. ADEQUAÇÃO: tem valor cognitivo — medido por fitness interna
     (reduz energia livre? ressoa com conceitos do CGE? comprime?)

Sem os dois, o padrão é descartado. Isto evita a armadilha de "achar
que qualquer padrão bonito é cognição".

Uso típico:
  injector = CymaticInjector(medium=cymatic, cge=cge)
  injector.sweep_and_inject(freq_range=(1, 40), n_freqs=20)
  incorporados = injector.get_incorporated()  # só os que passaram
"""
from __future__ import annotations
import math
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple

from .cymatic_medium import CymaticMedium, ResonantPattern


class CymaticInjector:
    """
    Portão de adequação entre o meio cimático e o sistema cognitivo.

    Cada padrão cimático candidato é avaliado por um FITNESS EVALUATOR
    (função que mede valor cognitivo). Só os que superam um threshold
    de adequação SÃO incorporados ao CGE/massinha.
    """

    def __init__(self, medium: CymaticMedium,
                 cge=None,
                 adequacy_threshold: float = 0.4,
                 fitness_evaluator: Optional[Callable] = None):
        self.medium = medium
        self.cge = cge
        self.adequacy_threshold = adequacy_threshold
        # Evaluator default: mede coerência estrutural + alinhamento com CGE
        self.fitness_evaluator = fitness_evaluator or self._default_evaluator

        self.candidates: List[Dict] = []      # todos os padrões testados
        self.incorporated: List[Dict] = []    # só os que passaram o gate
        self.rejected: List[Dict] = []        # estáveis mas sem adequação

    # ─── Avaliação de adequação ───────────────────────────────────────

    def _default_evaluator(self, pattern: ResonantPattern) -> float:
        """
        Evaluator padrão de valor cognitivo. Combina:

        1. SIMETRIA estrutural (0..1): padrões simétricos tendem a
           representar conceitos coerentes (vs. ruído).
        2. ALINHAMENTO COM CGE (0..1): se o padrão ressoa com conceitos
           já presentes no CGE, tem mais chance de ser útil.
        3. ENTROPIA ESPECTRAL (0..1): padrões com estrutura espectral
           rica (não monocromática) carregam mais informação.

        Retorna 0..1. Padrões abaixo do threshold são rejeitados.
        """
        v = pattern.vector.astype(np.float64)
        n = len(v)

        # 1. Simetria (reflexão + translação parcial)
        half = n // 2
        sym_reflect = float(np.mean(v[:half] == v[half:][::-1]))
        # Simetria translacional: compara primeiros 1/4 com segundos 1/4
        q = n // 4
        sym_trans = float(np.mean(v[:q] == v[q:2*q]))
        symmetry = (sym_reflect + sym_trans) / 2

        # 2. Alinhamento com CGE (se disponível)
        cge_alignment = 0.0
        if self.cge is not None and hasattr(self.cge, 'nodes'):
            # Compara o padrão com os vetores dos nós do CGE
            best_sim = 0.0
            count = 0
            for nid, ndata in self.cge.nodes.items():
                vec = ndata.get('vector')
                if vec is None:
                    continue
                try:
                    vec_np = vec.detach().cpu().numpy() if hasattr(vec, 'detach') else np.asarray(vec)
                    if len(vec_np) == n:
                        sim = float(np.dot(v, vec_np) / (n + 1e-9))
                        best_sim = max(best_sim, abs(sim))
                        count += 1
                except Exception:
                    continue
                if count > 50:  # amostra para velocidade
                    break
            cge_alignment = min(1.0, best_sim * 3.0)  # escala: sim 0.33 = align 1.0

        # 3. Entropia espectral
        spectrum = np.abs(np.fft.fft(v))[:n // 2]
        psd = spectrum / (spectrum.sum() + 1e-9)
        spectral_entropy = -float(np.sum(psd * np.log(psd + 1e-12)))
        max_entropy = math.log(n // 2)
        norm_entropy = spectral_entropy / max_entropy if max_entropy > 0 else 0.0
        # Padroes com entropia moderada (nem puro tom, nem ruido) sao mais ricos
        info_richness = 1.0 - abs(norm_entropy - 0.5) * 2  # pico em 0.5

        # Combinação ponderada — nenhum termo domina
        adequacy = (0.40 * symmetry
                    + 0.35 * cge_alignment
                    + 0.25 * info_richness)
        return float(np.clip(adequacy, 0.0, 1.0))

    # ─── Sweep com gate ───────────────────────────────────────────────

    def sweep_and_inject(self, freq_range: Tuple[float, float] = (1.0, 40.0),
                         n_freqs: int = 20, steps_per_freq: int = 10,
                         verbose: bool = False) -> Dict:
        """
        Varre frequências, gera padrões cimáticos, e para cada um:
          1. Mede estabilidade (attractor?)
          2. Mede adequação (valor cognitivo?)
          3. Só incorpora se AMBOS passarem

        Retorna relatório completo: candidatos, incorporados, rejeitados.
        """
        freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)

        for f in freqs:
            self.medium.reset()
            stab_series = self.medium.drive([float(f)], n_steps=steps_per_freq,
                                            damping=0.3)
            final_stab = stab_series[-1] if stab_series else 0.0

            pattern = self.medium.detect_resonances(
                min_stability=0.5, freqs=[float(f)],
                stability_series=stab_series
            )

            if pattern is None:
                continue

            # GATE 1: estabilidade
            if pattern.stability < 0.85:
                continue

            # GATE 2: adequação
            adequacy = float(self.fitness_evaluator(pattern))

            candidate = {
                "freq": float(f),
                "pattern_id": pattern.pattern_id,
                "stability": pattern.stability,
                "adequacy": adequacy,
                "passed": adequacy >= self.adequacy_threshold,
                "pattern": pattern,
            }
            self.candidates.append(candidate)

            if candidate["passed"]:
                pattern.cognitive_value = adequacy
                self._incorporate(pattern)
                self.incorporated.append(candidate)
                if verbose:
                    print(f"  [INCORPORADO] freq={f:.1f} stab={pattern.stability:.3f} "
                          f"adq={adequacy:.3f}")
            else:
                self.rejected.append(candidate)
                if verbose:
                    print(f"  [rejeitado]   freq={f:.1f} stab={pattern.stability:.3f} "
                          f"adq={adequacy:.3f} (abaixo de {self.adequacy_threshold})")

        return self._report()

    def _incorporate(self, pattern: ResonantPattern) -> None:
        """Incorpora o padrão ao CGE como novo nó (se disponível)."""
        if self.cge is None:
            return
        try:
            import torch
            vec = torch.tensor(pattern.vector, dtype=torch.float32,
                              device=getattr(self.cge, 'device', 'cpu'))
            node_name = f"cymatic_{pattern.pattern_id}"
            if hasattr(self.cge, 'create_node') and node_name not in getattr(self.cge, 'nodes', {}):
                self.cge.create_node(node_name, semantic_vec=vec)
                self.cge.nodes[node_name]["confidence"] = pattern.stability
                self.cge.nodes[node_name]["state"] = "imagined"
        except Exception:
            pass

    def _report(self) -> Dict:
        if not self.candidates:
            return {"n_candidates": 0, "n_incorporated": 0, "n_rejected": 0}
        return {
            "n_candidates": len(self.candidates),
            "n_incorporated": len(self.incorporated),
            "n_rejected": len(self.rejected),
            "incorporation_rate": len(self.incorporated) / len(self.candidates),
            "best_freq": max(self.candidates, key=lambda c: c["adequacy"])["freq"],
            "best_adequacy": max(c["adequacy"] for c in self.candidates),
            "mean_stability_of_incorporated": (
                float(np.mean([c["stability"] for c in self.incorporated]))
                if self.incorporated else 0.0
            ),
        }

    def summary(self) -> str:
        r = self._report()
        lines = [
            f"CymaticInjector report:",
            f"  Candidatos testados: {r.get('n_candidates', 0)}",
            f"  Incorporados (stab+adq): {r.get('n_incorporated', 0)}",
            f"  Rejeitados (sem adq): {r.get('n_rejected', 0)}",
            f"  Taxa de incorporação: {r.get('incorporation_rate', 0):.1%}",
            f"  Melhor frequência: {r.get('best_freq', 0):.1f} "
            f"(adq={r.get('best_adequacy', 0):.3f})",
        ]
        return "\n".join(lines)
