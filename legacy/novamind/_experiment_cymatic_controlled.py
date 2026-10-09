"""
_experiment_cymatic_controlled.py — A experiencia que responde:
   "A estrutura cimatica melhora ALGUM trial?"

Protocolo experimental honesto:
  - Mesma seed, mesmas condicoes, tudo identico
  - Unica diferenca: COM vs SEM injecao cimatica no CGE
  - Se o score muda significativamente, a cimatica tem efeito real
  - Se nao muda, o gargalo nao e estrutural (e funcional/funcional)

Isto e ciencia: variavel controlada, hipotese testavel, resultado
reportado honestamente seja qual for.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from core.cymatic_medium import CymaticMedium
from core.cymatic_injector import CymaticInjector


class FakeCGE:
    """CGE minimo para teste controlado. So precisa de .nodes com vetores."""
    def __init__(self):
        self.nodes = {}
        self.device = 'cpu'
    def create_node(self, node_id, semantic_vec=None, **kw):
        self.nodes[node_id] = {"vector": semantic_vec, **kw}


def run_cymatic_vs_baseline():
    """Compara o valor cognitivo dos padroes cimaticos vs baseline aleatorio."""

    print("=" * 60)
    print("EXPERIENCIA CONTROLADA: cimatica tem efeito cognitivo?")
    print("=" * 60)
    print()
    print("Hipotese: padroes cimaticos (estaveis + adequados) tem valor")
    print("cognitivo MAIOR que padroes aleatorios com a mesma estabilidade.")
    print("Se sim, a cimatica NAO e so estabilidade — carrega estrutura util.")
    print()

    # ─── Condicao 1: CGE vazio (baseline absoluto) ───────────────────
    cge_empty = FakeCGE()
    cm1 = CymaticMedium(dim=5000, seed=2026)
    inj1 = CymaticInjector(medium=cm1, cge=cge_empty, adequacy_threshold=0.4)
    print("[1] CGE VAZIO — sem conceitos para alinhar:")
    inj1.sweep_and_inject(freq_range=(1, 40), n_freqs=30, steps_per_freq=10, verbose=False)
    print(inj1.summary())
    print()

    # ─── Condicao 2: CGE com conceitos reais (simulados) ─────────────
    cge_seeded = FakeCGE()
    # Popula com alguns "conceitos" (vetores estruturados, nao aleatorios)
    import torch
    rng = np.random.RandomState(99)
    concept_names = ["tree", "water", "fire", "rock", "mind", "cycle",
                     "balance", "growth", "structure", "force"]
    for name in concept_names:
        # Conceito = padrao estruturado (mistura de poucos senos = coerente)
        n = 5000
        idx = np.arange(n)
        v = np.zeros(n)
        for k in rng.choice(range(1, 20), size=3, replace=False):
            v += np.sin(2 * np.pi * k * idx / n + rng.uniform(0, 6.28))
        v = np.sign(v)
        cge_seeded.nodes[name] = {"vector": torch.tensor(v, dtype=torch.float32)}

    cm2 = CymaticMedium(dim=5000, seed=2026)
    inj2 = CymaticInjector(medium=cm2, cge=cge_seeded, adequacy_threshold=0.4)
    print("[2] CGE COM CONCEITOS — estrutura para alinhar:")
    inj2.sweep_and_inject(freq_range=(1, 40), n_freqs=30, steps_per_freq=10, verbose=False)
    print(inj2.summary())
    print()

    # ─── Condicao 3: baseline aleatorio (controle negativo) ──────────
    print("[3] BASELINE ALEATORIO — padroes nao-cimaticos (controle):")
    rng = np.random.RandomState(2026)
    random_adqs = []
    for _ in range(30):
        # Vetor aleatorio puro, estabilidade maxima (nao muda)
        from core.cymatic_medium import ResonantPattern
        v = rng.choice([-1.0, 1.0], 5000)
        pat = ResonantPattern(pattern_id="rand", vector=v,
                             driving_freqs=(), stability=1.0, energy=0.0)
        adq = inj2.fitness_evaluator(pat)
        random_adqs.append(adq)
    print(f"  Adequacao media de 30 padroes ALEATORIOS: {np.mean(random_adqs):.3f}")
    print(f"  (vs cimaticos: {np.mean([c['adequacy'] for c in inj2.candidates]):.3f})")
    print()

    # ─── Analise estatistica ─────────────────────────────────────────
    cym_adqs = [c["adequacy"] for c in inj2.candidates]
    rand_adqs = random_adqs

    cym_mean, rand_mean = np.mean(cym_adqs), np.mean(rand_adqs)
    cym_std, rand_std = np.std(cym_adqs), np.std(rand_adqs)

    print("-" * 60)
    print("RESULTADO ESTATISTICO:")
    print(f"  Cimaticos:      media={cym_mean:.3f} ± {cym_std:.3f} (n={len(cym_adqs)})")
    print(f"  Aleatorios:     media={rand_mean:.3f} ± {rand_std:.3f} (n={len(rand_adqs)})")
    print(f"  Diferenca:      {cym_mean - rand_mean:+.3f}")

    # Teste t simplificado
    if len(cym_adqs) > 1 and len(rand_adqs) > 1:
        pooled_std = np.sqrt((cym_std**2 + rand_std**2) / 2)
        if pooled_std > 1e-9:
            t_stat = (cym_mean - rand_mean) / (pooled_std * np.sqrt(2/len(cym_adqs)))
            print(f"  t-statistic:    {t_stat:.2f}")
            if abs(t_stat) > 2.0:
                verdict = "SIGNIFICATIVO (p<0.05) — cimatica tem efeito real"
            elif abs(t_stat) > 1.0:
                verdict = "MARGINAL — tendencia mas nao conclusivo"
            else:
                verdict = "NAO SIGNIFICATIVO — cimatica nao difere de aleatorio"
            print(f"  Veredito:       {verdict}")

    print()
    print("CONCLUSAO HONESTA:")
    if cym_mean - rand_mean > 0.05:
        print("  Padroes cimaticos TEM valor cognitivo acima do aleatorio.")
        print("  A ideia de 'som gera estrutura util' e sustentada pelos dados.")
    elif cym_mean - rand_mean > 0:
        print("  Padroes cimaticos sao marginalmente melhores que aleatorio.")
        print("  Sinal fraco — precisa de mais estrutura ou melhor evaluator.")
    else:
        print("  Padroes cimaticos NAO superam aleatorio neste setup.")
        print("  A estabilidade sozinha nao gera valor cognitivo.")
    print("=" * 60)


if __name__ == "__main__":
    run_cymatic_vs_baseline()
