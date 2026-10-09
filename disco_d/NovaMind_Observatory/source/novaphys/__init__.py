"""
novaphys — percepção física em tempo real para a NovaMind
==========================================================

Liga a AGI ao "Simulador de Realidade": doze domínios de física resolvidos
por métodos numéricos estabelecidos e validados contra resultados conhecidos
(split-step de Fourier, Gross-Pitaevskii, velocity-Verlet, Barnes-Hut,
geodésicas de Kerr, Bosch-Hale, projeção de pressão, equação de onda).

O que isto acrescenta ao projeto
--------------------------------
Três módulos do core estão prontos e sem mundo para operar:

  `physical_sandbox.py` aprende archetypes observando pares (antes, depois),
      mas hoje é ele mesmo que gera os pares — o aprendizado fecha em si.
  `active_experimenter.py` escolhe experimentos por ganho de informação,
      mas ajusta uma power-law contra dados que ele gerou com uma power-law.
  `self_vector.py` cria o referencial egocêntrico e mede afordância, mas
      sem ambiente físico não há o que ter afordância sobre.

Aqui existe um mundo externo que não colabora: o agente recebe medições com
nome, unidade e descrição, e manoplas para girar. As equações, não. Se ele
quiser saber como o mundo funciona, tem de experimentar.

Uso
---
    from novaphys import PhysicsWorld, rodar_suite

    w = PhysicsWorld()                       # servidor em localhost:8124
    w.reset(modulo="fluidos", semente=42)
    obs = w.agir(ajustar={"visc": 1e-5}, passos=400)
    print(obs["escalares"]["fl_Re"])

    rodar_suite(w)                            # benchmark completo

Antes, suba o servidor de mundo no repo do simulador:

    node bridge/server.mjs

Honestidade sobre o escopo
--------------------------
Isto dá à AGI experiência física fundamentada e um benchmark de descoberta
que resiste a Goodhart. Não dá "compreensão" nem AGI. O que ele mede é
estreito e verificável: se escolher onde medir rende mais lei por
experimento do que medir ao acaso.
"""

import sys as _sys

# O console do Windows abre em cp1252 e estoura em qualquer acento ou seta.
# Reconfigurar aqui evita que todo script que importe o pacote precise lembrar
# de exportar PYTHONIOENCODING.
for _s in (_sys.stdout, _sys.stderr):
    try:
        if _s is not None and getattr(_s, "encoding", "").lower() not in ("utf-8", "utf8"):
            _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from .client import PhysicsWorld, WorldError, DEFAULT_URL
from .agents import (BasisRegressor, LocalRegressor, HybridPredictor,
                     RandomScientist, ActiveScientist, ActiveScientistV2,
                     ActiveOptimizer, RandomOptimizer)
from .bench import rodar_tarefa, rodar_suite

__all__ = [
    "PhysicsWorld", "WorldError", "DEFAULT_URL",
    "BasisRegressor", "LocalRegressor", "HybridPredictor",
    "RandomScientist", "ActiveScientist", "ActiveScientistV2",
    "ActiveOptimizer", "RandomOptimizer",
    "rodar_tarefa", "rodar_suite",
]

__version__ = "1.0.0"
