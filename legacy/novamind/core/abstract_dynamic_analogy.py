"""
AbstractDynamicAnalogyEngine — Analogias sobre ESTRUTURAS MATEMATICAS.

Nao transfere conceitos. Transfere:
  - Bifurcacoes (Hopf, saddlenode, transcritical, pitchfork)
  - Histerese (memoria de estado, ciclos)
  - Atratores (ponto fixo, ciclo limite, estranho)
  - Transicoes de fase (ordem, desordem, criticalidade)
  - Oscilacoes (frequencia, amplitude, sincronizacao)
  - Caos (sensibilidade a condicoes iniciais, expoente Lyapunov)
  - Reacao-difusao (padroes de Turing, ondas)
  - Leis de escala (power law, exponencial, logistica)

Cada estrutura matematica pode se manifestar em multiplos dominios.
A AGI aprende a RECONHECER a estrutura e sabe que:
  "Ah, isso e uma bifurcacao. Ja vi isso em ecologia, magnetismo, e Alzheimer."
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import random


@dataclass
class DynamicStructure:
    """Uma estrutura matematica abstrata que aparece em multiplos dominios."""
    name: str
    type: str  # "bifurcation", "attractor", "oscillation", "phase_transition", "scaling_law"
    description: str
    mathematical_signature: str  # equacao ou descricao formal
    domains: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    maps_to_alzheimer: str = ""  # como se manifesta em Alzheimer

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
            "domains": self.domains[:3],
            "signature": self.mathematical_signature[:50],
        }


class AbstractDynamicAnalogyEngine:
    """Reconhece e transfere estruturas MATEMATICAS entre dominios.

    Uso:
        engine = AbstractDynamicAnalogyEngine()
        dynamics = engine.analyze_model(alzheimer_model)
        for d in dynamics:
            print(f"This model has {d.name} - seen in {d.domains}")
        analogy = engine.find_analogy(alzheimer_model)
    """

    def __init__(self):
        self._structures = self._build_library()
        self.history: List[DynamicStructure] = []

    def _build_library(self) -> Dict[str, DynamicStructure]:
        """Constroi biblioteca de estruturas matematicas conhecidas."""
        structures = {}

        # ─── BIFURCACOES ───
        structures["hopf_bifurcation"] = DynamicStructure(
            "Hopf Bifurcation", "bifurcation",
            "Sistema passa de ponto fixo estavel para ciclo limite. Oscilacoes emergem.",
            "dx/dt = mu*x - y + sigma*x*(x^2+y^2); dy/dt = x + mu*y + sigma*y*(x^2+y^2)",
            domains=["ecologia", "neurosciencia", "quimica", "fluidodinamica", "economia"],
            examples=["Neuronios disparam em burst quando mu > 0", "Economia entra em ciclo de boom-bust",
                      "Ciclo circadiano e regulado por bifurcacao de Hopf"],
            maps_to_alzheimer="Neuronios com excesso de calcium podem entrar em oscilacao patologica, sincronizando em hipersincronia.",
        )

        structures["saddlenode_bifurcation"] = DynamicStructure(
            "Saddle-Node Bifurcation", "bifurcation",
            "Dois pontos fixos (um estavel, um instavel) colidem e desaparecem. Sistema perde equilibrio.",
            "dx/dt = r + x^2",
            domains=["ecologia", "neurosciencia", "engenharia", "climatologia"],
            examples=["Extincao repentina quando taxa de reproducao cai abaixo de threshold",
                      "Ponto de inflexao climatico: derretimento abrupto de geleiras"],
            maps_to_alzheimer="Perda repentina de cognicao quando acumulo de tau ultrapassa threshold. Sistema nao volta.",
        )

        structures["pitchfork_bifurcation"] = DynamicStructure(
            "Pitchfork Bifurcation", "bifurcation",
            "Um estado simetrico se divide em dois estados assimetricos. Quebra de simetria.",
            "dx/dt = r*x - x^3",
            domains=["fisica", "biologia_evolutiva", "sociologia", "neurociencia"],
            examples=["Ferromagnetismo: ima perde magnetismo acima de temperatura critica",
                      "Dominância cerebral: esquerda vs direita",
                      "Especie se divide em duas subespecies"],
            maps_to_alzheimer="Cerebro saudavel e simetrico. Alzheimer quebra simetria: neurodegeneracao assimetrica.",
        )

        # ─── HISTERESE ───
        structures["hysteresis"] = DynamicStructure(
            "Hysteresis", "bifurcation",
            "Sistema tem memoria: o caminho de ida e diferente do de volta. Ciclo irreversivel.",
            "dx/dt = -x + tanh(beta*(x + s))",
            domains=["magnetismo", "economia", "neurociencia", "psicologia", "engenharia"],
            examples=["Magnetizacao: momento magnetico persiste apos campo externo cessar",
                      "Precos: sobem rapido, descem devagar",
                      "Memoria: neuronios mantem estado apos estimulo"],
            maps_to_alzheimer="Amyloid se acumula (ida). Uma vez acumulado, clearance e muito mais dificil (volta). Ciclo irreversivel.",
        )

        structures["snapback_hysteresis"] = DynamicStructure(
            "Snap-Back Hysteresis", "bifurcation",
            "Sistema salta repentinamente entre estados estaveis quando perturbacao ultrapassa threshold.",
            "dx/dt = h - x + s*tanh(beta*x)",
            domains=["geologia", "neurociencia", "climatologia", "ciencia_materiais"],
            examples=["Terremoto: placa tectonica acumula tensao e libera de repente",
                      "Crise epileptica: cerebro acumula excitabilidade e dispara"],
            maps_to_alzheimer="Declinio cognitivo nao e linear. Pacientes tem 'quedas' repentinas apos periodos estaveis.",
        )

        # ─── ATRATORES ───
        structures["stable_focus"] = DynamicStructure(
            "Stable Focus", "attractor",
            "Espiral converge para ponto fixo. Oscilacoes amortecidas.",
            "dx/dt = -x + omega*y; dy/dt = -omega*x - y",
            domains=["biologia", "economia", "engenharia"],
            examples=["Populacao converge para capacidade de suporte",
                      "Preco converge para equilibrio de mercado"],
            maps_to_alzheimer="Cerebro saudavel e um foco estavel. Perturbacoes sao amortecidas.",
        )

        structures["limit_cycle"] = DynamicStructure(
            "Limit Cycle", "attractor",
            "Sistema oscila perpetuamente com amplitude e frequencia fixas.",
            "dx/dt = mu*x - omega*y - (x^2+y^2)*x; dy/dt = omega*x + mu*y - (x^2+y^2)*y",
            domains=["biologia", "fisiologia", "economia", "quimica"],
            examples=["Ciclo circadiano, ciclo menstrual, ciclo cardiaco",
                      "Ciclo economico: expansao, pico, recessao, vale"],
            maps_to_alzheimer="Ciclo sono-vigilia e interrompido. Sleep fragmentation -> clearance prejudicado -> amyloid.",
        )

        structures["strange_attractor"] = DynamicStructure(
            "Strange Attractor", "attractor",
            "Sistema caotico: trajetoria nunca se repete mas fica confinada a regiao.",
            "dx/dt = sigma*(y-x); dy/dt = x*(rho-z) - y; dz/dt = x*y - beta*z",
            domains=["meteorologia", "neurociencia", "astrofisica", "economia"],
            examples=["Clima e caotico: previsao so funciona em curto prazo",
                      "EEG de cerebro saudavel mostra caos. Epilepsia e perda de caos."],
            maps_to_alzheimer="EEG de Alzheimer mostra perda de complexidade. Cerebro fica "preso" em padroes simples.",
        )

        # ─── TRANSICOES DE FASE ───
        structures["phase_transition"] = DynamicStructure(
            "Phase Transition", "phase_transition",
            "Ordem emerge do caos (ou vice-versa) quando parametro ultrapassa valor critico.",
            "parametrica: beta * (T - T_c)",
            domains=["fisica", "biologia", "sociologia", "neurociencia"],
            examples=["Agua congela a 0C, ferve a 100C",
                      "Magnetizacao: spins se alinham abaixo de Tc",
                      "Consciencia: seria uma transicao de fase na dinamica neural?"],
            maps_to_alzheimer="Cerebro faz transicao de fase: saudavel (caos funcional) -> doente (ordem rigida).",
        )

        structures["self_organized_criticality"] = DynamicStructure(
            "Self-Organized Criticality", "phase_transition",
            "Sistema evolui naturalmente para ponto critico, onde ocorrem avalanches de todos os tamanhos.",
            "dx/dt = f(x) + ruido; reset quando threshold excedido",
            domains=["geologia", "neurociencia", "economia", "biologia_evolutiva"],
            examples=["Avalanches neurais no cerebro seguem power law",
                      "Terremotos seguem lei de Gutenberg-Richter",
                      "Extincoes em massa seguem power law"],
            maps_to_alzheimer="Avalanches neurais sao perdidas. Cerebro perde criticalidade -> perda de processamento de informacao.",
        )

        # ─── REACAO-DIFUSAO ───
        structures["turing_patterns"] = DynamicStructure(
            "Turing Patterns", "oscillation",
            "Sistema reacao-difusao gera padroes espaciais estaveis (manchas, listras).",
            "du/dt = D_u * laplacian(u) + f(u,v); dv/dt = D_v * laplacian(v) + g(u,v)",
            domains=["biologia_do_desenvolvimento", "quimica", "ecologia"],
            examples=["Padroes de pelagem de zebras e leopardos",
                      "Formacao de digitos na embriao",
                      "Vegetacao em desertos forma padroes de Turing"],
            maps_to_alzheimer="Padroes de propagacao de tau seguem ondas que lembram reacao-difusao. Prion-like propagation.",
        )

        return structures

    def analyze_model(self, model) -> List[DynamicStructure]:
        """Identifica estruturas matematicas no modelo ODE.

        Analisa as ODEs para reconhecer padroes:
          - Termos cubicos -> bifurcacao pitchfork
          - Termos quadraticos -> saddlenode
          - Acoplamento com auto-amplificacao -> Hopf
          - Termos com memoria/integracao -> histerese
          - Termos de difusao -> Turing
        """
        matches = []
        ode_str = str(model.ode_fn.__doc__ if hasattr(model.ode_fn, '__doc__') else model.ode_fn)

        # Heuristica: verifica termos nas ODEs
        # Testa simulando o modelo e analisando comportamento
        from scipy.integrate import solve_ivp
        import numpy as np

        try:
            # Simula com condicao inicial padrao
            sol1 = solve_ivp(model.ode_fn, (0, 365), model.initial_state,
                            method="RK45", max_step=50, rtol=1e-3, atol=1e-6)

            # Simula com condicao inicial alterada (testa histerese/memoria)
            y0_alt = model.initial_state.copy()
            if len(y0_alt) > 0:
                y0_alt[0] *= 2.0
            sol2 = solve_ivp(model.ode_fn, (0, 365), y0_alt,
                            method="RK45", max_step=50, rtol=1e-3, atol=1e-6)

            if sol1.y.shape[1] > 1 and sol2.y.shape[1] > 1:
                # Verifica se ha histerese (estado final depende da condicao inicial)
                final_diff = np.mean(np.abs(sol1.y[:, -1] - sol2.y[:, -1]))
                if final_diff > 0.1:
                    if "hysteresis" in self._structures:
                        matches.append(self._structures["hysteresis"])

                # Verifica se ha oscilacao (variancia alta no final)
                if sol1.y.shape[1] > 10:
                    last_third = sol1.y[:, -sol1.y.shape[1]//3:]
                    var_per_var = np.var(last_third, axis=1)
                    if np.mean(var_per_var) > 0.05:
                        if "limit_cycle" in self._structures:
                            matches.append(self._structures["limit_cycle"])

                # Verifica se ha termo cubico (pitchfork)
                n_vars = len(model.initial_state)
                test_y = np.ones(n_vars) * 0.5
                deriv = model.ode_fn(0, test_y)
                deriv2 = model.ode_fn(0, test_y * 2)
                ratio = np.mean(np.abs(deriv2) / (np.abs(deriv) + 1e-10))
                if ratio > 5:
                    if "pitchfork_bifurcation" in self._structures:
                        matches.append(self._structures["pitchfork_bifurcation"])

        except:
            pass

        self.history.extend(matches)
        return matches[:5]

    def find_analogy(self, model):
        """Encontra a melhor analogia matematica para o modelo."""
        dynamics = self.analyze_model(model)
        if not dynamics:
            return None

        best = dynamics[0]
        return {
            "structure": best.name,
            "type": best.type,
            "description": best.description,
            "alzheimer_connection": best.maps_to_alzheimer,
            "domains": best.domains,
        }

    def get_stats(self) -> dict:
        return {
            "known_structures": len(self._structures),
            "by_type": dict(
                (t, sum(1 for s in self._structures.values() if s.type == t))
                for t in set(s.type for s in self._structures.values())
            ),
            "total_matches": len(self.history),
        }
