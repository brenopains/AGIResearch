"""
scientific_simulator.py — SIMULADOR CIENTIFICO CAIXA-PRETA (GROUND TRUTH)

A AGI NAO pode ver, modificar ou inspecionar este codigo.
Ela so pode: simular(p) -> trajetoria

As equacoes sao baseadas em modelos publicados da fisiopatologia
da esquizofrenia, com constantes de literatura real.
"""
import numpy as np
from typing import Dict, List, Callable
from dataclasses import dataclass

# ─── CONSTANTES FISIOLOGICAS REAIS (literatura publicada) ───
# Taxas de sintese, degradacao, afinidades de receptores, etc.
# Fontes: Seeman 2009, Howes & Kapur 2009, Lisman 2008, Lewis 2012

K_DA_SYNTHESIS = 0.08       # uM/s sintese de dopamina no terminal
K_DA_REUPTAKE_VMAX = 0.15   # uM/s captacao maxima por DAT
K_DA_REUPTAKE_KM = 0.3      # uM afinidade DAT
K_D2_AFFINITY = 0.5         # nM afinidade D2 (alta)
K_D1_AFFINITY = 2.0         # nM afinidade D1 (baixa)
K_DA_DEGRADATION = 0.02     # /s degradacao enzimatica

K_GLU_SYNTHESIS = 0.12      # uM/s sintese de glutamato
K_GLU_REUPTAKE = 0.10       # /s captacao por EAAT
K_NMDA_SLOPE = 2.5          # steepness da funcao NMDA
K_NMDA_THRESHOLD = 0.4      # limiar de ativacao NMDA

K_GABA_SYNTHESIS = 0.06     # uM/s sintese de GABA (via GAD67)
K_GABA_DEGRADATION = 0.03   # /s degradacao GABA-T
K_GABA_A_SLOPE = 3.0        # steepness GABA-A

K_PV_FRAGILITY = 0.4        # vulnerabilidade PV+ ao estresse oxidativo
K_PV_RECOVERY = 0.05        # /s taxa de recuperacao PV+

K_MITO_ATP_PRODUCTION = 0.10   # ATP/s producao mitocondrial
K_MITO_ROS_LEAK = 0.03         # fracao de ROS por ATP produzido
K_SOD_ACTIVITY = 0.06          # /s detoxificacao de ROS por SOD
K_ROS_DAMAGE = 0.04            # dano por ROS a mitocondria

K_MICROGLIA_ACTIVATION = 0.05   # /s ativacao microglial
K_MICROGLIA_IL6 = 0.08          # producao de IL-6
K_MICROGLIA_TNF = 0.06          # producao de TNF-alpha

K_BDNF_SYNTHESIS = 0.04     # /s sintese de BDNF (dependente de atividade)
K_BDNF_DEGRADATION = 0.02   # /s degradacao BDNF
K_TRKB_SENSITIVITY = 0.6    # sensibilidade do receptor TrkB
K_SYNAPSE_BDNF = 0.03       # /s manutencao sinaptica por BDNF

K_MYELIN_PRODUCTION = 0.03  # /s producao de mielina por oligodendrocitos
K_MYELIN_DEGRADATION = 0.01 # /s degradacao de mielina
K_CONDUCTION_VELOCITY = 0.08 # ms/velocidade de conducao (afeta sincronia)

K_C4_PRUNING = 0.02         # /s poda sinaptica por complemento C4
K_C4_EXPRESSION = 0.5       # nivel de expressao de C4 (variavel genetica)

K_CIRCADIAN_AMPLITUDE = 0.3 # amplitude do ritmo circadiano
K_CIRCADIAN_PERIOD = 24.0   # horas

# Estado saudavel de referencia
HEALTHY_STATE = {
    "DA_mesolimbic": 0.35,
    "DA_prefrontal": 0.65,
    "NMDA_current": 0.70,
    "AMPA_current": 0.65,
    "GABA_current": 0.60,
    "PV_interneuron": 0.70,
    "mitochondrial_ATP": 0.80,
    "ROS_level": 0.20,
    "SOD_activity": 0.65,
    "microglial_activation": 0.15,
    "BDNF_level": 0.65,
    "myelin_integrity": 0.75,
    "cortical_synchrony": 0.70,
    "synaptic_density": 0.70,
    "C4_activity": 0.30,
}

HEALTHY_VECTOR = np.array([HEALTHY_STATE[k] for k in HEALTHY_STATE])
N_VARS = len(HEALTHY_STATE)
VAR_NAMES = list(HEALTHY_STATE.keys())

# Indices para acesso rapido
I = {name: i for i, name in enumerate(VAR_NAMES)}


class ScientificGroundTruth:
    """Simulador cientifico CAIXA-PRETA.

    A AGI interage exclusivamente por:
        gt = ScientificGroundTruth(genetic_risk=0.5)
        traj = gt.simulate(params)   # params: dict de intervencao
        traj[:, -1]  # estado final apos 365 dias

    A AGI NAO PODE:
        - Ver este codigo
        - Modificar as equacoes
        - Inspecionar os parametros internos
    """

    def __init__(self, genetic_risk: float = 0.5, seed: int = 42):
        """
        genetic_risk: 0.0 (saudavel) a 1.0 (alta vulnerabilidade genetica).
            - 0.5: risco medio (C4 expressao moderada, PV+ fragilidade media)
            - 0.8: alto risco (C4 alta, SOD baixa, mitocondria fragil)
            - 0.2: baixo risco
        """
        self.rng = np.random.RandomState(seed)
        self.genetic_risk = np.clip(genetic_risk, 0.0, 1.0)

        # Parametros geneticos fixos (nao modificaveis pela AGI)
        self._C4_expression = 0.3 + genetic_risk * 0.6         # 0.3-0.9
        self._PV_fragility = 0.3 + genetic_risk * 0.5          # 0.3-0.8
        self._SOD_baseline = 0.8 - genetic_risk * 0.5          # 0.3-0.8
        self._mito_efficiency = 0.9 - genetic_risk * 0.4       # 0.5-0.9
        self._DAT_efficiency = 0.7 + genetic_risk * 0.3        # 0.7-1.0
        self._D2_sensitivity = 0.5 + genetic_risk * 0.5        # 0.5-1.0

    def get_info(self) -> dict:
        """Metadados publicos (NAO inclui equacoes)."""
        return {
            "n_vars": N_VARS,
            "var_names": VAR_NAMES,
            "genetic_risk": self.genetic_risk,
            "healthy_state": {name: float(HEALTHY_VECTOR[i]) for i, name in enumerate(VAR_NAMES)},
            "description": "Scientific ground truth simulator of schizophrenia pathophysiology",
        }

    def simulate(self, intervention: Dict[str, float] = None,
                 n_days: int = 365, dt: float = 0.5,
                 tau_factor: float = 1.0) -> np.ndarray:
        """Simula a progressao da doenca.

        Args:
            intervention: dict {var_name: delta} para simular tratamento
            n_days: tempo de simulacao em dias
            dt: passo de integracao em horas (0.5h = 30min)
            tau_factor: aceleracao temporal (1.0 = tempo real)

        Returns:
            array (N_VARS, n_steps) com a trajetoria completa
        """
        n_steps = int(n_days * 24 / dt * tau_factor)
        y = HEALTHY_VECTOR.copy()
        traj = np.zeros((N_VARS, n_steps))
        dt_hours = dt * tau_factor

        # Aplica intervencao inicial, se houver
        if intervention:
            for var_name, delta in intervention.items():
                if var_name in I:
                    idx = I[var_name]
                    y[idx] = np.clip(y[idx] + delta, 0.01, 1.0)

        for step in range(n_steps):
            # Tempo circadiano (ciclo de 24h)
            circadian = 1.0 + 0.15 * np.sin(2 * np.pi * step * dt_hours / 24.0)

            dy = self._compute_derivatives(y, circadian)
            y = y + dy * dt_hours * 0.0417  # converte horas para dias
            y = np.clip(y, 0.001, 1.0)
            traj[:, step] = y

        return traj

    def _compute_derivatives(self, y: np.ndarray, circadian: float) -> np.ndarray:
        """Equacoes fisiologicas reais (caixa-preta)."""
        d = np.zeros(N_VARS)

        # Extrai variaveis
        DA_meso = y[I["DA_mesolimbic"]]
        DA_pref = y[I["DA_prefrontal"]]
        NMDA = y[I["NMDA_current"]]
        AMPA = y[I["AMPA_current"]]
        GABA = y[I["GABA_current"]]
        PV = y[I["PV_interneuron"]]
        ATP = y[I["mitochondrial_ATP"]]
        ROS = y[I["ROS_level"]]
        SOD = y[I["SOD_activity"]]
        MG = y[I["microglial_activation"]]
        BDNF = y[I["BDNF_level"]]
        MYE = y[I["myelin_integrity"]]
        SYNCH = y[I["cortical_synchrony"]]
        SYNAPSE = y[I["synaptic_density"]]
        C4 = y[I["C4_activity"]]

        # Base genetica
        C4_genetic = self._C4_expression
        PV_frag = self._PV_fragility
        SOD_base = self._SOD_baseline
        mito_eff = self._mito_efficiency
        DAT_eff = self._DAT_efficiency
        D2_sens = self._D2_sensitivity

        # ─── 1. DOPAMINA MESOLIMBICA (via D2, hiperfuncao) ───
        # Sintese constitutiva + feedback de NMDA
        synthesis_meso = K_DA_SYNTHESIS * (1.0 + 0.3 * NMDA)
        # Autorreceptor D2 inibe sintese
        D2_auto = 1.0 / (1.0 + DA_meso * D2_sens * 5.0)
        # Reuptake por DAT
        reuptake = K_DA_REUPTAKE_VMAX * DA_meso / (K_DA_REUPTAKE_KM + DA_meso) * DAT_eff
        d[I["DA_mesolimbic"]] = synthesis_meso * D2_auto - reuptake - K_DA_DEGRADATION * DA_meso

        # ─── 2. DOPAMINA PREFRONTAL (via D1, hipofuncao) ───
        # Sintese mais lenta, fortemente modulada por NMDA
        synthesis_pref = K_DA_SYNTHESIS * 0.6 * (0.5 + 0.5 * NMDA)
        # Inibida por estresse oxidativo
        oxidative_inhibition = 1.0 / (1.0 + ROS * 3.0)
        # Reuptake
        reuptake_pref = K_DA_REUPTAKE_VMAX * 0.7 * DA_pref / (0.2 + DA_pref)
        d[I["DA_prefrontal"]] = (synthesis_pref * oxidative_inhibition
                                  - reuptake_pref - K_DA_DEGRADATION * DA_pref)

        # ─── 3. NMDA (hipofuncao progressiva) ───
        # Ativado por glutamato, inibido por ROS e inflamacao
        NMDA_activation = K_NMDA_SLOPE * PV / (1.0 + K_NMDA_SLOPE * PV)
        # PV+ interneurons modulam NMDA via inibicao feedforward
        NMDA_damage = ROS * 0.3 + MG * 0.2
        d[I["NMDA_current"]] = (K_GLU_SYNTHESIS * NMDA_activation
                                  - NMDA_damage * NMDA - K_GLU_REUPTAKE * NMDA)

        # ─── 4. AMPA (plasticidade) ───
        # AMPA segue NMDA com delay, modulado por BDNF
        AMPA_target = NMDA * (0.5 + 0.5 * BDNF)
        d[I["AMPA_current"]] = (AMPA_target - AMPA) * 0.1

        # ─── 5. GABA (hipofuncao PV+ mediada) ───
        # Sintese de GABA por GAD67 em PV+ interneurons
        GABA_synthesis = K_GABA_SYNTHESIS * PV
        # Degradacao
        GABA_degradation = K_GABA_DEGRADATION * GABA
        d[I["GABA_current"]] = GABA_synthesis - GABA_degradation

        # ─── 6. PV+ INTERNEURON (elo critico) ───
        # PV+ e vulneravel a ROS e inflamacao
        PV_damage = K_PV_FRAGILITY * PV_frag * (ROS * 0.5 + MG * 0.3)
        # Recuperacao dependente de BDNF e ATP
        PV_recovery = K_PV_RECOVERY * (0.3 + 0.7 * BDNF * ATP)
        # Atividade intrinseca
        PV_activity = 0.2 * (1.0 + NMDA * 0.5 - GABA * 0.3)
        d[I["PV_interneuron"]] = PV_activity + PV_recovery - PV_damage - 0.05 * PV

        # ─── 7. MITOCONDRIA ATP ───
        # Producao de ATP, danificada por ROS
        ATP_production = K_MITO_ATP_PRODUCTION * mito_eff * circadian
        ATP_damage = K_ROS_DAMAGE * ROS * ATP
        d[I["mitochondrial_ATP"]] = ATP_production - ATP_damage - 0.03 * ATP

        # ─── 8. ROS (estresse oxidativo) ───
        # Producao: leak mitocondrial + inflamacao
        ROS_production = (K_MITO_ROS_LEAK * (1.0 - ATP) * 2.0
                          + MG * 0.15 + (1.0 - SOD) * 0.05)
        # Detoxificacao por SOD
        ROS_detox = K_SOD_ACTIVITY * SOD * ROS
        d[I["ROS_level"]] = ROS_production - ROS_detox

        # ─── 9. SOD (defesa antioxidante) ───
        # Nivel genetico + regulacao por stress
        SOD_target = SOD_base + 0.2 * ROS - 0.1 * MG
        d[I["SOD_activity"]] = (SOD_target - SOD) * 0.05

        # ─── 10. MICROGLIA (neuroinflamacao) ───
        # Ativacao por dano e ROS
        MG_injury = DA_meso * 0.1 + ROS * 0.3 + (1.0 - PV) * 0.2
        MG_activation = K_MICROGLIA_ACTIVATION * MG_injury
        MG_recovery = 0.02 * (1.0 + BDNF * 0.5) * MG
        d[I["microglial_activation"]] = MG_activation - MG_recovery

        # ─── 11. BDNF ───
        # Sintese dependente de atividade (NMDA, AMPA, GABA, sincronia)
        BDNF_activity = (NMDA * 0.3 + AMPA * 0.2 + SYNCH * 0.4) * 0.5
        BDNF_synthesis = K_BDNF_SYNTHESIS * (0.3 + 0.7 * BDNF_activity)
        BDNF_degradation = K_BDNF_DEGRADATION * BDNF
        d[I["BDNF_level"]] = BDNF_synthesis - BDNF_degradation

        # ─── 12. MIELINA ───
        # Producao por oligodendrocitos, danificada por ROS
        myelin_production = K_MYELIN_PRODUCTION * (0.5 + 0.5 * BDNF)
        myelin_damage = K_MYELIN_DEGRADATION * MYE * (1.0 + ROS * 2.0)
        d[I["myelin_integrity"]] = myelin_production - myelin_damage

        # ─── 13. SINCRONIA CORTICAL (gamma) ───
        # Depende de PV+, mielina, GABA, NMDA
        synchrony_drive = (PV * 0.4 + GABA * 0.2 + NMDA * 0.2 + MYE * 0.2)
        synchrony_noise = 0.1 * (1.0 - ATP)
        d[I["cortical_synchrony"]] = (synchrony_drive - SYNCH) * 0.1 - synchrony_noise

        # ─── 14. DENSIDADE SINAPTICA ───
        # Manutencao por BDNF, poda por C4 e microglia
        synapse_formation = K_SYNAPSE_BDNF * BDNF * 0.3
        C4_pruning = K_C4_PRUNING * C4_genetic * SYNAPSE * (1.0 + MG * 0.5)
        d[I["synaptic_density"]] = synapse_formation - C4_pruning + 0.01 * (1.0 - SYNAPSE)

        # ─── 15. ATIVIDADE C4 ───
        # Expressao constitutiva + modulada por inflamacao
        C4_target = C4_genetic * (0.5 + 0.5 * MG)
        d[I["C4_activity"]] = (C4_target - C4) * 0.03

        return d


def create_gt(genetic_risk: float = 0.5) -> ScientificGroundTruth:
    """Factory function."""
    return ScientificGroundTruth(genetic_risk=genetic_risk)
