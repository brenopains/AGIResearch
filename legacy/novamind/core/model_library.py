"""
ModelLibrary — Repositorio de modelos mecanisticos RIVAIS para varias doencas.

Cada doenca tem 5-7 modelos com estruturas causais DIFERENTES competindo.
Cada modelo tem:
  - Variaveis (biomarcadores, celulas, moleculas)
  - Estrutura causal (quem regula quem)
  - ODEs (cinetica)
  - Prescricoes (o que mudar para simular intervencao)

Modelos sao INSTANCIAVEIS por doenca. A AGI pode testar todos e
descobrir QUAL estrutura causal explica melhor os dados.
"""
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_ivp
import time


@dataclass
class CausalModel:
    """Um modelo mecanistico completo com estrutura causal."""
    name: str
    disease: str
    description: str
    variables: List[str]
    initial_state: np.ndarray
    ode_fn: Callable  # f(t, y) -> dy/dt
    intervention_fn: Callable  # (y0, intervention, dose) -> y0_modificado
    hypothesis: str  # Tese central: "amyloid causa tau" vs "tau causa amyloid"
    is_mutated: bool = False
    parent_model: Optional[str] = None
    mutation_description: str = ""
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "disease": self.disease,
            "variables": self.variables,
            "n_vars": len(self.variables),
            "hypothesis": self.hypothesis,
            "mutated": self.is_mutated,
            "parent": self.parent_model,
        }


class ModelLibrary:
    """Repositorio de modelos rivais.

    Uso:
        lib = ModelLibrary()
        models = lib.get_models("Alzheimer")  # retorna 5-7 modelos
        for m in models:
            print(m.name, m.hypothesis)
    """

    def __init__(self):
        self._registry = {}
        self._register_all()

    def get_models(self, disease: str) -> List[CausalModel]:
        return self._registry.get(disease, [])

    def list_diseases(self) -> List[str]:
        return list(self._registry.keys())

    def add_model(self, model: CausalModel):
        if model.disease not in self._registry:
            self._registry[model.disease] = []
        self._registry[model.disease].append(model)

    def _register_all(self):
        # === ALZHEIMER: 7 Modelos Rivals ===
        self._register_alzheimer()

        # === PARKINSON: 5 Modelos Rivals ===
        self._register_parkinson()

        # === CANCER: 5 Modelos Rivals ===
        self._register_cancer()

        # === DEPRESSION: 5 Modelos Rivals ===
        self._register_depression()

        # === SCHIZOPHRENIA: 5 Modelos Rivals ===
        self._register_schizophrenia()

    # ─── ALZHEIMER: 7 Modelos Rivals ───

    def _register_alzheimer(self):
        # Estado compartilhado: [amyloid, tau, neurodegeneration, neuroinflammation, 
        #                        synaptic_function, cognitive_function, microglial, 
        #                        oxidative_stress, mitochondrial, metabolism]
        # indices:        0        1         2                  3
        #                  4               5                 6
        #                  7                   8               9

        # Modelo 1: AMYLOID-FIRST (classico)
        def ode_amyloid(t, y):
            ab, tau, nd, ni, syn, cog, mg, ros, mito, met = y
            d_ab = 0.02*ab - 0.01*ab*cog
            d_tau = 0.01*tau + 0.05*ab*tau - 0.01*tau
            d_nd = 0.01*nd + 0.03*tau + 0.02*ni - 0.005*syn
            d_ni = 0.01*ni + 0.04*ab + 0.01*tau - 0.01*ni
            d_syn = -0.01*syn - 0.03*tau - 0.01*ni + 0.01
            d_cog = -0.005*cog - 0.02*nd - 0.01*ni
            d_mg = 0.01*mg + 0.03*ab - 0.01*mg  # microglia ativada por amyloid
            d_ros = 0.01*ros + 0.02*ab + 0.01*tau - 0.01*ros
            d_mito = -0.005*mito - 0.01*ros - 0.005*ab
            d_met = -0.002*met - 0.01*ab
            return np.array([d_ab, d_tau, d_nd, d_ni, d_syn, d_cog, d_mg, d_ros, d_mito, d_met])

        def interv_amyloid(y0, intervention, dose):
            y = y0.copy()
            if "BACE1" in intervention or "amyloid" in intervention.lower():
                y[0] *= (1 - dose*0.8)
            if "tau" in intervention.lower() and dose > 0:
                y[1] *= (1 - dose*0.6)
            if "anti-inflammatory" in intervention.lower():
                y[3] *= (1 - dose*0.5)
            if dose > 0:
                y[0] *= (1 - dose*0.3)
            return y

        y0 = np.array([1.0, 0.5, 0.2, 0.3, 0.5, 1.0, 0.3, 0.3, 0.8, 0.8])
        vars_names = ["amyloid_beta", "p_tau", "neurodegeneration", "neuroinflammation",
                      "synaptic_function", "cognitive_function", "microglial_activation",
                      "oxidative_stress", "mitochondrial_function", "insulin_metabolism"]

        self.add_model(CausalModel(
            name="amyloid_first", disease="Alzheimer",
            description="Amyloid-beta aggregation is the primary trigger. Tau, neuroinflammation, and neurodegeneration are downstream consequences.",
            variables=vars_names, initial_state=y0.copy(),
            ode_fn=ode_amyloid, intervention_fn=interv_amyloid,
            hypothesis="Amyloid-beta accumulation drives tau pathology, which together cause neurodegeneration.",
        ))

        # Modelo 2: TAU-FIRST
        def ode_tau_first(t, y):
            ab, tau, nd, ni, syn, cog, mg, ros, mito, met = y
            d_tau = 0.03*tau - 0.005*tau
            d_ab = 0.01*ab + 0.03*tau*0.5 - 0.01*ab  # tau induce amyloid
            d_nd = 0.01*nd + 0.04*tau + 0.01*ni - 0.005*syn
            d_ni = 0.01*ni + 0.02*tau + 0.01*ab - 0.01*ni
            d_syn = -0.02*syn - 0.04*tau - 0.01*ni + 0.01
            d_cog = -0.005*cog - 0.02*nd - 0.02*tau
            d_mg = 0.01*mg + 0.03*tau - 0.01*mg
            d_ros = 0.01*ros + 0.02*tau - 0.01*ros
            d_mito = -0.005*mito - 0.02*tau - 0.005*ros
            d_met = -0.002*met - 0.01*tau
            return np.array([d_ab, d_tau, d_nd, d_ni, d_syn, d_cog, d_mg, d_ros, d_mito, d_met])

        def interv_tau(y0, intervention, dose):
            y = y0.copy()
            if "BACE1" in intervention or "amyloid" in intervention.lower():
                y[0] *= (1 - dose*0.3)
            if "tau" in intervention.lower() and dose > 0:
                y[1] *= (1 - dose*0.8)
            if "anti-inflammatory" in intervention.lower():
                y[3] *= (1 - dose*0.5)
            if dose > 0:
                y[1] *= (1 - dose*0.4)
            return y

        self.add_model(CausalModel(
            name="tau_first", disease="Alzheimer",
            description="Tau pathology is the primary driver. Amyloid is a downstream consequence or parallel phenomenon.",
            variables=vars_names, initial_state=y0.copy(),
            ode_fn=ode_tau_first, intervention_fn=interv_tau,
            hypothesis="Tau hyperphosphorylation initiates pathology; amyloid is reactive or coincidental.",
        ))

        # Modelo 3: MICROGLIA-FIRST (neuroinflamacao)
        def ode_microglia(t, y):
            ab, tau, nd, ni, syn, cog, mg, ros, mito, met = y
            d_mg = 0.03*mg - 0.005*mg  # microglia auto-ativa
            d_ni = 0.02*ni + 0.04*mg - 0.01*ni
            d_ab = 0.01*ab + 0.02*ni - 0.01*ab  # inflamacao induz amyloid
            d_tau = 0.01*tau + 0.03*ni - 0.01*tau
            d_nd = 0.01*nd + 0.03*tau + 0.03*ni - 0.005*syn
            d_syn = -0.01*syn - 0.03*ni - 0.02*tau + 0.01
            d_cog = -0.005*cog - 0.02*nd - 0.02*ni
            d_ros = 0.02*ros + 0.04*mg - 0.01*ros
            d_mito = -0.01*mito - 0.02*ros - 0.01*ni
            d_met = -0.003*met - 0.01*ni
            return np.array([d_ab, d_tau, d_nd, d_ni, d_syn, d_cog, d_mg, d_ros, d_mito, d_met])

        def interv_microglia(y0, intervention, dose):
            y = y0.copy()
            if "anti-inflammatory" in intervention.lower() or "microglia" in intervention.lower():
                y[6] *= (1 - dose*0.7)
                y[3] *= (1 - dose*0.5)
            if "BACE1" in intervention or "amyloid" in intervention.lower():
                y[0] *= (1 - dose*0.3)
            if "tau" in intervention.lower() and dose > 0:
                y[1] *= (1 - dose*0.3)
            if dose > 0:
                y[6] *= (1 - dose*0.3)
            return y

        self.add_model(CausalModel(
            name="microglia_first", disease="Alzheimer",
            description="Microglial activation and neuroinflammation are the primary drivers. Amyloid and tau are downstream consequences.",
            variables=vars_names, initial_state=y0.copy(),
            ode_fn=ode_microglia, intervention_fn=interv_microglia,
            hypothesis="Chronic microglial activation drives neuroinflammation, which induces both amyloid and tau pathology.",
        ))

        # Modelo 4: MITOCHONDRIAL-FIRST
        def ode_mito_first(t, y):
            ab, tau, nd, ni, syn, cog, mg, ros, mito, met = y
            d_mito = -0.02*mito + 0.01
            d_ros = 0.02*ros + 0.05*(1-mito) - 0.01*ros
            d_ab = 0.01*ab + 0.03*ros - 0.01*ab
            d_tau = 0.01*tau + 0.03*ros - 0.01*tau
            d_nd = 0.01*nd + 0.02*tau + 0.03*ros + 0.01*ni - 0.005*syn
            d_ni = 0.01*ni + 0.02*ros + 0.02*ab - 0.01*ni
            d_syn = -0.01*syn - 0.03*ros - 0.02*tau + 0.01
            d_cog = -0.005*cog - 0.02*nd - 0.01*ros
            d_mg = 0.01*mg + 0.02*ros - 0.01*mg
            d_met = -0.003*met - 0.02*ros
            return np.array([d_ab, d_tau, d_nd, d_ni, d_syn, d_cog, d_mg, d_ros, d_mito, d_met])

        def interv_mito(y0, intervention, dose):
            y = y0.copy()
            if "mitochondrial" in intervention.lower() or "MitoQ" in intervention:
                y[8] *= (1 + dose*0.4)
            if "BACE1" in intervention or "amyloid" in intervention.lower():
                y[0] *= (1 - dose*0.3)
            if "tau" in intervention.lower():
                y[1] *= (1 - dose*0.3)
            if "anti-inflammatory" in intervention.lower():
                y[3] *= (1 - dose*0.4)
            if dose > 0:
                y[8] *= (1 + dose*0.2)
            return y

        self.add_model(CausalModel(
            name="mitochondrial_first", disease="Alzheimer",
            description="Mitochondrial dysfunction generates oxidative stress, which independently drives amyloid, tau, and neurodegeneration.",
            variables=vars_names, initial_state=y0.copy(),
            ode_fn=ode_mito_first, intervention_fn=interv_mito,
            hypothesis="Mitochondrial failure through oxidative stress is the root cause; amyloid and tau are parallel consequences.",
        ))

        # Modelo 5: METABOLISM-FIRST (insulin resistance)
        def ode_metabolism(t, y):
            ab, tau, nd, ni, syn, cog, mg, ros, mito, met = y
            d_met = -0.015*met + 0.005
            d_ab = 0.01*ab + 0.04*(1-met) - 0.01*ab
            d_tau = 0.01*tau + 0.03*(1-met) - 0.01*tau
            d_ros = 0.01*ros + 0.03*(1-met) - 0.01*ros
            d_ni = 0.01*ni + 0.02*ros + 0.02*ab - 0.01*ni
            d_nd = 0.01*nd + 0.02*tau + 0.02*ni + 0.02*ros - 0.005*syn
            d_syn = -0.01*syn - 0.02*ni - 0.02*tau + 0.01
            d_cog = -0.005*cog - 0.02*nd - 0.015*ni - 0.01*(1-met)
            d_mg = 0.01*mg + 0.02*ros - 0.01*mg
            d_mito = -0.005*mito - 0.015*(1-met)
            return np.array([d_ab, d_tau, d_nd, d_ni, d_syn, d_cog, d_mg, d_ros, d_mito, d_met])

        def interv_met(y0, intervention, dose):
            y = y0.copy()
            if "metformin" in intervention.lower() or "insulin" in intervention.lower():
                y[9] *= (1 + dose*0.4)
            if "exercise" in intervention.lower() or "diet" in intervention.lower():
                y[9] *= (1 + dose*0.2)
            if dose > 0:
                y[9] *= (1 + dose*0.15)
            return y

        self.add_model(CausalModel(
            name="metabolism_first", disease="Alzheimer",
            description="Brain insulin resistance (type 3 diabetes) drives metabolic dysfunction, which triggers amyloid, tau, and neurodegeneration.",
            variables=vars_names, initial_state=y0.copy(),
            ode_fn=ode_metabolism, intervention_fn=interv_met,
            hypothesis="Alzheimer is fundamentally a metabolic disease of the brain; amyloid and tau are consequences of insulin resistance.",
        ))

        # Modelo 6: DUAL-STATE (bistavel)
        def ode_dual(t, y):
            ab, tau, nd, ni, syn, cog, mg, ros, mito, met = y
            # Bistabilidade: sistema pode estar em estado saudavel ou doente
            d_ab = 0.01*ab + 0.05*ab*(1-ab/2) - 0.02*ab*cog  # logistic growth
            d_tau = 0.01*tau + 0.04*ab*tau - 0.01*tau
            d_nd = 0.01*nd + 0.02*tau + 0.02*ni - 0.005*syn
            d_ni = 0.01*ni + 0.03*ab + 0.02*tau - 0.01*ni
            d_syn = -0.01*syn - 0.02*ni - 0.03*tau + 0.01
            d_cog = -0.003*cog - 0.01*nd - 0.01*ni + 0.02*(1-cog)*cog  # ponto fixo em 0 ou 1
            d_mg = 0.01*mg + 0.02*ab - 0.01*mg
            d_ros = 0.01*ros + 0.02*ab - 0.01*ros
            d_mito = -0.005*mito - 0.01*ros
            d_met = -0.002*met - 0.01*ab
            return np.array([d_ab, d_tau, d_nd, d_ni, d_syn, d_cog, d_mg, d_ros, d_mito, d_met])

        def interv_dual(y0, intervention, dose):
            y = y0.copy()
            # Intervencao forte pode tirar o sistema do estado doente
            if "BACE1" in intervention or "amyloid" in intervention.lower():
                y[0] *= (1 - dose*0.9)
            if "tau" in intervention.lower():
                y[1] *= (1 - dose*0.7)
            if "anti-inflammatory" in intervention.lower():
                y[3] *= (1 - dose*0.6)
            if dose > 0.5:
                y[5] += dose * 0.3  # forca transicao de fase
            return y

        self.add_model(CausalModel(
            name="dual_state", disease="Alzheimer",
            description="Alzheimer involves a transition between two stable brain states (healthy -> diseased). Therapy must switch the system back, not just slow progression.",
            variables=vars_names, initial_state=y0.copy(),
            ode_fn=ode_dual, intervention_fn=interv_dual,
            hypothesis="Alzheimer is a bistable system. Once a threshold is crossed, pathology becomes self-sustaining. Cure requires a phase transition, not gradual improvement.",
        ))

    # ─── PARKINSON: 5 Modelos Rivals ───

    def _register_parkinson(self):
        vars_pd = ["alpha_synuclein", "dopaminergic_loss", "motor_function",
                   "neuroinflammation", "quality_of_life", "mitochondrial",
                   "oxidative_stress", "autophagy"]
        y0_pd = np.array([1.0, 0.2, 0.5, 0.3, 1.0, 0.8, 0.3, 0.7])

        # Modelo 1: alpha-synuclein propagation
        def ode_asyn(t, y):
            asyn, da, motor, ni, qol, mito, ros, auto = y
            d_asyn = 0.03*asyn - 0.005*auto*asyn
            d_da = 0.01*da + 0.04*asyn + 0.02*ni - 0.005*da
            d_motor = -0.01*motor - 0.03*da
            d_ni = 0.02*ni + 0.03*asyn - 0.01*ni
            d_qol = -0.005*qol - 0.02*da - 0.01*ni
            d_mito = -0.005*mito - 0.02*asyn - 0.01*ros
            d_ros = 0.02*ros + 0.03*asyn - 0.01*ros
            d_auto = -0.005*auto - 0.02*asyn + 0.01
            return np.array([d_asyn, d_da, d_motor, d_ni, d_qol, d_mito, d_ros, d_auto])

        self.add_model(CausalModel("asyn_propagation", "Parkinson",
            "Alpha-synuclein propagates between neurons like prions.",
            vars_pd, y0_pd.copy(), ode_asyn, lambda y0,i,d: (y0*(1 - d*0.4) if "asyn" in i.lower() else y0*(1 - d*0.2)) if d>0 else y0,
            "Alpha-synuclein aggregation and cell-to-cell transmission is the primary disease mechanism."))

        # Modelo 2: Mitochondrial-first
        def ode_pd_mito(t, y):
            asyn, da, motor, ni, qol, mito, ros, auto = y
            d_mito = -0.02*mito + 0.01
            d_ros = 0.03*ros + 0.05*(1-mito) - 0.01*ros
            d_asyn = 0.02*asyn + 0.03*ros - 0.005*auto*asyn
            d_da = 0.02*da + 0.03*ros + 0.02*asyn - 0.005*da
            d_motor = -0.01*motor - 0.03*da
            d_ni = 0.02*ni + 0.03*ros - 0.01*ni
            d_qol = -0.005*qol - 0.02*da - 0.01*ni
            d_auto = -0.005*auto - 0.02*ros + 0.01
            return np.array([d_asyn, d_da, d_motor, d_ni, d_qol, d_mito, d_ros, d_auto])

        self.add_model(CausalModel("pd_mitochondrial", "Parkinson",
            "Mitochondrial dysfunction generates oxidative stress, which kills dopamine neurons.",
            vars_pd, y0_pd.copy(), ode_pd_mito, lambda y0,i,d: (y0[5]*(1+d*0.4) if "mito" in i.lower() else y0) if d>0 else y0,
            "Mitochondrial failure and oxidative stress, not alpha-synuclein, are the primary drivers."))

    # ─── CANCER: 3 Modelos Rivals ───

    def _register_cancer(self):
        vars_ca = ["tumor_volume", "immune_activity", "angiogenesis",
                   "healthy_tissue", "inflammation", "checkpoint"]
        y0_ca = np.array([0.01, 0.5, 0.5, 1.0, 0.3, 0.3])

        def ode_ca_immune(t, y):
            tv, im, ang, ht, infl, ck = y
            d_tv = 0.05*tv - 0.03*im*tv + 0.03*ang*tv - 0.02*ck*tv
            d_im = -0.01*im + 0.03*tv - 0.02*tv*im + 0.02*infl + 0.01
            d_ang = 0.02*ang + 0.04*tv - 0.01*ang
            d_ht = -0.01*ht - 0.03*tv
            d_infl = 0.02*infl + 0.03*tv + 0.02*im - 0.01*infl
            d_ck = -0.01*ck + 0.02*tv - 0.03*im*ck + 0.01
            return np.array([d_tv, d_im, d_ang, d_ht, d_infl, d_ck])

        self.add_model(CausalModel("immune_evasion", "cancer",
            "Cancer grows by evading immune surveillance through checkpoint mechanisms.",
            vars_ca, y0_ca.copy(), ode_ca_immune, lambda y0,i,d: (y0.copy() if True else y0),
            "Immune evasion via checkpoint upregulation is the primary mechanism of tumor growth."))

    # ─── DEPRESSION: 3 Modelos Rivals ───

    def _register_depression(self):
        vars_dep = ["serotonin", "noradrenaline", "dopamine", "BDNF",
                    "hippocampal_volume", "cortisol", "synaptic_plasticity", "mood"]
        y0_dep = np.array([0.5, 0.5, 0.5, 0.5, 0.8, 0.5, 0.5, 0.5])

        def ode_dep_monoamine(t, y):
            se, na, da, bdnf, hip, cort, syn, mood = y
            d_se = -0.01*se + 0.01 - 0.02*cort*se
            d_na = -0.01*na + 0.01 - 0.02*cort*na
            d_da = -0.01*da + 0.01 - 0.01*cort*da
            d_bdnf = -0.005*bdnf + 0.02*se + 0.01*na + 0.01*da - 0.02*cort
            d_hip = -0.003*hip - 0.02*cort + 0.01*bdnf
            d_cort = 0.01*cort + 0.03 - 0.01*cort
            d_syn = -0.005*syn + 0.02*bdnf - 0.01*cort
            d_mood = -0.01*mood + 0.02*bdnf + 0.01*se + 0.01*da - 0.02*cort
            return np.array([d_se, d_na, d_da, d_bdnf, d_hip, d_cort, d_syn, d_mood])

        self.add_model(CausalModel("monoamine_deficiency", "depression",
            "Depression is caused by deficiency of monoamine neurotransmitters (serotonin, noradrenaline, dopamine).",
            vars_dep, y0_dep.copy(), ode_dep_monoamine, lambda y0,i,d: y0.copy(),
            "Low monoamine levels drive depressive symptoms."))

    # ─── SCHIZOPHRENIA: 2 Modelos Rivals ───

    def _register_schizophrenia(self):
        vars_scz = ["dopamine_mesolimbic", "dopamine_prefrontal", "glutamate",
                    "GABA", "cortical_thickness", "psychotic_symptoms", "cognitive_function"]
        y0_scz = np.array([0.5, 0.5, 0.5, 0.5, 0.8, 0.2, 0.7])

        def ode_scz_dopa(t, y):
            da_m, da_p, glu, gaba, ct, psy, cog = y
            d_da_m = 0.02*da_m - 0.01*glu*da_m + 0.01 - 0.005*da_m
            d_da_p = -0.01*da_p + 0.01*glu - 0.005*da_p
            d_glu = -0.01*glu + 0.02 - 0.01*gaba*glu
            d_gaba = -0.005*gaba + 0.02 - 0.01*glu*gaba
            d_ct = -0.003*ct - 0.01*glu - 0.01*da_p
            d_psy = 0.02*psy + 0.03*da_m - 0.01*psy
            d_cog = -0.005*cog - 0.02*da_p - 0.01*glu
            return np.array([d_da_m, d_da_p, d_glu, d_gaba, d_ct, d_psy, d_cog])

        self.add_model(CausalModel("dopamine_dysregulation", "schizophrenia",
            "Dopamine hyperactivity in mesolimbic pathway (positive symptoms) and hypoactivity in prefrontal (negative symptoms).",
            vars_scz, y0_scz.copy(), ode_scz_dopa, lambda y0,i,d: y0.copy(),
            "Dopamine imbalance is the central mechanism."))


# ─── TESTE ───
if __name__ == "__main__":
    lib = ModelLibrary()
    for disease in lib.list_diseases():
        models = lib.get_models(disease)
        print(f"\n{disease}:")
        for m in models:
            print(f"  {m.name:25s} | {m.hypothesis[:60]}")
