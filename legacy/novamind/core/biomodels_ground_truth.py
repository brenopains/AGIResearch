"""
biomodels_ground_truth.py — SIMULADOR CIENTIFICO REAL (caixa-preta para a AGI)

Usa modelos SBML curados do BioModels Database via Tellurium.
A AGI NUNCA ve as equacoes. Ela so pode:
    gt = BiomodelsGroundTruth()
    gt.list_models() -> descricao dos modelos (sem equacoes)
    gt.simulate(model_id, intervention) -> trajetoria

Cada modelo e um mecanismo biologico real publicado.
Juntos, formam um "body simulator" minimo mas cientifico.
"""
import os, sys, tempfile, json, zipfile, io, time
import numpy as np
import requests
import tellurium as te
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


# Cache local para modelos baixados
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "biomodels_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


@dataclass
class ModelInfo:
    """Informacoes publicas sobre um modelo (sem equacoes)."""
    id: str
    name: str
    n_species: int
    n_reactions: int
    n_parameters: int
    species_names: List[str]
    parameter_names: List[str]
    description: str = ""


class BiomodelsGroundTruth:
    """Ground truth simulator baseado em modelos SBML reais do BioModels.

    A AGI interage exclusivamente por esta interface.
    Internamente, cada modelo e um sistema de EDOs validado
    e publicado na literatura cientifica.
    """

    # Modelos verificados que funcionam com Tellurium
    # Organizados por relevancia para esquizofrenia
    AVAILABLE_MODELS = {
        "dopamine_signaling": {
            "id": "BIOMD0000000257",
            "name": "Dopamine signaling cascade",
            "description": "Dopamine receptor signaling via cAMP/PKA pathway",
        },
        "gpcR_signaling": {
            "id": "BIOMD0000000165",
            "name": "GPCR signaling (dopamine receptors)",
            "description": "G-protein coupled receptor cascade, dopamine D1/D2 pathways",
        },
        "synaptic_transmission": {
            "id": "BIOMD0000000355",
            "name": "Synaptic transmission (calcium/IP3)",
            "description": "Presynaptic calcium dynamics and neurotransmitter release",
        },
        "mitochondrial_metabolism": {
            "id": "BIOMD0000000430",
            "name": "Mitochondrial energy metabolism",
            "description": "Mitochondrial ATP production, ROS generation, metabolic flux",
        },
        "neurotransmitter_release": {
            "id": "BIOMD0000000594",
            "name": "Neurotransmitter release machinery",
            "description": "SNARE complex, vesicle fusion, neurotransmitter exocytosis",
        },
        "calcium_dynamics": {
            "id": "BIOMD0000000073",
            "name": "Intracellular calcium dynamics",
            "description": "IP3-mediated calcium release, calcium oscillations",
        },
        "mapk_signaling": {
            "id": "BIOMD0000000010",
            "name": "MAPK/ERK signaling cascade",
            "description": "Growth factor signaling, synaptic plasticity, cell survival",
        },
        "camp_signaling": {
            "id": "BIOMD0000000048",
            "name": "cAMP/PKA signaling",
            "description": "Cyclic AMP pathway, CREB activation, gene expression",
        },
        "apoptosis": {
            "id": "BIOMD0000000320",
            "name": "Apoptosis signaling (dopamine-induced)",
            "description": "Oxidative stress-induced apoptosis, mitochondrial permeability",
        },
        "hodgkin_huxley": {
            "id": "BIOMD0000000100",
            "name": "Hodgkin-Huxley neuron model",
            "description": "Action potential generation and propagation",
        },
        "cell_cycle": {
            "id": "BIOMD0000000143",
            "name": "Circadian/cell cycle regulation",
            "description": "Circadian rhythm, oxidative stress response, PER genes",
        },
        "simple_regulatory": {
            "id": "BIOMD0000000299",
            "name": "Simple gene regulatory network",
            "description": "Transcriptional regulation, feedback control",
        },
    }

    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._loaded_models: Dict[str, te.RoadRunner] = {}
        self._model_infos: Dict[str, ModelInfo] = {}

    def list_models(self) -> Dict[str, str]:
        """Retorna descricao publica dos modelos disponiveis (sem equacoes)."""
        return {k: v["description"] for k, v in self.AVAILABLE_MODELS.items()}

    def get_model_info(self, model_key: str) -> Optional[ModelInfo]:
        """Retorna informacoes publicas sobre um modelo."""
        if model_key not in self.AVAILABLE_MODELS:
            return None
        if model_key in self._model_infos:
            return self._model_infos[model_key]

        rr = self._load_model(model_key)
        if rr is None:
            return None

        info = ModelInfo(
            id=self.AVAILABLE_MODELS[model_key]["id"],
            name=self.AVAILABLE_MODELS[model_key]["name"],
            n_species=len(rr.getFloatingSpeciesIds()),
            n_reactions=len(rr.getReactionIds()),
            n_parameters=len(rr.getGlobalParameterIds()),
            species_names=list(rr.getFloatingSpeciesIds()),
            parameter_names=list(rr.getGlobalParameterIds()),
            description=self.AVAILABLE_MODELS[model_key]["description"],
        )
        self._model_infos[model_key] = info
        return info

    def simulate(self, model_key: str,
                 intervention: Dict[str, float] = None,
                 t_end: float = 100.0,
                 n_points: int = 100) -> Optional[np.ndarray]:
        """Simula o modelo e retorna trajetoria.

        Args:
            model_key: chave do modelo (ex: "dopamine_signaling")
            intervention: dict {nome_da_variavel_ou_parametro: novo_valor}
            t_end: tempo final de simulacao
            n_points: numero de pontos na trajetoria

        Returns:
            array (n_species+1, n_points) onde a primeira linha e o tempo
            None se o modelo nao existir ou falhar
        """
        rr = self._load_model(model_key)
        if rr is None:
            return None

        rr.reset()
        try:
            # Aplica intervencao
            if intervention:
                for target, value in intervention.items():
                    # Tenta como especie flutuante
                    if target in rr.getFloatingSpeciesIds():
                        rr[target] = value
                    # Tenta como parametro global
                    elif target in rr.getGlobalParameterIds():
                        rr[target] = value
                    # Tenta como parametro local (reacao)
                    elif target in rr.getReactionIds():
                        pass  # nao modificavel direto

            # Simula
            s = rr.simulate(0, t_end, n_points)
            if s is None or len(s) == 0:
                return None

            # Retorna (n_species+1, n_points) - tempo + especies
            trajectory = np.array(s).T
            return trajectory

        except Exception:
            return None

    def simulate_intervention_sweep(self, model_key: str,
                                    target: str,
                                    values: List[float],
                                    t_end: float = 100.0) -> Dict[float, np.ndarray]:
        """Varre um parametro e retorna trajetorias para cada valor."""
        results = {}
        for v in values:
            traj = self.simulate(model_key, {target: v}, t_end)
            if traj is not None:
                results[v] = traj
        return results

    def _load_model(self, model_key: str):  # -> roadrunner.RoadRunner
        """Carrega modelo do cache ou baixa do BioModels."""
        if model_key in self._loaded_models:
            return self._loaded_models[model_key]

        if model_key not in self.AVAILABLE_MODELS:
            return None

        model_id = self.AVAILABLE_MODELS[model_key]["id"]
        cache_path = os.path.join(self.cache_dir, f"{model_id}.xml")

        # Tenta cache primeiro
        if os.path.exists(cache_path):
            try:
                rr = te.loadSBMLModel(cache_path)
                self._loaded_models[model_key] = rr
                return rr
            except:
                pass

        # Baixa do BioModels
        try:
            url = f"https://www.ebi.ac.uk/biomodels/model/download/{model_id}"
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                return None

            z = zipfile.ZipFile(io.BytesIO(resp.content))
            for name in z.namelist():
                if name.endswith("_url.xml"):
                    sbml_content = z.read(name)
                    with open(cache_path, "wb") as f:
                        f.write(sbml_content)
                    rr = te.loadSBMLModel(cache_path)
                    self._loaded_models[model_key] = rr
                    return rr
        except Exception:
            return None

        return None

    def get_healthy_state(self, model_key: str) -> Optional[Dict[str, float]]:
        """Retorna o estado basal do modelo (sem perturbacao)."""
        rr = self._load_model(model_key)
        if rr is None:
            return None
        try:
            rr.reset()
            # Simula curto para atingir estado estacionario
            rr.simulate(0, 10, 10)
            species = rr.getFloatingSpeciesIds()
            values = [rr[s] for s in species]
            return dict(zip(species, values))
        except:
            return None

    def compute_steady_state(self, model_key: str,
                             intervention: Dict[str, float] = None,
                             max_time: float = 500.0) -> Optional[Dict[str, float]]:
        """Encontra estado estacionario apos intervencao."""
        rr = self._load_model(model_key)
        if rr is None:
            return None

        rr.reset()
        try:
            if intervention:
                for target, value in intervention.items():
                    if target in rr.getFloatingSpeciesIds():
                        rr[target] = value

            s = rr.simulate(0, max_time, 100)
            if s is None or len(s) == 0:
                return None

            species = rr.getFloatingSpeciesIds()
            final_values = [s[-1][i+1] for i in range(len(species))]
            return dict(zip(species, final_values))
        except:
            return None
