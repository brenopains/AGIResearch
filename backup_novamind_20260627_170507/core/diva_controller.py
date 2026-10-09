"""
DIVAController — Controle Motor da Fala Baseado em Feedback Sensorial
======================================================================
Implementa o modelo DIVA (Directions Into Velocities of Articulators)
para controle da fala com:

  1. Feedforward controller — programas motores internalizados
  2. Auditory feedback controller — corrige com base no que OUVIU
  3. Somatosensory feedback controller — corrige com base no que SENTIU
  4. Inverse model — mapeia som ouvido → comando motor
  5. Forward model — mapeia comando motor → consequência sensorial esperada

A AGI APRENDE igual um bebê:
  - Tenta um som → ouve o resultado → sente os articuladores
  - Compara com o que queria → calcula erro → corrige o comando
  - Com prática, o feedforward fica preciso e o feedback vira backup

Referência: Guenther (2016) "Neural Control of Speech" (DIVA model)
"""

import numpy as np
import time
from typing import Dict, Optional, Tuple, List
from collections import defaultdict, deque


class DIVAController:
    """
    Controlador DIVA para aprendizagem motora da fala.

    O ciclo de produção de um fonema:
      1. Feedforward: recupera comando motor do programa internalizado
      2. Executa: articulatory_synthesize gera som + feedback sensorial
      3. Auditory feedback: compara som REAL com som ALVO
      4. Somatosensory feedback: compara posição REAL com posição ALVO
      5. Corrige: combina erros auditivo + somatossensorial para ajustar
      6. Aprende: atualiza forward e inverse models
      7. Repete: a próxima tentativa já começa melhor
    """

    def __init__(self, learning_rate: float = 0.15):
        self.lr = learning_rate

        # Dicionários de alvos sensoriais para cada fonema
        # Chave: nome do fonema. Valor: vetor de features.
        self.auditory_targets: Dict[str, np.ndarray] = {}
        self.somatosensory_targets: Dict[str, np.ndarray] = {}

        # Programas motores feedforward (comando articulatório aprendido)
        self.feedforward_commands: Dict[str, np.ndarray] = {}

        # Modelo inverso: [f0, f1, f2, f3] → comando articulatório
        self.inverse_memory: deque = deque(maxlen=500)

        # Modelo forward: comando → predição acústica
        self.forward_memory: deque = deque(maxlen=500)

        # Histórico de erros para cada fonema (aprendizado por reforço)
        self.error_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))

        # Alvos aprendidos da mídia (YouTube, áudios externos)
        # Cada alvo: {id, f0, f1, f2, f3, source, timestamp, mastery}
        self.media_targets: deque = deque(maxlen=50)

        # Fonemas conhecidos com alvos auditivos iniciais (aproximações)
        self._init_prototype_targets()

    def _init_prototype_targets(self):
        """Inicializa alvos auditivos prototípicos para fonemas básicos."""
        prototypes = {
            "aa": np.array([200, 700, 1100, 2500]),
            "ee": np.array([200, 500, 1800, 2500]),
            "ii": np.array([250, 300, 2200, 3000]),
            "oo": np.array([200, 500, 900, 2500]),
            "uu": np.array([200, 300, 900, 2300]),
            "mmm": np.array([180, 200, 1000, 2000]),
        }
        for phoneme, target in prototypes.items():
            self.auditory_targets[phoneme] = target

    def produce(self, phoneme: str,
                articulatory_config: Optional[Dict] = None) -> Dict:
        """
        Produz um fonema completo com correção por feedback.

        1. Pega o programa motor feedforward (ou aleatório se novo fonema)
        2. Enriquece com feedback auditory + somatossensorial
        3. Retorna comando corrigido + resultado sensorial

        Args:
            phoneme: nome do fonema a produzir
            articulatory_config: configuração base opcional

        Returns:
            dict com {command, sensory_feedback, errors, corrections}
        """
        # Step 1: Feedforward command (programa motor)
        if articulatory_config is not None:
            cmd = articulatory_config.copy()
        elif phoneme in self.feedforward_commands:
            cmd = self._vector_to_config(self.feedforward_commands[phoneme])
        else:
            # Novo fonema: comando aleatório com leve tendência ao alvo
            cmd = self._random_command()
            target = self.auditory_targets.get(phoneme)
            if target is not None:
                cmd = self._approximate_from_target(target, cmd)

        # Step 2: Dados sensoriais esperados (forward model)
        expected_auditory = self._predict_auditory(cmd)

        # Step 3: Simula o resultado (na prática, chamaria articulatory_synthesize)
        sensory_result = self._simulate_sensory(cmd)

        # Step 4: Calcula erros de feedback
        auditory_error = 0.0
        somato_error = 0.0

        if phoneme in self.auditory_targets:
            actual = np.array([
                sensory_result["acoustic"]["f0"],
                sensory_result["acoustic"]["f1"],
                sensory_result["acoustic"]["f2"],
                sensory_result["acoustic"]["f3"],
            ])
            target_vec = self.auditory_targets[phoneme]
            auditory_error = float(np.mean(np.abs(actual - target_vec) / (target_vec + 1e-10)))

        # Somatossensorial: compara posição atual com a desejada para este fonema
        if phoneme in self.somatosensory_targets:
            art_vec = self._config_to_vector(cmd)
            somato_target = self.somatosensory_targets[phoneme]
            somato_error = float(np.mean(np.abs(art_vec - somato_target)))

        # Step 5: Corrige o comando com base nos erros
        correction = self._compute_correction(auditory_error, somato_error, cmd, phoneme)
        corrected_cmd = {}
        for k, v in cmd.items():
            corr = correction.get(k, 0.0)
            corrected_cmd[k] = np.clip(v + corr, 0.0, 1.0)

        # Step 6: Aprende
        self._learn(phoneme, cmd, sensory_result, auditory_error, somato_error)

        # Armazena erro no histórico
        mean_error = (auditory_error + somato_error) / 2
        self.error_history[phoneme].append(mean_error)

        return {
            "command": corrected_cmd,
            "original_command": cmd,
            "sensory_feedback": sensory_result,
            "errors": {
                "auditory": auditory_error,
                "somatosensory": somato_error,
                "mean": mean_error,
            },
            "corrections": correction,
            "phoneme": phoneme,
        }

    def _random_command(self) -> Dict:
        return {
            "tongue_dorsum_x": np.random.random(),
            "tongue_dorsum_y": np.random.random(),
            "lip_rounding": np.random.random() * 0.5,
            "jaw_open": np.random.random(),
            "velum": np.random.random() * 0.2,
            "vocal_fold_tension": 0.4 + np.random.random() * 0.4,
        }

    def _approximate_from_target(self, target: np.ndarray, base: Dict) -> Dict:
        """Aproxima uma configuração articulatória a partir do alvo auditivo."""
        f0, f1, f2, f3 = target
        cmd = base.copy()
        cmd["jaw_open"] = np.clip((f1 - 250) / 550, 0.0, 1.0)
        # Inverte F2 ≈ 800 + 1400 * frontness^1.5 * (1+ty*0.5) * (1-0.4*lip)
        # Assume lip=0, ty médio
        f2_adj = f2 * 1.0  # sem correção de lábio inicial
        frontness_norm = (f2_adj - 800) / 1400
        if frontness_norm > 0:
            frontness = max(0, min(1, frontness_norm ** (1/1.5)))
        else:
            frontness = 0.0
        cmd["tongue_dorsum_x"] = frontness
        cmd["tongue_dorsum_y"] = np.clip(1.0 - (f1 - 250) / 550, 0.0, 1.0)
        cmd["vocal_fold_tension"] = np.clip((f0 - 80) / 220, 0.3, 0.9)
        return cmd

    def _simulate_sensory(self, cmd: Dict) -> Dict:
        """Simula o feedback sensorial de uma configuração articulatória."""
        from .articulatory_synthesizer import articulatory_to_formants
        f0, f1, f2, f3 = articulatory_to_formants(
            cmd.get("tongue_dorsum_x", 0.5),
            cmd.get("tongue_dorsum_y", 0.5),
            cmd.get("lip_rounding", 0.0),
            cmd.get("jaw_open", 0.5),
            cmd.get("velum", 0.0),
            cmd.get("vocal_fold_tension", 0.6),
        )
        return {
            "acoustic": {
                "f0": f0, "f1": f1, "f2": f2, "f3": f3,
            },
            "articulatory": cmd,
        }

    def _predict_auditory(self, cmd: Dict) -> Optional[np.ndarray]:
        """Forward model: prevê a saída auditiva de um comando."""
        if len(self.forward_memory) < 5:
            return None
        nearby = []
        for mem_cmd, mem_acoustic in self.forward_memory:
            dist = sum(abs(cmd.get(k, 0) - mem_cmd.get(k, 0))
                       for k in ["tongue_dorsum_x", "tongue_dorsum_y",
                                  "jaw_open", "lip_rounding", "vocal_fold_tension"])
            if dist < 0.5:
                nearby.append((dist, mem_acoustic))
        if not nearby:
            return None
        nearby.sort(key=lambda x: x[0])
        return np.mean([n[1] for n in nearby[:5]], axis=0)

    def _compute_correction(self, aud_err: float, som_err: float,
                            cmd: Dict, phoneme: str) -> Dict:
        """Calcula correção no comando baseado nos erros sensoriais."""
        correction = {}
        for k in cmd:
            correction[k] = 0.0
        if aud_err > 0.1 and phoneme in self.auditory_targets:
            target = self.auditory_targets[phoneme]
            f0, f1, f2, f3 = target
            current = cmd
            # Heurística: se F1 muito alto, fecha mandíbula
            expected_f1 = 250 + 550 * current.get("jaw_open", 0.5)
            if expected_f1 > f1 + 50:
                correction["jaw_open"] = -self.lr * 0.3
                correction["tongue_dorsum_y"] = self.lr * 0.2
            elif expected_f1 < f1 - 50:
                correction["jaw_open"] = self.lr * 0.3
            expected_f2 = 800 + 1600 * current.get("tongue_dorsum_x", 0.5)
            if expected_f2 > f2 + 100:
                correction["tongue_dorsum_x"] = -self.lr * 0.3
                correction["lip_rounding"] = self.lr * 0.2
            elif expected_f2 < f2 - 100:
                correction["tongue_dorsum_x"] = self.lr * 0.3
        if som_err > 0.1 and phoneme in self.somatosensory_targets:
            target = self.somatosensory_targets[phoneme]
            art_vec = self._config_to_vector(cmd)
            diff = target[:6] - art_vec[:6]
            for i, k in enumerate(["tongue_dorsum_x", "tongue_dorsum_y",
                                    "lip_rounding", "jaw_open",
                                    "velum", "vocal_fold_tension"]):
                if i < len(diff):
                    correction[k] = correction.get(k, 0) + float(diff[i]) * self.lr * 0.3
        return correction

    def _config_to_vector(self, cmd: Dict) -> np.ndarray:
        return np.array([
            cmd.get("tongue_dorsum_x", 0.5),
            cmd.get("tongue_dorsum_y", 0.5),
            cmd.get("lip_rounding", 0.0),
            cmd.get("jaw_open", 0.5),
            cmd.get("velum", 0.0),
            cmd.get("vocal_fold_tension", 0.6),
        ])

    def _vector_to_config(self, vec: np.ndarray) -> Dict:
        names = ["tongue_dorsum_x", "tongue_dorsum_y", "lip_rounding",
                 "jaw_open", "velum", "vocal_fold_tension"]
        return {n: float(np.clip(vec[i] if i < len(vec) else 0.5, 0.0, 1.0))
                for i, n in enumerate(names)}

    def _learn(self, phoneme: str, cmd: Dict, result: Dict,
               aud_err: float, som_err: float):
        """Aprende com a tentativa: atualiza forward, inverse, feedforward."""
        acoustic_vec = np.array([
            result["acoustic"]["f0"],
            result["acoustic"]["f1"],
            result["acoustic"]["f2"],
            result["acoustic"]["f3"],
        ])
        art_vec = self._config_to_vector(cmd)

        # Forward model: comando → acústica
        self.forward_memory.append((cmd.copy(), acoustic_vec.copy()))

        # Inverse model: acústica → comando
        self.inverse_memory.append((acoustic_vec.copy(), art_vec.copy()))

        # Atualiza feedforward command se o erro foi pequeno (sucesso)
        if aud_err < 0.2 and phoneme not in self.feedforward_commands:
            self.feedforward_commands[phoneme] = art_vec.copy()
        elif aud_err < 0.15 and phoneme in self.feedforward_commands:
            existing = self.feedforward_commands[phoneme]
            self.feedforward_commands[phoneme] = existing + self.lr * (art_vec - existing)

        # Atualiza alvo somatossensorial
        if aud_err < 0.2:
            if phoneme not in self.somatosensory_targets:
                self.somatosensory_targets[phoneme] = art_vec.copy()
            else:
                existing = self.somatosensory_targets[phoneme]
                self.somatosensory_targets[phoneme] = existing + self.lr * 0.3 * (
                    art_vec[:6] - existing[:6])

    def inverse_map(self, f0: float, f1: float, f2: float, f3: float) -> Optional[Dict]:
        """
        Modelo inverso: dado um som ouvido, prediz o comando motor.
        Isso permite IMITAR sons ouvidos.
        """
        if len(self.inverse_memory) < 3:
            return None
        query = np.array([f0, f1, f2, f3])
        best = None
        best_dist = float("inf")
        for ac_vec, art_vec in self.inverse_memory:
            dist = float(np.mean(np.abs(query - ac_vec) / (ac_vec + 1e-10)))
            if dist < best_dist:
                best_dist = dist
                best = art_vec
        if best is not None and best_dist < 0.5:
            return self._vector_to_config(best)
        return None

    def get_mastery(self, phoneme: str) -> float:
        """Retorna o nível de maestria para um fonema (0-1)."""
        errors = list(self.error_history.get(phoneme, []))
        if not errors:
            return 0.0
        recent = errors[-5:]
        mean_err = np.mean(recent)
        mastery = 1.0 - mean_err
        has_feedforward = phoneme in self.feedforward_commands
        if has_feedforward:
            mastery = mastery * 0.7 + 0.3
        return float(np.clip(mastery, 0.0, 1.0))

    def get_state(self) -> Dict:
        return {
            "phonemes_known": len(self.feedforward_commands),
            "auditory_targets": len(self.auditory_targets),
            "somatosensory_targets": len(self.somatosensory_targets),
            "inverse_memory_size": len(self.inverse_memory),
            "forward_memory_size": len(self.forward_memory),
            "media_targets": len(self.media_targets),
            "mastery": {
                ph: self.get_mastery(ph)
                for ph in list(self.auditory_targets.keys())[:10]
            },
        }

    # ─── Desacoplamento Laringeo / Articulatório ──────────────────

    def _produce_laryngeal(self, phoneme: str, cmd: Dict) -> Dict:
        """
        Correção somente do componente laríngeo (f0 via vocal_fold_tension).

        Baseado em evidências de que feedback auditivo de pitch
        opera em escala de tempo diferente do feedback de formantes.
        (Tang et al. 2026 — feedback diferencial f0 vs f1/f2)
        """
        if phoneme not in self.auditory_targets:
            return {"vocal_fold_correction": 0.0}

        target = self.auditory_targets[phoneme]
        target_f0 = target[0]
        current_f0 = 150 + 200 * cmd.get("vocal_fold_tension", 0.5)
        err = target_f0 - current_f0
        correction = np.clip(err / 200, -0.2, 0.2) * self.lr
        return {"vocal_fold_correction": float(correction)}

    def _produce_articulatory(self, phoneme: str, cmd: Dict) -> Dict:
        """
        Correção somente do componente articulatório (f1/f2/f3 via tongue, lip, jaw).

        Rota separada da laríngea. Duas dinâmicas de aprendizado distintas.
        """
        if phoneme not in self.auditory_targets:
            return {}

        target = self.auditory_targets[phoneme]
        target_f1, target_f2 = target[1], target[2]

        # Estima formantes atuais
        current_f1 = 250 + 550 * cmd.get("jaw_open", 0.5)
        tx = cmd.get("tongue_dorsum_x", 0.5)
        ly = cmd.get("tongue_dorsum_y", 0.5)
        lip = cmd.get("lip_rounding", 0.0)
        current_f2 = 800 + 1600 * tx * (1 - 0.4 * lip)

        corrections = {}
        # F1: controla jaw_open e tongue_dorsum_y
        f1_err = target_f1 - current_f1
        corrections["jaw_open"] = np.clip(f1_err / 550, -0.3, 0.3) * self.lr
        corrections["tongue_dorsum_y"] = np.clip(-f1_err / 550, -0.2, 0.2) * self.lr

        # F2: controla tongue_dorsum_x e lip_rounding
        f2_err = target_f2 - current_f2
        corrections["tongue_dorsum_x"] = np.clip(f2_err / 1600, -0.3, 0.3) * self.lr
        if lip < 0.9:
            corrections["lip_rounding"] = np.clip(-f2_err / 1600, -0.1, 0.1) * self.lr * 0.5

        return {k: float(v) for k, v in corrections.items()}

    def produce_decoupled(self, phoneme: str,
                          articulatory_config: Optional[Dict] = None) -> Dict:
        """
        Produção com correção de dois loops DESACOPLADOS.
        """
        # Feedforward
        if articulatory_config is not None:
            cmd = articulatory_config.copy()
        elif phoneme in self.feedforward_commands:
            cmd = self._vector_to_config(self.feedforward_commands[phoneme])
        else:
            cmd = self._random_command()
            target = self.auditory_targets.get(phoneme)
            if target is not None:
                cmd = self._approximate_from_target(target, cmd)

        # Loop laríngeo
        laryngeal_corr = self._produce_laryngeal(phoneme, cmd)
        cmd["vocal_fold_tension"] = np.clip(
            cmd["vocal_fold_tension"] + laryngeal_corr.get("vocal_fold_correction", 0),
            0.0, 1.0
        )

        # Loop articulatório
        artic_corr = self._produce_articulatory(phoneme, cmd)
        for k, v in artic_corr.items():
            if k in cmd:
                cmd[k] = np.clip(cmd[k] + v, 0.0, 1.0)

        # Simula resultado
        sensory_result = self._simulate_sensory(cmd)

        # Calcula erros
        auditory_error = 0.0
        somato_error = 0.0
        if phoneme in self.auditory_targets:
            actual = np.array([
                sensory_result["acoustic"]["f0"],
                sensory_result["acoustic"]["f1"],
                sensory_result["acoustic"]["f2"],
                sensory_result["acoustic"]["f3"],
            ])
            target_vec = self.auditory_targets[phoneme]
            auditory_error = float(np.mean(np.abs(actual - target_vec) / (target_vec + 1e-10)))
        if phoneme in self.somatosensory_targets:
            art_vec = self._config_to_vector(cmd)
            somato_target = self.somatosensory_targets[phoneme]
            somato_error = float(np.mean(np.abs(art_vec - somato_target)))

        self._learn(phoneme, cmd, sensory_result, auditory_error, somato_error)
        self.error_history[phoneme].append((auditory_error + somato_error) / 2)

        return {
            "command": cmd,
            "sensory_feedback": sensory_result,
            "errors": {"auditory": auditory_error, "somatosensory": somato_error,
                       "mean": (auditory_error + somato_error) / 2},
            "corrections": {"laryngeal": laryngeal_corr, "articulatory": artic_corr},
            "phoneme": phoneme,
        }

    # ─── Aprendizado de Alvos da Mídia ──────────────────────────

    def learn_target_from_media(self, f0: float, f1: float, f2: float, f3: float,
                                 source: str = "youtube", label: str = "unknown"):
        """
        Aprende um alvo acústico extraído de mídia (YouTube, áudio).

        Alvos de mídia são "exemplos a imitar" — a AGI tenta reproduzí-los
        via babble + DIVA correction.

        Args:
            f0, f1, f2, f3: formantes extraídos pelo dorsal stream
            source: origem do alvo (youtube, audio_file, etc.)
            label: classificação opcional (ex: "vowel_aa")
        """
        target_id = f"media_{len(self.media_targets)}_{int(__import__('time').time())}"
        self.media_targets.append({
            "id": target_id,
            "f0": float(f0), "f1": float(f1), "f2": float(f2), "f3": float(f3),
            "source": source, "label": label,
            "timestamp": __import__("time").time(),
            "mastery": 0.0,
            "attempts": 0,
        })

    def imitate_target(self, target_idx: int = -1) -> Optional[Dict]:
        """
        Tenta imitar um alvo aprendido da mídia.

        Retorna o resultado da tentativa (mesmo formato de produce).
        """
        if not self.media_targets:
            return None
        target = self.media_targets[target_idx]
        target.attempts += 1
        target_vec = np.array([target["f0"], target["f1"],
                               target["f2"], target["f3"]])

        # Cria um nome de fonema virtual para aprender
        ph = f"_media_{target['id'][:16]}"
        self.auditory_targets[ph] = target_vec

        result = self.produce_decoupled(ph)
        target["mastery"] = 1.0 - result["errors"]["mean"]

        return result
