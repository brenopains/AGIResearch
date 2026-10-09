import numpy as np
import time
import os
import tempfile
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
from .vocal_synthesizer import save_wav_to
from .articulatory_synthesizer import (
    synthesize_articulatory, articulatory_to_formants,
    random_articulation, save_articulatory_wav, SAMPLE_RATE,
)


class MirrorNeuron:
    """
    A mirror neuron: fires both when performing an action AND when observing it.
    This is the basis of imitation, empathy, and cultural learning.
    """

    def __init__(self, action_primitive: str):
        self.primitive = action_primitive
        self.strength = 0.5  # How well this neuron models the action
        self.observation_count = 0
        self.execution_count = 0
        self.last_observed = 0.0
        self.last_executed = 0.0
        self.motor_program: List[str] = []  # Sequence of sub-actions
        self.prediction_error_history: deque = deque(maxlen=10)

    def observe(self, description: str, context: Dict = None):
        self.observation_count += 1
        self.last_observed = time.time()
        self.strength = min(1.0, self.strength + 0.02)

    def execute(self) -> str:
        self.execution_count += 1
        self.last_executed = time.time()
        self.strength = min(1.0, self.strength + 0.05)
        return self.primitive

    def to_dict(self) -> dict:
        return {
            "primitive": self.primitive,
            "strength": self.strength,
            "observations": self.observation_count,
            "executions": self.execution_count,
            "motor_program": self.motor_program[:3],
        }


class MotorCortex:
    """
    Layer 8: Motor Output + Mirror Neurons.

    Gives the AGI the ability to ACT on the world, not just observe.
    Like a child, it babbles, draws, and imitates — learning through doing.

    Modules:
    ─────────
    • MirrorNeuronLayer — maps observed actions to motor commands
    • VocalMotorSystem — articulatory synthesis (controls tongue, lips, jaw, etc.)
    • SketchMotorSystem — "doodles" spatial descriptions (no canvas, emergent)
    • PlayDrive — intrinsic motivation to explore motor space

    The key insight: the AGI generates OUTPUT descriptions that it then
    feeds back into its own perception, creating a closed perception-action
    loop. It learns by predicting the sensory consequences of its actions.

    Now with ARTICULATORY CONTROL: the AGI controls individual articulators
    (tongue position, lip rounding, jaw opening, velum, vocal fold tension)
    and LEARNS how each affects the sound produced.
    """

    # Mapa de sons → configurações articulatórias prototípicas
    # Baseado no modelo simplificado: língua (x=front/back, y=high/low),
    # lábios (rounding), mandíbula (open), véu (nasal), pregas (tensão)
    ARTICULATORY_MAP = {
        "aa": {"tongue_dorsum_x": 0.3, "tongue_dorsum_y": 0.3,
               "lip_rounding": 0.0, "jaw_open": 0.8, "velum": 0.0, "vocal_fold_tension": 0.5},
        "ee": {"tongue_dorsum_x": 0.65, "tongue_dorsum_y": 0.4,
               "lip_rounding": 0.0, "jaw_open": 0.5, "velum": 0.0, "vocal_fold_tension": 0.5},
        "ii": {"tongue_dorsum_x": 0.85, "tongue_dorsum_y": 0.8,
               "lip_rounding": 0.0, "jaw_open": 0.2, "velum": 0.0, "vocal_fold_tension": 0.6},
        "oo": {"tongue_dorsum_x": 0.2, "tongue_dorsum_y": 0.5,
               "lip_rounding": 0.7, "jaw_open": 0.5, "velum": 0.0, "vocal_fold_tension": 0.5},
        "uu": {"tongue_dorsum_x": 0.1, "tongue_dorsum_y": 0.7,
               "lip_rounding": 0.8, "jaw_open": 0.3, "velum": 0.0, "vocal_fold_tension": 0.6},
        "mmm": {"tongue_dorsum_x": 0.5, "tongue_dorsum_y": 0.5,
                "lip_rounding": 0.0, "jaw_open": 0.1, "velum": 0.8, "vocal_fold_tension": 0.4},
    }

    PRIMITIVES = [
        "vocalizar_aa", "vocalizar_ee", "vocalizar_ii",
        "vocalizar_oo", "vocalizar_uu", "vocalizar_mmm",
        "apontar", "agarrar", "soltar", "empurrar", "puxar",
        "virar_esquerda", "virar_direita", "avancar", "recuar",
        "sorrir", "franzir_teste", "acenar",
        "riscar_linha", "riscar_circulo", "riscar_ponto",
    ]

    def __init__(self, learning_rate: float = 0.1):
        self.mirror_neurons: Dict[str, MirrorNeuron] = {}
        for p in self.PRIMITIVES:
            self.mirror_neurons[p] = MirrorNeuron(p)

        self.learning_rate = learning_rate
        self.motor_babbling_index = 0
        self.babbling_cycle = 0
        self.vocalization_history: List[Dict] = []
        self.sketch_history: List[Dict] = []
        self.action_history: List[Dict] = []
        self._max_history = 100
        self._max_inverse = 200
        self._last_action_result = ""

        # Aparelho vocal real — controle articulatório
        self._vocal_count = 0
        self.last_audio_path = ""
        self._vocal_dir = r"D:\novamind_data\temp\novamind_vocal"
        os.makedirs(self._vocal_dir, exist_ok=True)
        self.auditory_cortex = None  # set by mind.py after creation
        self.diva_controller = None  # set by mind.py after creation
        self.somatosensory_cortex = None  # set by mind.py after creation
        self.premotor_cortex = None  # set by mind.py
        self.basal_ganglia = None  # set by mind.py
        self.cerebellum = None  # set by mind.py
        self.syllable_oscillator = None  # set by mind.py

        # Alvos auditivos extraídos da mídia (YouTube, áudio)
        self.auditory_targets: deque = deque(maxlen=50)

        # Inverse model: observation → action
        self.inverse_model: Dict[str, str] = {}
        # Forward model: action → predicted outcome
        self.forward_model: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
        self._last_babbled_cfg: Dict = {}
        self._last_formants: Tuple = (0, 0, 0, 0)
        self._last_inverse: Optional[Dict] = None

    def observe_action(self, description: str):
        """
        Mirror neuron activation: observe an action in the environment.

        Matches the description to known motor primitives and fires
        the corresponding mirror neurons — like a child watching
        someone clap and feeling the urge to clap too.
        """
        desc_lower = description.lower()
        for name, neuron in self.mirror_neurons.items():
            # Loose matching: if the primitive keyword appears in description
            if any(word in desc_lower for word in name.split("_")):
                neuron.observe(description)

                # Learn inverse model: observed action → motor primitive
                self.inverse_model[description[:50]] = name
                if len(self.inverse_model) > self._max_inverse:
                    for k in list(self.inverse_model)[:len(self.inverse_model)//2]:
                        del self.inverse_model[k]

    def babble(self) -> str:
        """
        Motor babbling: generate a random action primitive.

        Like a baby discovering its own body — tries random movements
        and observes the results. This is how the AGI learns the
        consequences of its own actions (forward model learning).

        Vocal actions geram áudio REAL via síntese de formantes.
        """
        primitives = list(self.mirror_neurons.keys())
        action = primitives[self.motor_babbling_index % len(primitives)]
        self.motor_babbling_index += 1

        neuron = self.mirror_neurons[action]

        if action.startswith("vocalizar_"):
            vowel = action.replace("vocalizar_", "")
            cfg = self.ARTICULATORY_MAP.get(vowel, None)
            if cfg is not None:
                tiny_jitter = {
                    k: np.clip(v + np.random.uniform(-0.1, 0.1), 0.0, 1.0)
                    for k, v in cfg.items()
                }
            else:
                tiny_jitter = random_articulation()
            desc, wav = self.vocalize(
                target_sound=vowel,
                duration=0.4 + (self.babbling_cycle % 3) * 0.2,
                auditory_cortex=self.auditory_cortex,
                articulatory_config=tiny_jitter,
            )
            result = desc
            self._last_babbled_cfg = tiny_jitter
        else:
            result = neuron.execute()

        self.action_history.append({
            "time": time.time(),
            "action": action,
            "type": "babble",
        })
        if len(self.action_history) > self._max_history:
            self.action_history = self.action_history[-self._max_history:]
        return result

    def imitate(self, description: str) -> Optional[str]:
        """
        Try to imitate an observed action using mirror neurons.

        Returns the motor command if a matching primitive is found.
        """
        self.observe_action(description)
        desc_lower = description.lower()

        best_match = None
        best_score = 0
        for name, neuron in self.mirror_neurons.items():
            score = 0
            for word in name.split("_"):
                if word in desc_lower:
                    score += 1
            if neuron.strength > 0.3:
                score += neuron.strength * 0.5
            if score > best_score:
                best_score = score
                best_match = name

        if best_match and best_score > 0.5:
            neuron = self.mirror_neurons[best_match]
            action = neuron.execute()
            self.action_history.append({
                "time": time.time(),
                "action": best_match,
                "type": "imitate",
                "trigger": description[:50],
            })
            if len(self.action_history) > self._max_history:
                self.action_history = self.action_history[-self._max_history:]
            return action
        return None

    def vocalize(self, target_sound: str = "", pitch: float = 0,
                 duration: float = 0.5, auditory_cortex=None,
                 articulatory_config: Optional[Dict] = None,
                 consonant_type: Optional[str] = None,
                 consonant_place: Optional[str] = None) -> Tuple[str, Optional[bytes]]:
        """
        Gera áudio REAL via SÍNTESE ARTICULATÓRIA.
        Retorna (descrição textual, wav_bytes) para auto-audição.

        A AGI agora controla ARTICULADORES INDIVIDUAIS (não formantes diretos):
          tongue_dorsum_x (0-1: posterior→anterior)
          tongue_dorsum_y (0-1: baixa→alta)
          lip_rounding    (0-1: esticado→arredondado)
          jaw_open        (0-1: fechado→aberto)
          velum           (0-1: oral→nasal)
          vocal_fold_tension (0-1: relaxado→tenso)

        O DIVA Controller refina o comando com feedback auditivo + somatossensorial,
        e o resultado é realimentado pelo AuditoryCortex (self-hearing em tempo real).
        """
        if not target_sound and articulatory_config is None:
            vocs = [p.replace("vocalizar_", "") for p in self.PRIMITIVES if p.startswith("vocalizar_")]
            target_sound = vocs[self.babbling_cycle % len(vocs)]

        # Step 1: Determina configuração articulatória
        if articulatory_config is not None:
            cfg = articulatory_config.copy()
        elif target_sound in self.ARTICULATORY_MAP:
            cfg = self.ARTICULATORY_MAP[target_sound].copy()
        else:
            cfg = random_articulation()

        # Step 2: Aplica DIVA (feedback correction) se disponível
        if self.diva_controller is not None and target_sound:
            diva_result = self.diva_controller.produce(target_sound, cfg)
            cfg = diva_result["command"]
            diva_errors = diva_result["errors"]
        else:
            diva_errors = {"auditory": 0.0, "somatosensory": 0.0, "mean": 0.0}

        # Step 3: Gera o waveform a partir da configuração articulatória
        if pitch > 0:
            cfg["vocal_fold_tension"] = np.clip((pitch - 80) / 220, 0.0, 1.0)

        waveform, sensory_feedback = synthesize_articulatory(
            **cfg,
            amplitude=0.5 + (self._vocal_count % 3) * 0.15,
            duration=duration,
            voiced=True,
            consonant_type=consonant_type,
            consonant_place=consonant_place,
        )

        # Step 4: Processamento somatossensorial (AGI SENTE os articuladores)
        somato_desc = ""
        if self.somatosensory_cortex is not None:
            somato_result = self.somatosensory_cortex.sense(
                cfg, sensory_feedback["acoustic"]
            )
            somato_desc = somato_result.get("self_description", "")

        # Step 5: Salva WAV
        self._vocal_count += 1
        self.babbling_cycle += 1
        wav_bytes = self._waveform_to_wav_bytes(waveform)

        fname = f"vocal_{self._vocal_count:04d}.wav"
        fpath = os.path.join(self._vocal_dir, fname)
        save_articulatory_wav(waveform, fpath)
        self.last_audio_path = fpath

        try:
            wavs = sorted([os.path.join(self._vocal_dir, f) for f in os.listdir(self._vocal_dir) if f.endswith('.wav')], key=os.path.getmtime)
            while len(wavs) > 20:
                os.remove(wavs.pop(0))
        except Exception:
            pass

        # Step 6: Self-hearing — AGI ouve a própria voz EM TEMPO REAL
        self._last_wav_bytes = wav_bytes
        self._last_wav_sr = SAMPLE_RATE
        if auditory_cortex is not None:
            try:
                hear_result = auditory_cortex.process(waveform, SAMPLE_RATE)
                self._last_hear_desc = hear_result.get("self_description", "")
                # Dual-stream: extrai representação articulatória do som ouvido
                if self.diva_controller is not None:
                    ac = sensory_feedback["acoustic"]
                    inv_cfg = self.diva_controller.inverse_map(
                        ac["f0"], ac["f1"], ac["f2"], ac["f3"]
                    )
                    self._last_inverse = inv_cfg
            except Exception:
                self._last_hear_desc = ""
                self._last_inverse = None
        else:
            self._last_hear_desc = ""
            self._last_inverse = None

        # Step 7: Gera descrição consolidada
        ac = sensory_feedback["acoustic"]
        f0, f1, f2, f3 = ac["f0"], ac["f1"], ac["f2"], ac["f3"]
        self._last_formants = (f0, f1, f2, f3)

        if target_sound:
            sound_info = target_sound
        else:
            sound_info = f"{f0:.0f}Hz_f1={f1:.0f}"

        desc_parts = [f"[Vocal: {sound_info}"]
        desc_parts.append(f"lingua({cfg.get('tongue_dorsum_x',0):.1f},{cfg.get('tongue_dorsum_y',0):.1f})")
        desc_parts.append(f"mand={cfg.get('jaw_open',0):.1f} lab={cfg.get('lip_rounding',0):.1f}")
        desc_parts.append(f"f1={f1:.0f} f2={f2:.0f}")
        if somato_desc:
            desc_parts.append(somato_desc[:40])
        if diva_errors["mean"] > 0:
            desc_parts.append(f"erro={diva_errors['mean']:.2f}")
        desc_parts.append("]")

        desc = " ".join(desc_parts)

        self.vocalization_history.append({
            "time": time.time(),
            "output": desc,
            "audio_file": fpath,
            "formants": (f0, f1, f2, f3),
            "articulatory_config": cfg,
            "duration": duration,
            "diva_errors": diva_errors,
        })
        if len(self.vocalization_history) > self._max_history:
            self.vocalization_history = self.vocalization_history[-self._max_history:]
        return desc, wav_bytes

    def _waveform_to_wav_bytes(self, waveform: np.ndarray) -> bytes:
        import struct, io
        signal_int = (waveform * 32767).astype(np.int16)
        n = len(waveform)
        buf = io.BytesIO()
        data_size = n * 2
        buf.write(struct.pack('<4sI4s', b'RIFF', 36 + data_size, b'WAVE'))
        buf.write(struct.pack('<4sIHHIIHH', b'fmt ', 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16))
        buf.write(struct.pack('<4sI', b'data', data_size))
        buf.write(signal_int.tobytes())
        return buf.getvalue()

    def sketch(self, target: str = "") -> str:
        """
        Autonomous sketching: generates a visual/spatial description
        WITHOUT a canvas. The AGI describes what it *intends* to draw.

        Like a child doodling — it's learning the mapping between
        motor commands and visual patterns. The forward model learns:
        riscar_circulo → see_circle_shape.
        """
        if not target:
            draws = [p for p in self.PRIMITIVES if p.startswith("riscar_")]
            neuron = self.mirror_neurons[draws[self.babbling_cycle % len(draws)]]
            action = neuron.execute()
            self.sketch_history.append({
                "time": time.time(),
                "output": action,
            })
            if len(self.sketch_history) > self._max_history:
                self.sketch_history = self.sketch_history[-self._max_history:]
            return action
        return ""

    def forward_model_predict(self, action: str) -> float:
        """Predict the sensory outcome of an action. Returns expected surprise."""
        history = self.forward_model.get(action, deque(maxlen=5))
        if not history:
            return 1.0  # Unknown action → high surprise
        return 1.0 - float(np.mean(history))

    def forward_model_learn(self, action: str, outcome_surprise: float):
        """Update forward model: action → outcome prediction."""
        self.forward_model[action].append(outcome_surprise)

    def play(self, context: str = "") -> str:
        """
        Play mode: autonomous exploration of the motor space.

        The AGI cycles through:
        1. Babble — try random actions
        2. Imitate — mirror observed actions
        3. Practice — repeat actions with high prediction error
        4. Mastery — refine actions with low prediction error

        This mimics childhood play: no external reward, just
        the joy of learning to control one's own body and
        predict the sensory consequences of actions.
        """
        cycle_type = self.babbling_cycle % 4
        self.babbling_cycle += 1

        if cycle_type == 0:
            return self.babble()
        elif cycle_type == 1 and context:
            imitation = self.imitate(context)
            if imitation:
                return imitation
        elif cycle_type == 2:
            # Practice high-error actions
            high_error = [(a, self.forward_model_predict(a))
                          for a in self.PRIMITIVES]
            high_error.sort(key=lambda x: x[1], reverse=True)
            if high_error and high_error[0][1] > 0.5:
                return self.mirror_neurons[high_error[0][0]].execute()
        else:
            # Practice actions that need repetition
            known_actions = [a for a in self.PRIMITIVES
                             if self.mirror_neurons[a].execution_count > 0]
            if known_actions:
                least_practiced = min(known_actions,
                                      key=lambda a: self.mirror_neurons[a].execution_count)
                return self.mirror_neurons[least_practiced].execute()

        return self.babble()

    def imitate_sound(self, f0: float, f1: float, f2: float, f3: float,
                      source: str = "midia") -> Optional[Dict]:
        """
        Imita um som ouvido usando o pipeline completo:
          PremotorCortex (planejamento) → BasalGanglia (seleção) →
          Cerebellum (timing) → SyllableOscillator (ritmo) → DIVA (execução+correção)

        Este é o ciclo auto-didata central:
          1. AuditoryCortex.dorsal extrai [f0,f1,f2,f3] de um vídeo/áudio
          2. MotorCortex.imitate_sound() tenta reproduzir
          3. DIVA corrige com feedback auditivo + somatossensorial
          4. BasalGanglia atualiza valor da ação baseado no erro
          5. Com prática, feedforward fica preciso

        Args:
            f0, f1, f2, f3: formantes alvo extraídos pelo dorsal stream
            source: origem do alvo (youtube, audio, self)

        Returns:
            dict com resultado da tentativa ou None
        """
        if self.diva_controller is None:
            return None

        # 1. Aprende o alvo no DIVA
        self.diva_controller.learn_target_from_media(f0, f1, f2, f3, source=source)
        self.auditory_targets.append({"f0": f0, "f1": f1, "f2": f2, "f3": f3, "source": source})

        # 2. Planeja gestos via PremotorCortex (se disponível)
        gestures = None
        if self.premotor_cortex is not None:
            gestures = self.premotor_cortex.plan_target(f0, f1, f2, f3)
            if gestures and self.cerebellum is not None:
                gestures = self.premotor_cortex.apply_timing(gestures, self.cerebellum)

        # 3. Decide via BasalGanglia (se disponível) se deve tentar
        action = "imitate_sound"
        go = True
        if self.basal_ganglia is not None:
            decision = self.basal_ganglia.decide(action, context={"target_formants": [f0, f1, f2, f3]})
            go = decision.get("go", True)

        if not go:
            return {"action": "inhibited", "reason": "basal_ganglia_nogo"}

        # 4. Aplica ritmo silábico (se disponível)
        rhythm_dur = 0.5
        if self.syllable_oscillator is not None:
            rhythm = self.syllable_oscillator.tick()
            rhythm_dur = 0.3 + rhythm.get("theta_phase_normalized", 0.5) * 0.4
            if len(rhythm.get("gamma_pulses", [])) > 2:
                # Se em burst gama, subdivide o gesto
                for g in gestures:
                    g["duration_ms"] = g.get("duration_ms", 200) * 0.6

        # 5. Determina fonema mais próximo para executar
        from .articulatory_synthesizer import articulatory_to_formants
        best_ph = []
        if gestures:
            for g in gestures:
                ph_type = g["type"]
                if ph_type in ("plosive", "fricative", "nasal"):
                    best_ph.append(g.get("consonant_type", "plosive"))
                elif ph_type == "vowel":
                    # Mapeia formantes do gesto de volta para nome
                    a = g.get("articulators", {})
                    vf0, vf1, vf2, _ = articulatory_to_formants(
                        a.get("tongue_dorsum_x", 0.5), a.get("tongue_dorsum_y", 0.5),
                        a.get("lip_rounding", 0), a.get("jaw_open", 0.5),
                        a.get("velum", 0), a.get("vocal_fold_tension", 0.5),
                    )
                    best_ph.append(self._closest_vowel_name(vf1, vf2))

        if not best_ph:
            best_ph = [self._closest_vowel_name(f1, f2)]

        # 6. Executa cada fonema na sequência
        results = []
        for i, ph in enumerate(best_ph):
            # Obtém configuração articulatória para o fonema
            if ph in self.ARTICULATORY_MAP:
                cfg = self.ARTICULATORY_MAP[ph].copy()
            else:
                cfg = random_articulation()

            # Aplica gesto específico se disponível
            if gestures and i < len(gestures):
                g = gestures[i]
                if g["type"] != "silence":
                    for k, v in g.get("articulators", {}).items():
                        cfg[k] = v
                    consonant_type = g.get("consonant_type")
                    consonant_place = g.get("consonant_place")
                else:
                    continue
            else:
                consonant_type = None
                consonant_place = None

            duration = max(0.15, rhythm_dur / max(1, len(best_ph)))

            # Executa com DIVA decoupled
            if self.diva_controller is not None:
                diva_result = self.diva_controller.produce_decoupled(ph, cfg)
                cfg = diva_result["command"]
                diva_errors = diva_result["errors"]
            else:
                diva_errors = {"auditory": 0.0, "somatosensory": 0.0, "mean": 0.0}

            # Gera waveform
            waveform, sf = synthesize_articulatory(
                **cfg, amplitude=0.5, duration=duration,
                voiced=True,
                consonant_type=consonant_type,
                consonant_place=consonant_place,
            )

            # Self-hearing
            if self.auditory_cortex is not None:
                try:
                    hear_result = self.auditory_cortex.process(waveform, SAMPLE_RATE)
                    hear_desc = hear_result.get("self_description", "")
                except Exception:
                    hear_desc = ""
            else:
                hear_desc = ""

            results.append({
                "phoneme": ph,
                "cfg": cfg,
                "consonant_type": consonant_type,
                "consonant_place": consonant_place,
                "diva_errors": diva_errors,
                "duration": duration,
                "hear_desc": hear_desc,
            })

        # 7. Atualiza BasalGanglia com resultado
        if self.basal_ganglia is not None:
            mean_errors = [r["diva_errors"]["mean"] for r in results]
            avg_err = float(np.mean(mean_errors)) if mean_errors else 0.5
            reward = 1.0 - avg_err
            self.basal_ganglia.update("imitate_sound", action, reward)

        return {
            "action": action,
            "target": {"f0": f0, "f1": f1, "f2": f2, "f3": f3},
            "gestures": gestures,
            "results": results,
            "duration": rhythm_dur,
            "source": source,
        }

    def _closest_vowel_name(self, f1: float, f2: float) -> str:
        best = "schwa"
        best_dist = float("inf")
        for name, cfg in self.ARTICULATORY_MAP.items():
            from .articulatory_synthesizer import articulatory_to_formants
            vf0, vf1, vf2, _ = articulatory_to_formants(
                cfg["tongue_dorsum_x"], cfg["tongue_dorsum_y"],
                cfg["lip_rounding"], cfg["jaw_open"],
                cfg["velum"], cfg["vocal_fold_tension"],
            )
            dist = abs(f1 - vf1) / 500 + abs(f2 - vf2) / 1500
            if dist < best_dist:
                best_dist = dist
                best = name
        return best

    def get_play_state(self) -> str:
        """Returns a description of what the AGI is 'playing' with."""
        recent = self.action_history[-3:] if self.action_history else []
        if not recent:
            return "[Motor Cortex: explorando espaco motor]"
        actions = [a["action"] for a in recent]
        return f"[Brincando: {', '.join(actions)}]"

    def get_state(self) -> dict:
        active_neurons = [
            n.to_dict() for n in self.mirror_neurons.values()
            if n.strength > 0.5
        ]
        return {
            "mirror_neurons_active": len(active_neurons),
            "total_actions": len(self.action_history),
            "vocalizations": len(self.vocalization_history),
            "sketches": len(self.sketch_history),
            "babbling_cycle": self.babbling_cycle,
            "current_action": self.action_history[-1] if self.action_history else None,
            "play_state": self.get_play_state(),
            "inverse_model_size": len(self.inverse_model),
            "articulatory_enabled": True,
            "last_formants": self._last_formants,
        }
