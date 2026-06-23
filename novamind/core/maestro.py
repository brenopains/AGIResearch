"""
MAESTRO — Orquestrador Cognitivo Paralelo em Tempo Real
=========================================================
Cada córtex (visual, auditivo, cognitivo, memória, mundo modelo,
arquétipos, motor) roda em THREAD PRÓPRIA, em PARALELO.

O Maestro:
1. Coleta saídas de todos os atores continuamente
2. Salience Network decide o que é relevante AGORA
3. Global Workspace mantém o conteúdo "consciente" atual
4. Executive Control monitora e decide estratégia

Referências: Baars GWT, Friston FEP, Seeley Salience Network
"""
import os
import threading
import queue
import time
import random
import traceback
import re
from collections import deque
from typing import Dict, List, Optional, Any, Tuple


# ═══════════════════════════════════════════════════════════════
# ATOR BASE — cada módulo cognitivo roda em sua própria thread
# ═══════════════════════════════════════════════════════════════
class Actor(threading.Thread):
    """Um módulo cognitivo que roda em loop contínuo na própria thread."""
    def __init__(self, name: str, mind=None, knowledge=None, playground=None):
        super().__init__(daemon=True)
        self.name = name
        self.mind = mind
        self.knowledge = knowledge
        self.playground = playground
        self.input_q: queue.Queue = queue.Queue()
        self.output_q: queue.Queue = queue.Queue()
        self.running = True
        self.state: Dict = {}
        self._pause = threading.Event()
        self._pause.set()  # começa rodando

    def send(self, data):
        self.input_q.put(data)

    def recv(self, timeout=0.1):
        try:
            return self.output_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def pause(self):
        self._pause.clear()

    def resume(self):
        self._pause.set()

    def stop(self):
        self.running = False
        self._pause.set()

    def run(self):
        while self.running:
            self._pause.wait()
            try:
                data = self.input_q.get(timeout=0.3)
                result = self.process(data)
                if result is not None:
                    self.output_q.put(result)
            except queue.Empty:
                self.idle_tick()
            except Exception as e:
                print(f"[MAESTRO:{self.name}] Erro: {e}")
                traceback.print_exc()

    def process(self, data) -> Optional[Any]:
        return None

    def idle_tick(self):
        pass

    def get_state(self) -> Dict:
        return self.state


# ═══════════════════════════════════════════════════════════════
# ATORES DA PERCEPÇÃO (rodam em tempo real)
# ═══════════════════════════════════════════════════════════════

class PerceptualActor(Actor):
    """
    Ator de percepção: busca conhecimento continuamente (YouTube, Wikipedia, etc.)
    e alimenta a mente com o que encontra — em tempo real, sem esperar.
    """
    def __init__(self, mind, knowledge):
        super().__init__("percepcao", mind=mind, knowledge=knowledge)
        self.current_topic = ""
        self.current_source = ""
        self.topics_completed = 0
        self.sentences_processed = 0
        self.crutch_mode = True  # Whisper/BLIP ativos

    def process(self, data):
        if data == "next_topic":
            return self._fetch_and_process()
        elif isinstance(data, str) and data.startswith("force_topic:"):
            forced = data.split(":", 1)[1]
            return self._fetch_and_process(forced_topic=forced)
        return None

    def idle_tick(self):
        # Ocioso: apenas mantém estado
        pass

    def _fetch_and_process(self, forced_topic=None) -> Optional[Dict]:
        """Busca UM tópico e retorna as sentenças processadas."""
        if forced_topic:
            topic = forced_topic
        else:
            curiosities = list(self.mind.unanswered_questions) if self.mind.unanswered_questions else []
            topic = self.knowledge.pick_next_topic(curiosities, self.mind)

        approved = True
        if hasattr(self.mind, 'permission_handler') and self.mind.permission_handler:
            details = f"Pesquisar '{topic}' na Web / Wikipedia / YouTube"
            approved = self.mind.permission_handler("web_search", details)
            
        if not approved:
            print(f"[MAESTRO:percepcao] Permissao negada para buscar '{topic}'. Ignorando.")
            return None

        self.current_topic = topic
        print(f"\n[MAESTRO:percepcao] === Tópico: {topic} ===")

        result = self.knowledge.fetch_knowledge(topic, self.mind)
        content = result.get("content", "")
        source = result.get("source", "")
        content_type = result.get("type", "text")

        self.current_source = f"{source} ({content_type})"
        self.sentences_processed = 0

        # ─── PROCESSAMENTO SENSORIAL REAL — INPUT PRIMÁRIO ───────────
        # A AGI vê e ouve por SI MESMA. A transcrição Whisper/BLIP
        # é apenas rótulo secundário. O que alimenta mind.think() é
        # a auto-descrição sensorial, não o texto da muleta.
        sensory_lines = []

        if content_type == "youtube":
            # Córtex visual: processa frames brutos → input primário
            frames_raw = result.get("frames_raw", [])
            for i, fb in enumerate(frames_raw[:5]):  # primeiros 5 frames
                vresult = self.mind.visual_cortex.process(fb)
                sensory_lines.append(f"[Visao t={i*20}s]: {vresult['self_description']}")

            # Córtex auditivo: processa waveform bruto → input primário
            audio_raw = result.get("audio_raw")
            audio_sr = result.get("audio_sample_rate", 16000)
            if audio_raw is not None:
                aresult = self.mind.auditory_cortex.process(audio_raw, audio_sr)
                sensory_lines.append(f"[Audio t=0s]: {aresult['self_description']}")

            # Extrai alvos auditivos do dorsal stream para imitação
            if audio_raw is not None:
                try:
                    ds = self.mind.auditory_cortex._extract_dorsal(audio_raw, audio_sr)
                    est_f1 = ds.get("estimated_f1", 0)
                    est_f2 = ds.get("estimated_f2", 0)
                    est_pitch = ds.get("estimated_pitch", 0)
                    if est_f1 > 0 and est_f2 > 0:
                        self.mind.motor.imitate_sound(
                            f0=est_pitch, f1=est_f1, f2=est_f2, f3=0,
                            source=f"youtube_{topic[:20]}"
                        )
                except Exception:
                    pass

        # O conteúdo primário é a auto-descrição sensorial
        if sensory_lines:
            content = " | ".join(sensory_lines)
            print(f"[MAESTRO:percepcao] Input sensorial primario: {content[:120]}...")
        # Transcrição da muleta (Whisper) vai como contexto secundário
        whisper_text = result.get("content", "")
        if whisper_text and len(whisper_text) > 20:
            content += "\n[Transcricao: " + whisper_text[:500] + "]"
        # ─── FIM PROCESSAMENTO SENSORIAL ────────────────────────────────

        if not content or len(content) < 20:
            return {"topic": topic, "source": source, "type": content_type,
                    "sentences": [], "confidence": 1.0}

        # Extrai confiança do Whisper (se for YouTube)
        confidence = 1.0
        if content_type == "youtube":
            conf_match = re.search(r'^\[AudioConfidence:\s*([\d.]+)\]', content, re.MULTILINE)
            if conf_match:
                confidence = float(conf_match.group(1))

        sentences = self.knowledge.__class__.split_into_sentences(content)
        self.sentences_processed = len(sentences)
        self.topics_completed += 1

        return {
            "topic": topic, "source": source, "type": content_type,
            "sentences": sentences, "confidence": confidence,
        }

    def get_state(self) -> Dict:
        return {
            "current_topic": self.current_topic,
            "current_source": self.current_source,
            "topics_completed": self.topics_completed,
            "sentences_processed": self.sentences_processed,
            "crutch_mode": self.crutch_mode,
        }


class CognitiveActor(Actor):
    """
    Ator cognitivo: processa conteúdo através do NovaMind.
    Recebe sentenças da percepção, roda mind.think(), e gera pensamentos.
    """
    def __init__(self, mind):
        super().__init__("cognitivo", mind=mind)
        self.current_sentence = ""
        self.cycle_count = 0
        self.last_thought = None

    def process(self, data) -> Optional[Dict]:
        if isinstance(data, dict) and "sentence" in data:
            sentence = data["sentence"]
            confidence = data.get("confidence", 1.0)
            self.current_sentence = sentence[:100]
            self.cycle_count += 1
            thought = self.mind.think(sentence, confidence=confidence)
            self.last_thought = thought
            return {
                "type": "thought",
                "cycle": self.cycle_count,
                "sentence": sentence[:100],
                "thought": thought,
                "attention": thought.get('predictive_coding', {}).get('attention_weight', 0.5),
                "prediction_error": thought.get('predictive_coding', {}).get('prediction_error', 0),
            }
        return None

    def remote_think(self, sentence, confidence=1.0):
        """Permite que outros atores enviem conteúdo diretamente."""
        self.send({"sentence": sentence, "confidence": confidence})

    def get_state(self) -> Dict:
        return {
            "current_sentence": self.current_sentence,
            "cycle_count": self.cycle_count,
        }


class MotorActor(Actor):
    """
    Ator motor: gera ações motoras e interage com o playground.
    Roda ciclos de brincadeira entre processamentos cognitivos.
    """
    def __init__(self, mind, playground):
        super().__init__("motor", mind=mind, playground=playground)
        self.last_action = ""
        self.last_result = ""
        self.play_count = 0

    def process(self, data) -> Optional[Dict]:
        if data == "play":
            return self._play_tick()
        elif isinstance(data, dict) and data.get("type") == "imitate":
            return self._imitate(data.get("description", ""))
        return None

    def _play_tick(self) -> Dict:
        observation = self.playground.observe()
        bg = getattr(self.mind, 'basal_ganglia', None)
        use_bg = bg is not None and self.mind.motor.babbling_cycle % 4 != 0

        if use_bg and hasattr(self.mind, 'auditory_cortex'):
            # BasalGanglia decide se vocaliza ou faz outra ação
            bg_decisions = []
            for act in self.mind.motor.PRIMITIVES:
                decision = bg.decide(act, {"observation": observation[:50]})
                if decision.get("go"):
                    bg_decisions.append((act, decision["score"]))
            if bg_decisions:
                bg_decisions.sort(key=lambda x: -x[1])
                selected = bg_decisions[0][0]
                if selected.startswith("vocalizar_"):
                    action = self.mind.motor.babble()
                else:
                    neuron = self.mind.motor.mirror_neurons[selected]
                    action = neuron.execute()
                result = self.playground.apply_action(action)
                bg.update("motor_context", selected, 0.3)
            else:
                action = self.mind.motor.babble()
                result = self.playground.apply_action(action)
        elif self.mind.motor.babbling_cycle % 4 == 0 and hasattr(self.mind, 'auditory_cortex'):
            action = self.mind.motor.babble()
            result = self.playground.apply_action(action)
            # Self-hearing: AGI ouve o proprio som e pensa sobre ele
            hear_desc = getattr(self.mind.motor, '_last_hear_desc', '')
            if hear_desc:
                sensory_input = f"[Auto-audicao]: {hear_desc}\n{result}"
                if self.mind:
                    self.mind.think(sensory_input, confidence=0.6)
        else:
            action = self.mind.motor.play(observation)
            result = self.playground.apply_action(action)
        self.last_action = action
        self.last_result = result
        self.play_count += 1

        return {
            "type": "motor",
            "action": action,
            "result": result[:100],
        }

    def _imitate(self, description: str) -> Dict:
        action = self.mind.motor.imitate(description)
        if action:
            result = self.playground.apply_action(action)
            self.last_action = action
            self.last_result = result
            return {"type": "motor_imitate", "action": action, "result": result[:100]}
        return {"type": "motor_imitate", "action": None, "result": "nada acontece"}

    def get_state(self) -> Dict:
        return {
            "last_action": self.last_action,
            "last_result": self.last_result,
            "play_count": self.play_count,
            "last_audio_path": self.mind.motor.last_audio_path if (self.mind and hasattr(self.mind, 'motor')) else "",
            "position": self.playground.position if self.playground else None,
            "held_object": self.playground.held_object if self.playground else None,
        }


class SleepActor(Actor):
    """
    Ator de sono: consolida memória + arquétipos periodicamente.
    Roda em background, acordado pelo maestro a cada N ciclos.
    """
    def __init__(self, mind):
        super().__init__("sono", mind=mind)
        self.last_sleep_time = 0
        self.sleep_count = 0
        self.new_archetypes = []

    def process(self, data) -> Optional[Dict]:
        if data == "sleep":
            return self._sleep_cycle()
        return None

    def _sleep_cycle(self) -> Dict:
        results = {"memory": False, "archetypes": []}

        # Consolidação de memória (hippocampal replay)
        try:
            self.mind.memory.trigger_sleep_cycle()
            results["memory"] = True
        except Exception as e:
            print(f"[MAESTRO:sono] Memory error: {e}")

        # Consolidação de arquétipos (MDL compression)
        try:
            new_arqs = self.mind.archetypes.consolidate()
            if new_arqs:
                results["archetypes"] = new_arqs
                self.new_archetypes.extend(new_arqs)
                print(f"[MAESTRO:sono] +{len(new_arqs)} arquétipos: {', '.join(new_arqs)}")
        except Exception as e:
            print(f"[MAESTRO:sono] Archetype error: {e}")

        self.last_sleep_time = time.time()
        self.sleep_count += 1
        return results

    def get_state(self) -> Dict:
        return {
            "sleep_count": self.sleep_count,
            "last_sleep": self.last_sleep_time,
            "new_archetypes": self.new_archetypes[-5:],
        }


class LanguageActor(Actor):
    """
    Ator de linguagem emergente: gera auto-fala, aprende fonemas,
    e realimenta o sistema auditivo com seus próprios sons.
    """
    def __init__(self, mind):
        super().__init__("linguagem", mind=mind)

    def process(self, data) -> Optional[Dict]:
        if data == "babble":
            return self._babble_cycle()
        return None

    def _babble_cycle(self) -> Dict:
        seq = self.mind.emergent_lang.babble_sequence(length=2 + (self.mind.cycle_count % 3))
        self.mind.emergent_lang.learn_from_self_play(seq)
        wav_path = self.mind.emergent_lang.vocalize_sequence(seq)
        result = {
            "type": "language",
            "sequence": seq,
            "audio_path": wav_path,
            "lexicon_size": len(self.mind.emergent_lang.lexicon),
        }
        if wav_path:
            self.mind.motor.last_audio_path = wav_path
            # Self-hearing: ouve a própria fala
            if hasattr(self.mind, 'auditory_cortex'):
                try:
                    import wave, numpy as npp
                    with wave.open(wav_path, 'rb') as wf:
                        sr_wav = wf.getframerate()
                        frames = wf.readframes(wf.getnframes())
                    raw_sig = npp.frombuffer(frames, dtype=npp.int16).astype(npp.float32) / 32768.0
                    hear = self.mind.auditory_cortex.process(raw_sig, sr_wav)
                    hear_desc = hear.get("self_description", "")
                    if hear_desc:
                        self.mind.think(f"[Auto-audicao linguagem]: {hear_desc}", confidence=0.6)
                except Exception:
                    pass
        return result

    def get_state(self) -> Dict:
        return self.mind.emergent_lang.get_state() if self.mind else {}


class CausalActor(Actor):
    """
    Ator de raciocínio causal: explora contra-factuais e planejamento
    em background, sem bloquear o ciclo principal.
    """
    def __init__(self, mind):
        super().__init__("causal", mind=mind)

    def process(self, data) -> Optional[Dict]:
        if data == "reason":
            return self._causal_cycle()
        return None

    def _causal_cycle(self) -> Dict:
        results = {"counterfactuals": [], "plans": []}
        try:
            concepts = list(self.mind.perception.concept_registry.keys())[:10]
            if len(concepts) >= 2:
                for i in range(min(3, len(concepts) - 1)):
                    cf = self.mind.causal.simulate_counterfactual(
                        self.mind.world_model, concepts[i], concepts[i + 1]
                    )
                    results["counterfactuals"].append(cf)
        except Exception:
            pass
        return results


# ═══════════════════════════════════════════════════════════════
# O MAESTRO — orquestrador principal
# ═══════════════════════════════════════════════════════════════
from .neuromodulator import Neuromodulator

# Novos módulos de incorporação (import defensivo: não quebra se ausentes)
try:
    from .neuromodulator_state import NeuromodulatorState
except ImportError:
    NeuromodulatorState = None

try:
    from .multiscale_hdv import MultiScaleHDV
except ImportError:
    MultiScaleHDV = None

try:
    from .temporal_binder import TemporalBinder
except ImportError:
    TemporalBinder = None

try:
    from .simulated_body import SimulatedBody, Actuator
except ImportError:
    SimulatedBody = None
    Actuator = None

try:
    from .body_encoder import BodyEncoder
except ImportError:
    BodyEncoder = None

try:
    from .cerebellum import Cerebellum
except ImportError:
    Cerebellum = None


class CognitiveMaestro:
    """
    O Maestro — orquestrador cognitivo paralelo em tempo real.

    Gerencia:
    - Percepção: busca conhecimento continuamente
    - Cognição: processa conteúdo através do NovaMind
    - Motor: gera ações e interage com o playground
    - Sono: consolida memória e arquétipos
    - Salience: decide o que é relevante agora
    - Executive: monitora progresso e decide estratégia
    - Límbico: neuromoduladores [DA, NA, ACh, 5HT] dirigem atenção,
      plasticidade e comportamento

    Tudo roda em PARALELO, cada módulo em sua própria thread.
    O Maestro coordena: coleta resultados, decide o que processar,
    e mantém o ciclo de aprendizado contínuo.
    """
    def __init__(self, mind, knowledge_manager, playground):
        self.mind = mind
        self.knowledge = knowledge_manager
        self.playground = playground

        # Estado compartilhado com a UI
        self.state = {
            "status": "booting",
            "current_topic": "",
            "current_source": "",
            "current_sentence": "",
            "sentences_processed": 0,
            "total_sentences": 0,
            "topics_completed": 0,
            "stream_vision": "Standby",
            "stream_audio": "Standby",
            "stream_reading": "Standby",
            "media_type": None,
            "media_audio_url": None,
            "media_frame_url": None,
            "media_caption": None,
            "media_timestamp": None,
        }

        self.autonomous_learning_enabled = False

        # Cria atores
        self.percepcao = PerceptualActor(mind, knowledge_manager)
        self.cognitivo = CognitiveActor(mind)
        self.motor = MotorActor(mind, playground)
        self.sono = SleepActor(mind)
        self.linguagem = LanguageActor(mind)
        self.causal = CausalActor(mind)

        self.actors = {
            "percepcao": self.percepcao,
            "cognitivo": self.cognitivo,
            "motor": self.motor,
            "sono": self.sono,
            "linguagem": self.linguagem,
            "causal": self.causal,
        }

        # Estatísticas do maestro
        self.cycle_count = 0
        self.topics_completed = 0
        self.thought_history = deque(maxlen=50)
        self.running = False

        # === Sistema Límbico: Neuromodulação Multi-Dimensional ===
        self.neuromodulator = Neuromodulator()        # 4 NTs legado (compat)
        self._last_pe = 0.5  # prediction error anterior (para delta)

        # === Fase 1: NT State com 6 dimensões + causalidade contextual ===
        self.nt_state = NeuromodulatorState() if NeuromodulatorState else None

        # === Fase 2: Memória Multi-Escala L1/L2/L3 ===
        l3_path = None
        try:
            import os
            l3_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'knowledge', 'ltm_l3.npz'
            )
        except Exception:
            pass
        self.multiscale_hdv = MultiScaleHDV(l3_path=l3_path) if MultiScaleHDV else None

        # === Fase 3: Binding Temporal para Memória Episódica ===
        self.temporal_binder = TemporalBinder(seed=77) if TemporalBinder else None

        # === Fase 4: Corpo Simulado + Codificador Corporal ===
        self.body = SimulatedBody() if SimulatedBody else None
        self.body_encoder = BodyEncoder(seed=88) if BodyEncoder else None
        self._body_concept = "stable"    # conceito corporal atual
        self._body_nt_update_interval = 5  # a cada 5 ciclos

        # === Fase 5: Cerebelo (Forward + Inverse Model) ===
        self.cerebellum = Cerebellum() if Cerebellum else None

    def start(self):
        """Inicia todos os atores em paralelo."""
        print("\n" + "=" * 60)
        print("  MAESTRO: Orquestrador Cognitivo Paralelo")
        print("  Percepção | Cognição | Motor | Sono | Linguagem | Causal")
        print("=" * 60 + "\n")

        self.running = True
        self.state["status"] = "listening"

        for name, actor in self.actors.items():
            actor.start()
            print(f"[MAESTRO] Ator '{name}' iniciado")

        # Dispara o primeiro tópico se o aprendizado autônomo estiver ativado
        if self.autonomous_learning_enabled:
            self.percepcao.send("next_topic")

    def stop(self):
        """Para todos os atores."""
        self.running = False
        for actor in self.actors.values():
            actor.stop()
        print("[MAESTRO] Todos os atores parados")

    def tick(self):
        """
        Um ciclo do maestro: coleta saídas dos atores, decide processamento.
        Chamado pelo loop principal (não é thread própria).
        """
        self.cycle_count += 1

        # 1. Coleta resultado da cognição
        thought_result = self.cognitivo.recv(timeout=0.05)
        if thought_result:
            self._on_thought(thought_result)

        # 2. Coleta resultado da percepção (tópico completo)
        perception_result = self.percepcao.recv(timeout=0.05)
        if perception_result:
            self._on_topic_complete(perception_result)

        # 3. Coleta resultado motor
        motor_result = self.motor.recv(timeout=0.05)
        if motor_result:
            self._on_motor(motor_result)

        # 4. Coleta resultado do sono
        sleep_result = self.sono.recv(timeout=0.05)
        if sleep_result:
            self._on_sleep(sleep_result)

        # 5. Coleta resultado da linguagem emergente
        lang_result = self.linguagem.recv(timeout=0.05)
        if lang_result:
            self.state["current_sentence"] = f"Fala: {lang_result.get('sequence', '')[:60]}"
            audio_path = lang_result.get("audio_path", "")
            if audio_path:
                fname = os.path.basename(audio_path)
                self.state["vocal_audio_url"] = f"http://localhost:8000/static/vocal/{fname}"

        # 6. Coleta resultado do raciocínio causal
        causal_result = self.causal.recv(timeout=0.05)
        if causal_result:
            cfs = causal_result.get("counterfactuals", [])
            if cfs:
                self.state["current_sentence"] = f"Causal: {cfs[0].get('concept_a','?')}->{cfs[0].get('concept_b','?')}"

        # 7. Decisões do maestro
        self._decide_next()

        # 8. [NOVO] Loop corpo→NT→gain (a cada N ciclos para não sobrecarregar)
        if self.cycle_count % self._body_nt_update_interval == 0:
            self._body_nt_loop()

    def _on_thought(self, result: Dict):
        """Um pensamento foi gerado pelo córtex cognitivo."""
        thought = result["thought"]
        attention = result.get("attention", 0.5)
        prediction_error = result.get("prediction_error", 0)
        sentence = result["sentence"]

        self.state["current_sentence"] = sentence
        self.state["sentences_processed"] = result["cycle"]

        # Media state parsing para YouTube (atualiza frame/caption por sentença)
        if self.state.get("media_type") == "video":
            vis_match = re.match(r'^\[Visual t=(\d+)s\]:\s*(.*)', sentence)
            aud_match = re.match(r'^\[Audio t=(\d+)s\]:\s*(.*)', sentence)
            if vis_match:
                t_sec = int(vis_match.group(1))
                cap = vis_match.group(2)
                frame_idx = t_sec // 20 + 1
                self.state["media_timestamp"] = t_sec
                self.state["media_caption"] = f"[VISÃO] {cap}"
                self.state["media_frame_url"] = f"http://localhost:8000/static/yt/frames/frame_{frame_idx:03d}.jpg"
            elif aud_match:
                t_sec = int(aud_match.group(1))
                cap = aud_match.group(2)
                frame_idx = t_sec // 20 + 1
                self.state["media_timestamp"] = t_sec
                self.state["media_caption"] = f"[ÁUDIO] {cap}"
                self.state["media_frame_url"] = f"http://localhost:8000/static/yt/frames/frame_{frame_idx:03d}.jpg"

        # Atenção adaptativa via Neuromodulação
        if self.neuromodulator.attention_narrow():
            time.sleep(0.15)  # foco estreito: processa devagar
        elif attention > 0.3:
            time.sleep(0.02)
        # else: segue rápido (baixa atenção, baixa NA)

        # Registra no histórico
        nm_state = self.neuromodulator.as_dict()
        self.thought_history.append({
            "cycle": result["cycle"],
            "input": sentence,
            "attention": attention,
            "prediction_error": prediction_error,
            "neuromodulation": nm_state,
        })

        # === Sistema Límbico: atualiza neuromoduladores ===
        delta_pe = prediction_error - self._last_pe
        self._last_pe = prediction_error
        self.neuromodulator.step(prediction_error, delta_pe)

        # Dispara Forja se Surpresa + Plasticidade
        forge, reason = self.neuromodulator.should_forge()
        if forge:
            concepts = thought.get("layers", {}).get("perception", {}).get("concepts", [])
            unfamiliar = [c for c in concepts if c.get("access_count", 1) <= 1]
            concept_to_learn = unfamiliar[0].get("label") if unfamiliar else sentence[:30]
            print(f"[MAESTRO:limbico] Forja disparada! mood={self.neuromodulator.mood} "
                  f"NM={nm_state} conceito='{concept_to_learn}'")
            # Forja cria novo archetype para reduzir o PE
            try:
                if hasattr(self.mind, 'archetypes') and hasattr(self.mind.archetypes, 'forge'):
                    self.mind.archetypes.forge(concept_to_learn, prediction_error)
            except Exception as e:
                print(f"[MAESTRO:forge] Erro: {e}")
            # Busca ativa para entender o conceito
            self.percepcao.send(f"force_topic:{concept_to_learn}")

    def _on_topic_complete(self, result: Dict):
        """Um tópico foi completamente processado."""
        topic = result["topic"]
        source = result["source"]
        content_type = result["type"]
        sentences = result["sentences"]
        confidence = result["confidence"]

        self.state["current_topic"] = topic
        self.state["current_source"] = source
        self.state["total_sentences"] = len(sentences)
        self.state["sentences_processed"] = 0

        # Atualiza streams para UI
        if content_type == "youtube":
            self.state["stream_vision"] = f"Assistindo: {topic}"
            self.state["stream_audio"] = f"Ouvindo: {topic}"
            self.state["stream_reading"] = "—"
            self.state["media_type"] = "video"
            self.state["media_audio_url"] = "http://localhost:8000/static/yt/audio.wav"
            self.state["media_frame_url"] = "http://localhost:8000/static/yt/frames/frame_001.jpg"
            self.state["media_caption"] = "Carregando..."
            self.state["media_timestamp"] = 0
        elif content_type == "wikipedia":
            self.state["stream_vision"] = "—"
            self.state["stream_audio"] = "—"
            self.state["stream_reading"] = f"Lendo: {source} — {topic}"
            self.state["media_type"] = "text"
            self.state["media_audio_url"] = None
            self.state["media_frame_url"] = None
            self.state["media_caption"] = None
            self.state["media_timestamp"] = None
        else:
            self.state["stream_vision"] = "—"
            self.state["stream_audio"] = "—"
            self.state["stream_reading"] = f"Navegando: {source} — {topic}"
            self.state["media_type"] = "text"
            self.state["media_audio_url"] = None
            self.state["media_frame_url"] = None
            self.state["media_caption"] = None
            self.state["media_timestamp"] = None

        # Alimenta sentenças no córtex cognitivo (uma por tick)
        self._pending_sentences = [
            {"sentence": s, "confidence": confidence}
            for s in sentences
            if not s.startswith("[AudioConfidence") and not s.startswith("[Assistindo Vídeo")
        ]
        self._current_sentence_queue = iter(self._pending_sentences)
        self._topic_sentences_remaining = len(self._pending_sentences)
        self.state["total_sentences"] = self._topic_sentences_remaining

        print(f"[MAESTRO] Tópico '{topic}' → {self._topic_sentences_remaining} sentenças na fila")

        # Feedback: mira curiosidade respondida
        curiosities = list(self.mind.unanswered_questions) if self.mind.unanswered_questions else []
        if curiosities:
            try:
                self.mind.unanswered_questions.pop(0)
            except IndexError:
                pass

        self.topics_completed += 1
        self.state["topics_completed"] = self.topics_completed

    def _on_motor(self, result: Dict):
        """Uma ação motora foi executada."""
        self.state["stream_vision"] = f"Brincando: {result.get('action', '?')}"
        self.state["current_sentence"] = result.get("result", "")[:100]
        audio_path = self.motor.get_state().get("last_audio_path", "")
        if audio_path:
            fname = os.path.basename(audio_path)
            self.state["vocal_audio_url"] = f"http://localhost:8000/static/vocal/{fname}"
        else:
            self.state["vocal_audio_url"] = None

    def _on_sleep(self, result: Dict):
        """Ciclo de sono concluído."""
        if result.get("archetypes"):
            self.state["current_sentence"] = f"Sono: +{len(result['archetypes'])} arquétipos"

    def _decide_next(self):
        """Decisões do maestro dirigidas por neuromodulação."""
        # Se há sentenças pendentes da percepção, alimenta uma no cognitivo
        if hasattr(self, '_pending_sentences') and self._pending_sentences:
            if self._current_sentence_queue:
                try:
                    item = next(self._current_sentence_queue)
                    self.cognitivo.send(item)
                    self.state["sentences_processed"] += 1
                except StopIteration:
                    self._current_sentence_queue = None
                    self._pending_sentences = []
                    self._on_topic_done()
            return

        # Sem sentenças: decisões dirigidas por neuromodulação
        nm = self.neuromodulator

        # 1. Sono: confiança baixa + plasticidade alta
        sleep_b, sleep_reason = nm.should_sleep()
        if sleep_b:
            if self.cycle_count % 2 == 0:  # espaçamento para não travar
                print(f"[MAESTRO:limbico] Sono necessário. mood={nm.mood} NM={nm.as_dict()}")
                self.sono.send("sleep")
            return

        # 2. Forja: surpresa + plasticidade (já tratado em _on_thought)
        #    Aqui só reforçamos se não houver sentenças para processar

        # 3. Exploração vs Aprofundamento
        if nm.should_explore():
            # Dopamina alta → confiante para explorar coisas novas
            if self.cycle_count % 2 == 0 and self.autonomous_learning_enabled:
                self.linguagem.send("babble")
            if self.cycle_count % 3 == 0:
                self.motor.send("play")
        elif nm.should_exploit():
            # Dopamina baixa → focar no que já temos
            if self.cycle_count % 3 == 0:
                self.causal.send("reason")
        else:
            # Neutro: alterna entre atividades
            if self.cycle_count % 2 == 0:
                self.linguagem.send("babble")
            if self.cycle_count % 3 == 0:
                self.motor.send("play")
            elif self.cycle_count % 5 == 0:
                self.causal.send("reason")

        # 4. Troca de tópico por tédio ou falta de progresso
        if nm.should_switch_topic():
            if self.cycle_count % 3 == 0 and self.autonomous_learning_enabled:
                print(f"[MAESTRO:limbico] Tédio detectado. Buscando novo tópico. NM={nm.as_dict()}")
                self.percepcao.send("next_topic")

        # 5. Fallback: busca novo tópico periódica se autônomo
        if self.cycle_count % 7 == 0 and self.autonomous_learning_enabled:
            if self.topics_completed % 3 == 0 and self.topics_completed > 0:
                self.sono.send("sleep")
            self.percepcao.send("next_topic")

    def _on_topic_done(self):
        """Um tópico foi completamente processado pela cognição."""
        print(f"[MAESTRO] Tópico concluído. Total: {self.topics_completed}")

        # Sono a cada 3 tópicos
        if self.topics_completed % 3 == 0:
            self.sono.send("sleep")

        # Playground: brincadeira entre tópicos
        self.motor.send("play")

        # Limpeza YouTube
        try:
            if hasattr(self.knowledge, 'adapters') and "youtube" in self.knowledge.adapters:
                self.knowledge.adapters["youtube"].cleanup_temp_files()
        except Exception:
            pass

        # Próximo tópico se ativado
        if self.autonomous_learning_enabled:
            self.percepcao.send("next_topic")

    def get_state(self) -> Dict:
        """Estado completo para broadcast na UI."""
        actor_states = {name: actor.get_state() for name, actor in self.actors.items()}
        return {
            "maestro": {
                "cycle": self.cycle_count,
                "topics_completed": self.topics_completed,
                "status": self.state["status"],
                "actors": actor_states,
                "autonomous_learning_enabled": self.autonomous_learning_enabled,
                "neuromodulation": self.neuromodulator.as_dict(),
                "mood": self.neuromodulator.mood,
            },
            "learning": dict(self.state),
        }

    def get_learning_state(self) -> Dict:
        """Apenas o estado de aprendizado (para compatibilidade)."""
        return dict(self.state)

    # ─── Loop de Incorporação: Corpo → NT → Gain Control → Ação ──────────────

    def _body_nt_loop(self):
        """
        Ciclo de incorporação:
          1. Lê sensores corporais
          2. Atualiza NeuromodulatorState via corpo
          3. Ajusta gain de operações cognitivas
          4. Guarda episódio no MultiScaleHDV + TemporalBinder
          5. Atualiza Cerebelo com consequência da última ação
        """
        if self.body is None:
            return

        try:
            import time

            # 1. Lê sensores
            reading = self.body.read_sensors()

            # 2. Atualiza NT via corpo
            if self.nt_state is not None:
                self.nt_state.update_from_body(reading.to_dict())
                nt_dict = self.nt_state.as_dict()
                self._body_concept = (
                    self.body_encoder.get_body_concept(reading)
                    if self.body_encoder else "unknown"
                )

                # Propagate mood to state for UI
                self.state["body_concept"] = self._body_concept
                self.state["nt_state_6d"] = nt_dict
                self.state["nt_mood"] = self.nt_state.mood

                # 3. Gain control: ajusta intervalos de sono e aprendizado
                threat = self.nt_state.get_gain("threat_bias")
                if threat > 0.7 and self.nt_state.should_flee():
                    # Ameaça alta: entra em modo de foco estreito
                    print(f"[MAESTRO:corpo] Ameaça detectada! '{self._body_concept}' "
                          f"CRH={nt_dict.get('CRH', 0):.2f} → modo alerta")

                # 4. Guarda episódio na memória multi-escala
                if self.body_encoder and self.multiscale_hdv:
                    body_vec = self.body_encoder.encode_sensors(reading)
                    self.multiscale_hdv.write(
                        key=f"body:{self._body_concept}:{int(time.time())}",
                        vector=body_vec,
                        nt_state=nt_dict,
                        metadata={"concept": self._body_concept,
                                  "pain": reading.pain,
                                  "balance": reading.balance,
                                  "reward": reading.reward}
                    )

                # 5. Guarda episódio no TemporalBinder
                if self.temporal_binder and self.body_encoder:
                    body_vec = self.body_encoder.encode_sensors(reading)
                    self.temporal_binder.write_episodic(
                        concept_vec=body_vec,
                        context_keys=list(getattr(self.nt_state, '_active_contexts', [])),
                        nt_state=nt_dict,
                        t=time.time(),
                        metadata={"concept": self._body_concept}
                    )

                # 6. Atualiza Cerebelo com leitura atual
                if self.cerebellum:
                    # Converte leitura em vetor de estado e registra erro
                    state_vec = reading.to_array()[:15]
                    # Ação neutra (corpo parado) — o cerebelo aprende a prever repouso
                    action_vec = [0.0, 0.0, 0.0, 0.0]
                    import numpy as np
                    err = self.cerebellum.get_prediction_error()
                    if err > 0:
                        self.state["cerebellum_error"] = round(err, 4)

                # Tick temporal na memória multi-escala
                if self.multiscale_hdv:
                    self.multiscale_hdv.tick()

        except Exception as e:
            # Não quebra o loop principal por causa do corpo
            import traceback
            if self.cycle_count % 100 == 0:  # log apenas ocasionalmente
                print(f"[MAESTRO:corpo] Erro no loop: {e}")

    def get_embodiment_state(self) -> Dict:
        """Estado completo do sistema de incorporação para a UI."""
        out = {}

        if self.body:
            out["body"] = self.body.get_state()
            out["body_concept"] = self._body_concept

        if self.nt_state:
            out["nt_state_6d"] = self.nt_state.as_dict()
            out["nt_mood"] = self.nt_state.mood
            out["nt_gains"] = {
                "learning_rate":   round(self.nt_state.get_gain("learning_rate"), 3),
                "attention_width": round(self.nt_state.get_gain("attention_width"), 3),
                "threat_bias":     round(self.nt_state.get_gain("threat_bias"), 3),
                "trust_level":     round(self.nt_state.get_gain("trust_level"), 3),
            }

        if self.multiscale_hdv:
            out["memory_scales"] = self.multiscale_hdv.get_state()

        if self.temporal_binder:
            out["episodic_binder"] = self.temporal_binder.get_state()

        if self.cerebellum:
            out["cerebellum"] = self.cerebellum.get_state()

        return out
