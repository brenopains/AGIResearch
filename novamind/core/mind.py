"""
NovaMind — The Central Mind (Pipeline Orchestrator)
====================================================

This is the "consciousness" of NovaMind — the central process that
orchestrates all 7 cognitive layers into a unified processing pipeline.

For each input, the mind executes a COGNITIVE CYCLE:

    1. PERCEIVE  → Convert input to geometric concepts (Layer 1)
    2. MODEL     → Integrate into causal world model (Layer 2)
    3. REMEMBER  → Store in episodic memory, check compaction (Layer 3)
    4. REASON    → Hybrid reasoning (neural+symbolic+geometric) (Layer 4)
    5. REFLECT   → Metacognitive self-assessment (Layer 5)
    6. MOTIVATE  → Update goals and intrinsic drives (Layer 6)
    7. LEARN     → Adapt and improve continuously (Layer 7)

Each cycle produces a complete "thought trace" that shows exactly
what happened at every level — full cognitive transparency.
"""

import time
import json
import re
from typing import Dict, Optional, List

from .perception import PerceptionEngine
from .world_model import WorldModel
from .memory import MemorySystem
from .reasoning import HybridReasoningEngine
from .metacognition import MetacognitionSystem
from .goals import GoalSystem
from .learning import ContinuousLearningEngine
from .language_cortex import NativeLanguageCortex
from .grounded_semantics import CulturalRuleCompiler
from .predictive_coding import PredictiveCodingEngine
from .active_inference import ActiveInferenceEngine
from .motor_cortex import MotorCortex
from .archetypes import ArchetypeEngine
from .visual_cortex import VisualCortex
from .auditory_cortex import AuditoryCortex
from .working_memory import WorkingMemory
from .intrinsic_rl import IntrinsicRL
from .causal_reasoning import CausalReasoningEngine
from .world_model_4d import WorldModel4D
from .emergent_language import EmergentLanguage
from .diva_controller import DIVAController
from .somatosensory_cortex import SomatosensoryCortex
from .premotor_cortex import PremotorCortex
from .basal_ganglia import BasalGanglia
from .cerebellum import Cerebellum
from .syllable_oscillator import SyllableOscillator

# HDV Memory Integration
try:
    from .hdc_memory import HDCMemory
    HDCMEMORY_AVAILABLE = True
except ImportError:
    HDCMEMORY_AVAILABLE = False

# NovaCrush Integration
try:
    from .novacrush.genetic_memory import GeneticOptimizer, Genome
    NOVACRUSH_AVAILABLE = True
except ImportError:
    NOVACRUSH_AVAILABLE = False


class NovaMind:
    """
    The central cognitive architecture — 7 layers unified.
    Resource: ~40MB VRAM, ~900MB RAM, ~500ms/cycle on RTX 3050.
    """

    def __init__(self, max_concepts: int = 2000, max_episodes: int = 200,
                 use_semantic_recall: bool = True):
        self.max_concepts = max_concepts
        self.max_episodes = max_episodes
        self.cleanup_counter = 0
        self.cleanup_interval = 10

        self.perception = PerceptionEngine()
        self.world_model = WorldModel()
        self.memory = MemorySystem()
        self.memory.start_consolidation()
        self.predictor = PredictiveCodingEngine()
        self.actor = ActiveInferenceEngine()
        self.motor = MotorCortex()
        self.visual_cortex = VisualCortex()
        self.auditory_cortex = AuditoryCortex()
        self.motor.auditory_cortex = self.auditory_cortex  # self-hearing loop
        self.diva = DIVAController()
        self.somatosensory = SomatosensoryCortex()
        self.premotor = PremotorCortex()
        self.basal_ganglia = BasalGanglia()
        self.cerebellum = Cerebellum()
        self.syllable_oscillator = SyllableOscillator()
        self.motor.diva_controller = self.diva
        self.motor.somatosensory_cortex = self.somatosensory
        self.motor.premotor_cortex = self.premotor
        self.motor.basal_ganglia = self.basal_ganglia
        self.motor.cerebellum = self.cerebellum
        self.motor.syllable_oscillator = self.syllable_oscillator

        # Real semantic recall (MiniLM embeddings) instead of Jaccard/NCD.
        # ~99MB VRAM, ~770MB RAM, ~5ms/query warm. Disable for ultra-low-RAM.
        if use_semantic_recall:
            try:
                from .semantic_encoder import attach_encoder
                attach_encoder(self.memory)
            except Exception as e:
                print(f"[NovaMind] semantic recall unavailable ({e}); using string overlap")
        self.reasoning = HybridReasoningEngine()
        self.metacognition = MetacognitionSystem()
        self.goals = GoalSystem()
        self.learning = ContinuousLearningEngine()

        self.language = NativeLanguageCortex(substrate=getattr(self.perception, 'neural_substrate', None))
        self.rule_compiler = CulturalRuleCompiler()
        self.semantic_facts: List[Dict] = []
        self._max_semantic_facts = 500
        self.unanswered_questions: List[Dict] = []

        if NOVACRUSH_AVAILABLE:
            self.genetic_optimizer = GeneticOptimizer(population_size=10)
            self.genome_checkpoint_interval = 50
            self.last_genome = None
        else:
            self.genetic_optimizer = None

        self.archetypes = ArchetypeEngine()
        self.wm = WorkingMemory()
        self.intrinsic_rl = IntrinsicRL()

        # Memória HDV (conhecimento compilado do disco D)
        if HDCMEMORY_AVAILABLE:
            try:
                self.hdc_memory = HDCMemory(device=self.perception.device if hasattr(self.perception, 'device') else None)
                if not self.hdc_memory.matrices:
                    print("[NovaMind] Nenhum .hdv encontrado — rode coglang_compiler.ingest_all() primeiro")
            except Exception as e:
                print(f"[NovaMind] HDCMemory indisponível ({e})")
                self.hdc_memory = None
        else:
            self.hdc_memory = None

        # Injeta GroundingEngine + HDCMemory no PredictiveCoding para erro HDC
        if self.hdc_memory is not None and hasattr(self.hdc_memory, 'engine'):
            self.predictor.set_engine(self.hdc_memory.engine)
            self.predictor.set_hdc_memory(self.hdc_memory)

        self.causal = CausalReasoningEngine()
        self.world_4d = WorldModel4D()
        self.emergent_lang = EmergentLanguage()
        self.cycle_count = 0
        self.thought_history: List[Dict] = []
        self.start_time = time.time()
        self._last_emotional_valence = 0.0

    def think(self, raw_input: str, confidence: float = 1.0) -> Dict:
        cycle_start = time.time()
        self.cycle_count += 1

        thought = {
            'cycle': self.cycle_count,
            'input': raw_input,
            'timestamp': cycle_start,
            'layers': {},
        }

        # ═══════════════════════════════════════════════
        # LAYER 0: PREDICTIVE CODING (Top-down expectation)
        # ═══════════════════════════════════════════════
        prediction = self.predictor.predict_next(raw_input, self.world_model, self.memory)
        pred_error = self.predictor.compute_prediction_error(raw_input, prediction)
        # Uncertainty weighting: low confidence → amplify prediction error → more learning
        uncertainty_multiplier = 1.0 + max(0, (1.0 - confidence) * 2.0)
        pred_error = min(1.0, pred_error * uncertainty_multiplier)
        self.predictor.update_precision(pred_error)
        attention_wt = self.predictor.get_attention_weight()
        lr_mod = self.predictor.get_learning_rate_modulator()

        thought['predictive_coding'] = {
            'prediction': prediction[:100] if prediction else "",
            'prediction_error': pred_error,
            'precision': self.predictor.precision,
            'attention_weight': attention_wt,
            'learning_rate_modulator': lr_mod,
        }

        # ═══════════════════════════════════════════════
        # LAYER 1: PURE GEOMETRIC PERCEPTION (No LLMs)
        # ═══════════════════════════════════════════════
        perception_report = self.perception.perceive(raw_input)
        thought['layers']['perception'] = perception_report

        # Observe concepts in archetype engine (padrões → arquétipos)
        concept_labels = [c.get('label', '') for c in perception_report.get('concepts', [])]
        self.archetypes.observe(concept_labels)

        # Decay old concept activations + homeostatic regulation
        self.perception.decay_activations(rate=0.03)
        self.perception.homeostatic_regulation()

        # ═══════════════════════════════════════════════
        # LAYER 2: WORLD MODEL
        # ═══════════════════════════════════════════════
        self.world_model.integrate_perception(perception_report)
        world_model_data = self.world_model.get_graph_data()

        # Structural plasticity: prune/grow based on prediction error
        self.world_model.apply_structural_plasticity(pred_error)
        self.world_model.homeostatic_plasticity()

        # Extract semantic facts from declarative statements
        new_facts = self._extract_semantic_facts(raw_input)
        self._detect_contradictions(new_facts, thought)
        for fact in new_facts:
            self.memory.store_semantic_rule(fact)
            self.semantic_facts.append(fact)
            if len(self.semantic_facts) > self._max_semantic_facts:
                self.semantic_facts = self.semantic_facts[-self._max_semantic_facts:]
            # Link concepts in world model
            if fact.get('type') == 'cognitive_rule':
                # Complex rule, handled by advanced world model logic/compilation
                continue
            
            subj_id = self._concept_id_for_label(fact.get('subject', ''))
            obj_id = self._concept_id_for_label(fact.get('object', ''))
            if subj_id and obj_id and 'relation' in fact:
                self.world_model.add_causal_link(
                    subj_id, obj_id, fact['relation'], 0.8
                )

        # Check if new input satisfies any pending knowledge goal
        satisfied = self._check_pending_goals(raw_input, new_facts)
        if satisfied:
            thought['goal_satisfied'] = satisfied

        # Derive inferences from semantic rules (forward chaining)
        inferences = self._derive_inferences()
        if inferences:
            thought['inferences'] = inferences
        thought['layers']['world_model'] = world_model_data

        # ═══════════════════════════════════════════════
        # LAYER 3: MEMORY (Adaptive Relevance)
        # ═══════════════════════════════════════════════
        episode = self.memory.store_episode(
            content={
                'input': raw_input,
                'concepts': [c['label'] for c in perception_report.get('concepts', [])],
                'relations': len(perception_report.get('relations', [])),
                'type': 'interaction',
            },
            episode_type='interaction',
            importance=0.5 + 0.3 * (perception_report.get('new_concepts', 0) > 0),
            emotional_valence=self._last_emotional_valence,
            surprise=pred_error,
            expected_input=prediction,
        )

        relevant_episodes = self.memory.recall_episodic({'input': raw_input}, top_k=3)
        relevant_rules = self.memory.recall_semantic({'input': raw_input}, top_k=3)

        memory_report = {
            'stored_episode': episode.id,
            'recalled_episodes': [ep.to_dict() for ep in relevant_episodes],
            'recalled_rules': [r.to_dict() for r in relevant_rules],
            'stats': self.memory.get_stats(),
        }
        thought['layers']['memory'] = memory_report
        self.memory.decay_all()

        # ═══════════════════════════════════════════════
        # LAYER 3.5: HDC MEMORY (Conhecimento Compilado)
        # ═══════════════════════════════════════════════
        hdc_context = ""
        if hasattr(self, 'hdc_memory') and self.hdc_memory is not None and self.hdc_memory.matrices:
            try:
                hdc_context = self.hdc_memory.get_context(raw_input, top_k=3)
                hdc_mem_stats = self.hdc_memory.get_stats()
                if hdc_context:
                    # Alimenta working memory com o contexto HDV
                    self.wm.focus(hdc_context, ["hdc_memory"], "hdc", attention=0.6)
            except Exception as e:
                hdc_context = ""
                hdc_mem_stats = {"error": str(e)}
        else:
            hdc_mem_stats = {"sources": [], "total_items": 0}

        thought['layers']['hdc_memory'] = {
            'context': hdc_context[:300],
            'stats': hdc_mem_stats,
        }

        # ═══════════════════════════════════════════════
        # LAYER 4: REASONING 
        # ═══════════════════════════════════════════════
        knowledge_base = self._build_knowledge_base(
            perception_report, world_model_data, relevant_rules, hdc_context
        )
        reasoning_report = self.reasoning.reason(
            perception_report.get('concepts', []),
            knowledge_base
        )
        thought['layers']['reasoning'] = reasoning_report

        # ═══════════════════════════════════════════════
        # LAYER 5: METACOGNITION
        # ═══════════════════════════════════════════════
        # Extrai a melhor similaridade HDV para alimentar a confiança
        hdc_retrieval_score = 0.0
        hdc_ctx_str = thought.get('layers', {}).get('hdc_memory', {}).get('context', '')
        if hdc_ctx_str:
            import re as _re
            scores = _re.findall(r'sim=([\d.]+)', hdc_ctx_str)
            if scores:
                hdc_retrieval_score = min(1.0, max(float(s) for s in scores))

        metacognition_report = self.metacognition.assess(
            perception_report, reasoning_report, memory_report['stats'], world_model_data,
            hdc_retrieval_score=hdc_retrieval_score
        )
        thought['layers']['metacognition'] = metacognition_report

        # Update emotional valence for memory system (used in next cycle)
        meta_emotion = metacognition_report.get('emotional_state', {})
        satisfaction = meta_emotion.get('satisfaction', 0.5)
        confusion = meta_emotion.get('confusion', 0.0)
        self._last_emotional_valence = max(-1.0, min(1.0, satisfaction - confusion * 0.5 - 0.3))
        self.memory.set_emotional_context(self._last_emotional_valence)

        # ═══════════════════════════════════════════════
        # LAYER 6: GOALS
        # ═══════════════════════════════════════════════
        self.goals.update_from_metacognition(metacognition_report)
        current_goal = self.goals.get_next_goal()
        if current_goal:
            current_goal.update_progress(0.1, f"Processed cycle {self.cycle_count}")
        goal_report = self.goals.get_state()
        thought['layers']['goals'] = goal_report

        # ═══════════════════════════════════════════════
        # LAYER 7: LEARNING
        # ═══════════════════════════════════════════════
        learning_report = self.learning.learn(
            perception_report, reasoning_report, metacognition_report, memory_report['stats']
        )
        thought['layers']['learning'] = learning_report

        # Active Inference: suggest next action
        suggested_action = self.actor.suggest_action(
            self.world_model, self.metacognition, self.memory, self.predictor
        )
        thought['action'] = suggested_action

        # ═══════════════════════════════════════════════
        # LAYER 8: MOTOR CORTEX (Mirror Neurons + Play + Articulatory)
        # ═══════════════════════════════════════════════
        self.motor.observe_action(raw_input)

        motor_output = ""
        if self.cycle_count % 5 == 0:
            motor_output = self.motor.play(raw_input)

        # Aplica DIVA learning nos sons produzidos recentemente
        diva_state = {}
        if self.cycle_count % 3 == 0 and self.diva:
            diva_state = self.diva.get_state()

        # Self-hearing: alimenta dorsal stream no DIVA para aprendizado inverso
        if self.cycle_count % 2 == 0 and hasattr(self.motor, '_last_formants'):
            f0, f1, f2, f3 = self.motor._last_formants
            if f0 > 0:
                self.diva.inverse_map(f0, f1, f2, f3)

        # Syllable Oscillator: mantém ritmo interno
        if self.cycle_count % 2 == 0:
            self.syllable_oscillator.tick()

        # BasalGanglia: atualiza ciclo de dopamina tônica
        if self.cycle_count % 10 == 0:
            self.basal_ganglia.tonic_update()

        thought['motor'] = {
            **self.motor.get_state(),
            "latest_output": motor_output,
            "diva": diva_state,
            "premotor": self.premotor.get_state(),
            "basal_ganglia": self.basal_ganglia.get_state(),
            "cerebellum": self.cerebellum.get_state(),
            "oscillator": self.syllable_oscillator.get_state(),
        }

        # Somatosensory: atualiza a propriocepção
        if self.motor._last_babbled_cfg:
            somato = self.somatosensory.sense(
                self.motor._last_babbled_cfg,
                {"f0": 0, "f1": 0, "f2": 0, "f3": 0},  # será preenchido pelo próximo ciclo
            )
            thought['somatosensory'] = somato.get("self_description", "")

        # ═══════════════════════════════════════════════
        # LAYER 9: WORKING MEMORY
        # ═══════════════════════════════════════════════
        self.wm.focus(raw_input, concept_labels, "text",
                       attention=attention_wt)
        self.wm.decay()
        thought['working_memory'] = self.wm.get_state()

        # ═══════════════════════════════════════════════
        # LAYER 10: INTRINSIC RL
        # ═══════════════════════════════════════════════
        reward = self.intrinsic_rl.compute_reward(
            pred_error,
            learning_report.get('compression_progress', [0])[-1] if learning_report.get('compression_progress') else 0,
            learning_report.get('novelty_analysis', {}).get('is_novel', False),
            attention_wt
        )
        state_str = "|".join(concept_labels[:5]) if concept_labels else "empty"
        action_str = suggested_action.get("action", "observar") if suggested_action else "observar"
        self.intrinsic_rl.observe(state_str, action_str, reward, state_str)
        thought['intrinsic_rl'] = self.intrinsic_rl.get_state()

        # ═══════════════════════════════════════════════
        # LAYER 11: CAUSAL REASONING
        # ═══════════════════════════════════════════════
        causal_insight = {}
        if self.cycle_count % 5 == 0 and len(concept_labels) >= 2:
            cause = concept_labels[0]
            effect = concept_labels[-1]
            if cause != effect:
                cf = self.causal.simulate_counterfactual(self.world_model, cause, effect)
                causal_insight['counterfactual'] = cf
                plan = self.causal.plan(self.world_model, concept_labels, effect)
                causal_insight['plan'] = plan
        thought['causal'] = causal_insight

        # ═══════════════════════════════════════════════
        # LAYER 12: 4D WORLD MODEL
        # ═══════════════════════════════════════════════
        if motor_output and "avancar" in motor_output or "recuar" in motor_output or "virar" in motor_output:
            sim = self.world_4d.simulate_step(motor_output, (0, 0, 0))
            thought['world_4d'] = sim
        thought['world_4d_observe'] = self.world_4d.observe((0, 0, 0))

        # ═══════════════════════════════════════════════
        # LAYER 13: EMERGENT LANGUAGE
        # ═══════════════════════════════════════════════
        lang_output = ""
        if self.cycle_count % 7 == 0:
            seq = self.emergent_lang.babble_sequence(length=2)
            self.emergent_lang.learn_from_self_play(seq)
            lang_output = seq
        thought['emergent_language'] = {
            "output": lang_output,
            "state": self.emergent_lang.get_state(),
        }

        # ═══════════════════════════════════════════════
        # NovaCrush: Periodic genetic compression checkpoint (saves genome snapshot)
        if (self.genetic_optimizer and 
            self.cycle_count % self.genome_checkpoint_interval == 0 and
            self.cycle_count > 0):
            try:
                import torch
                substrate = self.perception.neural_substrate
                weights = {
                    'embeddings': substrate.embeddings.data.clone(),
                }
                if hasattr(substrate, 'transition'):
                    weights['transition'] = substrate.transition.weight.data.clone()
                self.last_genome = self.genetic_optimizer.compress_weights(
                    weights, generations=10
                )
            except Exception:
                pass  # Non-critical — genome checkpoint is optional

        # Apply archetypes to compress thinking (padrões → atalhos)
        compressed_concepts = self.archetypes.apply_archetypes(concept_labels)
        if compressed_concepts != concept_labels:
            thought['archetype_compression'] = {
                'before': len(concept_labels),
                'after': len(compressed_concepts),
                'ratio': len(compressed_concepts) / max(1, len(concept_labels)),
            }

        # ═══════════════════════════════════════════════
        # SYNTHESIS (Generating Native Signal Array)
        # ═══════════════════════════════════════════════
        response = self._synthesize_response(thought)
        thought['response'] = response
        thought['cycle_time_ms'] = (time.time() - cycle_start) * 1000

        self.thought_history.append({
            'cycle': self.cycle_count,
            'input': raw_input[:100],
            'response_preview': response['text'][:200],
            'confidence': metacognition_report.get('confidence', {}).get('overall', 0),
            'cycle_time_ms': thought['cycle_time_ms'],
        })
        if len(self.thought_history) > 50:
            self.thought_history = self.thought_history[-50:]

        self.cleanup_counter += 1
        if self.cleanup_counter >= self.cleanup_interval:
            self._resource_cleanup()
            self.cleanup_counter = 0

        return thought

    def get_causal_chains_with_labels(self, min_length: int = 2) -> List[List[str]]:
        """Causal chains with concept labels instead of hash IDs."""
        chains = self.world_model.get_causal_chains(min_length)
        nodes = self.world_model.nodes
        labeled = []
        for chain in chains:
            labeled_chain = []
            for cid in chain:
                node = nodes.get(cid)
                labeled_chain.append(node.label if node else cid[:8])
            labeled.append(labeled_chain)
        return labeled

    SEMANTIC_PATTERNS = [
        (r'(?:my )?(\w+) is named (\w+)', 'named'),
        (r'(i) live in (\w+(?:\s+\w+)?)', 'lives_in'),
        (r'(i) like (\w+(?:\s+\w+)?)', 'likes'),
        (r'(i) am learning about (.+)', 'learning_about'),
        (r'(i) am studying (.+)', 'learning_about'),
        (r'(\w+) is the (\w+) of (\w+)', lambda g: f'{g[1]}_of'),
        (r'(\w+) (?:is|are) (?:located )?in (\w+(?:\s+\w+)?)', 'in'),
        (r'(\w+) is an? (\w+)', 'is_a'),  # matches both "is a" and "is an"
        (r'(\w+) has (?:a |an |the )?(\w+)', 'has'),
        (r'(\w+) have (?:a |an |the )?(\w+)', 'has'),
    ]

    def _concept_id_for_label(self, label: str) -> str:
        """Find concept ID by label from the perception registry."""
        for cid, c in self.perception.concept_registry.items():
            if c.label.lower() == label.lower():
                return cid
        return None

    def _extract_semantic_facts(self, text: str) -> List[Dict]:
        """Extract semantic triples from declarative statements (not questions)."""
        text_lower = text.lower().strip()
        first_word = text_lower.split()[0] if text_lower else ''
        if first_word in ('what','why','how','when','where','who','which','do','does','did','is','are','can','tell','define','explain'):
            return []
        
        # 1. Extract cultural/cognitive rules (IF, CAUSE, PURPOSE)
        facts = self.rule_compiler.compile_text(text)
        
        # 2. Extract standard semantic triples (is_a, has, in)
        for pattern, relation in self.SEMANTIC_PATTERNS:
            m = re.search(pattern, text_lower)
            if m:
                groups = m.groups()
                if callable(relation):
                    rel = relation(groups)
                else:
                    rel = relation
                if len(groups) == 2:
                    subj, obj = groups
                elif len(groups) == 3:
                    subj, obj = groups[0], groups[2]
                else:
                    continue
                # Clean up determiners
                subj = subj.replace('my ', '').strip()
                obj = obj.strip()
                if subj and obj:
                    display = self._format_fact(subj, rel, obj)
                facts.append({
                        'subject': subj, 'relation': rel, 'object': obj,
                        'display': display,
                        'source_text': text[:100],
                    })
                # Secondary extraction: "X is named Y" → is_a(Y, X)
                if rel == 'named' and subj and obj:
                    secondary_display = self._format_fact(obj, 'is_a', subj)
                    facts.append({
                        'subject': obj, 'relation': 'is_a', 'object': subj,
                        'display': secondary_display,
                        'source_text': f'(inferred from: {text[:80]})',
                    })
        return facts

    def _detect_contradictions(self, new_facts: List[Dict], thought: Dict):
        """Check if any new fact contradicts an existing semantic rule."""
        for nf in new_facts:
            ns = nf.get('subject', '').lower()
            nr = nf.get('relation', '')
            no = nf.get('object', '').lower()
            if not ns or not nr or not no:
                continue
            for i, existing in enumerate(self.memory.semantic):
                c = existing.content
                if not isinstance(c, dict):
                    continue
                es = c.get('subject', '').lower()
                er = c.get('relation', '')
                eo = c.get('object', '').lower()
                if es == ns and er == nr and eo != no:
                    thought['contradiction'] = {
                        'old': c.get('display', f'{es} {er} {eo}'),
                        'new': nf.get('display', f'{ns} {nr} {no}'),
                        'subject': ns,
                        'relation': nr,
                    }
                    self.memory.semantic.pop(i)
                    break

    def _check_pending_goals(self, input_text: str, new_facts: List[Dict]) -> str:
        """Check if input satisfies any unanswered question. Returns goal text if yes."""
        if not self.unanswered_questions:
            return None
        input_keywords = set(self._content_words(input_text))
        # Also collect keywords from extracted facts
        for fact in new_facts:
            input_keywords.update(self._content_words(fact.get('subject', '') + ' ' + fact.get('object', '')))

        best_match = None
        best_ratio = 0
        for qi, q in enumerate(self.unanswered_questions):
            qkw = set(q['keywords'])
            if not qkw:
                continue
            overlap = len(input_keywords & qkw)
            ratio = overlap / len(qkw)
            if ratio >= 0.3 and ratio > best_ratio:
                best_ratio = ratio
                best_match = qi

        if best_match is not None:
            q = self.unanswered_questions.pop(best_match)
            topic = q['text'][:60]
            goal = self.goals.add_goal(
                f"Learned about: {topic}", goal_type='epistemic', priority=0.7
            )
            self.goals.complete_goal(goal.id, result=input_text[:100])
            return topic
        return None

    @staticmethod
    def _singular(w: str) -> str:
        """Crude singularization for inference matching."""
        w = w.lower().strip('.')
        if w.endswith('ies'):
            return w[:-3] + 'y'
        if w.endswith('ves'):
            return w[:-3] + 'f'
        if w.endswith('es') and len(w) > 4:
            return w[:-2]
        if w.endswith('s') and len(w) > 3 and not w.endswith('ss'):
            return w[:-1]
        return w

    # Relations whose chains compose transitively: a R b, b R c  ⇒  a R c
    TRANSITIVE_RELATIONS = frozenset(('is_a', 'in', 'part_of'))
    # Properties inherited down an is_a chain: x is_a y, y R z  ⇒  x R z
    INHERITABLE_RELATIONS = frozenset(('has', 'likes', 'lives_in', 'in'))
    MAX_INFERENCE_ITERS = 25

    def _derive_inferences(self) -> List[str]:
        """Derive new facts by computing the deductive closure of known facts.

        Generalizes beyond hand-written rule combos: it repeatedly applies
        (1) transitive composition for transitive relations and
        (2) property inheritance down is_a chains, until a fixpoint is reached.
        This lets arbitrarily long chains resolve (e.g. Rex→dog→mammal→animal).
        Surface forms are preserved; singular/plural is reconciled only for the
        chaining test, so 'Paris' is never mangled to 'Pari'.
        """
        # Gather base facts as (subject, relation, object) surface triples.
        base = set()
        for rule in self.memory.semantic:
            c = rule.content
            if isinstance(c, dict) and 'relation' in c:
                s = c.get('subject', '').lower().strip().strip('.')
                o = c.get('object', '').lower().strip().strip('.')
                r = c.get('relation', '')
                if s and o and r:
                    base.add((s, r, o))
        if len(base) < 2:
            return []

        sing = self._singular

        def chain(a: str, b: str) -> bool:
            """Can the object of one fact bind to the subject of another?"""
            return a == b or sing(a) == sing(b)

        facts = set(base)
        for _ in range(self.MAX_INFERENCE_ITERS):
            new = set()
            for (s1, r1, o1) in facts:
                if r1 in self.TRANSITIVE_RELATIONS:
                    for (s2, r2, o2) in facts:
                        if r2 == r1 and chain(o1, s2):
                            new.add((s1, r1, o2))
                if r1 == 'is_a':
                    for (s2, r2, o2) in facts:
                        if r2 in self.INHERITABLE_RELATIONS and chain(o1, s2):
                            new.add((s1, r2, o2))
                if r1 == 'capital_of':
                    new.add((o1, 'has_capital', s1))
            if new <= facts:
                break  # fixpoint reached
            facts |= new

        derived = facts - base
        if not derived:
            return []

        # Avoid restoring inferences already in semantic memory.
        already = {
            (c.get('subject', '').lower().strip('.'),
             c.get('relation', ''),
             c.get('object', '').lower().strip('.'))
            for c in (r.content for r in self.memory.semantic)
            if isinstance(c, dict) and c.get('source_text') == '(inferred)'
        }

        inferences = []
        for (subj, rel, obj) in sorted(derived):
            if (subj, rel, obj) in already:
                continue
            cont = {'subject': subj, 'relation': rel, 'object': obj,
                    'display': self._format_fact(subj, rel, obj), 'source_text': '(inferred)'}
            rule = self.memory.store_semantic_rule(cont, source='inference')
            rule.confidence = 0.5
            inferences.append(self._format_fact(subj, rel, obj))
        return inferences

    def _resource_cleanup(self):
        if len(self.perception.concept_registry) > self.max_concepts:
            sorted_c = sorted(
                self.perception.concept_registry.values(),
                key=lambda c: c.access_count + c.activation
            )
            for c in sorted_c[:len(sorted_c) // 4]:
                if c.id in self.perception.concept_registry:
                    del self.perception.concept_registry[c.id]

        if len(self.memory.episodic) > self.max_episodes:
            self.memory.episodic = sorted(
                self.memory.episodic, key=lambda e: e.activation * e.importance
            )[-self.max_episodes:]

        if len(self.memory.semantic) > 2000:
            self.memory.semantic = sorted(
                self.memory.semantic, key=lambda r: r.confidence * r.activation
            )[-2000:]

        if len(self.memory.procedural) > 200:
            self.memory.procedural = self.memory.procedural[-200:]

        if len(self.wm.slots) > 0:
            self.wm.decay()

        import gc
        gc.collect()
        if __import__('torch').cuda.is_available():
            __import__('torch').cuda.empty_cache()

    def _build_knowledge_base(self, perception: Dict, world_model: Dict,
                               semantic_rules: List, hdc_context: str = "") -> Dict:
        return {
            'concepts': self.perception.get_all_concepts(),
            'causal_edges': world_model.get('edges', []),
            'causal_chains': self.world_model.get_causal_chains(),
            'semantic_rules': [r.to_dict() for r in semantic_rules],
            'hdc_context': hdc_context,
        }

    STOPWORDS = {'the','a','an','is','are','was','were','be','been','has','have',
                 'had','do','does','did','will','would','can','could','may','might',
                 'what','why','how','when','where','who','which','this','that','its',
                 'it','i','you','he','she','we','they','me','him','not','no','yes','to',
                 'of','in','on','for','with','by','as','at','from'}

    QWORDS = frozenset(('what','why','how','when','where','who','which','do','does','did','is','are','can','tell','define','explain'))

    @staticmethod
    def _is_question(text: str) -> bool:
        words = text.strip().lower().split()
        return bool(words and words[0] in NovaMind.QWORDS)

    def _content_words(self, text: str) -> List[str]:
        return [w for w in text.lower().split()
                if w not in self.STOPWORDS and len(w) > 2]

    SEMANTIC_RECALL_THRESHOLD = 0.35  # min cosine to accept a semantic match

    def _find_best_recall(self, input_content: set, recalled_inputs: list, input_text: str) -> str:
        """Find the best matching recall from episodic memory, skipping questions.

        When a semantic encoder is attached, rank candidates by embedding cosine
        similarity (captures meaning across paraphrases). Otherwise fall back to
        content-word overlap.
        """
        # Candidates: stored declarative statements (skip questions, skip self)
        candidates = [
            ri for ri in recalled_inputs
            if ri and ri != input_text
            and not (ri.strip().lower().split()[:1] and ri.strip().lower().split()[0] in self.QWORDS)
        ]
        if not candidates:
            return None

        encoder = getattr(self.memory, 'encoder', None)
        if encoder is not None:
            q_vec = encoder.encode_one(input_text)
            cand_vecs = encoder.encode(candidates)
            sims = cand_vecs @ q_vec
            best_i = int(sims.argmax())
            if float(sims[best_i]) >= self.SEMANTIC_RECALL_THRESHOLD:
                return candidates[best_i]
            return None

        # Fallback: content-word overlap
        best_recall, best_overlap = None, 0
        for ri in candidates:
            overlap = len(input_content & set(self._content_words(ri)))
            if overlap > best_overlap:
                best_overlap = overlap
                best_recall = ri
        if best_recall and best_overlap >= 1:
            return best_recall
        return None

    def _format_fact(self, subj: str, rel: str, obj: str) -> str:
        obj = obj.strip('.')
        display_rel = rel.replace('_', ' ')
        subj_disp = subj.title() if subj != 'i' else 'I'
        if rel in ('is_a',):
            article = 'an' if obj[0] in 'aeiou' else 'a'
            return f"{subj_disp} is {article} {obj}."
        if rel == 'lives_in':
            return f"{subj_disp} live in {obj}." if subj == 'i' else f"{subj_disp} lives in {obj}."
        if rel == 'capital_of':
            return f"{subj_disp} is the capital of {obj}."
        if rel == 'learning_about':
            return f"I am learning about {obj}." if subj == 'i' else f"{subj_disp} is learning about {obj}."
        if rel == 'named':
            return f"{subj_disp} is named {obj}."
        if rel == 'likes':
            return f"I like {obj}." if subj == 'i' else f"{subj_disp} likes {obj}."
        if rel == 'has':
            return f"{subj_disp} has {obj}."
        if rel == 'in':
            return f"{subj_disp} is in {obj}."
        if rel == 'has_capital':
            return f"{subj_disp} has capital {obj}."
        return f"{subj_disp} {display_rel} {obj}."

    def _match_semantic_rule(self, input_text: str, input_content: set) -> str:
        """Check if a question matches a stored semantic fact. Returns best match."""
        q_words = input_text.lower().split()
        q_text = input_text.lower()
        has_i = 'i' in q_words or 'my' in q_words

        best_score = -1
        best_text = None

        for rule in self.memory.semantic:
            cont = rule.content
            if not isinstance(cont, dict) or 'relation' not in cont:
                continue
            rel = cont['relation']
            subj = cont.get('subject', '').lower()
            obj = cont.get('object', '').strip('.').lower()

            score = 0
            rel_word = rel.replace('_', ' ')

            # Synonym map for relation matching
            SYN_REL = {'study': 'learning', 'studying': 'learning', 'enjoy': 'like',
                       'liked': 'like', 'enjoys': 'like', 'lives': 'live',
                       'living': 'live', 'learn': 'learning', 'name': 'named',
                       'names': 'named', 'called': 'named'}

            # Check if any question word is semantically related to the relation
            for w in q_words:
                if len(w) > 2 and rel.startswith(w):
                    score += 3
                elif len(w) > 2 and w in rel:
                    score += 2
                elif w in SYN_REL and SYN_REL[w] in rel:
                    score += 3
            # Relation appears in question text
            if rel_word in q_text:
                score += 4
            # Object appears in question
            if obj in q_words or obj in input_content:
                score += 2
            # Subject matches self-reference
            if subj == 'i' and has_i:
                score += 1

            if score > best_score:
                best_score = score
                best_text = cont.get('display') or self._format_fact(subj, rel, obj)

        if best_score >= 1:
            return best_text

        # Fallback: NCD-based semantic recall (only if content words overlap)
        query_dict = {'input': input_text}
        recalled_rules = self.memory.recall_semantic(query_dict, top_k=1)
        for rule in recalled_rules:
            cont = rule.content
            if isinstance(cont, dict) and 'relation' in cont:
                subj = cont.get('subject', '').lower()
                rel = cont.get('relation', '')
                obj = cont.get('object', '').lower()
                if subj and obj:
                    # Only use if subj or obj appears in input content words
                    if subj in input_content or obj in input_content:
                        return cont.get('display') or self._format_fact(cont['subject'], rel, obj)
        return None

    def _meta_str(self, thought: Dict, confidence: float, is_novel: bool) -> str:
        parts = [f"cycle #{thought['cycle']}"]
        if is_novel:
            parts.append("NEW")
        parts.append(f"{confidence:.0%}")
        return f" [{' | '.join(parts)}]"

    RELATION_WORDS = frozenset(('named','called','known','has','have','had','is','are','was',
                                 'were','been','become','became','seems','appears','said',
                                 'told','asked','answered','replied','thought','felt',
                                 'want','wants','needs','need','can','could','will','would',
                                 'shall','should','may','might','must','does','did','doing',
                                 'made','make','makes','takes','take','took','given','gives',
                                 'give','get','gets','got','go','goes','went','come','came',
                                 'see','saw','seen','use','used','using','say','says',
                                 'live','lives','living','like','likes','liked','enjoy',
                                 'enjoys','enjoyed','love','loves','loved','hate','hates',
                                 'learning','learn','learns','learned','studying','study',
                                 'studies','studied','teach','teaches','teaching','taught',
                                 'about','with','without','into','onto','upon','from',
                                 'and','or','but','if','then','else','nor','yet','so',
                                 'because','since','while','during','before','after'))

    def _maybe_ask_question(self, thought: Dict, response_text: str) -> str:
        """Append a follow-up question if cognitive state suggests curiosity."""
        meta = thought['layers'].get('metacognition', {})
        emotion = meta.get('emotional_state', {})
        confusion = emotion.get('confusion', 0)
        perception = thought['layers'].get('perception', {})
        concepts = perception.get('concepts', [])

        # Don't add questions to question responses (recursive)
        input_text = thought.get('input', '')
        if self._is_question(input_text):
            return response_text

        # High confusion: ask for clarification
        if confusion > 0.5:
            return response_text + " I'm a bit confused. Could you explain more?"

        # Novel concept: ask to learn (skip relation/grammar words)
        for c in concepts[:5]:
            label = c.get('label', '')
            is_new = c.get('access_count', 1) == 0
            if (len(label) >= 3 and label not in self.STOPWORDS
                    and label not in self.RELATION_WORDS and is_new):
                known = any(
                    isinstance(r.content, dict) and
                    (label in r.content.get('subject','') or label in r.content.get('object',''))
                    for r in self.memory.semantic
                )
                if not known:
                    return response_text + f" Tell me about {label}."

        return response_text

    def _synthesize_response(self, thought: Dict) -> Dict:
        perception_data = thought['layers'].get('perception', {})
        memory_data = thought['layers'].get('memory', {})
        metacognition = thought['layers'].get('metacognition', {})
        learning = thought['layers'].get('learning', {})
        reasoning = thought['layers'].get('reasoning', {})

        input_text = thought.get('input', '')
        concepts = [c.get('label', '') for c in perception_data.get('concepts', [])]
        confidence = metacognition.get('confidence', {}).get('overall', 0.5)
        is_novel = learning.get('novelty_analysis', {}).get('is_novel', False)
        consensus = reasoning.get('consensus', {})
        is_question = self._is_question(input_text)

        recruited = memory_data.get('recalled_episodes', [])
        recalled_inputs = [ep.get('content', {}).get('input', '') for ep in recruited if ep.get('content')]
        input_content = set(self._content_words(input_text))

        # For questions: inferred rules → semantic rules → episodic recall → "I don't know"
        if is_question:
            # Check inferred rules first (subject+object both in input)
            for rule in self.memory.semantic:
                c = rule.content
                if isinstance(c, dict) and c.get('source_text') == '(inferred)':
                    subj = c.get('subject','').lower()
                    obj = c.get('object','').lower()
                    if subj in input_content and obj in input_content:
                        text = c.get('display') or self._format_fact(c.get('subject',''), c.get('relation',''), c.get('object',''))
                        text += self._meta_str(thought, confidence, is_novel)
                        return {'text': text, 'confidence': confidence,
                                'primary_paradigm': consensus.get('primary_paradigm', 'unknown'), 'novel_input': is_novel}
            # Then check extracted semantic rules (precise match)
            sem_match = self._match_semantic_rule(input_text, input_content)
            if sem_match:
                text = sem_match
                text += self._meta_str(thought, confidence, is_novel)
                return {'text': text, 'confidence': confidence,
                        'primary_paradigm': consensus.get('primary_paradigm', 'unknown'), 'novel_input': is_novel}
            # Then episodic recall as fallback
            best_recall = self._find_best_recall(input_content, recalled_inputs, input_text)
            if best_recall:
                text = best_recall
                text += self._meta_str(thought, confidence, is_novel)
                return {'text': text, 'confidence': confidence,
                        'primary_paradigm': consensus.get('primary_paradigm', 'unknown'), 'novel_input': is_novel}
        else:
            # Non-questions: episodic recall first, then novelty
            best_recall = self._find_best_recall(input_content, recalled_inputs, input_text)
            if best_recall:
                text = f"I remember: {best_recall}"
                text += self._meta_str(thought, confidence, is_novel)
                return {'text': text, 'confidence': confidence,
                        'primary_paradigm': consensus.get('primary_paradigm', 'unknown'), 'novel_input': is_novel}

        # POLICY 3: Question without recall/semantic → admit uncertainty, create knowledge goal
        if is_question:
            topic_keywords = self._content_words(input_text)
            if topic_keywords and len(topic_keywords) >= 1:
                self.unanswered_questions.append({
                    'text': input_text[:100],
                    'keywords': topic_keywords,
                    'timestamp': time.time(),
                })
            text = "I don't know." + self._meta_str(thought, confidence, is_novel)
            text = self._maybe_ask_question(thought, text)
            return {'text': text, 'confidence': confidence,
                    'primary_paradigm': consensus.get('primary_paradigm', 'unknown'), 'novel_input': is_novel}

        # Check if a goal was satisfied this cycle
        goal_topic = thought.get('goal_satisfied')
        if goal_topic:
            is_novel = False  # It's satisfying a goal, not just novelty
            text = f"I notice new information. This tells me about {goal_topic}, which I was curious about."
            text += self._meta_str(thought, confidence, is_novel)
            return {'text': text, 'confidence': confidence,
                    'primary_paradigm': consensus.get('primary_paradigm', 'unknown'), 'novel_input': False}

        # POLICY 4: Novel input → acknowledge (include inferences if any)
        if is_novel:
            contradiction = thought.get('contradiction')
            if contradiction:
                text = (f"I notice new information. I previously thought "
                        f"{contradiction['old']}, but now I understand {contradiction['new']}.")
            else:
                text = "I notice new information."
            inferences = thought.get('inferences', [])
            if inferences:
                text += f" I infer: {inferences[0]}"
            text += self._meta_str(thought, confidence, is_novel)
            text = self._maybe_ask_question(thought, text)
            return {'text': text, 'confidence': confidence,
                    'primary_paradigm': consensus.get('primary_paradigm', 'unknown'), 'novel_input': is_novel}

        # POLICY 4b: Contradiction detected (even if not novel → e.g. known concepts change)
        contradiction = thought.get('contradiction')
        if contradiction:
            text = (f"I notice new information. I previously thought "
                    f"{contradiction['old']}, but now I understand {contradiction['new']}.")
            text += self._meta_str(thought, confidence, is_novel)
            return {'text': text, 'confidence': confidence,
                    'primary_paradigm': consensus.get('primary_paradigm', 'unknown'), 'novel_input': is_novel}

        # POLICY 5: Clear symbolic reasoning consensus → reasoned response (neural/geometric are too noisy)
        paradigm = consensus.get('primary_paradigm')
        conclusion = consensus.get('conclusion', '')
        if paradigm == 'symbolic' and confidence > 0.5 and len(conclusion) >= 15:
            text = conclusion[:200]
            text += self._meta_str(thought, confidence, is_novel)
            return {'text': text, 'confidence': confidence,
                    'primary_paradigm': paradigm, 'novel_input': is_novel}

        # POLICY 5: NGram fallback with template safety net
        if concepts:
            base = self.language.generate_fluent_thought(
                input_text, concepts[:5], semantic_skeleton=None,
                confidence=confidence, is_novel=False
            )
        else:
            base = "Processing your input."
        base += self._meta_str(thought, confidence, is_novel)
        return {'text': base, 'confidence': confidence,
                'primary_paradigm': paradigm or 'unknown', 'novel_input': is_novel}

    def get_full_state(self) -> Dict:
        """Return the complete mind state for the dashboard."""
        state = {
            'cycle_count': self.cycle_count,
            'uptime_seconds': time.time() - self.start_time,
            'perception': {
                'total_concepts': len(self.perception.concept_registry),
                'all_concepts': self.perception.get_all_concepts(),
                'novacrush_enabled': getattr(self.perception, 'novacrush_enabled', False),
            },
            'world_model': self.world_model.get_graph_data(),
            'memory': {
                'stats': self.memory.get_stats(),
                'contents': self.memory.get_all_memories(),
            },
            'reasoning': {
                'paradigm_weights': self.reasoning.paradigm_weights,
                'history': self.reasoning.reasoning_history[-10:],
            },
            'metacognition': self.metacognition.get_full_state(),
            'goals': self.goals.get_state(),
            'learning': self.learning.get_stats(),
            'predictive_coding': self.predictor.get_state(),
            'active_inference': self.actor.get_state(),
            'motor_cortex': self.motor.get_state(),
            'visual_cortex': self.visual_cortex.get_state(),
            'auditory_cortex': self.auditory_cortex.get_state(),
            'working_memory': self.wm.get_state(),
            'intrinsic_rl': self.intrinsic_rl.get_state(),
            'causal_reasoning': self.causal.get_state(),
            'world_4d': self.world_4d.get_state(),
            'emergent_language': self.emergent_lang.get_state(),
            'diva_controller': self.diva.get_state(),
            'somatosensory_cortex': self.somatosensory.get_state(),
            'thought_history': self.thought_history,
        }

        # NovaCrush stats
        if NOVACRUSH_AVAILABLE and hasattr(self.perception, 'novacrush_enabled'):
            if self.perception.novacrush_enabled:
                substrate = self.perception.neural_substrate
                state['novacrush'] = {
                    'compression': substrate.get_compression_stats(),
                    'forward_forward': substrate.get_ff_stats(),
                    'last_genome_bytes': len(self.last_genome.serialize()) if self.last_genome else None,
                }

        return state
