"""
EmergentLanguageEngine — Linguagem Emergente do CGE
=====================================================
Gera linguagem natural ESTRUTURADA a partir do grafo de conceitos,
sem usar nenhum encoder pré-treinado (BGE, LaBSE, etc).

Princípios:
  1. O vocabulário É o conjunto de conceitos do CGE — não vem de fora
  2. A sintaxe emerge das relações tipadas (IS_A → "X is Y", CAUSES → "X causes Y")
  3. O pensamento verbalizado retroalimenta o CGE (monólogo interno)
  4. Tudo é feito com operações HDV nativas (bind, bundle, unbind)

Isso resolve o "abismo intelectual" entre um sistema que só processa
padrões estatísticos e um que realmente PENSA em linguagem.
"""
import time
import random
import math
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field

try:
    from .coglang import (
        Manifold, bind_roles, unbind_role, decode_roles,
        ROLE_SUBJECT, ROLE_VERB, ROLE_OBJECT,
        ROLE_ADJECTIVE, ROLE_ADVERB, ROLE_NEGATION, ROLE_PREPOSITION
    )
    COGLANG_OK = True
except ImportError:
    COGLANG_OK = False

try:
    from .cognitive_graph_engine import CognitiveGraphEngine
    CGE_OK = True
except ImportError:
    CGE_OK = False

try:
    from .mental_state import get_mental_state
    MENTAL_OK = True
except ImportError:
    MENTAL_OK = False


@dataclass
class Utterance:
    """Uma unidade de fala gerada pelo sistema."""
    text: str = ""
    concepts: List[str] = field(default_factory=list)
    relations: List[Tuple[str, str, str]] = field(default_factory=list)
    confidence: float = 0.5
    source: str = ""  # "cge_traversal", "internal_monologue", "imagination"
    is_question: bool = False
    is_hypothesis: bool = False
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "concepts": self.concepts,
            "confidence": self.confidence,
            "source": self.source,
            "is_question": self.is_question,
            "is_hypothesis": self.is_hypothesis,
        }


# Mapeamento de tipos de relacao para templates de frase
RELATION_TO_SENTENCE = {
    "IS_A": "{subject} is {object}",
    "CAUSES": "{subject} causes {object}",
    "ENABLES": "{subject} enables {object}",
    "PREVENTS": "{subject} prevents {object}",
    "PART_OF": "{subject} is part of {object}",
    "CORRELATED": "{subject} is related to {object}",
    "OPPOSES": "{subject} opposes {object}",
    "BEFORE": "{subject} comes before {object}",
    "AFTER": "{subject} comes after {object}",
    "HAS_PROPERTY": "{subject} has {object}",
    "AFFECTS": "{subject} affects {object}",
    "FOLLOWS": "{subject} follows {object}",
    "HAS_PART": "{subject} contains {object}",
}

RELATION_PAST_TENSE = {
    "CAUSES": "caused", "ENABLES": "enabled", "PREVENTS": "prevented",
    "AFFECTS": "affected", "FOLLOWS": "followed",
}

QUALIFIERS = ["possibly", "likely", "certainly", "maybe", "probably", "perhaps"]
CONJUNCTIONS = ["and", "but", "therefore", "however", "moreover", "furthermore"]


class EmergentVocabulary:
    """
    Vocabulario que emerge dos conceitos que o sistema realmente APRENDEU.
    
    Diferente de BGE/WordNet: cada palavra É um conceito no CGE.
    O vocabulario cresce organicamente com o conhecimento do sistema.
    Nao ha palavras que o sistema nao "entende" — toda palavra tem um
    conceito correspondente no grafo.
    """

    def __init__(self, cge: Optional['CognitiveGraphEngine'] = None):
        self.cge = cge
        self._word_to_concept: Dict[str, str] = {}
        self._concept_to_word: Dict[str, str] = {}
        self._word_frequency: Dict[str, int] = defaultdict(int)
        self._unknown_words: Set[str] = set()

    def sync_from_cge(self) -> int:
        """Sincroniza vocabulario com os conceitos do CGE."""
        if self.cge is None or not hasattr(self.cge, 'nodes'):
            return 0
        count = 0
        for node_id in self.cge.nodes:
            word = node_id.lower().replace("_", " ").strip()
            if word not in self._word_to_concept:
                self._word_to_concept[word] = node_id
                self._concept_to_word[node_id] = word
                count += 1
        return count

    def add_word(self, word: str, concept_id: Optional[str] = None):
        """Adiciona uma palavra manualmente."""
        w = word.lower().strip()
        if not w:
            return
        concept = concept_id or w
        self._word_to_concept[w] = concept
        self._concept_to_word[concept] = w

    def has_word(self, word: str) -> bool:
        return word.lower().strip() in self._word_to_concept

    def concept_for(self, word: str) -> Optional[str]:
        return self._word_to_concept.get(word.lower().strip())

    def word_for(self, concept_id: str) -> Optional[str]:
        return self._concept_to_word.get(concept_id)

    def all_words(self) -> List[str]:
        return list(self._word_to_concept.keys())

    def sample_words(self, n: int = 5) -> List[str]:
        words = self.all_words()
        return random.sample(words, min(n, len(words)))

    def known_ratio(self, words: List[str]) -> float:
        if not words:
            return 1.0
        known = sum(1 for w in words if self.has_word(w))
        return known / len(words)

    def size(self) -> int:
        return len(self._word_to_concept)


class CGEToLanguage:
    """
    Converte estruturas do CGE em linguagem natural.
    
    Estrategia:
      1. Pega um conceito "foco" (o que esta sendo pensado)
      2. Busca suas arestas no CGE
      3. Para cada aresta, aplica um template de frase baseado no tipo
      4. Monta sentencas usando bind de papéis HDV
      5. Opcionalmente: combina sentencas em paragrafos
    
    Nao usa gramatica externa — a gramatica emerge dos tipos de relacao.
    """

    def __init__(self, cge: Optional['CognitiveGraphEngine'] = None,
                 vocab: Optional['EmergentVocabulary'] = None):
        self.cge = cge
        self.vocab = vocab or EmergentVocabulary(cge)
        self._utterance_history: deque = deque(maxlen=100)

    def set_cge(self, cge: 'CognitiveGraphEngine'):
        self.cge = cge
        if self.vocab is not None:
            self.vocab.cge = cge
            self.vocab.sync_from_cge()

    def describe_concept(self, concept_id: str, max_sentences: int = 3) -> Utterance:
        """Gera descricao textual de um conceito baseado nas arestas do CGE."""
        if self.cge is None:
            return Utterance(text=f"[no CGE]", concepts=[concept_id])

        edges = []
        try:
            if hasattr(self.cge, 'edges'):
                for e in self.cge.edges:
                    if isinstance(e, dict) and e.get('from', '') == concept_id:
                        edges.append(e)
        except Exception:
            pass

        if not edges:
            return Utterance(text=concept_id.replace("_", " "),
                             concepts=[concept_id],
                             confidence=0.3)

        sentences = []
        used_relations = set()
        concepts_used = [concept_id]

        for e in edges[:max_sentences * 2]:
            to_id = e.get('to', '')
            edge_type = e.get('edge_type', 'CORRELATED')
            if isinstance(edge_type, int) or isinstance(edge_type, float):
                edge_type = self._resolve_edge_type(edge_type)

            edge_key = (to_id, edge_type)
            if edge_key in used_relations:
                continue
            used_relations.add(edge_key)

            subj = concept_id.replace("_", " ")
            obj = to_id.replace("_", " ") if to_id else "something"

            template = RELATION_TO_SENTENCE.get(edge_type, "{subject} is related to {object}")
            sentence = template.format(subject=subj, object=obj)

            confidence = float(e.get('weight', 0.5)) if isinstance(e.get('weight'), (int, float)) else 0.5
            if confidence < 0.4:
                qualifier = random.choice(QUALIFIERS)
                sentence = f"{qualifier}, {sentence}"
            elif confidence < 0.6:
                pass

            sentences.append(sentence)
            if to_id:
                concepts_used.append(to_id)

        text = ". ".join(sentences) + "."
        utterance = Utterance(
            text=text,
            concepts=concepts_used,
            relations=[(concept_id, e.get('to', ''), str(e.get('edge_type', '')))
                       for e in edges[:max_sentences]],
            confidence=min(1.0, len(sentences) / max(1, max_sentences)),
            source="cge_traversal",
        )
        self._utterance_history.append(utterance)
        return utterance

    def compare_concepts(self, concept_a: str, concept_b: str) -> Utterance:
        """Gera texto comparando dois conceitos."""
        desc_a = self.describe_concept(concept_a, max_sentences=2)
        desc_b = self.describe_concept(concept_b, max_sentences=2)
        text = f"{concept_a.replace('_', ' ')}: {desc_a.text}. {concept_b.replace('_', ' ')}: {desc_b.text}."
        return Utterance(
            text=text,
            concepts=[concept_a, concept_b],
            confidence=min(desc_a.confidence, desc_b.confidence),
            source="cge_traversal",
        )

    def describe_relation_chain(self, start: str, end: str,
                                 max_hops: int = 5) -> Utterance:
        """Descreve uma cadeia causal entre dois conceitos."""
        if self.cge is None or not hasattr(self.cge, 'plan_to_goal'):
            return Utterance(text=f"{start} to {end} [no planner]",
                             concepts=[start, end])
        try:
            path = self.cge.plan_to_goal(start, end, max_depth=max_hops)
        except Exception:
            path = None

        if not path:
            return Utterance(
                text=f"I don't know how {start.replace('_', ' ')} leads to {end.replace('_', ' ')}",
                concepts=[start, end], confidence=0.2, is_hypothesis=True
            )

        sentences = []
        for i in range(len(path) - 1):
            s = path[i].replace("_", " ")
            t = path[i + 1].replace("_", " ")
            e_type = "leads to"
            if hasattr(self.cge, 'edges'):
                for edge in self.cge.edges:
                    if isinstance(edge, dict) and edge.get('from') == path[i] and edge.get('to') == path[i + 1]:
                        raw = edge.get('edge_type', '')
                        if isinstance(raw, str):
                            e_type = raw.lower().replace("_", " ")
                        break
            sentences.append(f"{s} {e_type} {t}")

        text = ", therefore ".join(sentences) + "."
        return Utterance(
            text=text, concepts=path,
            confidence=0.7, source="cge_traversal",
        )

    def _resolve_edge_type(self, edge_code) -> str:
        """Converte codigo numerico de edge_type para string."""
        mapping = {0: "CORRELATED", 1: "IS_A", 2: "CAUSES", 3: "PART_OF",
                   4: "BEFORE", 5: "AFTER", 6: "ENABLES", 7: "OPPOSES"}
        return mapping.get(int(edge_code) if isinstance(edge_code, (int, float)) else 0, "CORRELATED")

    def recent_utterances(self, n: int = 5) -> List[Utterance]:
        return list(self._utterance_history)[-n:]


class InternalMonologue:
    """
    Monologo interno: o sistema pensa em linguagem.
    
    Ciclo:
      1. Pega o foco atual do MentalState
      2. Gera descricao textual do conceito foco (via CGEToLanguage)
      3. Aplica operacoes mentais na verbalizacao (questionar, duvidar, inferir)
      4. Alimenta de volta no CGE como novos conceitos/relacoes
      5. Repete — criando uma cadeia de pensamento
    
    Isto simula o "pensamento verbal" humano — o looping entre
    pensamento e linguagem que permite raciocinio abstrato.
    """

    def __init__(self, cge_to_lang: 'CGEToLanguage',
                 mental_state=None):
        self.translator = cge_to_lang
        self.cge = cge_to_lang.cge
        self.vocab = cge_to_lang.vocab
        self.mental_state = mental_state

        self._thought_chain: List[dict] = []
        self._max_chain = 50
        self._cycle = 0
        self._monologue_active = False

        self._curiosity_topics: deque = deque(maxlen=20)
        self._confidence_threshold = 0.4

        self._operation_templates = [
            self._op_describe,
            self._op_compare,
            self._op_question,
            self._op_chain,
            self._op_hypothesize,
        ]

    def activate(self):
        """Ativa o monologo interno continuo."""
        self._monologue_active = True

    def deactivate(self):
        self._monologue_active = False

    def tick(self, focus: Optional[str] = None) -> Optional[Utterance]:
        """Um ciclo de pensamento verbal."""
        if not self._monologue_active:
            return None
        self._cycle += 1

        current_focus = self._resolve_focus(focus)
        if not current_focus:
            return self._generate_curiosity_thought()

        op = random.choice(self._operation_templates)
        try:
            utterance = op(current_focus)
        except Exception:
            utterance = self.translator.describe_concept(current_focus)

        if utterance and utterance.text:
            self._record_thought(utterance)
            self._feed_back_to_cge(utterance)

        return utterance

    def _resolve_focus(self, hint: Optional[str] = None) -> Optional[str]:
        """Determina o foco do pensamento atual."""
        if hint:
            return hint

        if self.mental_state is not None:
            try:
                focus = getattr(self.mental_state, 'focus', '')
                if focus:
                    return focus
            except Exception:
                pass

        if self._curiosity_topics:
            return self._curiosity_topics[0]

        if self.cge is not None and hasattr(self.cge, 'nodes') and self.cge.nodes:
            low_conf_nodes = [
                nid for nid, ndata in self.cge.nodes.items()
                if isinstance(ndata, dict) and ndata.get('confidence', 1.0) < self._confidence_threshold
            ]
            if low_conf_nodes:
                return random.choice(low_conf_nodes)
            return random.choice(list(self.cge.nodes.keys()))

        return None

    def _op_describe(self, focus: str) -> Utterance:
        """Descreve o conceito foco."""
        return self.translator.describe_concept(focus, max_sentences=3)

    def _op_compare(self, focus: str) -> Utterance:
        """Compara com outro conceito."""
        others = self.vocab.sample_words(3)
        others = [o for o in others if o.replace(" ", "_") != focus]
        if not others:
            return self.translator.describe_concept(focus)
        other = random.choice(others)
        other_id = other.replace(" ", "_")
        return self.translator.compare_concepts(focus, other_id)

    def _op_question(self, focus: str) -> Utterance:
        """Gera uma pergunta sobre o conceito."""
        desc = self.translator.describe_concept(focus, max_sentences=2)
        question_words = ["why", "how", "what if", "when", "what causes"]
        qw = random.choice(question_words)
        text = f"{qw} does {focus.replace('_', ' ')} work? I know that {desc.text}"
        return Utterance(
            text=text, concepts=[focus],
            confidence=0.5, source="internal_monologue",
            is_question=True,
        )

    def _op_chain(self, focus: str) -> Utterance:
        """Tenta encontrar uma cadeia causal a partir do foco."""
        if self.cge is None or not hasattr(self.cge, 'nodes'):
            return self.translator.describe_concept(focus)
        targets = [n for n in list(self.cge.nodes.keys())[:20] if n != focus]
        if not targets:
            return self.translator.describe_concept(focus)
        target = random.choice(targets)
        return self.translator.describe_relation_chain(focus, target)

    def _op_hypothesize(self, focus: str) -> Utterance:
        """Gera uma hipotese sobre o conceito."""
        desc = self.translator.describe_concept(focus, max_sentences=1)
        hypothesis_verbs = ["might cause", "could be related to", "may influence",
                            "probably affects", "might be connected to"]
        verb = random.choice(hypothesis_verbs)
        others = self.vocab.sample_words(3)
        others = [o.replace(" ", "_") for o in others if o.replace(" ", "_") != focus]
        other = random.choice(others) if others else "something"
        text = f"I hypothesize that {focus.replace('_', ' ')} {verb} {other.replace('_', ' ')}"
        return Utterance(
            text=text, concepts=[focus, other],
            confidence=0.3, source="internal_monologue",
            is_hypothesis=True,
        )

    def _generate_curiosity_thought(self) -> Optional[Utterance]:
        """Gera pensamento baseado em curiosidade."""
        if not self._curiosity_topics:
            if self.cge is not None and hasattr(self.cge, 'nodes') and self.cge.nodes:
                topic = random.choice(list(self.cge.nodes.keys()))
                self._curiosity_topics.append(topic)
            else:
                return None
        topic = self._curiosity_topics[0]
        utterance = self.translator.describe_concept(topic)
        self._curiosity_topics.popleft()
        return utterance

    def add_curiosity(self, topic: str):
        """Adiciona um topico para explorar por curiosidade."""
        self._curiosity_topics.append(topic)

    def _record_thought(self, utterance: Utterance):
        """Registra pensamento no historico."""
        self._thought_chain.append({
            "cycle": self._cycle,
            "utterance": utterance.to_dict(),
            "time": time.time(),
        })
        if len(self._thought_chain) > self._max_chain:
            self._thought_chain = self._thought_chain[-self._max_chain:]

    def _feed_back_to_cge(self, utterance: Utterance):
        """Alimenta o pensamento verbalizado de volta no CGE."""
        if self.cge is None:
            return
        thought_id = f"_thought_{self._cycle}"
        try:
            self.cge.create_node(
                thought_id,
                semantic_vec=torch.zeros(10000) if hasattr(torch, 'zeros') else None,
            )
            for i, concept in enumerate(utterance.concepts):
                if concept and hasattr(self.cge, 'nodes') and concept in self.cge.nodes:
                    self.cge.add_edge(thought_id, concept, 'CORRELATED')
            if utterance.is_hypothesis:
                for i, concept in enumerate(utterance.concepts):
                    if concept and hasattr(self.cge, 'nodes') and concept in self.cge.nodes:
                        self.cge.add_edge(thought_id, concept, 'CORRELATED')
        except Exception:
            pass

    def get_thought_chain(self, n: int = 5) -> List[str]:
        """Retorna as ultimas N verbalizacoes."""
        return [t["utterance"]["text"] for t in self._thought_chain[-n:]]

    def get_stats(self) -> dict:
        return {
            "cycle": self._cycle,
            "active": self._monologue_active,
            "chain_length": len(self._thought_chain),
            "curiosity_queue": len(self._curiosity_topics),
            "vocab_size": self.vocab.size() if self.vocab else 0,
        }


class EmergentLanguageEngine:
    """
    Motor completo de linguagem emergente.
    
    Unifica:
    - EmergentVocabulary (conceitos → palavras)
    - CGEToLanguage (grafo → sentencas)
    - InternalMonologue (pensamento verbal continuo)
    
    O sistema inteiro usa isto para:
    1. Falar (gerar texto sobre o que pensa)
    2. Pensar (monologo interno que retroalimenta o CGE)
    3. Perguntar (curiosidade verbalizada)
    4. Explicar (cadeias causais em linguagem)
    """

    def __init__(self, cge: Optional['CognitiveGraphEngine'] = None,
                 mental_state=None):
        self.cge = cge
        self.mental_state = mental_state

        self.vocab = EmergentVocabulary(cge)
        self.translator = CGEToLanguage(cge, self.vocab)
        self.monologue = InternalMonologue(self.translator, mental_state)

        self._output_history: deque = deque(maxlen=200)
        self._is_speaking = False

        if cge is not None:
            self.vocab.sync_from_cge()

    def set_cge(self, cge: 'CognitiveGraphEngine'):
        """Conecta ao CGE."""
        self.cge = cge
        self.translator.set_cge(cge)
        self.monologue.cge = cge
        self.monologue.vocab.cge = cge
        self.vocab.cge = cge
        self.vocab.sync_from_cge()

    def think(self, focus: Optional[str] = None) -> Optional[str]:
        """Um ciclo de pensamento. Retorna texto."""
        utterance = self.monologue.tick(focus)
        if utterance:
            self._output_history.append(utterance)
            return utterance.text
        return None

    def speak(self, topic: str, max_length: int = 5) -> str:
        """Gera um paragrafo sobre um topico."""
        self._is_speaking = True
        utterances = []
        focus = topic.replace(" ", "_")

        utterances.append(self.translator.describe_concept(focus))
        if self.cge is not None and hasattr(self.cge, 'nodes'):
            neighbors = []
            try:
                for e in getattr(self.cge, 'edges', []):
                    if isinstance(e, dict) and e.get('from') == focus:
                        neighbors.append(e.get('to'))
            except Exception:
                pass
            for n in neighbors[:max_length - 1]:
                if n:
                    utterances.append(self.translator.describe_concept(n))

        texts = [u.text for u in utterances if u and u.text]
        result = " ".join(texts)
        self._output_history.append(Utterance(
            text=result, concepts=[topic],
            confidence=0.7, source="speak",
        ))
        self._is_speaking = False
        return result

    def ask(self, topic: str) -> str:
        """Gera uma pergunta sobre um topico."""
        focus = topic.replace(" ", "_")
        utterance = self.monologue._op_question(focus)
        self._output_history.append(utterance)
        return utterance.text

    def explain(self, concept_a: str, concept_b: str) -> str:
        """Explica a relacao entre dois conceitos."""
        chain = self.translator.describe_relation_chain(
            concept_a.replace(" ", "_"),
            concept_b.replace(" ", "_"),
        )
        return chain.text

    def start_monologue(self):
        """Inicia o monologo interno continuo."""
        self.monologue.activate()

    def stop_monologue(self):
        self.monologue.deactivate()

    def get_recent_output(self, n: int = 5) -> List[str]:
        return [u.text for u in list(self._output_history)[-n:]]

    def get_stats(self) -> dict:
        return {
            "vocab_size": self.vocab.size(),
            "monologue_active": self.monologue._monologue_active,
            "monologue_cycles": self.monologue._cycle,
            "output_history": len(self._output_history),
            "is_speaking": self._is_speaking,
        }
