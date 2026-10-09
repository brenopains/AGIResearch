"""
OpenIE Knowledge Extractor — Extrator de Conhecimento com Relacoes Tipadas
==========================================================================
Substitui o extrator de co-ocorrencia textual (janela deslizante de bigramas)
por parsing de dependencias gramaticais (spaCy DependencyParser + OpenIE).

Para cada frase, extrai:
  - Sujeito → verbo → objeto (relacao CAUSES, ENABLES, PRECEDES, etc.)
  - Sujeito → "is" → predicado nominal (relacao IS_A)
  - Entidade → "has/have" → atributo (relacao HAS_PROPERTY)

Confianca calculada por frequencia de extracao + consistencia gramatical.

Uso:
    extractor = OpenIEKnowledgeExtractor()
    triples = extractor.extract_from_text("Gravity pulls objects toward Earth.")
    # [('gravity', 'CAUSES', 'pull'), ('pull', 'AFFECTS', 'objects'), ...]
"""
import re, math, time
from collections import defaultdict
from typing import List, Tuple, Optional, Dict


class OpenIEKnowledgeExtractor:
    """
    Extrator de conhecimento com relacoes tipadas baseado em parser de dependencias.
    
    Usa spaCy para analise gramatical (sujeito-verbo-objeto) e mapeia
    verbos para tipos de relacao semantica (CAUSES, ENABLES, PRECEDES, etc.)
    """
    
    # Mapeamento verbo -> tipo de relacao
    VERB_RELATION_MAP = {
        # Causa
        'cause': 'CAUSES', 'causes': 'CAUSES', 'caused': 'CAUSES',
        'result': 'CAUSES', 'results': 'CAUSES', 'resulted': 'CAUSES',
        'lead': 'CAUSES', 'leads': 'CAUSES', 'led': 'CAUSES',
        'produce': 'CAUSES', 'produces': 'CAUSES', 'produced': 'CAUSES',
        'create': 'CAUSES', 'creates': 'CAUSES', 'created': 'CAUSES',
        'generate': 'CAUSES', 'generates': 'CAUSES', 'generated': 'CAUSES',
        'induce': 'CAUSES', 'induces': 'CAUSES', 'induced': 'CAUSES',
        'trigger': 'CAUSES', 'triggers': 'CAUSES', 'triggered': 'CAUSES',
        'make': 'CAUSES', 'makes': 'CAUSES', 'made': 'CAUSES',
        'force': 'CAUSES', 'forces': 'CAUSES', 'forced': 'CAUSES',
        
        # Habilitacao
        'enable': 'ENABLES', 'enables': 'ENABLES', 'enabled': 'ENABLES',
        'allow': 'ENABLES', 'allows': 'ENABLES', 'allowed': 'ENABLES',
        'permit': 'ENABLES', 'permits': 'ENABLES', 'permitted': 'ENABLES',
        'facilitate': 'ENABLES', 'facilitates': 'ENABLES', 'facilitated': 'ENABLES',
        'support': 'ENABLES', 'supports': 'ENABLES', 'supported': 'ENABLES',
        
        # Precedencia temporal
        'precede': 'PRECEDES', 'precedes': 'PRECEDES', 'preceded': 'PRECEDES',
        'follow': 'FOLLOWS', 'follows': 'FOLLOWS', 'followed': 'FOLLOWS',
        'come': 'PRECEDES', 'comes': 'PRECEDES',
        'occur': 'PRECEDES', 'occurs': 'PRECEDES', 'occurred': 'PRECEDES',
        
        # Influencia
        'influence': 'INFLUENCES', 'influences': 'INFLUENCES', 'influenced': 'INFLUENCES',
        'affect': 'AFFECTS', 'affects': 'AFFECTS', 'affected': 'AFFECTS',
        'impact': 'AFFECTS', 'impacts': 'AFFECTS', 'impacted': 'AFFECTS',
        'change': 'AFFECTS', 'changes': 'AFFECTS', 'changed': 'AFFECTS',
        'modify': 'AFFECTS', 'modifies': 'AFFECTS', 'modified': 'AFFECTS',
        
        # Prevencao
        'prevent': 'PREVENTS', 'prevents': 'PREVENTS', 'prevented': 'PREVENTS',
        'block': 'PREVENTS', 'blocks': 'PREVENTS', 'blocked': 'PREVENTS',
        'inhibit': 'PREVENTS', 'inhibits': 'PREVENTS', 'inhibited': 'PREVENTS',
        'reduce': 'PREVENTS', 'reduces': 'PREVENTS', 'reduced': 'PREVENTS',
        'stop': 'PREVENTS', 'stops': 'PREVENTS', 'stopped': 'PREVENTS',
    }
    
    def __init__(self):
        self.nlp = None  # Lazy init
        self._stats = {'sentences_parsed': 0, 'triples_extracted': 0}
    
    def _ensure_nlp(self):
        """Load spaCy model on first use (lazy)."""
        if self.nlp is None:
            import spacy
            print("[OpenIE] Loading spaCy model...")
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except:
                import subprocess, sys
                subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
                self.nlp = spacy.load('en_core_web_sm')
            print("[OpenIE] Ready")
    
    def extract_from_text(self, text: str) -> List[Dict]:
        """
        Extrai triplas de conhecimento tipadas de um texto.
        
        Returns:
            Lista de dicts: {subject, relation, object, confidence, source_sentence}
        """
        self._ensure_nlp()
        doc = self.nlp(text)
        triples = []
        
        for sent in doc.sents:
            self._stats['sentences_parsed'] += 1
            
            # Extract SVO triples from dependency parse
            sent_triples = self._extract_svo(sent)
            triples.extend(sent_triples)
            
            # Extract copular constructions (X is Y → IS_A)
            cop_triples = self._extract_copular(sent)
            triples.extend(cop_triples)
            
            # Extract possession (X has Y → HAS_PROPERTY)
            poss_triples = self._extract_possession(sent)
            triples.extend(poss_triples)
        
        self._stats['triples_extracted'] += len(triples)
        return triples
    
    def _extract_svo(self, sent) -> List[Dict]:
        """
        Extract Subject-Verb-Object triples from a sentence.
        Uses spaCy's dependency parse (nsubj + dobj relations).
        """
        triples = []
        seen = set()
        
        for token in sent:
            # Find verbs with subjects and objects
            if token.pos_ == 'VERB':
                subjects = []
                objects = []
                prep_objects = []
                
                for child in token.children:
                    if child.dep_ in ('nsubj', 'nsubjpass', 'agent'):
                        subjects.append(self._extract_phrase(child))
                    elif child.dep_ in ('dobj', 'dative', 'attr'):
                        objects.append(self._extract_phrase(child))
                    elif child.dep_ == 'prep':
                        # Extract objects of prepositions
                        for prep_child in child.children:
                            if prep_child.dep_ == 'pobj':
                                prep_objects.append(self._extract_phrase(prep_child))
                
                if subjects and (objects or prep_objects):
                    all_objects = objects + prep_objects
                    verb_lemma = token.lemma_.lower()
                    rel_type = self.VERB_RELATION_MAP.get(verb_lemma, 'AFFECTS')
                    
                    for subj in subjects:
                        for obj in all_objects:
                            key = (subj, rel_type, obj)
                            if key not in seen:
                                seen.add(key)
                                confidence = self._compute_confidence(token, sent)
                                triples.append({
                                    'subject': subj,
                                    'relation': rel_type,
                                    'object': obj,
                                    'confidence': confidence,
                                    'source': sent.text,
                                    'extraction_type': 'svo',
                                })
        return triples
    
    def _extract_copular(self, sent) -> List[Dict]:
        """Extract 'X is Y' → IS_A triples."""
        triples = []
        for token in sent:
            if token.lemma_ == 'be' and token.pos_ == 'AUX':
                subjects = []
                attributes = []
                
                for child in token.children:
                    if child.dep_ == 'nsubj':
                        subjects.append(self._extract_phrase(child))
                    elif child.dep_ in ('attr', 'acomp', 'advmod'):
                        attributes.append(self._extract_phrase(child))
                
                for subj in subjects:
                    for attr in attributes:
                        triples.append({
                            'subject': subj,
                            'relation': 'IS_A',
                            'object': attr,
                            'confidence': 0.7,
                            'source': sent.text,
                            'extraction_type': 'copular',
                        })
        return triples
    
    def _extract_possession(self, sent) -> List[Dict]:
        """Extract 'X has Y' → HAS_PROPERTY triples."""
        triples = []
        for token in sent:
            if token.lemma_ in ('have', 'has', 'contain', 'include', 'possess'):
                subjects = []
                objects = []
                
                for child in token.children:
                    if child.dep_ == 'nsubj':
                        subjects.append(self._extract_phrase(child))
                    elif child.dep_ == 'dobj':
                        objects.append(self._extract_phrase(child))
                
                for subj in subjects:
                    for obj in objects:
                        triples.append({
                            'subject': subj,
                            'relation': 'HAS_PROPERTY',
                            'object': obj,
                            'confidence': 0.65,
                            'source': sent.text,
                            'extraction_type': 'possession',
                        })
        return triples
    
    def _extract_phrase(self, token) -> str:
        """Extract the full noun phrase for a token (including modifiers)."""
        words = [token.text]
        for child in token.children:
            if child.dep_ in ('amod', 'compound', 'poss', 'nummod', 'det'):
                words.insert(0, child.text) if child.i < token.i else words.append(child.text)
        return ' '.join(words).lower()
    
    def _compute_confidence(self, verb_token, sent) -> float:
        """
        Compute confidence for an SVO extraction based on:
        - Grammatical certainty: direct dependency paths are more certain
        - Negation: negated verbs reduce confidence
        - Modal auxiliaries: might/could reduce confidence
        """
        base = 0.7
        
        # Check for negation
        for child in verb_token.children:
            if child.dep_ == 'neg':
                base *= 0.5  # Negated verb → lower confidence
        
        # Check for modal auxiliaries
        for child in verb_token.children:
            if child.pos_ == 'AUX' and child.lemma_ in ('may', 'might', 'could', 'would', 'should'):
                base *= 0.7
        
        # Check passive voice
        if verb_token.dep_ == 'ROOT':
            for child in verb_token.children:
                if child.dep_ == 'nsubjpass':
                    base *= 0.8  # Passive is less certain than active
        
        return min(0.99, max(0.1, base))
    
    def extract_from_corpus(self, corpus_path: str, max_sentences: int = 10000) -> List[Dict]:
        """Extract triples from a corpus file (one sentence per line)."""
        all_triples = []
        count = 0
        with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line and len(line) > 20:
                    triples = self.extract_from_text(line)
                    all_triples.extend(triples)
                    count += 1
                    if count % 100 == 0:
                        print(f"[OpenIE] Processed {count} lines, "
                              f"{len(all_triples)} triples extracted")
                    if count >= max_sentences:
                        break
        return all_triples
    
    def get_stats(self) -> Dict:
        return dict(self._stats)
