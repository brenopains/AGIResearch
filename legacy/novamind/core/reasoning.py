"""
NovaMind — Layer 4: Hybrid Reasoning Engine
=============================================

Implements three-paradigm reasoning: Neural + Symbolic + Geometric,
with a consensus mechanism that integrates results.

Why hybrid reasoning?
---------------------
Each reasoning paradigm has different strengths:

    NEURAL (Pattern-based):
        ✅ Analogies, similarity, intuition, generalization
        ❌ Logical consistency, mathematical proofs, exact counting

    SYMBOLIC (Logic-based):
        ✅ Deduction, constraint checking, rule following, proofs
        ❌ Handling uncertainty, learning from examples, creativity

    GEOMETRIC (Structure-based):
        ✅ Spatial reasoning, structural analogies, topology
        ❌ Sequential processing, linguistic reasoning

No single paradigm is sufficient for general intelligence.
NovaMind runs all three IN PARALLEL and uses a learned
consensus mechanism to integrate their outputs.

The reasoning trace is fully transparent — you can see which
paradigm contributed what to the final answer.

References:
    - Kahneman (2011): "Thinking, Fast and Slow" (System 1 vs System 2)
    - Marcus & Davis (2019): "Rebooting AI" (neuro-symbolic integration)
    - Garcez et al. (2019): "Neural-Symbolic Computing"
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
import time
import hashlib

from .math.geometric_algebra import CliffordAlgebra, MultiVector
from .math.hyperbolic import PoincareBall
from .math.topology import SimplicialComplex, PersistentHomology

CONCEPT_DIM = 32
GA_DIM = 8


class ReasoningResult:
    """Result from a single reasoning paradigm."""

    def __init__(self, paradigm: str, conclusion: str, confidence: float,
                 trace: List[str], supporting_evidence: List[str] = None):
        self.paradigm = paradigm
        self.conclusion = conclusion
        self.confidence = confidence
        self.trace = trace
        self.supporting_evidence = supporting_evidence or []
        self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            'paradigm': self.paradigm,
            'conclusion': self.conclusion,
            'confidence': self.confidence,
            'trace': self.trace,
            'supporting_evidence': self.supporting_evidence,
        }


class NeuralReasoner:
    """
    Neural (Pattern-based) reasoning — System 1 / Fast thinking.

    Uses geometric similarity in the Poincaré ball and Clifford algebra
    to find analogies, similarities, and pattern-based inferences.

    This is "intuition" — fast, approximate, based on pattern matching.
    """

    def __init__(self):
        self.ball = PoincareBall(dim=CONCEPT_DIM)
        self.algebra = CliffordAlgebra(p=GA_DIM)

    def reason(self, query_concepts: List[Dict], knowledge_base: Dict) -> ReasoningResult:
        """
        Neural reasoning: find the most relevant patterns and analogies.

        Process:
        1. Embed the query in concept space
        2. Find nearest neighbors (most similar known concepts)
        3. Check for analogical patterns (A:B :: C:?)
        4. Generate conclusion based on strongest patterns
        """
        trace = []
        trace.append("⚡ NEURAL REASONING (Pattern-based / System 1)")

        # Step 1: Identify key concepts and their positions
        positions = {}
        for concept in query_concepts:
            if concept.get('position'):
                positions[concept['label']] = np.array(concept['position'])
                trace.append(f"  📍 Located '{concept['label']}' in hyperbolic space "
                           f"(depth: {np.linalg.norm(concept['position']):.3f})")

        if not positions:
            return ReasoningResult("neural", "Insufficient concept embeddings",
                                   0.2, trace)

        # Step 2: Find nearest concepts in the knowledge base
        neighbors = self._find_nearest_concepts(positions, knowledge_base)
        for neighbor in neighbors[:3]:
            trace.append(f"  🔗 Similar to: '{neighbor['label']}' "
                       f"(similarity: {neighbor['similarity']:.3f})")

        # Step 3: Detect analogical patterns
        analogies = self._detect_analogies(positions, knowledge_base)
        for analogy in analogies[:2]:
            trace.append(f"  🔄 Analogy detected: {analogy['description']}")

        # Step 4: Generate intuitive conclusion
        conclusion, confidence = self._generate_intuition(
            query_concepts, neighbors, analogies
        )
        trace.append(f"  💡 Intuition: {conclusion} (confidence: {confidence:.2f})")

        return ReasoningResult(
            "neural", conclusion, confidence, trace,
            [n['label'] for n in neighbors[:3]]
        )

    def _find_nearest_concepts(self, positions: Dict[str, np.ndarray],
                                knowledge_base: Dict) -> List[Dict]:
        """Find nearest concepts in the knowledge graph."""
        neighbors = []
        known_concepts = knowledge_base.get('concepts', [])

        for label, pos in positions.items():
            for known in known_concepts:
                if known.get('position') and known['label'] != label:
                    known_pos = np.array(known['position'])
                    dist = float(self.ball.distance(
                        pos.reshape(1, -1), known_pos.reshape(1, -1)
                    ).mean())
                    similarity = np.exp(-dist)
                    neighbors.append({
                        'label': known['label'],
                        'similarity': similarity,
                        'distance': dist,
                    })

        neighbors.sort(key=lambda x: x['similarity'], reverse=True)
        return neighbors

    def _detect_analogies(self, positions: Dict[str, np.ndarray],
                          knowledge_base: Dict) -> List[Dict]:
        """
        Detect analogical patterns using geometric algebra.

        An analogy A:B :: C:D exists when the transformation from A to B
        (a rotor in Clifford algebra) is similar to the transformation from C to D.
        """
        analogies = []
        labels = list(positions.keys())

        if len(labels) >= 2:
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    a_label, b_label = labels[i], labels[j]
                    a_pos = positions[a_label][:GA_DIM]
                    b_pos = positions[b_label][:GA_DIM]

                    a_vec = self.algebra.vector(a_pos)
                    b_vec = self.algebra.vector(b_pos)
                    similarity = self.algebra.concept_similarity(a_vec, b_vec)

                    analogies.append({
                        'description': f"'{a_label}' relates to '{b_label}' "
                                      f"(geometric similarity: {similarity:.3f})",
                        'concepts': (a_label, b_label),
                        'similarity': abs(similarity),
                    })

        analogies.sort(key=lambda x: x['similarity'], reverse=True)
        return analogies

    def _generate_intuition(self, query_concepts: List[Dict],
                             neighbors: List[Dict],
                             analogies: List[Dict]) -> Tuple[str, float]:
        """Generate an intuitive conclusion from patterns."""
        query_labels = [c['label'] for c in query_concepts]

        if neighbors:
            related_labels = [n['label'] for n in neighbors[:3]]
            avg_similarity = np.mean([n['similarity'] for n in neighbors[:3]])

            conclusion = (f"Based on pattern matching, {', '.join(query_labels)} "
                        f"relates to {', '.join(related_labels)}. "
                        f"The concepts share structural similarity in concept space.")
            confidence = float(min(0.9, avg_similarity + 0.2))
        else:
            conclusion = (f"Novel concepts: {', '.join(query_labels)}. "
                        f"No strong pattern matches found — this is genuinely new information.")
            confidence = 0.3

        return conclusion, confidence


class SymbolicReasoner:
    """
    Symbolic (Logic-based) reasoning — System 2 / Slow thinking.

    Applies formal logical rules, performs deduction, checks constraints,
    and handles explicit rule-based inference.

    This is "deliberation" — slow, precise, guaranteed consistent.
    """

    # Relações que NÃO compõem por transitividade. Uma aresta destes tipos é
    # um fato válido e continua aparecendo como proposição — o que ela não faz
    # é servir de elo numa cadeia derivada.
    #
    # A lista é curta e explícita porque o custo dos dois erros é assimétrico:
    # deixar passar uma relação não-transitiva produz uma afirmação causal
    # falsa; bloquear uma transitiva por engano só deixa de derivar algo. Na
    # dúvida, o vocabulário desconhecido passa.
    NAO_TRANSITIVAS = {
        'correlaciona_com', 'correlates_with', 'correlated_with', 'correlation',
        'associado_a', 'associated_with', 'association',
        'similar_a', 'similar_to', 'semelhante_a',
        'co_ocorre', 'co_occurs_with', 'coocorre',
        'proximo_de', 'near', 'adjacent_to',
        'contrasta_com', 'contrasts_with',
    }

    def __init__(self):
        self.rules: List[Dict] = []
        self.facts: Dict[str, List[str]] = defaultdict(list)

    def add_rule(self, rule: Dict):
        """Add a logical rule to the knowledge base."""
        self.rules.append(rule)

    def add_fact(self, subject: str, predicate: str, obj: str):
        """Add a fact (subject, predicate, object) triple."""
        self.facts[subject].append(f"{predicate}:{obj}")

    def reason(self, query_concepts: List[Dict], knowledge_base: Dict) -> ReasoningResult:
        """
        Symbolic reasoning: apply logical rules and deduction.

        Process:
        1. Extract logical propositions from the query
        2. Match against known rules
        3. Apply forward and backward chaining
        4. Check for contradictions
        5. Generate logically grounded conclusion
        """
        trace = []
        trace.append("🔢 SYMBOLIC REASONING (Logic-based / System 2)")

        # Step 1: Extract propositions
        propositions = self._extract_propositions(query_concepts, knowledge_base)
        query_labels = [c.get('label', '') for c in query_concepts if c.get('label')]
        for prop in propositions:
            trace.append(f"  📝 Proposition: {prop}")

        # Step 2: Apply rule-based inference
        todas = self._forward_chain(propositions, knowledge_base, query_labels)
        # As auto-notas NÃO são derivações. Se entrassem na mesma lista,
        # entrariam também na contagem "derived from N inference(s)" — e o
        # motor passaria a inflar a própria confiança justamente ao admitir
        # uma limitação, que é o contrário do ponto.
        inferences = [i for i in todas if not i.get('meta')]
        limitacoes = [i['conclusion'] for i in todas if i.get('meta')]
        for inf in inferences:
            trace.append(f"  ⟹ Inferred: {inf['conclusion']} "
                       f"(from rule: {inf['rule']})")
        for lim in limitacoes:
            trace.append(f"  ⚠️ Self-report: {lim}")

        # Step 3: Check for contradictions
        contradictions = self._check_contradictions(propositions + [i['conclusion'] for i in inferences])
        if contradictions:
            for c in contradictions:
                trace.append(f"  ⚠️ CONTRADICTION: {c}")

        # Step 4: Build conclusion
        conclusion, confidence = self._build_conclusion(propositions, inferences,
                                                        contradictions, limitacoes)
        trace.append(f"  📐 Conclusion: {conclusion} (confidence: {confidence:.2f})")

        return ReasoningResult(
            "symbolic", conclusion, confidence, trace,
            [i['rule'] for i in inferences[:3]]
        )

    def _extract_propositions(self, concepts: List[Dict], kb: Dict) -> List[str]:
        """Extract logical propositions from concepts and their relations."""
        propositions = []

        for concept in concepts:
            connections = concept.get('connections', {})
            for rel_type, targets in connections.items():
                for target in targets:
                    propositions.append(f"{concept['label']} {rel_type} {target}")

        # Add facts from the world model
        world_edges = kb.get('causal_edges', [])
        for edge in world_edges[:10]:
            propositions.append(
                f"{edge.get('source', '?')} {edge.get('type', 'relates_to')} {edge.get('target', '?')}"
            )

        return propositions

    def _forward_chain(self, propositions: List[str], kb: Dict,
                       query_labels: Optional[List[str]] = None) -> List[Dict]:
        """
        Forward chaining: apply rules to derive new facts.

        If we know A and we have a rule A → B, conclude B.
        """
        inferences = []
        semantic_rules = kb.get('semantic_rules', [])

        for rule in semantic_rules:
            rule_content = rule.get('content', {})
            rule_type = rule_content.get('type', '')
            abstraction = rule_content.get('abstraction', {})

            if rule_type == 'sequential':
                from_type = abstraction.get('from_type', '')
                to_type = abstraction.get('to_type', '')
                freq = abstraction.get('frequency', 0)

                for prop in propositions:
                    if from_type.lower() in prop.lower():
                        inferences.append({
                            'conclusion': f"Likely followed by {to_type} "
                                        f"(frequency: {freq:.0%})",
                            'rule': f"{from_type} → {to_type}",
                            'confidence': freq,
                        })

            elif rule_type == 'attribute':
                attr = abstraction.get('attribute', '')
                value = abstraction.get('value', '')
                freq = abstraction.get('frequency', 0)

                # A regra precisa TOCAR alguma proposição da consulta.
                # Antes bastava freq > 0.5 e ela disparava sempre, sem olhar as
                # premissas — o que não é inferência, é repetir a regra de volta.
                # Medido na bancada de dedução: com 6 regras-distrator na base,
                # 86% das derivações reivindicadas não tinham relação nenhuma
                # com a consulta (precisão de 14%).
                relevante = any(
                    (attr and attr.lower() in prop.lower()) or
                    (value and value.lower() in prop.lower())
                    for prop in propositions
                )
                if freq > 0.5 and relevante:
                    inferences.append({
                        'conclusion': f"Common attribute: {attr} = {value}",
                        'rule': f"Frequency-based: {attr}={value} ({freq:.0%})",
                        'confidence': freq,
                    })

        # Apply causal chain inference
        causal_chains = kb.get('causal_chains', [])
        for chain in causal_chains[:5]:
            if len(chain) >= 2:
                for prop in propositions:
                    if chain[0].lower() in prop.lower():
                        inferences.append({
                            'conclusion': f"Causal chain: {' → '.join(chain)}",
                            'rule': 'Causal transitivity',
                            'confidence': 0.7 ** (len(chain) - 1),  # Decreases with chain length
                        })
                        break   # uma cadeia deriva UMA conclusão, não uma por premissa

        # ── Fecho transitivo sobre as arestas causais ────────────────────────
        # Isto é a dedução mais elementar que existe: se A→B e B→C estão na
        # base, então A→C se segue, mesmo que ninguém tenha escrito.
        #
        # Faltava. Medido na bancada de dedução: com as duas arestas dadas
        # SEPARADAMENTE, o motor derivava 0% dos pares compostos, enquanto um
        # fecho transitivo de dez linhas derivava 100%. O motor acertava 100%
        # quando a cadeia [A,B,C] vinha pronta numa lista — mas aí ele estava
        # ecoando, não compondo.
        # O TIPO DA ARESTA DECIDE SE ELA COMPÕE.
        # Causalidade compõe com causalidade. Correlação não compõe com nada:
        # "A causa B" mais "B correlaciona-se com C" NÃO dá "A causa C" — é o
        # erro elementar da inferência causal, e o fecho o cometia por
        # construção, porque `e.get('type')` era lido para a proposição e
        # jogado fora na hora de fechar.
        #
        # Medido em novaphys/generalizacao.py, padrão P8: 100% de violação —
        # em todos os itens o motor afirmava o par que a mistura de tipos
        # proíbe. Isto é anterior à unificação dos arcos: a primeira versão do
        # fecho já descartava o tipo.
        #
        # A política é conservadora de propósito: tipo desconhecido é tratado
        # como componível, porque `mind.py` alimenta arestas de vocabulário
        # aberto e travar o que não se conhece calaria o motor. O que bloqueia
        # é a lista explícita de relações que sabidamente não compõem.
        arestas = set()
        for e in kb.get('causal_edges', []):
            s, t = e.get('source'), e.get('target')
            tipo = str(e.get('type', '') or '').lower()
            if s and t and tipo not in self.NAO_TRANSITIVAS:
                arestas.add((s, t))
        for ch in kb.get('causal_chains', []):
            for i in range(len(ch) - 1):
                arestas.add((ch[i], ch[i + 1]))

        # ── Arcos de atributo ────────────────────────────────────────────────
        # "C tem a propriedade P" é uma RELAÇÃO ENTRE DOIS TERMOS, exatamente
        # como "C leva a P". Tratá-las como coisas de naturezas diferentes foi
        # o erro que produziu a versão anterior deste bloco: eu tinha escrito
        # uma regra à mão para o caso Reach∘Attribute, e ela resolvia esse caso
        # e nenhum outro.
        #
        # Medido em novaphys/generalizacao.py, cinco formatos de composição:
        #     P1 reach∘attr        (a regra que eu escrevi) ......... 100%
        #     P2 attr∘attr .................................... 0%
        #     P3 attr∘reach ................................... 0%
        #     P4 reach∘attr∘attr .............................. 0%
        #     P5 chain∘attr (mesma composição, outro campo) ... 100%
        # Ou seja: transferia para o mesmo caminho noutro campo, e para mais
        # nada. Era uma regra, não uma capacidade.
        #
        # Unificar os dois tipos de arco num grafo só resolve os cinco com UM
        # mecanismo em vez de cinco casos especiais, e é a razão de a versão
        # anterior ter sido apagada em vez de estendida.
        arcos_attr = {}
        for rule in semantic_rules:
            rc = rule.get('content', {})
            if rc.get('type') != 'attribute':
                continue
            ab = rc.get('abstraction', {})
            attr, val, freq = ab.get('attribute'), ab.get('value'), ab.get('frequency', 0)
            if attr and val and freq > 0.5:
                arcos_attr[(attr, val)] = float(freq)

        # ── Bloqueios: o raciocínio é DERROTÁVEL ─────────────────────────────
        # Fecho transitivo é monotônico — saber mais nunca o faz concluir menos.
        # Conhecimento real não é assim: "A leva a B, B leva a C" mais "B impede
        # C" não dá "A leva a C", dá uma exceção.
        #
        # Medido: sem este bloco, o motor afirmava a conclusão derrotada em
        # 100% dos itens de exceção (P6). Não é um caso raro — é a diferença
        # entre fechar um grafo e raciocinar sobre ele.
        bloqueios = set()
        for b in kb.get('bloqueios', []):
            if b.get('blocker') and b.get('blocked'):
                bloqueios.add((b['blocker'], b['blocked']))
        for rule in semantic_rules:
            rc = rule.get('content', {})
            if rc.get('type') in ('blocking', 'prevents'):
                ab = rc.get('abstraction', {})
                if ab.get('blocker') and ab.get('blocked'):
                    bloqueios.add((ab['blocker'], ab['blocked']))

        unificados = arestas | set(arcos_attr.keys())
        # ── Autoconhecimento: o motor sabe onde ele PAROU ────────────────────
        # O fecho para em 8 iterações e a saída é cortada em 40 derivações.
        # Ambos os cortes são conhecidos no instante em que acontecem — e eram
        # descartados em silêncio, com a conclusão saindo com a mesma confiança
        # de sempre.
        #
        # Medido em novaphys/calibracao.py: numa cadeia de 60 nós o alvo cai
        # fora do corte, o acerto é 0% e a confiança reportada era 0,850. O
        # sistema cortou e não sabia que tinha cortado. Isso não é um limite de
        # capacidade — é um limite que ele conhece e não conta.
        truncou = False
        if unificados:
            alcanca = set(unificados)
            for i in range(8):
                novos = {(a, d) for (a, b) in alcanca for (c, d) in alcanca
                         if b == c and (a, d) not in alcanca and a != d}
                if not novos:
                    break
                if len(alcanca) > 4000 or i == 7:
                    truncou = True
                    break
                alcanca |= novos
            derivados = alcanca - unificados

            # Um par derivado cai se algum bloqueador do destino estiver no
            # caminho — isto é, se a origem alcança o bloqueador. É a checagem
            # sã e barata: não reconstrói os caminhos, mas nenhuma conclusão
            # sobrevive a um bloqueador alcançável, que é o que importa.
            if bloqueios:
                caidos = set()
                for (a, d) in derivados:
                    for (bl, bd) in bloqueios:
                        if bd == d and (a == bl or (a, bl) in alcanca):
                            caidos.add((a, d))
                            break
                if caidos:
                    derivados = derivados - caidos
                    for (a, d) in sorted(caidos)[:5]:
                        inferences.append({
                            'conclusion': f"Defeated: {a} ⇝ {d} does not hold — "
                                          f"a blocking premise lies on the path",
                            'rule': 'Defeasible reasoning: blocked derivation',
                            'confidence': 0.6,
                        })
            # FILTRA primeiro, corta depois. A ordem importa: cortando antes,
            # o teto alfabético descartava as derivações relevantes junto com o
            # resto. Numa cadeia de 60 nós, "a → n59" ficava fora do corte
            # porque n1, n10, n11... vêm antes em ordem alfabética, e o motor
            # reprovava um caso que ele na verdade sabia derivar.
            # A relevância é medida contra os CONCEITOS DA CONSULTA, não
            # contra as proposições. `_extract_propositions` despeja todas as
            # arestas causais da base, inclusive as de componentes que não têm
            # ligação nenhuma com a pergunta — então filtrar por proposição
            # deixava passar derivações de um grafo desconexo. Medido: 50% de
            # contaminação entre componentes separados.
            alvo = [q.lower() for q in (query_labels or [])] or                    [p.lower() for p in propositions]
            relevantes = [(a, d) for (a, d) in derivados
                          if any(a.lower() == q or a.lower() in q for q in alvo)]

            # Proveniência: um par derivado só por arestas causais é uma
            # transitividade; um que precisou de pelo menos um arco de atributo
            # é uma composição de mecanismos, e a frase tem de dizer qual dos
            # dois foi, senão a saída não permite auditar o raciocínio.
            so_causal = set(arestas)
            for _ in range(8):
                extra = {(a, d) for (a, b) in so_causal for (c, d) in so_causal
                         if b == c and (a, d) not in so_causal and a != d}
                if not extra or len(so_causal) > 4000:
                    break
                so_causal |= extra

            if len(relevantes) > 40:
                truncou = True
            for (a, d) in sorted(relevantes)[:40]:
                if (a, d) in so_causal:
                    inferences.append({
                        'conclusion': f"Transitive: {a} → {d}",
                        'rule': 'Transitive closure over causal edges',
                        'confidence': 0.75,
                    })
                else:
                    conf = min([f for (k, f) in arcos_attr.items()] or [0.9])
                    inferences.append({
                        'conclusion': f"Composed: {a} ⇝ {d} — reached by chaining "
                                      f"causal and attribute relations",
                        'rule': 'Closure over unified relation graph (reach ∪ has)',
                        'confidence': 0.75 * max(0.0, min(1.0, conf)),
                    })

        # Deduplica: a mesma conclusão derivada por caminhos diferentes continua
        # sendo uma conclusão só. Sem isto, uma regra que casa com três
        # proposições era contada três vezes, e o número inflado ia direto para
        # a frase "derived from N inference(s)", inflando junto a confiança de
        # quem lesse. Medido: 3,00× de inflação na bancada de dedução.
        # ── Autoconhecimento: regras que o motor NÃO SABE LER ────────────────
        # Uma regra de tipo desconhecido é silenciosamente ignorada, e o
        # resultado fica idêntico ao de uma base vazia: "No applicable rules
        # found", confiança 0,20.
        #
        # São situações opostas. "Não havia o que derivar" e "havia e eu não
        # consigo processar" são exatamente os dois casos que mais importam
        # para quem depende da resposta — e o sistema os reportava com o mesmo
        # número. Medido em novaphys/calibracao.py: era essa colisão que fazia
        # o AUC da confiança variar de 0,438 a 1,000 só conforme a proporção de
        # itens de cada tipo na prova.
        conhecidos = {'sequential', 'attribute', 'blocking', 'prevents',
                      'conjunction_handled'}
        nao_lidas = {}
        for rule in semantic_rules:
            t = str(rule.get('content', {}).get('type', '') or '').lower()
            if t and t not in conhecidos:
                nao_lidas[t] = nao_lidas.get(t, 0) + 1

        limitacoes = []
        if truncou:
            limitacoes.append('closure truncated — some derivations were not computed')
        for t, k in sorted(nao_lidas.items()):
            limitacoes.append(f"{k} rule(s) of unsupported type '{t}' were ignored")
        for lim in limitacoes:
            inferences.append({'conclusion': lim, 'rule': 'self-report',
                               'confidence': 0.0, 'meta': True})

        vistas, unicas = set(), []
        for inf in inferences:
            chave = inf['conclusion']
            if chave in vistas:
                continue
            vistas.add(chave)
            unicas.append(inf)
        return unicas

    def _check_contradictions(self, statements: List[str]) -> List[str]:
        """Check for logical contradictions among statements."""
        contradictions = []

        # Check for opposite assertions (simplified)
        negation_pairs = [
            ('causes', 'prevents'),
            ('is', 'is not'),
            ('has', 'lacks'),
            ('true', 'false'),
        ]

        for i, s1 in enumerate(statements):
            for j, s2 in enumerate(statements):
                if i >= j:
                    continue
                for pos, neg in negation_pairs:
                    if pos in s1.lower() and neg in s2.lower():
                        # Check if they're about the same subject
                        s1_words = set(s1.lower().split())
                        s2_words = set(s2.lower().split())
                        overlap = s1_words & s2_words
                        if len(overlap) >= 2:
                            contradictions.append(f"'{s1}' contradicts '{s2}'")

        return contradictions

    def _build_conclusion(self, propositions: List[str], inferences: List[Dict],
                          contradictions: List[str],
                          limitacoes: Optional[List[str]] = None) -> Tuple[str, float]:
        """
        Build a logically grounded conclusion.

        A confiança devolvida aqui é o que um humano lê e usa para decidir se
        acredita. Antes ela tinha três valores possíveis (0,20 / 0,85 / 0,95) e
        nenhum deles olhava para o que tinha acontecido na derivação — nem para
        os cortes, nem para o que o motor não conseguiu processar. O resultado
        medido em novaphys/calibracao.py: a confiança separava acerto de erro
        no nível do acaso, e o AUC oscilava de 0,438 a 1,000 apenas conforme a
        proporção de tipos de item na prova.
        """
        limitacoes = limitacoes or []
        if contradictions:
            conclusion = (f"Logical analysis found {len(contradictions)} contradiction(s). "
                        f"The knowledge base may need revision. "
                        f"Main issue: {contradictions[0]}")
            confidence = 0.3
        elif inferences:
            top_inf = sorted(inferences, key=lambda x: x['confidence'], reverse=True)
            conclusion = (f"Logical deduction yields: {top_inf[0]['conclusion']}. "
                        f"Derived from {len(inferences)} inference(s) using "
                        f"{len(propositions)} known proposition(s).")
            confidence = float(min(0.95, top_inf[0]['confidence'] + 0.1))
        elif limitacoes:
            # Nada derivado E o motor sabe por quê. É diferente de nada derivado
            # porque nada se seguia — e a diferença tem de aparecer no número,
            # não só na prosa, porque é o número que o leitor usa.
            conclusion = (f"No conclusion reached, and this may be a limitation of the "
                        f"reasoner rather than of the premises: {limitacoes[0]}. "
                        f"Known facts: {len(propositions)}.")
            confidence = 0.1
        else:
            # Silêncio confiante: havia premissas, o fecho rodou até o fim, toda
            # regra era legível — e ainda assim nada se segue. Isso é uma
            # RESPOSTA, não uma falha, e é frequentemente a resposta certa
            # ("causa mais correlação não dá causa"). Tratá-la como 0,20 punia
            # o motor por acertar, e colocava uma derivação truncada (0,45)
            # acima de uma negativa correta.
            #
            # 0,70 e não 0,95 de propósito: não haver limitação DECLARADA não
            # prova que não havia limitação. O motor sabe que não sabe ler uma
            # regra de tipo estranho; não sabe o que ele nem representa — a
            # inferência de aridade 2, por exemplo, some sem deixar rastro.
            # Essa diferença entre "sei que não sei" e "não sei que não sei" é
            # exatamente o que estes 0,25 de folga significam.
            conclusion = (f"No applicable rules found for the given propositions. "
                        f"Known facts: {len(propositions)}.")
            confidence = 0.7

        if limitacoes and confidence > 0.1:
            # Derivou alguma coisa, mas sabe que não derivou tudo. A confiança
            # tem de cair abaixo da linha de "pode usar sem conferir".
            confidence = round(min(confidence, 0.45), 4)
            conclusion += f" (partial: {limitacoes[0]})"

        return conclusion, confidence


class GeometricReasoner:
    """
    Geometric (Topology-based) reasoning — Structural thinking.

    Uses topological data analysis and the geometric structure of
    the knowledge graph to reason about:
        - Structural relationships between concept clusters
        - Missing connections (knowledge gaps)
        - Circular reasoning detection
        - Hierarchical inference via entailment cones
    """

    def __init__(self):
        self.ball = PoincareBall(dim=CONCEPT_DIM)

    def reason(self, query_concepts: List[Dict], knowledge_base: Dict) -> ReasoningResult:
        """
        Geometric reasoning: analyze the topology of concept space.

        Process:
        1. Build simplicial complex from concept positions
        2. Compute persistent homology (topological features)
        3. Analyze hierarchical relationships (entailment cones)
        4. Detect structural patterns (clusters, gaps, loops)
        """
        trace = []
        trace.append("📐 GEOMETRIC REASONING (Topology-based / Structural)")

        # Collect all concept positions
        all_concepts = knowledge_base.get('concepts', []) + query_concepts
        positioned_concepts = [c for c in all_concepts if c.get('position')]

        if len(positioned_concepts) < 3:
            return ReasoningResult(
                "geometric",
                "Insufficient concepts for topological analysis (need ≥ 3)",
                0.1, trace
            )

        # Step 1: Build distance matrix in hyperbolic space
        n = min(len(positioned_concepts), 30)  # Limit for computation
        positions = np.array([c['position'] for c in positioned_concepts[:n]])
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                d = float(self.ball.distance(
                    positions[i].reshape(1, -1),
                    positions[j].reshape(1, -1)
                ).mean())
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        trace.append(f"  📊 Built {n}×{n} hyperbolic distance matrix")

        # Step 2: Build Vietoris-Rips complex and compute persistent homology
        complex = SimplicialComplex()
        max_filtration = float(np.percentile(dist_matrix[dist_matrix > 0], 75)) if dist_matrix.any() else 2.0
        complex.build_from_distance_matrix(dist_matrix, max_dim=2, max_filtration=max_filtration)

        ph = PersistentHomology(complex)
        persistence = ph.compute(max_dim=2)

        betti = ph.betti_numbers()
        trace.append(f"  🔵 Betti numbers: β₀={betti.get(0, 0)} (clusters), "
                    f"β₁={betti.get(1, 0)} (loops), β₂={betti.get(2, 0)} (voids)")

        # Step 3: Analyze topological features
        gaps = ph.knowledge_gaps()
        cycles = ph.circular_reasoning_detector()
        entropy = ph.persistence_entropy()

        trace.append(f"  📈 Topological complexity (persistence entropy): {entropy:.3f}")

        if gaps:
            trace.append(f"  🕳️ Knowledge gaps detected: {len(gaps)} persistent voids")
        if cycles:
            trace.append(f"  🔄 Circular reasoning paths: {len(cycles)} persistent cycles")

        # Step 4: Hierarchical analysis
        hierarchy_report = self._analyze_hierarchy(positioned_concepts[:n])
        for h in hierarchy_report[:3]:
            trace.append(f"  🌳 {h}")

        # Step 5: Build conclusion
        conclusion = self._build_geometric_conclusion(betti, gaps, cycles, entropy, hierarchy_report)
        confidence = min(0.85, 0.3 + entropy * 0.3 + (0.1 if not cycles else 0))
        trace.append(f"  🔷 Geometric conclusion: {conclusion}")

        return ReasoningResult(
            "geometric", conclusion, confidence, trace,
            [f"β₀={betti.get(0, 0)}", f"β₁={betti.get(1, 0)}", f"entropy={entropy:.3f}"]
        )

    def _analyze_hierarchy(self, concepts: List[Dict]) -> List[str]:
        """Analyze hierarchical relationships in hyperbolic space."""
        reports = []

        # Sort by depth (distance from origin = specificity)
        depths = []
        for c in concepts:
            pos = np.array(c['position']).reshape(1, -1)
            origin = np.zeros_like(pos)
            depth = float(self.ball.distance(origin, pos).mean())
            depths.append((c['label'], depth))

        depths.sort(key=lambda x: x[1])

        if depths:
            most_abstract = depths[0]
            most_specific = depths[-1]
            reports.append(f"Most abstract concept: '{most_abstract[0]}' (depth: {most_abstract[1]:.3f})")
            reports.append(f"Most specific concept: '{most_specific[0]}' (depth: {most_specific[1]:.3f})")
            avg_depth = np.mean([d[1] for d in depths])
            reports.append(f"Average concept depth: {avg_depth:.3f} (hierarchy spread)")

        return reports

    def _build_geometric_conclusion(self, betti: Dict, gaps: List, cycles: List,
                                     entropy: float, hierarchy: List[str]) -> str:
        """Build a conclusion from topological analysis."""
        parts = []

        clusters = betti.get(0, 0)
        if clusters > 1:
            parts.append(f"The knowledge forms {clusters} disconnected clusters "
                        f"that may need bridging connections")
        elif clusters == 1:
            parts.append("Knowledge is well-connected (single component)")

        if gaps:
            parts.append(f"There are {len(gaps)} knowledge gap(s) that could be filled")

        if cycles:
            parts.append(f"Warning: {len(cycles)} potential circular reasoning path(s) detected")

        if entropy > 0.5:
            parts.append("The knowledge structure is topologically complex — rich internal organization")
        elif entropy > 0:
            parts.append("The knowledge structure is relatively simple — may benefit from enrichment")

        return ". ".join(parts) if parts else "Knowledge topology is still forming."


class HybridReasoningEngine:
    """
    The integrated reasoning engine that runs all three paradigms
    and builds a consensus.

    This is where "thinking" happens — not as a single process but as
    a PARLIAMENT of different reasoning styles, each contributing
    their perspective, then voting on the best answer.
    """

    def __init__(self):
        self.neural = NeuralReasoner()
        self.symbolic = SymbolicReasoner()
        self.geometric = GeometricReasoner()

        # Learned weights for each paradigm (updated by metacognition)
        self.paradigm_weights = {
            'neural': 0.4,
            'symbolic': 0.35,
            'geometric': 0.25,
        }

        # History for learning from past reasoning
        self.reasoning_history: List[Dict] = []
        self._max_reasoning_history = 100

    def reason(self, query_concepts: List[Dict], knowledge_base: Dict) -> Dict:
        """
        Run all three reasoning paradigms and build consensus.

        Returns a comprehensive reasoning report with full trace.
        """
        start_time = time.time()

        # Run all three paradigms
        neural_result = self.neural.reason(query_concepts, knowledge_base)
        symbolic_result = self.symbolic.reason(query_concepts, knowledge_base)
        geometric_result = self.geometric.reason(query_concepts, knowledge_base)

        # Consensus building
        consensus = self._build_consensus(neural_result, symbolic_result, geometric_result)

        reasoning_time = time.time() - start_time

        report = {
            'paradigms': {
                'neural': neural_result.to_dict(),
                'symbolic': symbolic_result.to_dict(),
                'geometric': geometric_result.to_dict(),
            },
            'consensus': consensus,
            'paradigm_weights': self.paradigm_weights.copy(),
            'reasoning_time_ms': reasoning_time * 1000,
            'combined_trace': (
                neural_result.trace +
                ["---"] +
                symbolic_result.trace +
                ["---"] +
                geometric_result.trace +
                ["---"] +
                [f"🎯 CONSENSUS: {consensus['conclusion']} "
                 f"(confidence: {consensus['confidence']:.2f})"]
            ),
        }

        self.reasoning_history.append({
            'timestamp': time.time(),
            'consensus_confidence': consensus['confidence'],
            'paradigm_confidences': {
                'neural': neural_result.confidence,
                'symbolic': symbolic_result.confidence,
                'geometric': geometric_result.confidence,
            },
        })
        if len(self.reasoning_history) > self._max_reasoning_history:
            self.reasoning_history = self.reasoning_history[-self._max_reasoning_history:]

        return report

    def _build_consensus(self, neural: ReasoningResult,
                         symbolic: ReasoningResult,
                         geometric: ReasoningResult) -> Dict:
        """
        Build consensus from three reasoning paradigms.

        Uses confidence-weighted integration with conflict detection.
        """
        results = [
            ('neural', neural, self.paradigm_weights['neural']),
            ('symbolic', symbolic, self.paradigm_weights['symbolic']),
            ('geometric', geometric, self.paradigm_weights['geometric']),
        ]

        # Weighted confidence
        total_weight = sum(r[2] * r[1].confidence for r in results)
        normalized_contributions = {}

        for name, result, weight in results:
            contribution = weight * result.confidence / (total_weight + 1e-10)
            normalized_contributions[name] = contribution

        # Select primary conclusion from highest-contributing paradigm
        best_paradigm = max(results, key=lambda r: r[2] * r[1].confidence)

        # Check for conflicts
        conclusions = [neural.conclusion, symbolic.conclusion, geometric.conclusion]
        conflict = self._detect_reasoning_conflict(conclusions)

        # Build consensus conclusion
        if conflict:
            consensus_conclusion = (
                f"MIXED: {best_paradigm[1].conclusion}. "
                f"Note: reasoning paradigms produced different perspectives. "
                f"Primary = {best_paradigm[0]} (contribution: {normalized_contributions[best_paradigm[0]]:.0%})"
            )
            consensus_confidence = best_paradigm[1].confidence * 0.7
        else:
            consensus_conclusion = (
                f"{best_paradigm[1].conclusion}. "
                f"All reasoning paradigms converge (agreement: "
                f"{sum(1 for r in results if r[1].confidence > 0.3)}/3)"
            )
            consensus_confidence = min(0.95, total_weight * 1.2)

        return {
            'conclusion': consensus_conclusion,
            'confidence': consensus_confidence,
            'primary_paradigm': best_paradigm[0],
            'contributions': normalized_contributions,
            'conflict_detected': conflict,
        }

    def _detect_reasoning_conflict(self, conclusions: List[str]) -> bool:
        """Detect if reasoning paradigms produced conflicting conclusions."""
        from .math.information import InformationEngine

        if len(conclusions) < 2:
            return False

        # Use NCD to check if conclusions are very different
        for i in range(len(conclusions)):
            for j in range(i + 1, len(conclusions)):
                ncd = InformationEngine.normalized_compression_distance(
                    conclusions[i], conclusions[j]
                )
                if ncd > 0.8:
                    return True

        return False

    def update_weights(self, feedback: Dict):
        """Update paradigm weights based on metacognitive feedback."""
        for paradigm in self.paradigm_weights:
            if paradigm in feedback:
                # Exponential moving average update
                alpha = 0.1
                self.paradigm_weights[paradigm] = (
                    (1 - alpha) * self.paradigm_weights[paradigm] +
                    alpha * feedback[paradigm]
                )

        # Normalize
        total = sum(self.paradigm_weights.values())
        for paradigm in self.paradigm_weights:
            self.paradigm_weights[paradigm] /= total
