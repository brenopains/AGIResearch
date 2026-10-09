"""
SymbolicEngine — Motor de Inferência Simbólica sobre o CGE
===========================================================
Implementa as operações de raciocínio dedutivo que o bind/bundle NÃO capturam:

    1. UNIFICAÇÃO: casar padrão de papéis semânticos no CGE
    2. MODUS PONENS: se A→B e A está ativo, inferir B
    3. MODUS TOLLENS: se A→B e ¬B está ativo, inferir ¬A
    4. INSTANCIAÇÃO UNIVERSAL: ∀x P(x)→Q(x), P(Sócrates) ⊢ Q(Sócrates)
    5. SILOGISMO HIPOTÉTICO: A→B, B→C ⊢ A→C

O motor opera como uma CAMADA sobre o CGE. Ele lê nós e arestas,
aplica regras, e escreve novas arestas de inferência de volta no CGE.

Não substitui a geometria HDC — a geometria é o substrato da percepção.
A inferência simbólica é o substrato do raciocínio dedutivo.
"""
import torch
import time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field


@dataclass
class InferenceStep:
    """Um passo de inferência com rastreabilidade completa."""
    rule: str           # "modus_ponens", "modus_tollens", "universal_instantiation", etc.
    premises: List[str] # IDs dos nós/arestas usados como premissa
    conclusion: str     # ID do nó/conclusão gerado
    confidence: float   # 0-1, derivado da confiança das premissas
    timestamp: float = field(default_factory=time.time)
    proof_chain: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "rule": self.rule,
            "premises": self.premises,
            "conclusion": self.conclusion,
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp,
        }


class SymbolicEngine:
    """
    Motor de inferência simbólica sobre o CGE.

    Opera em 3 níveis:
        N1 — Matching: encontra nós e arestas que satisfazem padrões
        N2 — Regras: aplica regras de inferência sobre os matches
        N3 — Escrita: escreve conclusões de volta no CGE como nós/arestas

    Uso:
        engine = SymbolicEngine(cge)
        engine.register_universal("human", "mortal")  # ∀x Human(x)→Mortal(x)
        engine.assert_fact("socrates", "human")        # Human(Socrates)
        result = engine.infer("socrates", "mortal")    # Mortal(Socrates)?
        # → InferenceStep(rule="universal_instantiation", conclusion="mortal", confidence=0.95)
    """

    def __init__(self, cge, bge=None):
        self.cge = cge
        self.bge = bge

        # Registro de regras universais: {subject: {property: confidence}}
        self.universal_rules: Dict[str, Dict[str, float]] = {}
        # Regras relacionais: {rel_type: [(subject, object, confidence)]}
        self.relational_rules: Dict[str, List[Tuple[str, str, float]]] = {}
        # Negações: {entity: {property: confidence}}
        self.negations: Dict[str, Dict[str, float]] = {}

        # Histórico de inferências
        self.inference_history: List[InferenceStep] = []
        self._proof_counter = 0

        # Cache BGE embeddings (evita encode repetido)
        self._bge_cache: Dict[str, torch.Tensor] = {}
        self._bge_string_sim: Dict[Tuple[str, str], float] = {}

        # Marcadores semânticos para tipos de aresta
        self.EDGE_IS_A = None       # "is a" (subclass)
        self.EDGE_HAS_PROP = None   # "has property"
        self.EDGE_NEGATION = None   # "not"
        self.EDGE_SUBJECT = None    # papel semântico: sujeito
        self.EDGE_VERB = None       # papel semântico: verbo
        self.EDGE_OBJECT = None     # papel semântico: objeto

    # ─── Configuração de Papéis Semânticos ──────────────────────────

    def register_semantic_roles(self, is_a, has_property, negation,
                                subj_role, verb_role, obj_role):
        """Registra os vetores HDV para papéis semânticos no CGE."""
        self.EDGE_IS_A = is_a
        self.EDGE_HAS_PROP = has_property
        self.EDGE_NEGATION = negation
        self.EDGE_SUBJECT = subj_role
        self.EDGE_VERB = verb_role
        self.EDGE_OBJECT = obj_role

    # ─── Registro de Conhecimento ───────────────────────────────────

    def register_universal(self, subject: str, property: str,
                           confidence: float = 0.9):
        """
        Registra uma regra universal: "todo subject tem property".
        Equivalente lógico: ∀x Subject(x) → Property(x)
        """
        if subject not in self.universal_rules:
            self.universal_rules[subject] = {}
        self.universal_rules[subject][property] = max(
            confidence,
            self.universal_rules[subject].get(property, 0)
        )

    def assert_fact(self, entity: str, property: str,
                    confidence: float = 0.9):
        """
        Registra um fato: "entity tem a propriedade property".
        Cria um nó entity no CGE e uma aresta HAS_PROP.
        """
        vec = self._bge_vec(entity)
        if vec is not None and entity not in self.cge.nodes:
            self.cge.create_node(entity, vec)
        prop_vec = self._bge_vec(property)
        if prop_vec is not None and property not in self.cge.nodes:
            self.cge.create_node(property, prop_vec)
        # Cria aresta is_a entre entity e property
        if self.EDGE_IS_A is not None and entity in self.cge.nodes and property in self.cge.nodes:
            is_a = self.EDGE_IS_A.to(self.cge.device)
            subj_vec = self.cge.nodes[entity]["vector"]
            prop_vec = self.cge.nodes[property]["vector"]
            if subj_vec is not None and prop_vec is not None:
                edge_vec = torch.sign(subj_vec.to(self.cge.device) * is_a * prop_vec.to(self.cge.device))
                self.cge.add_edge(entity, property, edge_vec)

    def assert_negation(self, entity: str, property: str,
                        confidence: float = 0.9):
        """Registra ¬property(entity). Usado por Modus Tollens."""
        if entity not in self.negations:
            self.negations[entity] = {}
        self.negations[entity][property] = max(
            confidence,
            self.negations[entity].get(property, 0)
        )

        # Adiciona aresta is_a ou has_property
        if self.EDGE_IS_A is not None:
            is_a = self.EDGE_IS_A.to(self.cge.device)
            subj_vec = self.cge.nodes[entity]["vector"] if entity in self.cge.nodes else None
            prop_vec = self.cge.nodes[property]["vector"] if property in self.cge.nodes else None
            if subj_vec is not None and prop_vec is not None:
                edge_vec = torch.sign(subj_vec * is_a * prop_vec)
                self.cge.add_edge(entity, property, edge_vec)

    # ─── Regras de Inferência ───────────────────────────────────────

    def infer(self, entity: str, property: str) -> Optional[InferenceStep]:
        """
        Tenta provar que entity tem a propriedade property.
        Suporta: Universal Instantiation, Modus Ponens, Barbara (HS).

        Para Barbara, encadeia regras universais transitivamente:
            se ∀x P(x)→Q(x) e ∀x Q(x)→R(x), então ∀x P(x)→R(x)
        """
        proofs = []

        # --- Estratégia 1: Instanciação Universal direta ---
        for universal_subj, props in self.universal_rules.items():
            if property in props:
                rule_conf = props[property]
                subj_match = self._check_is_a(entity, universal_subj)
                if subj_match:
                    conf = min(rule_conf, subj_match)
                    proofs.append(self._make_step(
                        "universal_instantiation",
                        [entity, universal_subj, property],
                        f"{entity}_{property}", conf))

        # --- Estratégia 2: Modus Ponens ---
        for edge in self.cge.edges:
            if not isinstance(edge, dict):
                continue
            if str(edge.get("from", "")).lower() == entity.lower():
                to_id = edge.get("to", "")
                if to_id and self._unify(to_id, property):
                    conf = edge.get("weight", 0.5)
                    if isinstance(conf, torch.Tensor):
                        conf = conf.mean().item()
                    proofs.append(self._make_step(
                        "modus_ponens", [entity, to_id],
                        f"{entity}_{to_id}", float(conf)))

        # --- Estratégia 3: Barbara (Silogismo Hipotético) via regras universais ---
        # Se entity satisfaz P, e P→Q→R, então entity satisfaz R
        for p_subj, p_props in self.universal_rules.items():
            for p_prop, p_conf in p_props.items():
                if self._check_is_a(entity, p_subj):
                    # p_prop é a propriedade intermediária (ex: "has zorb")
                    # Verifica se p_prop é sujeito de outra regra universal
                    if p_prop in self.universal_rules:
                        for q_prop, q_conf in self.universal_rules[p_prop].items():
                            if q_prop == property or self._unify(q_prop, property):
                                conf = min(float(p_conf), float(q_conf)) * 0.85
                                proofs.append(self._make_step(
                                    "hypothetical_syllogism",
                                    [p_subj, p_prop, q_prop],
                                    f"{entity}_{q_prop}", conf))

        # --- Estratégia 4: Barbara via BGE chain ---
        # Se entity é semanticamente similar a P, e P tem regra Q, e Q tem regra R,
        # tenta ver se a cadeia inteira leva a property.
        for p_subj, p_props in self.universal_rules.items():
            p_sim = self._bge_sim(entity, p_subj)
            if p_sim is not None and p_sim >= 0.2:
                for p_prop, p_conf in p_props.items():
                    if p_prop in self.universal_rules:
                        for q_prop, q_conf in self.universal_rules[p_prop].items():
                            if q_prop == property or self._unify(q_prop, property):
                                conf = float(p_sim) * min(float(p_conf), float(q_conf))
                                proofs.append(self._make_step(
                                    "hypothetical_syllogism_bge",
                                    [p_subj, p_prop, q_prop],
                                    f"{entity}_{q_prop}", conf))

        # Retorna o melhor proof
        if proofs:
            best = max(proofs, key=lambda s: s.confidence)
            self.inference_history.append(best)
            return best
        return None

    def infer_relational(self, entity_a: str, relation: str,
                         entity_b: str) -> Optional[InferenceStep]:
        """
        Infere relação transitiva: se A rel B e B rel C, então A rel C.
        Procura no CGE por cadeias relacionais.
        """
        for edge in self.cge.edges:
            if not isinstance(edge, dict):
                continue
            # Edge from A to X
            if str(edge.get("from", "")).lower() == entity_a.lower():
                mid = str(edge.get("to", ""))
                # mid could be B (the chain target)
                if self._unify(mid, entity_b):
                    conf = edge.get("weight", 0.5)
                    if isinstance(conf, torch.Tensor):
                        conf = conf.mean().item()
                    step = self._make_step(
                        "relational_direct",
                        [entity_a, mid],
                        f"{entity_a}_rel_{entity_b}",
                        float(conf))
                    self.inference_history.append(step)
                    return step
                # Chain: A → X and X → B
                for edge2 in self.cge.edges:
                    if not isinstance(edge2, dict):
                        continue
                    if str(edge2.get("from", "")).lower() == mid.lower() and \
                       str(edge2.get("to", "")).lower() == entity_b.lower():
                        conf1 = edge.get("weight", 0.5)
                        conf2 = edge2.get("weight", 0.5)
                        if isinstance(conf1, torch.Tensor): conf1 = conf1.mean().item()
                        if isinstance(conf2, torch.Tensor): conf2 = conf2.mean().item()
                        step = self._make_step(
                            "relational_chain",
                            [entity_a, mid, entity_b],
                            f"{entity_a}_rel_{entity_b}",
                            float(min(conf1, conf2)))
                        self.inference_history.append(step)
                        return step
        return None

    def infer_all(self, entity: str) -> List[InferenceStep]:
        """Infere todas as propriedades possíveis para uma entidade."""
        results = []
        # Coleta todas as propriedades conhecidas como targets de universal rules
        all_properties = set()
        for subj, props in self.universal_rules.items():
            for prop in props:
                all_properties.add(prop)
        # Também coleta do CGE
        for edge in self.cge.edges:
            if isinstance(edge, dict) and "to" in edge:
                all_properties.add(str(edge["to"]))

        for prop in all_properties:
            result = self.infer(entity, prop)
            if result:
                results.append(result)
        return results

    def modus_tollens(self, entity: str, property: str) -> Optional[InferenceStep]:
        """
        Modus Tollens (MT):
            ∀x (property(x) → consequence(x))   regra universal
            ¬consequence(entity)                negação registrada
            ⊢ ¬property(entity)                 conclusão
        
        Exemplo típico:
            ∀x (grom(x) → has_zorb(x))   universal: grom → has zorb
            ¬has_zorb(entity)             negação
            ⊢ ¬grom(entity)               conclusão
        
        Estratégia:
            1. Coleta todas as propriedades negadas para entity
            2. Para cada, encontra regras universais onde neg_prop é consequente
            3. O sujeito da regra é o que se conclui ¬(entity)
        """
        # 1. Obtém propriedades negadas para entity
        negated = self.negations.get(entity, {})
        
        # Se entity tem ¬property diretamente (caso raro)
        if property in negated:
            conf = max(negated[property], 0.5)
            step = self._make_step("modus_tollens",
                [entity, property, f"not_{property}"],
                f"not_{entity}", conf)
            self.inference_history.append(step)
            return step
        
        # 2. Para cada propriedade negada, busca regra universal onde ela é consequente
        for neg_prop, neg_conf in negated.items():
            for rule_subj, rule_props in self.universal_rules.items():
                if neg_prop in rule_props and self._unify(rule_subj, property):
                    conf = float(rule_props[neg_prop]) * float(neg_conf)
                    step = self._make_step("modus_tollens",
                        [entity, rule_subj, neg_prop],
                        f"not_{entity}", conf)
                    self.inference_history.append(step)
                    return step
        
        return None

    def _bge_vec(self, text: str) -> Optional[torch.Tensor]:
        """BGE encode com cache."""
        if text in self._bge_cache:
            return self._bge_cache[text]
        if self.bge:
            vec = self.bge.encode_bipolar([text])[0].to(self.cge.device)
            self._bge_cache[text] = vec
            return vec
        return None

    def _bge_sim(self, a: str, b: str) -> Optional[float]:
        """Similaridade BGE com cache."""
        key = (a, b) if a < b else (b, a)
        if key in self._bge_string_sim:
            return self._bge_string_sim[key]
        va = self._bge_vec(a)
        vb = self._bge_vec(b)
        if va is not None and vb is not None:
            sim = torch.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
            self._bge_string_sim[key] = sim
            return sim
        return None

    # ─── Métodos Auxiliares ─────────────────────────────────────────

    def _check_is_a(self, entity: str, category: str,
                    threshold: float = 0.25) -> Optional[float]:
        """
        Verifica se entity é da categoria 'category'.
        
        Ordem de precedência:
          1. Aresta explícita no CGE (assert_fact criou link)
          2. Matching de string exato
          3. BGE similarity (apenas fallback)
          
        A aresta explícita é o único critério forte. A similaridade BGE
        é falível porque palavras aleatórias podem ter alta similaridade
        no espaço semântico sem relação factual.
        """
        # 1. Verifica arestas explícitas no CGE
        for edge in self.cge.edges:
            if not isinstance(edge, dict):
                continue
            if (str(edge.get("from", "")).lower() == entity.lower() and
                str(edge.get("to", "")).lower() == category.lower()):
                # Found direct edge
                if entity in self.cge.nodes and category in self.cge.nodes:
                    vec_e = self.cge.nodes[entity].get("vector")
                    vec_c = self.cge.nodes[category].get("vector")
                    if vec_e is not None and vec_c is not None and isinstance(vec_e, torch.Tensor):
                        return float(torch.cosine_similarity(
                            vec_e.flatten().unsqueeze(0),
                            vec_c.flatten().unsqueeze(0)
                        ).item())
                return 0.8
        
        # 2. Matching de string exato
        if entity.lower() == category.lower():
            return 1.0
        if category.lower() in entity.lower():
            return 0.85
        
        # 3. Fallback BGE (threshold mais alto para evitar falsos positivos)
        sim = self._bge_sim(entity, category)
        if sim is not None and sim >= 0.5:  # threshold mais alto para fallback
            return float(sim)
        
        return None

    def _unify(self, a: str, b: str, threshold: float = 0.3) -> bool:
        """Verifica se dois conceitos são unificáveis."""
        if a is None or b is None:
            return False
        if a.lower() == b.lower():
            return True
        # Verifica similaridade no CGE
        if a in self.cge.nodes and b in self.cge.nodes:
            vec_a = self.cge.nodes[a].get("vector")
            vec_b = self.cge.nodes[b].get("vector")
            if vec_a is not None and vec_b is not None and isinstance(vec_a, torch.Tensor):
                sim = torch.cosine_similarity(
                    vec_a.flatten().unsqueeze(0),
                    vec_b.flatten().unsqueeze(0)
                ).item()
                return sim >= threshold
        # BGE fallback com cache
        sim = self._bge_sim(a, b)
        if sim is not None:
            return sim >= threshold
        return b.lower() in a.lower() or a.lower() in b.lower()

    def _make_step(self, rule: str, premises: List[str],
                   conclusion: str, confidence: float) -> InferenceStep:
        """Cria um passo de inferência padronizado."""
        self._proof_counter += 1
        return InferenceStep(
            rule=rule,
            premises=premises,
            conclusion=conclusion,
            confidence=min(max(confidence, 0.0), 1.0),
        )

    # ─── Utilitários ────────────────────────────────────────────────

    def get_proof_tree(self, entity: str, property: str) -> List[dict]:
        """Retorna a árvore de prova para uma inferência."""
        tree = []
        for step in self.inference_history:
            if step.conclusion == f"{entity}_{property}" or entity in step.conclusion:
                tree.append(step.to_dict())
        return tree

    def extract_universals_from_graph(self):
        """
        Varre o CGE em busca de padrões que indicam regras universais.
        Ex: se nó "human" tem aresta para "mortal" com alta confiança,
        registra como ∀human mortal.
        """
        for edge in self.cge.edges:
            if not isinstance(edge, dict):
                continue
            frm = str(edge.get("from", ""))
            to = str(edge.get("to", ""))
            weight = edge.get("weight", 0.5)
            if isinstance(weight, torch.Tensor):
                weight = weight.mean().item()
            if weight > 0.6 and frm and to and len(frm) > 2 and len(to) > 2:
                if "causes" in str(edge.get("edge_type", "")).lower() or \
                   "is_a" in str(edge.get("type", "")).lower():
                    self.register_universal(frm, to, float(weight))

    def reset(self, full=False):
        """Reseta o estado do motor simbólico entre raciocínios.
        Se full=True, limpa também cache BGE (mais agressivo).
        Mantém CGE e BGE."""
        self.universal_rules.clear()
        self.relational_rules.clear()
        self.negations.clear()
        self.inference_history.clear()
        self._proof_counter = 0
        if full:
            self._bge_cache.clear()
            self._bge_string_sim.clear()

    def clear_inferences(self):
        """Limpa o histórico de inferências."""
        self.inference_history.clear()
        self._proof_counter = 0
