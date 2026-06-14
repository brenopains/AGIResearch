"""
NovaMind — Layer 2: World Model (Causal Knowledge Graph)
=========================================================

Maintains a dynamic causal model of the world based on Judea Pearl's
causal calculus (do-calculus) and structural causal models (SCMs).

Why a causal model instead of a correlation model?
---------------------------------------------------
LLMs learn correlations: P(Y | X)  — "Y tends to follow X"
But correlation ≠ causation: roosters crow before dawn, but don't cause it.

A genuine intelligence must distinguish:
    - P(Y | X):     "Y given that X was observed" (correlation)
    - P(Y | do(X)): "Y given that X was deliberately set" (intervention)
    - P(Y_x | X'):  "Y would have been if X were different" (counterfactual)

Pearl's Causal Hierarchy:
    Level 1: Association — P(Y|X) — "seeing" (LLMs can do this)
    Level 2: Intervention — P(Y|do(X)) — "doing" (requires causal model)
    Level 3: Counterfactual — P(Y_x|X') — "imagining" (requires full SCM)

NovaMind implements all three levels.

References:
    - Pearl (2009): "Causality: Models, Reasoning, and Inference"
    - Pearl & Mackenzie (2018): "The Book of Why"
    - Bareinboim & Pearl (2016): "Causal Inference and the Data-Fusion Problem"
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import time

EPSILON = 1e-10


class CausalLink:
    """
    A directed causal edge in the world model.

    Encodes: X --[strength, type]--> Y
    meaning X causes/influences Y with given strength and mechanism type.
    """

    def __init__(self, source: str, target: str, link_type: str = "causes",
                 strength: float = 0.5, mechanism: str = "unknown"):
        self.source = source
        self.target = target
        self.link_type = link_type  # causes, prevents, enables, modulates
        self.strength = strength   # [0, 1] how strong the causal influence
        self.mechanism = mechanism  # description of the mechanism
        self.evidence_count = 1
        self.confidence = 0.5
        self.creation_time = time.time()
        self.last_updated = time.time()

    def update(self, new_strength: float, new_confidence: float):
        """Bayesian update of causal strength."""
        # Weighted average with more weight on accumulated evidence
        weight = self.evidence_count / (self.evidence_count + 1)
        self.strength = weight * self.strength + (1 - weight) * new_strength
        self.confidence = weight * self.confidence + (1 - weight) * new_confidence
        self.evidence_count += 1
        self.last_updated = time.time()

    def to_dict(self) -> dict:
        return {
            'source': self.source,
            'target': self.target,
            'type': self.link_type,
            'strength': self.strength,
            'confidence': self.confidence,
            'mechanism': self.mechanism,
            'evidence_count': self.evidence_count,
        }


class WorldNode:
    """A node in the world model — represents an entity or state variable."""

    def __init__(self, concept_id: str, label: str, node_type: str = "variable"):
        self.concept_id = concept_id
        self.label = label
        self.node_type = node_type  # variable, intervention, outcome
        self.state: Optional[float] = None  # Current state value
        self.prior_distribution = np.ones(10) / 10  # Discretized prior
        self.observed = False
        self.intervened = False
        self.creation_time = time.time()

    def to_dict(self) -> dict:
        return {
            'id': self.concept_id,
            'label': self.label,
            'type': self.node_type,
            'state': self.state,
            'observed': self.observed,
            'intervened': self.intervened,
        }


class WorldModel:
    """
    Layer 2: Causal Knowledge Graph.

    This is a Structural Causal Model (SCM) that maintains:
        - A directed acyclic graph (DAG) of causal relationships
        - Structural equations for each variable
        - Capability for interventional and counterfactual reasoning

    The world model continuously updates as new information arrives,
    and can answer three types of queries:
        1. Associational: "What is X given that I see Y?"
        2. Interventional: "What happens to X if I set Y to y?"
        3. Counterfactual: "Would X have been different if Y had been y'?"
    """

    def __init__(self, max_nodes: int = 500):
        self.max_nodes = max_nodes
        self.nodes: Dict[str, WorldNode] = {}
        self.edges: Dict[Tuple[str, str], CausalLink] = {}
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)  # outgoing edges
        self.reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)  # incoming edges
        self.temporal_events: List[Dict] = []  # History of events for temporal reasoning
        self.update_count = 0

    def add_node(self, concept_id: str, label: str, node_type: str = "variable") -> WorldNode:
        """Add a node to the world model, with LRU pruning to prevent RAM explosion."""
        if concept_id not in self.nodes:
            # Prevent O(N^2) explosion by keeping node count bounded (DeepSeek advice 2.1 & 2.5)
            if len(self.nodes) >= self.max_nodes:
                self._prune_graph()
                
            self.nodes[concept_id] = WorldNode(concept_id, label, node_type)
            # Track node activity/recency
            self.nodes[concept_id].last_active = time.time()
            self.nodes[concept_id].usage_count = 1
        else:
            self.nodes[concept_id].last_active = time.time()
            self.nodes[concept_id].usage_count = getattr(self.nodes[concept_id], 'usage_count', 0) + 1
            
        return self.nodes[concept_id]
        
    def _prune_graph(self):
        """
        DeepSeek Fix 2.1: Semantic continuous pruning.
        Removes the least recently used / least useful nodes and their edges
        to keep the memory footprint bounded.
        """
        # Score nodes based on recency and usage (simple LRU/LFU hybrid)
        now = time.time()
        scored_nodes = []
        for n_id, node in self.nodes.items():
            age = now - getattr(node, 'last_active', node.creation_time)
            usage = getattr(node, 'usage_count', 1)
            score = usage / (age + 1)
            scored_nodes.append((n_id, score))
            
        # Sort by score ascending (lowest score first)
        scored_nodes.sort(key=lambda x: x[1])
        
        # Remove bottom 20% of nodes
        nodes_to_remove = int(self.max_nodes * 0.2)
        pruned_ids = set([x[0] for x in scored_nodes[:nodes_to_remove]])
        
        # Remove nodes
        for n_id in pruned_ids:
            del self.nodes[n_id]
            
        # Remove associated edges
        edges_to_remove = []
        for edge_key in self.edges.keys():
            if edge_key[0] in pruned_ids or edge_key[1] in pruned_ids:
                edges_to_remove.append(edge_key)
                
        for edge_key in edges_to_remove:
            del self.edges[edge_key]
            
        # Clean up adjacency lists
        for n_id in pruned_ids:
            if n_id in self.adjacency:
                del self.adjacency[n_id]
            if n_id in self.reverse_adjacency:
                del self.reverse_adjacency[n_id]
                
        for adj_list in self.adjacency.values():
            adj_list.difference_update(pruned_ids)
        for rev_adj_list in self.reverse_adjacency.values():
            rev_adj_list.difference_update(pruned_ids)

    def add_causal_link(self, source: str, target: str, link_type: str = "causes",
                        strength: float = 0.5, mechanism: str = "unknown"):
        """
        Add or update a causal link.

        Checks for cycles (causal graphs must be DAGs) and updates
        existing links if they already exist.
        """
        # Ensure nodes exist
        if source not in self.nodes:
            self.add_node(source, source)
        if target not in self.nodes:
            self.add_node(target, target)

        edge_key = (source, target)

        if edge_key in self.edges:
            # Update existing link
            self.edges[edge_key].update(strength, 0.5 + 0.5 * strength)
        else:
            # Check for cycles before adding
            if not self._would_create_cycle(source, target):
                self.edges[edge_key] = CausalLink(source, target, link_type, strength, mechanism)
                self.adjacency[source].add(target)
                self.reverse_adjacency[target].add(source)

        self.update_count += 1

    def _would_create_cycle(self, source: str, target: str) -> bool:
        """Check if adding source → target would create a cycle."""
        # BFS from target to see if we can reach source
        visited = set()
        queue = [target]
        while queue:
            current = queue.pop(0)
            if current == source:
                return True
            if current in visited:
                continue
            visited.add(current)
            queue.extend(self.adjacency.get(current, set()))
        return False

    def integrate_perception(self, perception_report: Dict):
        """
        Integrate a perception report into the world model.

        Converts perceived concepts and relations into causal nodes and links.
        """
        # Add concepts as nodes
        for concept in perception_report.get('concepts', []):
            self.add_node(concept['id'], concept['label'], concept['type'])

        # Convert relations to causal links
        relation_to_causal = {
            'causes': ('causes', 0.7),
            'is_a': ('is_a', 0.9),
            'has': ('has', 0.8),
            'part_of': ('part_of', 0.8),
            'implies': ('implies', 0.6),
            'requires': ('requires', 0.7),
            'co_occurs': ('correlates', 0.3),
        }

        for rel in perception_report.get('relations', []):
            rel_type = rel.get('type', 'correlates')
            causal_type, default_strength = relation_to_causal.get(rel_type, ('correlates', 0.3))

            # Find node IDs for source and target labels
            source_id = None
            target_id = None
            for concept in perception_report['concepts']:
                if concept['label'] == rel.get('source'):
                    source_id = concept['id']
                elif concept['label'] == rel.get('target'):
                    target_id = concept['id']

            if source_id and target_id:
                self.add_causal_link(
                    source_id, target_id, causal_type,
                    default_strength * rel.get('confidence', 1.0),
                    mechanism=rel.get('extracted_from', 'perception')
                )

        # Record temporal event
        self.temporal_events.append({
            'time': time.time(),
            'concepts': [c['label'] for c in perception_report.get('concepts', [])],
            'relations': len(perception_report.get('relations', [])),
        })

    def query_association(self, target: str, evidence: Dict[str, float]) -> Dict:
        """
        Level 1 (Pearl's Ladder): Associational query.

        P(target | evidence) — What is target given that we observe evidence?

        Uses belief propagation over the causal graph.
        """
        if target not in self.nodes:
            return {'probability': 0.5, 'confidence': 0.0, 'detail': 'Unknown target'}

        # Set observed values
        for node_id, value in evidence.items():
            if node_id in self.nodes:
                self.nodes[node_id].state = value
                self.nodes[node_id].observed = True

        # Simple belief propagation (message passing)
        belief = self._propagate_beliefs(target, evidence)

        # Clear observations
        for node_id in evidence:
            if node_id in self.nodes:
                self.nodes[node_id].observed = False
                self.nodes[node_id].state = None

        return {
            'query_type': 'association',
            'target': target,
            'evidence': evidence,
            'probability': belief,
            'confidence': self._compute_query_confidence(target, evidence),
            'path': self._causal_path(list(evidence.keys())[0] if evidence else target, target),
        }

    def query_intervention(self, target: str, interventions: Dict[str, float]) -> Dict:
        """
        Level 2 (Pearl's Ladder): Interventional query.

        P(target | do(interventions)) — What happens to target if we SET
        variables to specific values?

        This is fundamentally different from association:
        do(X=x) means we CUT all incoming edges to X and set X=x.
        This eliminates confounding.
        """
        if target not in self.nodes:
            return {'probability': 0.5, 'confidence': 0.0, 'detail': 'Unknown target'}

        # Perform the do-calculus "surgery":
        # Remove all incoming edges to intervened variables
        saved_edges = {}
        for node_id, value in interventions.items():
            if node_id in self.nodes:
                self.nodes[node_id].state = value
                self.nodes[node_id].intervened = True

                # Save and remove incoming edges (the "do" operation)
                incoming = list(self.reverse_adjacency.get(node_id, set()))
                for parent in incoming:
                    edge_key = (parent, node_id)
                    if edge_key in self.edges:
                        saved_edges[edge_key] = self.edges[edge_key]

        # Propagate in the mutilated graph
        belief = self._propagate_beliefs(target, interventions)

        # Restore edges
        for edge_key, edge in saved_edges.items():
            self.edges[edge_key] = edge
        for node_id in interventions:
            if node_id in self.nodes:
                self.nodes[node_id].intervened = False
                self.nodes[node_id].state = None

        return {
            'query_type': 'intervention',
            'target': target,
            'interventions': interventions,
            'probability': belief,
            'confidence': self._compute_query_confidence(target, interventions),
            'causal_effect': belief - 0.5,  # Deviation from base rate
        }

    def query_counterfactual(self, target: str, actual: Dict[str, float],
                             hypothetical: Dict[str, float]) -> Dict:
        """
        Level 3 (Pearl's Ladder): Counterfactual query.

        P(Y_{x'} | X=x, Y=y) — "Given that X was x and Y was y,
        what WOULD Y have been if X had been x'?"

        Three-step process:
            1. Abduction: infer exogenous variables U from evidence
            2. Action: modify the model according to the hypothetical
            3. Prediction: compute the counterfactual outcome
        """
        if target not in self.nodes:
            return {'probability': 0.5, 'confidence': 0.0, 'detail': 'Unknown target'}

        # Step 1: Abduction — compute what the "background conditions" were
        # given the actual observations
        background = self._abduction(actual)

        # Step 2: Action — modify the model
        # Step 3: Prediction — compute Y under hypothetical with same background
        counterfactual_belief = self._predict_with_background(target, hypothetical, background)
        actual_belief = self._propagate_beliefs(target, actual)

        return {
            'query_type': 'counterfactual',
            'target': target,
            'actual_world': actual,
            'hypothetical_world': hypothetical,
            'actual_outcome': actual_belief,
            'counterfactual_outcome': counterfactual_belief,
            'causal_attribution': actual_belief - counterfactual_belief,
            'confidence': self._compute_query_confidence(target, actual) * 0.7,  # Lower for counterfactuals
        }

    def _propagate_beliefs(self, target: str, evidence: Dict[str, float]) -> float:
        """
        Simple belief propagation through the causal graph.

        In a full implementation, this would use exact inference (junction trees)
        or approximate inference (loopy BP, variational). Here we use a forward
        pass through the topological order.
        """
        # Topological sort
        topo_order = self._topological_sort()
        if not topo_order:
            return 0.5

        # Initialize beliefs
        beliefs = {}
        for node_id in topo_order:
            if node_id in evidence:
                beliefs[node_id] = evidence[node_id]
            else:
                beliefs[node_id] = 0.5  # Prior

        # Forward propagation
        for node_id in topo_order:
            if node_id in evidence:
                continue

            parents = self.reverse_adjacency.get(node_id, set())
            if not parents:
                continue

            # Combine parent influences
            total_influence = 0.0
            total_weight = 0.0

            for parent in parents:
                edge_key = (parent, node_id)
                if edge_key in self.edges:
                    link = self.edges[edge_key]
                    parent_belief = beliefs.get(parent, 0.5)

                    if link.link_type == 'causes':
                        influence = parent_belief * link.strength
                    elif link.link_type == 'prevents':
                        influence = (1 - parent_belief) * link.strength
                    elif link.link_type == 'enables':
                        influence = parent_belief * link.strength * 0.5 + 0.5
                    elif link.link_type == 'modulates':
                        influence = 0.5 + (parent_belief - 0.5) * link.strength
                    else:
                        influence = parent_belief * link.strength * 0.5 + 0.25

                    total_influence += influence * link.confidence
                    total_weight += link.confidence

            if total_weight > 0:
                beliefs[node_id] = np.clip(total_influence / total_weight, 0, 1)

        return beliefs.get(target, 0.5)

    def _abduction(self, observations: Dict[str, float]) -> Dict[str, float]:
        """
        Abduction step: infer "background conditions" (exogenous variables)
        from observations.
        """
        background = {}
        for node_id, value in observations.items():
            # Compute residual (unexplained by parents)
            predicted = self._propagate_beliefs(node_id, {
                k: v for k, v in observations.items() if k != node_id
            })
            background[f"u_{node_id}"] = value - predicted
        return background

    def _predict_with_background(self, target: str, hypothetical: Dict[str, float],
                                  background: Dict[str, float]) -> float:
        """Predict target under hypothetical conditions with fixed background."""
        # Combine hypothetical with background noise
        adjusted = dict(hypothetical)
        for key, noise in background.items():
            node_id = key.replace("u_", "")
            if node_id in adjusted:
                adjusted[node_id] = np.clip(adjusted[node_id] + noise, 0, 1)
        return self._propagate_beliefs(target, adjusted)

    def _topological_sort(self) -> List[str]:
        """Topological sort of the causal DAG."""
        in_degree = defaultdict(int)
        for node_id in self.nodes:
            in_degree[node_id] = len(self.reverse_adjacency.get(node_id, set()))

        queue = [n for n in self.nodes if in_degree[n] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for child in self.adjacency.get(node, set()):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return result

    def _causal_path(self, source: str, target: str) -> List[str]:
        """Find a causal path from source to target (BFS)."""
        if source not in self.nodes or target not in self.nodes:
            return []

        visited = set()
        queue = [(source, [source])]

        while queue:
            current, path = queue.pop(0)
            if current == target:
                return path
            if current in visited:
                continue
            visited.add(current)
            for child in self.adjacency.get(current, set()):
                queue.append((child, path + [child]))

        return []

    def _compute_query_confidence(self, target: str, evidence: Dict[str, float]) -> float:
        """Estimate confidence in a query result."""
        if target not in self.nodes:
            return 0.0

        # Confidence based on: path existence, link strengths, evidence count
        path_confidence = 0.0
        for ev_node in evidence:
            path = self._causal_path(ev_node, target)
            if path:
                # Product of edge confidences along path
                conf = 1.0
                for i in range(len(path) - 1):
                    edge_key = (path[i], path[i + 1])
                    if edge_key in self.edges:
                        conf *= self.edges[edge_key].confidence
                path_confidence = max(path_confidence, conf)

        return path_confidence

    def get_graph_data(self) -> Dict:
        """Return the causal graph for visualization."""
        return {
            'nodes': [n.to_dict() for n in self.nodes.values()],
            'edges': [e.to_dict() for e in self.edges.values()],
            'node_count': len(self.nodes),
            'edge_count': len(self.edges),
            'update_count': self.update_count,
        }

    def get_causal_chains(self, min_length: int = 2) -> List[List[str]]:
        """Find all causal chains of at least min_length."""
        chains = []
        topo = self._topological_sort()

        for start in topo:
            self._dfs_chains(start, [start], chains, min_length)

        return chains

    def _dfs_chains(self, node: str, current_path: List[str],
                     chains: List[List[str]], min_length: int):
        """DFS to find causal chains."""
        children = self.adjacency.get(node, set())
        if not children:
            if len(current_path) >= min_length:
                chains.append(list(current_path))
            return

        for child in children:
            if child not in current_path:  # Avoid cycles
                current_path.append(child)
                self._dfs_chains(child, current_path, chains, min_length)
                current_path.pop()

        if len(current_path) >= min_length:
            chains.append(list(current_path))
            
    def get_active_semantic_skeleton(self, context_concepts: List[str] = None, max_length: int = 5) -> List[str]:
        """
        DeepSeek Fix 3: Grounding.
        Extracts the most active causal chain to act as the semantic skeleton for Broca's Area.
        """
        if not self.edges:
            return context_concepts[:max_length] if context_concepts else []
            
        # Find the most recently active or strongest causal chain
        chains = self.get_causal_chains(min_length=2)
        if not chains:
            # Fallback to strongest edges
            sorted_edges = sorted(self.edges.values(), key=lambda e: e.strength * e.confidence, reverse=True)
            if not sorted_edges:
                return context_concepts[:max_length] if context_concepts else []
            best = sorted_edges[0]
            return [best.source, best.link_type, best.target]
            
        # Sort chains by average edge strength
        scored_chains = []
        for chain in chains:
            score = 0
            for i in range(len(chain)-1):
                edge_key = (chain[i], chain[i+1])
                if edge_key in self.edges:
                    score += self.edges[edge_key].strength * self.edges[edge_key].confidence
            scored_chains.append((score / len(chain), chain))
            
        scored_chains.sort(key=lambda x: x[0], reverse=True)
        best_chain = scored_chains[0][1]
        
        # Limit length
        return best_chain[:max_length]
