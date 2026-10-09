"""
CognitiveGraphEngine — Motor de Grafos Cognitivos Multimodais
==============================================================
Arquitetura Híbrida: Grafo Explícito (memória de trabalho) ↔ Holograma (forja/analogia).

Filosofia:
  O HDC não armazena conteúdo (pixels, ondas, palavras).
  O HDC armazena RELAÇÕES. O grafo explícito segura a cena.
  O holograma comprimido permite a Forja operar fora do fecho.

Fluxo:
  1. PERCEPÇÃO → cria Nós (objetos) e Arestas (relações) no grafo explícito
  2. CONSOLIDAÇÃO → compress_to_hologram() → vetor único HDV
  3. FORJA → opera no holograma (bind, analogia, generalização)
  4. IMAGINAÇÃO → decompress_from_hologram() → grafo reconstruído para inspeção

Referências:
  Damásio (Somatic Marker Hypothesis) — emoção bindada ao conceito
  Kahneman (System 1/2) — grafo = System 2, holograma = System 1
  Hawkins (Thousand Brains) — referência relativa entre patches
"""

import torch
import torchhd
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any

from .coglang import Manifold

DIM = 10000


class CognitiveGraphEngine:
    def __init__(self, hdc_memory=None, device: str = "cuda", bge_encoder=None):
        self.dim = DIM
        self.device = torch.device(device)
        self.memory = hdc_memory
        self.bge = bge_encoder  # BGEEncoder opcional para text -> HDV real

        # Papéis Espaciais (arestas do grafo visual)
        self.EDGE_RIGHT_OF = torchhd.random(1, DIM, device=self.device).squeeze(0)
        self.EDGE_LEFT_OF = torchhd.random(1, DIM, device=self.device).squeeze(0)
        self.EDGE_ABOVE = torchhd.random(1, DIM, device=self.device).squeeze(0)
        self.EDGE_BELOW = torchhd.random(1, DIM, device=self.device).squeeze(0)
        self.EDGE_BEHIND = torchhd.random(1, DIM, device=self.device).squeeze(0)
        self.EDGE_PART_OF = torchhd.random(1, DIM, device=self.device).squeeze(0)

        # Papéis Temporais
        self.EDGE_BEFORE = torchhd.random(1, DIM, device=self.device).squeeze(0)
        self.EDGE_AFTER = torchhd.random(1, DIM, device=self.device).squeeze(0)
        self.EDGE_TRANSITION = torchhd.random(1, DIM, device=self.device).squeeze(0)

        # Papéis Teológicos (causa e efeito)
        self.EDGE_CAUSES = torchhd.random(1, DIM, device=self.device).squeeze(0)
        self.EDGE_CAUSED_BY = torchhd.random(1, DIM, device=self.device).squeeze(0)

        # O Grafo Explícito (memória de trabalho, NÃO superbundle)
        self.nodes: Dict[str, dict] = {}
        self.edges: List[dict] = []
        self._last_update = time.time()

        print(f"[COGGRAPH] Motor de Grafos Cognitivos ativo (device={device})")

    # ─── Criação de Nós ──────────────────────────────────────────────

    def create_node(self, node_id: str, semantic_vec: torch.Tensor,
                    perceptual_vec: Optional[torch.Tensor] = None,
                    nt_vec: Optional[torch.Tensor] = None) -> dict:
        """
        Cria um nó semântico no grafo.

        O nó carrega:
        - semantic_vec: o conceito (ex: "bola", "fogo", "cachorro")
        - perceptual_vec: a aparência (ex: patch visual, espectrograma)
        - nt_vec: o marcador somático (ex: dor → medo)

        O bind final = conceito ⊗ aparência ⊗ emoção
        Isso garante que "fogo" com NT de dor seja diferente de "fogo" com NT de curiosidade.
        """
        vec = semantic_vec.flatten().to(self.device)

        if perceptual_vec is not None:
            vec = torch.sign(vec * perceptual_vec.flatten().to(self.device))

        if nt_vec is not None:
            vec = torch.sign(vec * nt_vec.flatten().to(self.device))

        self.nodes[node_id] = {
            "id": node_id,
            "vector": vec,
            "semantic": semantic_vec,
            "perceptual": perceptual_vec,
            "nt": nt_vec,
            "emotional_valence": 0.0,  # -1 (neg) a +1 (pos)
            "emotional_arousal": 0.0,  # 0 (calmo) a 1 (excitado)
            "confidence": 1.0,
            "state": "visible",
            "created_at": time.time(),
            "last_accessed": time.time(),
        }
        return self.nodes[node_id]

    def update_node_state(self, node_id: str, state: str = "visible",
                          confidence: Optional[float] = None):
        """Atualiza o estado de um nó (visible, hidden, occluded, imagined)."""
        if node_id in self.nodes:
            self.nodes[node_id]["state"] = state
            if confidence is not None:
                self.nodes[node_id]["confidence"] = confidence

    # ─── Criação de Arestas ──────────────────────────────────────────

    def add_edge(self, from_id: str, to_id: str, edge_type: torch.Tensor) -> Optional[dict]:
        """
        Cria uma aresta explícita no grafo.

        A aresta é o bind triplo: Nó_A ⊗ Relação ⊗ Nó_B
        Isso permite que o unbinding recupere qualquer dos três elementos.
        """
        if from_id not in self.nodes or to_id not in self.nodes:
            return None

        vec_a = self.nodes[from_id]["vector"]
        vec_b = self.nodes[to_id]["vector"]
        edge_vec = torch.sign(vec_a * edge_type.to(self.device) * vec_b)

        edge_data = {
            "from": from_id,
            "to": to_id,
            "edge_type": edge_type,
            "vector": edge_vec,
            "confidence": 1.0,
        }
        self.edges.append(edge_data)
        return edge_data

    def add_spatial_edge(self, from_id: str, to_id: str,
                         relation: str) -> Optional[dict]:
        """Atalho: adiciona aresta espacial por nome."""
        type_map = {
            "right_of": self.EDGE_RIGHT_OF,
            "left_of": self.EDGE_LEFT_OF,
            "above": self.EDGE_ABOVE,
            "below": self.EDGE_BELOW,
            "behind": self.EDGE_BEHIND,
            "part_of": self.EDGE_PART_OF,
        }
        edge_type = type_map.get(relation)
        if edge_type is None:
            return None
        return self.add_edge(from_id, to_id, edge_type)

    def add_temporal_edge(self, from_id: str, to_id: str,
                          relation: str) -> Optional[dict]:
        """Atalho: adiciona aresta temporal por nome."""
        type_map = {
            "before": self.EDGE_BEFORE,
            "after": self.EDGE_AFTER,
            "transitions_to": self.EDGE_TRANSITION,
        }
        edge_type = type_map.get(relation)
        if edge_type is None:
            return None
        return self.add_edge(from_id, to_id, edge_type)

    def add_causal_edge(self, from_id: str, to_id: str,
                        relation: str = "causes") -> Optional[dict]:
        """Atalho: adiciona aresta causal."""
        edge_type = self.EDGE_CAUSES if relation == "causes" else self.EDGE_CAUSED_BY
        return self.add_edge(from_id, to_id, edge_type)

    # ─── Atenção do Maestro ──────────────────────────────────────────

    def query_attention(self, concept_vec: torch.Tensor,
                        top_k: int = 5) -> List[Tuple[str, dict, float]]:
        """
        O Maestro pergunta: "Qual nó se parece com este conceito?"

        Retorna os top_k nós mais similares.
        Não precisa decodificar a cena inteira — busca direta no grafo.
        """
        cv = concept_vec.flatten().to(self.device)
        scored = []
        for nid, ndata in self.nodes.items():
            sim = float(torch.cosine_similarity(
                cv.unsqueeze(0), ndata["vector"].unsqueeze(0)))
            scored.append((nid, ndata, sim))

        scored.sort(key=lambda x: -x[2])
        return scored[:top_k]

    def query_edges(self, node_id: str) -> List[dict]:
        """Retorna todas as arestas que envolvem um nó específico."""
        return [e for e in self.edges
                if e["from"] == node_id or e["to"] == node_id]

    def query_neighbors(self, node_id: str) -> List[Tuple[str, str, dict]]:
        """Retorna os vizinhos de um nó com o tipo de relação."""
        neighbors = []
        for e in self.edges:
            if e["from"] == node_id:
                neighbors.append((e["to"], "->", e))
            if e["to"] == node_id:
                neighbors.append((e["from"], "<-", e))
        return neighbors

    # ─── Consolidação: Grafo → Holograma ─────────────────────────────

    def compress_to_hologram(self) -> torch.Tensor:
        """
        CONSOLIDAÇÃO DA MEMÓRIA: esmaga o grafo explícito num único vetor HDC.

        A Forja (FractalMaestro) opera neste holograma para:
        - Bind analógico (cachorro ⊗ carro = dog-car imaginário)
        - Generalização fora do fecho
        - Busca associativa

        ATENÇÃO: o holograma é um superbundle e tem ruído de crosstalk.
        Mas isso é INTENCIONAL — a Forja opera melhor no ruído rico
        do que em structs rígidas. O cérebro faz exatamente isso:
        o hipocampo comprime para o neocórtex durante o sono.
        """
        if not self.nodes and not self.edges:
            return torch.zeros(DIM, device=self.device)

        accumulator = torch.zeros(DIM, device=self.device)
        n_terms = 0

        for ndata in self.nodes.values():
            accumulator = accumulator + ndata["vector"] * ndata["confidence"]
            n_terms += 1

        for edata in self.edges:
            accumulator = accumulator + edata["vector"] * edata["confidence"]
            n_terms += 1

        return torch.sign(accumulator) if n_terms > 0 else accumulator

    def decompress_from_hologram(self, hologram: torch.Tensor,
                                 top_k_nodes: int = 10) -> Dict:
        """
        IMAGINAÇÃO: desempacota um holograma de volta para um grafo explícito.

        O Maestro pode "olhar" o que a Forja inventou:
        1. Busca no HDV os conceitos que ressoam no holograma
        2. Testa quais relações espaciais/temporais estão ativas
        3. Reconstrói o grafo na memória de trabalho

        Returns:
            dict com nodes e edges reconstruídos
        """
        holo = hologram.flatten().to(self.device)
        reconstructed = {"nodes": {}, "edges": []}

        # 1a. Recupera nós da memoria HDV (se disponivel)
        if self.memory is not None:
            results = self.memory.query_vec(holo, top_k=top_k_nodes, min_sim=0.10)
            for sim, meta, source in results:
                nid = meta.get("id", f"node_{len(reconstructed['nodes'])}")
                text = meta.get("text", "")[:50]
                recovered_vec = meta.get("_vec")
                if recovered_vec is not None:
                    recovered_vec = recovered_vec.to(self.device)
                    reconstructed["nodes"][nid] = {
                        "id": nid,
                        "vector": recovered_vec,
                        "text": text,
                        "confidence": sim,
                        "state": "imagined",
                        "source": source,
                    }

        # 1b. Recupera nos do proprio grafo (CGE local)
        # Importante para HDVs aleatorios que nao estao no HDCMemory
        for nid, ndata in self.nodes.items():
            if nid in reconstructed["nodes"]:
                continue
            vec = ndata["vector"].to(self.device)
            sim = float(torch.cosine_similarity(holo.unsqueeze(0), vec.unsqueeze(0)))
            if sim > 0.10:
                reconstructed["nodes"][nid] = {
                    "id": nid,
                    "vector": vec,
                    "text": nid,
                    "confidence": sim,
                    "state": ndata["state"],
                    "source": "cge_local",
                }

        # 2. Testa arestas entre os nós recuperados
        node_ids = list(reconstructed["nodes"].keys())
        edge_types = [
            ("right_of", self.EDGE_RIGHT_OF),
            ("left_of", self.EDGE_LEFT_OF),
            ("above", self.EDGE_ABOVE),
            ("below", self.EDGE_BELOW),
            ("behind", self.EDGE_BEHIND),
            ("part_of", self.EDGE_PART_OF),
            ("before", self.EDGE_BEFORE),
            ("causes", self.EDGE_CAUSES),
        ]

        for i, a_id in enumerate(node_ids):
            for b_id in node_ids[i + 1:]:
                va = reconstructed["nodes"][a_id]["vector"]
                vb = reconstructed["nodes"][b_id]["vector"]

                for rel_name, edge_vec in edge_types:
                    test = torch.sign(va * edge_vec * vb)
                    sim = float(torch.cosine_similarity(
                        holo.unsqueeze(0), test.unsqueeze(0)))

                    if sim > 0.15:
                        reconstructed["edges"].append({
                            "from": a_id,
                            "to": b_id,
                            "relation": rel_name,
                            "confidence": sim,
                        })

        return reconstructed

    # ─── Manipulação Imaginativa ─────────────────────────────────────

    def imagine_spatial_change(self, node_id: str, new_relation: str,
                                target_id: str) -> Optional[torch.Tensor]:
        """
        IMAGINAÇÃO ESPACIAL: "E se a bola estivesse à ESQUERDA da caixa?"

        Não modifica o grafo real. Retorna o holograma modificado.
        O Maestro pode medir a Free Energy da nova configuração.
        """
        if node_id not in self.nodes or target_id not in self.nodes:
            return None

        current_holo = self.compress_to_hologram()

        # Remove as arestas atuais envolvendo node_id (simulado)
        # Na prática: subtrai o bundle das arestas antigas
        old_edges = self.query_edges(node_id)
        for oe in old_edges:
            current_holo = torch.sign(current_holo - oe["vector"])

        # Cria a nova aresta imaginada
        type_map = {
            "right_of": self.EDGE_RIGHT_OF,
            "left_of": self.EDGE_LEFT_OF,
            "above": self.EDGE_ABOVE,
            "below": self.EDGE_BELOW,
            "behind": self.EDGE_BEHIND,
        }
        edge_type = type_map.get(new_relation)
        if edge_type is None:
            return None

        new_edge = self.add_edge(node_id, target_id, edge_type)
        if new_edge is None:
            return current_holo

        imagined_holo = torch.sign(current_holo + new_edge["vector"])
        return imagined_holo

    def imagine_temporal_transition(self, action_vec: torch.Tensor) -> torch.Tensor:
        """
        IMAGINAÇÃO TEMPORAL: "E se eu aplicar esta ação?"

        Percorre as arestas de transição temporal e aplica a ação,
        retornando o holograma do estado futuro predito.
        """
        current_holo = self.compress_to_hologram()
        av = action_vec.flatten().to(self.device)

        for edge in self.edges:
            if edge["edge_type"] is self.EDGE_TRANSITION:
                predicted = torch.sign(edge["vector"] * av)
                current_holo = torch.sign(current_holo + predicted)

        return current_holo

    # ─── Interface com o Maestro ─────────────────────────────────────

    def get_scene_text(self) -> str:
        """Retorna uma descrição textual do grafo atual para a Broca."""
        lines = []
        for nid, ndata in self.nodes.items():
            status = f"[{ndata['state']}]" if ndata["state"] != "visible" else ""
            lines.append(f"  {nid} {status} (conf={ndata['confidence']:.2f})")

        for edge in self.edges:
            rel_name = "?"
            for name, attr in [("right_of", self.EDGE_RIGHT_OF),
                                ("left_of", self.EDGE_LEFT_OF),
                                ("above", self.EDGE_ABOVE),
                                ("behind", self.EDGE_BEHIND),
                                ("part_of", self.EDGE_PART_OF),
                                ("before", self.EDGE_BEFORE),
                                ("causes", self.EDGE_CAUSES)]:
                if attr is edge["edge_type"] or torch.equal(attr, edge["edge_type"]):
                    rel_name = name
                    break
            lines.append(f"  {edge['from']} --{rel_name}--> {edge['to']}")

        return "\n".join(lines) if lines else "(grafo vazio)"

    # ─── Forja: Consolidação de Padrões ─────────────────────────────

    def forge_consolidate(self, min_chain_confidence: float = 0.3) -> int:
        """
        FORJA: Consolida cadeias transitivas em arestas diretas.

        Diferente da abordagem holografica (que falha com HDVs ortogonais),
        a Forja percorre o GRAFO EXPLICITO e detecta padroes A -> B -> C.
        Quando encontra uma cadeia com confianca suficiente, cria A -> C.

        Isso e o analogo ao hipocampo consolidando memorias:
        deteccao de padrao -> criacao de atalho -> abstracao.

        Returns:
            int: numero de novas arestas criadas
        """
        created = 0
        for mid_id in list(self.nodes.keys()):
            incoming = [(e["from"], e) for e in self.edges if e["to"] == mid_id]
            outgoing = [(e["to"], e) for e in self.edges if e["from"] == mid_id]

            if not incoming or not outgoing:
                continue

            for from_id, in_edge in incoming:
                for to_id, out_edge in outgoing:
                    if from_id == to_id:
                        continue

                    if any(e["from"] == from_id and e["to"] == to_id for e in self.edges):
                        continue

                    if in_edge["edge_type"] is not out_edge["edge_type"]:
                        continue

                    # Confianca da cadeia = produto das confiancas
                    chain_conf = in_edge.get("confidence", 1.0) * out_edge.get("confidence", 1.0)
                    if chain_conf < min_chain_confidence:
                        continue

                    edge_type = in_edge["edge_type"]
                    new_edge = self.add_edge(from_id, to_id, edge_type)
                    if new_edge is not None:
                        new_edge["confidence"] = chain_conf
                        created += 1

        return created

    # ─── Forja: Abstração (Criação de Conceitos Novos) ───────────

    def forge_abstract(self, instance_ids: List[str], concept_name: str,
                       min_sim: float = 0.15) -> Optional[dict]:
        """
        FORJA: Cria um conceito abstrato a partir de multiplas instancias.

        Funciona como o cerebro: apos ver varias cadeiras especificas,
        extrai o padrao comum (cadeira) como um novo conceito via bundle.

        1. Bundle das instancias → vetor do conceito abstrato
        2. Cria no para o conceito
        3. Cria arestas 'part_of' das instancias → conceito
        4. Detecta propriedades comuns e cria arestas do conceito → propriedades

        Args:
            instance_ids: lista de IDs dos nos instancia
            concept_name: nome do novo conceito
            min_sim: similaridade minima para considerar propriedade comum

        Returns:
            dict do no criado, ou None se falhar
        """
        # Validar que todas as instancias existem
        valid_ids = [i for i in instance_ids if i in self.nodes]
        if len(valid_ids) < 2:
            return None

        if concept_name in self.nodes:
            return self.nodes[concept_name]

        # 1. Bundle das instancias
        vectors = [self.nodes[i]["vector"].to(self.device) for i in valid_ids]
        stacked = torch.stack(vectors)
        bundle = torch.sign(stacked.sum(dim=0))

        # 2. Criar no para o conceito abstrato
        self.create_node(concept_name, bundle)
        concept_node = self.nodes[concept_name]
        concept_node["confidence"] = 0.8  # comeca com confianca moderada
        concept_node["state"] = "imagined"

        # 3. Criar arestas 'part_of' das instancias → conceito
        for iid in valid_ids:
            self.add_edge(iid, concept_name, self.EDGE_PART_OF)

        # 4. Detectar propriedades comuns
        #    Se 2+ instancias tem aresta CAUSES para o mesmo no,
        #    criar aresta do conceito abstrato para esse no
        part_of_targets = {concept_name}
        property_votes: Dict[str, float] = {}
        for edge in self.edges:
            if edge["from"] in valid_ids and edge["to"] not in part_of_targets:
                prop = edge["to"]
                if prop not in property_votes:
                    property_votes[prop] = 0
                property_votes[prop] += 1

        for prop, votes in property_votes.items():
            if votes >= 2:  # 2+ instancias compartilham
                self.add_edge(concept_name, prop, self.EDGE_CAUSES)

        return concept_node

    # ─── Forja: Ciclo de Consolidação (Sono / Ócio) ─────────────

    def consolidation_cycle(self, min_chain_conf: float = 0.3,
                            min_abstract_sim: float = 0.20,
                            prune_below: float = 0.1) -> dict:
        """
        Ciclo completo de consolidacao, analogo ao sono hipocampal.

        1. FORJA CONSOLIDATE: detecta padroes A->B->C, cria A->C direto
        2. FORJA ABSTRACT: agrupa nos similares em conceitos abstratos
        3. DECAY: reduz confianca de nos/arestas nao usados
        4. PRUNE: remove nos/arestas abaixo do limiar

        Returns:
            dict com estatisticas: edges_created, concepts_created, pruned
        """
        stats = {"edges_created": 0, "concepts_created": 0,
                 "pruned_nodes": 0, "pruned_edges": 0}

        # 1. Consolidar cadeias
        stats["edges_created"] = self.forge_consolidate(min_chain_confidence=min_chain_conf)

        # 2. Abstrair clusters de nos com arestas compartilhadas
        #    Agrupa nos que compartilham 3+ vizinhos (common neighbors)
        node_list = list(self.nodes.keys())
        clusters = []
        for i, a in enumerate(node_list):
            if a.startswith("abstract_"):
                continue
            neighbors_a = set()
            for e in self.edges:
                if e["from"] == a: neighbors_a.add(e["to"])
                if e["to"] == a: neighbors_a.add(e["from"])
            for b in node_list[i + 1:]:
                if b.startswith("abstract_"):
                    continue
                neighbors_b = set()
                for e in self.edges:
                    if e["from"] == b: neighbors_b.add(e["to"])
                    if e["to"] == b: neighbors_b.add(e["from"])
                shared = neighbors_a & neighbors_b
                if len(shared) >= 3:
                    clusters.append((a, b, list(shared)))

        # Para cada cluster, criar abstração
        seen = set()
        for a, b, shared in clusters:
            if a in seen or b in seen:
                continue
            abstract_name = f"abstract_{a}_{b}"
            result = self.forge_abstract([a, b], abstract_name,
                                          min_sim=min_abstract_sim)
            if result is not None:
                seen.add(a)
                seen.add(b)
                stats["concepts_created"] += 1

        # 3. Decay de confianca (nos nao utilizados perdem forca)
        for nid, ndata in self.nodes.items():
            if ndata["state"] == "visible" and ndata["confidence"] > 0.1:
                ndata["confidence"] *= 0.99  # decay suave
            elif ndata["state"] != "visible":
                ndata["confidence"] *= 0.95  # decay mais rapido

        # 4. Poda: remover nos/arestas abaixo do limiar
        to_remove = [nid for nid, ndata in self.nodes.items()
                     if ndata["confidence"] < prune_below and ndata["state"] != "visible"]
        for nid in to_remove:
            del self.nodes[nid]
            stats["pruned_nodes"] += 1

        # Remover arestas que referenciam nos removidos
        self.edges = [e for e in self.edges
                      if e["from"] in self.nodes and e["to"] in self.nodes
                      and e.get("confidence", 1.0) >= prune_below]
        stats["pruned_edges"] = len(to_remove)  # aproximado

        return stats

    # ─── Tag Emocional ─────────────────────────────────────────

    def tag_emotion(self, node_id: str, valence: float, arousal: float):
        """Marca um no com estado emocional (NT state)."""
        if node_id in self.nodes:
            self.nodes[node_id]["emotional_valence"] = max(-1.0, min(1.0, valence))
            self.nodes[node_id]["emotional_arousal"] = max(0.0, min(1.0, arousal))

    def recall_by_emotion(self, target_valence: float, top_k: int = 5) -> List[Dict]:
        """Recupera nos com valencia emocional similar ao estado atual."""
        scored = []
        for nid, ndata in self.nodes.items():
            sim = 1.0 - abs(ndata["emotional_valence"] - target_valence)
            scored.append((sim, nid, ndata))
        scored.sort(key=lambda x: -x[0])
        return [{"id": nid, "confidence": round(s, 3), "valence": nd["emotional_valence"]}
                for s, nid, nd in scored[:top_k]]

    # ─── Inibição Lateral ──────────────────────────────────────

    def lateral_inhibition(self, strength: float = 0.3, sim_threshold: float = 0.50):
        """
        INIBICAO LATERAL: nos com alta sobreposicao de vizinhos competem.

        Analogia: no cerebro, quando um neuronio dispara, ele inibe
        neuronios vizinhos (GABA). Aqui, nos que compartilham 60%+
        dos vizinhos tem o mais fraco suprimido.

        Args:
            strength: quanto o perdedor perde de confianca (0-1)
            sim_threshold: fracao de vizinhos compartilhados para competir
        """
        node_list = list(self.nodes.keys())
        for i, a in enumerate(node_list):
            neighbors_a = self._get_neighbors(a)
            if not neighbors_a:
                continue
            for b in node_list[i + 1:]:
                if b not in self.nodes:
                    continue
                neighbors_b = self._get_neighbors(b)
                if not neighbors_b:
                    continue
                shared = neighbors_a & neighbors_b
                union = neighbors_a | neighbors_b
                jaccard = len(shared) / len(union) if union else 0
                if jaccard >= sim_threshold:
                    conf_a = self.nodes[a].get("confidence", 1.0)
                    conf_b = self.nodes[b].get("confidence", 1.0)
                    if conf_a > conf_b:
                        self.nodes[b]["confidence"] *= (1.0 - strength)
                    elif conf_b > conf_a:
                        self.nodes[a]["confidence"] *= (1.0 - strength)

    def _get_neighbors(self, node_id: str) -> set:
        """Retorna IDs dos vizinhos de um no."""
        neighbors = set()
        for e in self.edges:
            if e["from"] == node_id:
                neighbors.add(e["to"])
            if e["to"] == node_id:
                neighbors.add(e["from"])
        return neighbors

    # ─── Escrita Automatica no CGE ─────────────────────────────

    def write_text_concepts(self, text: str, emotional_valence: float = 0.0,
                            emotional_arousal: float = 0.0) -> int:
        """
        ESCREVE conceitos de um texto no CGE automaticamente.

        Extrai palavras significativas, cria nos para cada uma,
        e arestas CAUSES entre palavras que co-ocorrem.

        Args:
            text: texto para extrair conceitos
            emotional_valence: estado emocional atual (-1 a +1)
            emotional_arousal: excitacao atual (0 a 1)

        Returns:
            numero de nos criados
        """
        # Tokenizacao simples (sem depender de spaCy aqui)
        words = text.lower().split()
        # Filtra stopwords basicas
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                     "at", "to", "for", "of", "with", "and", "or", "but", "not",
                     "this", "that", "these", "those", "it", "its", "be", "been",
                     "being", "have", "has", "had", "do", "does", "did", "will",
                     "would", "could", "should", "may", "might", "shall", "can"}
        words = [w.strip(".,!?;:()[]{}'\"") for w in words
                 if w.strip(".,!?;:()[]{}'\"") not in stopwords
                 and len(w.strip(".,!?;:()[]{}'\"")) > 2]

        created = 0
        seen = set()
        for w in words:
            if w in seen:
                continue
            seen.add(w)
            if w not in self.nodes:
                if self.bge is not None:
                    vec = self.bge.encode_bipolar([w]).squeeze(0).to(self.device)
                else:
                    import hashlib
                    h = hashlib.md5(w.encode()).hexdigest()
                    seed = int(h[:8], 16)
                    rng = torch.Generator(device=self.device)
                    rng.manual_seed(seed)
                    vec = torch.sign(torch.randn(self.dim, generator=rng, device=self.device))
                self.create_node(w, vec)
                self.tag_emotion(w, emotional_valence, emotional_arousal)
                created += 1

        # Criar arestas entre palavras que co-ocorrem na mesma frase
        for i in range(len(words) - 1):
            a, b = words[i], words[i + 1]
            if a in self.nodes and b in self.nodes:
                if a != b:
                    already = any(e["from"] == a and e["to"] == b for e in self.edges)
                    if not already:
                        self.add_edge(a, b, self.EDGE_CAUSES)

        return created

    # ─── Ativação em Espalhamento (Spreading Activation) ─────────

    def spread_activation(self, seed_ids: List[str], decay: float = 0.5,
                          max_hops: int = 2) -> Dict[str, float]:
        """
        SPREADING ACTIVATION: ativa nos vizinhos a partir de sementes.

        Analogia: no cerebro, quando um conceito e ativado,
        conceitos relacionados sao pre-ativados (preparation).

        Args:
            seed_ids: nos que comecam ativados
            decay: fator de decaimento por hop (0.5 = 50% a cada hop)
            max_hops: distancia maxima de propagacao

        Returns:
            dict {node_id: activation_level}
        """
        activation = {nid: 0.0 for nid in self.nodes}
        for sid in seed_ids:
            if sid in activation:
                activation[sid] = 1.0

        current_frontier = set(seed_ids) & set(self.nodes.keys())
        for hop in range(1, max_hops + 1):
            next_frontier = set()
            for nid in current_frontier:
                for edge in self.edges:
                    neighbor = None
                    if edge["from"] == nid:
                        neighbor = edge["to"]
                    elif edge["to"] == nid:
                        neighbor = edge["from"]
                    if neighbor and neighbor in activation:
                        new_act = activation[nid] * decay
                        if new_act > activation[neighbor]:
                            activation[neighbor] = new_act
                            next_frontier.add(neighbor)
            current_frontier = next_frontier
            if not current_frontier:
                break

        return activation

    # ─── Grounding Causal: Intervenção e Teste de Modelo ───────

    def causal_intervention(self, node_id: str, remove: bool = True) -> Dict:
        """
        INTERVENCAO CAUSAL: simula o que acontece se remover um no.

        Diferente de associacao (A CAUSES B), intervencao testa:
        "Se eu remover A, B ainda acontece?"

        Returns:
            dict com predicoes: {vizinho_id: mudanca_esperada}
        """
        predictions = {}
        if node_id not in self.nodes:
            return predictions

        if remove:
            for edge in self.edges:
                if edge["from"] == node_id:
                    predictions[edge["to"]] = -edge.get("confidence", 1.0)

        return predictions

    def test_causal_model(self, intervention: Dict, observation: Dict) -> float:
        """
        TESTE DE MODELO CAUSAL: compara predicao com observacao real.

        Returns:
            erro de predicao (0 = perfeito, 1 = totalmente errado)
        """
        if not intervention:
            return 0.0
        errors = []
        for node_id, predicted_change in intervention.items():
            observed = observation.get(node_id, 0)
            error = abs(predicted_change - observed) / 2.0
            errors.append(min(1.0, error))
        return sum(errors) / len(errors) if errors else 0.0

    def plan_to_goal(self, start: str, target: str,
                     max_depth: int = 5) -> Optional[List[str]]:
        """
        PLANEJAMENTO: BFS no grafo CAUSES de start ate target.

        Returns:
            lista de nos no caminho, ou None
        """
        if start not in self.nodes or target not in self.nodes:
            return None
        visited = {start}
        queue = [(start, [start])]
        while queue:
            current, path = queue.pop(0)
            if current == target:
                return path
            for edge in self.edges:
                if edge["from"] == current and edge["to"] not in visited:
                    visited.add(edge["to"])
                    queue.append((edge["to"], path + [edge["to"]]))
            if len(path) > max_depth:
                continue
        return None

    # ─── Compressão Conceitual Robusta ─────────────────────────

    def refine_abstraction(self, concept_name: str,
                           counterexample_id: str) -> Optional[dict]:
        """
        REFINA conceito abstrato baseado em contraexemplo.

        Se "cadeira" foi abstraido de instancias e aparece "banco",
        este metodo cria distincao.

        Returns:
            dict com refinamento
        """
        if concept_name not in self.nodes or counterexample_id not in self.nodes:
            return None
        concept_vec = self.nodes[concept_name]["vector"]
        counter_vec = self.nodes[counterexample_id]["vector"]
        sim = float(torch.cosine_similarity(
            concept_vec.unsqueeze(0), counter_vec.unsqueeze(0)))
        if sim < 0.3:
            return None

        concept_props = set()
        counter_props = set()
        for edge in self.edges:
            if edge["from"] == concept_name:
                concept_props.add(edge["to"])
            if edge["from"] == counterexample_id:
                counter_props.add(edge["to"])

        differentiating = concept_props - counter_props
        if not differentiating:
            return None

        refined_vec = torch.sign(concept_vec + counter_vec)
        refined_name = f"{concept_name}_refined"
        if refined_name not in self.nodes:
            self.create_node(refined_name, refined_vec)
            self.nodes[refined_name]["state"] = "imagined"
            self.nodes[refined_name]["confidence"] = 0.7
            shared = concept_props & counter_props
            for prop in shared:
                self.add_edge(refined_name, prop, self.EDGE_CAUSES)

        return {
            "original": concept_name,
            "refined": refined_name,
            "sim_to_counterexample": round(sim, 3),
            "differentiating_props": list(differentiating),
        }

    def get_concept_hierarchy(self, root: str, depth: int = 0,
                              max_depth: int = 5) -> Dict:
        """Retorna hierarquia conceitual a partir de um no raiz."""
        if root not in self.nodes or depth > max_depth:
            return {"name": root, "children": []}
        ndata = self.nodes[root]
        children = []
        visited = {root}
        for edge in self.edges:
            if edge["from"] == root and edge["to"] not in visited:
                visited.add(edge["to"])
                child = self.get_concept_hierarchy(
                    edge["to"], depth + 1, max_depth)
                if child["name"] != root:
                    children.append(child)
        return {
            "name": root,
            "confidence": round(ndata.get("confidence", 1.0), 2),
            "state": ndata.get("state", "visible"),
            "children": children,
        }

    def clear(self):
        """Limpa o grafo (memória de trabalho é volátil)."""
        self.nodes.clear()
        self.edges.clear()

    def get_hologram_size(self) -> int:
        """Número de termos no holograma atual."""
        return len(self.nodes) + len(self.edges)

    def get_state(self) -> dict:
        """Estado completo do motor para debug e visualização."""
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "nodes": {k: {"state": v["state"], "conf": v["confidence"]}
                      for k, v in self.nodes.items()},
            "edges": [{"from": e["from"], "to": e["to"]} for e in self.edges],
            "hologram_terms": self.get_hologram_size(),
        }
