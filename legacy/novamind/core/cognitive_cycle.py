"""
CognitiveCycle — Loop cognitivo continuo com parada por estagnacao.

Filosofia:
  - Nao ha ordem fixa de execucao.
  - Cada modulo le do MentalState, processa, escreve no MentalState.
  - O loop continua enquanto o estado mental estiver mudando.
  - Para quando: Δerror ≈ 0, sem novos conceitos, sem novas hipoteses, OU orcamento acabou.

Fluxo conceitual (nao sequencial, todos interagem pelo MentalState):
  MentalState
    ├── SENSE: percepcao → HDV
    ├── RECALL: HNSW → episodios similares
    ├── CGE: ativacao → subgrafo causal
    ├── FORJA: observa estado → novo conceito?
    ├── WORLD MODEL: simula → novos links causais?
    ├── NEUROFIELD: modula DA/NA/ACH/5HT
    └── tudo escreve de volta no MentalState
"""
import time
import traceback
import hashlib
from typing import Dict, Any, Optional
import numpy as np
import torch

from .mental_state import get_mental_state, MentalState


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[COG {ts}] {msg}")


# Constantes do ciclo
MAX_INNER_ITERATIONS = 5        # orcamento cognitivo maximo por tick
STAGNATION_LIMIT = 2            # quantas iteracoes sem mudanca antes de parar
PREDICTION_ERROR_EPSILON = 1e-4 # minimo Δerror considerado "mudanca"


STOPWORDS = _SKIP_WORDS if "_SKIP_WORDS" in dir() else frozenset({
    "what","why","how","when","where","who","which",
    "a","an","the","is","are","was","were","be",
    "it","its","they","them","he","she","we","you","i",
    "to","of","in","on","at","by","for","with","without",
    "and","but","or","not","so","if","because","as",
    "this","that","these","those","do","does","did","will",
    "would","could","should","may","might","can","shall",
    "no","yes","just","very","all","each","every","some","any",
})


class CognitiveCycle:
    """Orquestrador do ciclo cognitivo continuo.

    Uso:
        ciclo = CognitiveCycle()
        ciclo.register_module("memory", mem)
        ciclo.tick("ola mundo")
        # ciclo continua rodando internamente ate estagnar
    """

    def __init__(self):
        self.ms = get_mental_state()
        self.modules: Dict[str, Any] = {}
        self._store = None  # EpisodeStore (carregado sob demanda)
        self._cge = None    # CGE instance (lazy)
        self._proj_text = None  # text-to-HDV projection
        self._ke = None     # KnowledgeExtractor (lazy)
        self._vis_mem = None  # VisualMemory (lazy)
        self._vis_enc = None  # VisualEncoder (lazy)
        self._audit = None  # CognitiveAudit (lazy, always passive)
        self._audit_deep = False
        self._recall_policy = None  # RecallPolicy (lazy)
        self._tick_count = 0
        self._sleep_counter = 0
        self._curiosity_queue = []
        self._recorder = None  # FlightRecorder (legacy)

    def register_module(self, name: str, module: Any):
        """Registra um modulo cognitivo."""
        self.modules[name] = module
        self.ms._modules[name] = module
        self.ms.register(name, {"status": "registered"})

    # ─── Tick principal ─────────────────────────────────────

    def tick(self, sensory_input: str = "", modality: str = "text",
             perception_data: Any = None) -> MentalState:
        """Um tick cognitivo completo. Roda o loop interno ate estagnar."""
        t0 = time.time()
        self.ms.tick += 1
        self.ms.total_cycles += 1
        self._tick_count += 1
        self._sleep_counter += 1

        # Inicia Flight Recorder (legacy)
        recorder = self._get_recorder()
        recorder.start_cycle(sensory_input, modality=modality)

        # Inicia CognitiveAudit (sempre passivo)
        audit = self._get_audit()
        audit.start_cycle(sensory_input, modality=modality)
        if self._audit_deep:
            audit.set_mental_state(tick=self._tick_count, total_cycles=self.ms.total_cycles)

        # 1. SENSE
        self._sense(sensory_input, modality, perception_data)
        if self._audit_deep:
            audit.add_candidate(self.ms.focus, 0.5, "sense")

        # 2. Se tem curiosidade pendente, injeta (antes do loop)
        if not sensory_input and self._curiosity_queue:
            curiosity = self._curiosity_queue.pop(0)
            self._sense(curiosity, "text")

        # Flags para detectar estagnacao
        prev_error = self.ms.prediction_error
        prev_hypotheses = self._count_hypotheses()
        prev_concepts = self._count_concepts()
        stagnation = 0
        iterations = 0

        # 3. Loop interno continuo
        for _ in range(MAX_INNER_ITERATIONS):
            iterations += 1

            # RECALL
            self._recall()
            recorder.record_pe(self.ms.prediction_error)
            recorder.record_neurofield(self.ms.da, self.ms.na, self.ms.ach, self.ms.ht5)

            # Audit timeline: registra estado a cada iteracao
            if self._audit_deep:
                cge = self.ms.read("cge") or {}
                active = cge.get("active_nodes", [])
                audit.set_mental_state(
                    iteration=iterations,
                    focus=self.ms.focus,
                    prediction_error=self.ms.prediction_error,
                    da=self.ms.da, na=self.ms.na,
                    cge_active=active[:5],
                )
                for n in active[:10]:
                    audit.record_activation(n, 0.5, "cge", self.ms.prediction_error)

            # CGE activate
            self._cge_activate()

            # CONHECIMENTO extraido do input (triplas, relacoes - alimenta CGE)
            self._extract_knowledge()

            # FORJA (sempre observa)
            self._forja_observe()

            # WORLD MODEL (gera hipoteses, escreve no CGE se descobrir algo)
            self._worldmodel_simulate()

            # NEUROFIELD
            self._neurofield_update()

            # Verificar estagnacao
            error_delta = abs(self.ms.prediction_error - prev_error)
            new_hypotheses = self._count_hypotheses() > prev_hypotheses
            new_concepts = self._count_concepts() > prev_concepts

            if error_delta < PREDICTION_ERROR_EPSILON and not new_hypotheses and not new_concepts:
                stagnation += 1
            else:
                stagnation = 0

            prev_error = self.ms.prediction_error
            prev_hypotheses = self._count_hypotheses()
            prev_concepts = self._count_concepts()

            if stagnation >= STAGNATION_LIMIT:
                break

        self.ms.cycle_iterations = iterations
        self.ms.last_cycle_duration = time.time() - t0
        self.ms.timestamp = time.time()

        # COMPETICAO DE HIPOTESES (antes da resposta, usa o CGE ativo)
        self._compete()

        # Finaliza e salva Flight Recorder (legacy)
        cge_state = self.ms.read("cge") or {}
        recall = self.ms.read("recall") or {}
        recorder.record_recall(
            recall.get("episodes", []),
            method=recall.get("method", ""),
        )
        recorder.record_cge_activation(
            seed=cge_state.get("seed", ""),
            active_nodes=cge_state.get("active_nodes", []),
            active_edges=cge_state.get("active_edges", []),
            total_nodes=cge_state.get("total_nodes", 0),
            total_edges=cge_state.get("total_edges", 0),
        )
        recorder.end_cycle(
            output=f"[{cge_state.get('total_nodes',0)} nos] {', '.join(cge_state.get('active_nodes',[])[:5])}",
            iterations=iterations,
        )

        # Finaliza CognitiveAudit (sempre passivo, nunca altera o ciclo)
        output = f"[{cge_state.get('total_nodes',0)} nos] {', '.join(cge_state.get('active_nodes',[])[:5])}"
        audit.end_cycle(output=output, iterations=iterations)

        # 4. Sono / consolidacao a cada N ticks
        if self._sleep_counter >= 20:
            self._sleep_cycle()

        # 5. Curiosidade: se prediction error alto e sem input externo, gera pergunta
        if not sensory_input and self.ms.prediction_error > 0.08:
            self._generate_curiosity()

        return self.ms

    def _get_recorder(self):
        if self._recorder is None:
            from .flight_recorder import FlightRecorder
            self._recorder = FlightRecorder()
        return self._recorder

    def _get_audit(self):
        """Retorna o CognitiveAudit (cria sob demanda, sempre passivo)."""
        if self._audit is None:
            from .cognitive_audit import CognitiveAudit
            self._audit = CognitiveAudit()
        return self._audit

    def set_audit_deep(self, enabled: bool = True):
        """Ativa/desativa auditoria profunda (mais detalhes, mais recursos)."""
        self._audit_deep = enabled
        if enabled:
            log("Auditoria profunda ativada")
        else:
            log("Auditoria profunda desativada")

    def get_audit_report(self, audit_id: str = None) -> str:
        """Retorna relatorio completo de auditoria."""
        audit = self._get_audit()
        from .cognitive_audit import AuditReplay
        replay = AuditReplay(audit)
        if audit_id:
            return replay.show(audit_id)
        last = audit.current.get("audit_id", "")
        if last:
            return replay.show(last)
        return "(nenhuma auditoria disponivel)"

    def get_replay(self, cycle_id: str = None):
        """Retorna o CognitiveReplay para visualizar um ciclo."""
        from .flight_recorder import CognitiveReplay
        return CognitiveReplay(self._get_recorder())

    def list_recorded_cycles(self, n: int = 10):
        return self._get_recorder().list_cycles(n)

    # ─── Modulos internos ───────────────────────────────────

    def _sense(self, inp: str, modality: str, data: Any):
        """Percepcao → MentalState."""
        self.ms.current_input = inp
        self.ms.register("perception", {
            "modality": modality,
            "raw": data,
            "timestamp": time.time(),
        })
        # Define focus a partir do input (pulando stopwords, prioriza ultima palavra significativa)
        if inp:
            words = inp.lower().split()
            content_words = [w.strip("?.,!;:") for w in words if w.strip("?.,!;:") not in STOPWORDS and len(w) > 2]
            if content_words:
                # Para perguntas tipo "what is a whale?", pega "whale" (ultima palavra de conteudo)
                # Para "is cat a mammal", pega "mammal" ou "cat"
                self.ms.focus = content_words[-1]  # ultima palavra de conteudo
            else:
                self.ms.focus = words[-1].strip("?.,!;:") if words else ""
        if not inp:
            self.ms.focus = ""
            self.ms.prediction_error = 0.0

    def _recall(self):
        """Busca episodios similares via EpisodeStore.
        
        Usa query_text (busca textual) em vez de HNSW porque o encoder
        estrutural mudou e o indice HNSW antigo nao corresponde.
        """
        if self._store is None:
            try:
                from .episode_store import EpisodeStore
                s = EpisodeStore()
                if s.load_backup():
                    s.build_hnsw()
                    self._store = s
            except Exception as e:
                pass

        if not (self._store and self.ms.current_input):
            return

        try:
            # Usa o focus (ultima palavra de conteudo) em vez do texto completo
            query = self.ms.focus if self.ms.focus else self.ms.current_input
            raw_results = self._store.query_text(query, top_k=5)

            # RecallPolicy decide se o recall e utilizavel
            if self._recall_policy is None:
                from .cognition.recall_policy import RecallPolicy
                self._recall_policy = RecallPolicy()
            decision = self._recall_policy.evaluate(raw_results, "text")

            if decision.is_empty:
                # Recall irrelevante: registra como vazio
                self.ms.register("recall", {
                    "status": "empty",
                    "decision": decision.to_dict(),
                })
                log(f"Recall: EMPTY ({decision.reason}, best={decision.best_similarity:.3f})")
                return

            # Recall util: registra normalmente
            results = decision.episodes
            if results:
                self.ms.register("recall", {
                    "episodes": results, "count": len(results),
                    "method": "text", "status": "accepted",
                    "decision": decision.to_dict(),
                })
                avg_surprise = float(np.mean([r.get("surprise", 0.1) for r in results]))
                self.ms.prediction_error = avg_surprise * 0.3 + self.ms.prediction_error * 0.7
                return
        except Exception as e:
            log(f"Recall error: {e}")

    def _text_to_hdv(self, text: str) -> np.ndarray:
        """Converte texto em HDV ESTRUTURADO via parser sintatico + bind de papeis.

        Diferenca do encoder anterior:
          - "cao morde homem" != "homem morde cao" (ordem preservada)
          - Suporta negacao, quantificadores, condicionais
          - Polissemia via contexto
        """
        if not hasattr(self, "_struct_encoder"):
            from .structured_encoder import StructuredHDVEncoder
            self._struct_encoder = StructuredHDVEncoder()
        hdv, _ = self._struct_encoder.encode(text)
        return hdv

    def _extract_terms(self, text: str) -> set:
        """Extrai termos significativos de um texto."""
        return {w.lower().strip("?.,!;:'\"") for w in text.split()
                if len(w) > 2 and w.lower() not in STOPWORDS and w.isalpha()}

    def _cge_activate(self):
        """CGE dinamico: carrega conceitos sob demanda do recall."""
        recall = self.ms.read("recall") or {}
        episodes = recall.get("episodes", [])
        no_recall = len(episodes) == 0

        all_terms = set()
        episode_terms = []

        if not no_recall:
            for ep in episodes:
                content = ep.get("content", {}) if isinstance(ep, dict) else {}
                text = str(content.get("input", "") or content.get("caption", "") or "")
                terms = self._extract_terms(text)
                all_terms.update(terms)
                episode_terms.append(terms)

        # Mesmo sem recall, usa o focus para ativar CGE
        focus = self.ms.focus
        if focus and focus not in STOPWORDS:
            all_terms.add(focus.lower())

        if not all_terms:
            return

        # Cria CGE sob demanda
        if self._cge is None:
            from .cognitive_graph_engine import CognitiveGraphEngine
            self._cge = CognitiveGraphEngine(device="cpu")
            self.register_module("cge", self._cge)
            self._load_taxonomy()

        # RESETA ativacao anterior (limpa estados "active" e "visible")
        for nid in self._cge.nodes:
            if self._cge.nodes[nid].get("state", "") in ("active", "visible"):
                self._cge.nodes[nid]["state"] = "hidden"
            self._cge.nodes[nid]["access_count"] = 0

        # Para cada termo, cria no CGE se nao existir
        edge_type = self._cge.EDGE_RELATED_TO if hasattr(self._cge, "EDGE_RELATED_TO") else self._cge.EDGE_CAUSES

        for term in all_terms:
            if term not in self._cge.nodes:
                hdv = self._text_to_hdv(term)
                vec = torch.from_numpy(hdv).float()
                self._cge.create_node(term, semantic_vec=vec)
                self._cge.nodes[term]["confidence"] = 0.5

        # Cria arestas de co-ocorrencia
        for terms_in_ep in episode_terms:
            terms_list = [t for t in terms_in_ep if t in all_terms]
            for i in range(len(terms_list)):
                for j in range(i + 1, len(terms_list)):
                    t1, t2 = terms_list[i], terms_list[j]
                    edge_exists = any(
                        e.get("from") == t1 and e.get("to") == t2
                        for e in self._cge.edges[-100:]
                    )
                    if not edge_exists:
                        try:
                            self._cge.add_edge(t1, t2, edge_type)
                        except Exception:
                            pass

        # Ativa paisagem de energia
        if focus and focus in self._cge.nodes:
            seed = focus
        else:
            freq = {}
            for terms in episode_terms:
                for t in terms:
                    if t in self._cge.nodes:
                        freq[t] = freq.get(t, 0) + 1
            seed = max(freq, key=freq.get) if freq else list(all_terms)[0]

        if seed in self._cge.nodes:
            try:
                # Propagacao contextual iterativa
                self._cge_contextual_propagate(seed)
            except Exception as e:
                log(f"CGE activate error: {e}")

        # Coleta ativacoes com valores reais (armazenados nos nos)
        active_nodes_info = []
        for nid in self._cge.nodes:
            node = self._cge.nodes[nid]
            state = node.get("state", "")
            if state in ("visible", "active"):
                act_level = node.get("activation_level", 0.5)
                source = node.get("activation_source", "cge")
                active_nodes_info.append((nid, act_level, source))
        # Ordena por nivel de ativacao (maior primeiro)
        active_nodes_info.sort(key=lambda x: x[1], reverse=True)
        
        active = [n[0] for n in active_nodes_info[:15]]
        activation_map = {n[0]: round(n[1], 3) for n in active_nodes_info[:15]}
        source_map = {n[0]: n[2] for n in active_nodes_info[:15]}

        # Arestas entre nos ativos
        active_edges = []
        if len(active) > 1:
            aset = set(active)
            for e in self._cge.edges:
                if e.get("from") in aset and e.get("to") in aset:
                    active_edges.append({"from": e["from"], "to": e["to"]})
        self.ms.register("cge", {
            "total_nodes": len(self._cge.nodes),
            "total_edges": len(self._cge.edges),
            "active_nodes": active,
            "activation_map": activation_map,
            "source_map": source_map,
            "active_edges": active_edges[:15],
            "seed": seed,
        })

    def _cge_contextual_propagate(self, seed: str, max_iterations: int = 10):
        """Propagacao contextual iterativa no CGE.

        Em vez de uma unica ativacao, executa multiplos ciclos de:
          Ativar → Recalcular energia → Propagar → Verificar estagnacao
        """
        if self._cge is None or seed not in self._cge.nodes:
            return

        landscape = self._cge.compute_energy_landscape() if hasattr(self._cge, "compute_energy_landscape") else {}

        # Parametros contextuais (modulados por DA/NA)
        temperature = max(0.1, 0.5 - self.ms.da * 0.3)  # DA alta → foco estreito
        max_energy = 10.0 + self.ms.na * 5.0  # NA alta → mais energia (alerta)

        # Ativacao inicial
        prev_active_set = set()
        for iteration in range(max_iterations):
            # Propaga (sempre usa a landscape atualizada)
            activation = self._cge.activate_with_landscape(
                [seed], landscape=landscape,
                temperature=temperature, max_energy=max_energy,
            )

            # Aplica ativacao nos estados dos nos (com valores reais)
            active_set = set()
            for nid, act_level in activation.items():
                if act_level > 0.1:
                    if nid in self._cge.nodes:
                        self._cge.nodes[nid]["state"] = "active"
                        self._cge.nodes[nid]["activation_level"] = round(float(act_level), 3)
                        self._cge.nodes[nid]["access_count"] += 1
                        active_set.add(nid)
                else:
                    if nid in self._cge.nodes and self._cge.nodes[nid].get("state") != "hidden":
                        self._cge.nodes[nid]["state"] = "hidden"
                        self._cge.nodes[nid]["activation_level"] = 0.0

            # Verifica estagnacao
            if iteration > 0 and active_set == prev_active_set:
                break

            prev_active_set = active_set

            # Recalcula paisagem de energia
            landscape = self._cge.compute_energy_landscape() if hasattr(self._cge, "compute_energy_landscape") else {}

            # Reforca ativos, decai inativos
            for nid in self._cge.nodes:
                if nid in active_set:
                    landscape[nid] = landscape.get(nid, 0) + 0.3
                else:
                    landscape[nid] = landscape.get(nid, 0) - 0.2

    def _load_taxonomy(self):
        """Carrega taxonomia WordNet no CGE (relacoes is-a, causa)."""
        import json, os
        tax_file = r"D:\AGI2\memories\taxonomy_cge_state.json"
        if not os.path.exists(tax_file):
            # Tenta o arquivo primario
            tax_file = r"D:\AGI2\memories\taxonomy.json"
        if not os.path.exists(tax_file):
            return
        try:
            with open(tax_file) as f:
                data = json.load(f)
            if "node_count" in data:
                # Formato CGE state: ja populado
                log(f"CGE taxonomy carregada: {data['node_count']} nodes, {data['edge_count']} edges")
                return  # ja carregado pelo _taxonomy_curriculum.py --load
            # Formato taxonomy.json: criar nos e arestas
            if isinstance(data, list) and len(data) > 0:
                import torch
                edge_type = self._cge.EDGE_CAUSES
                count = 0
                for t in data:
                    child, parent = t["child"], t["parent"]
                    if child not in self._cge.nodes:
                        hdv = self._text_to_hdv(child)
                        self._cge.create_node(child, semantic_vec=torch.from_numpy(hdv).float())
                        self._cge.nodes[child]["confidence"] = 0.9
                    if parent not in self._cge.nodes:
                        hdv = self._text_to_hdv(parent)
                        self._cge.create_node(parent, semantic_vec=torch.from_numpy(hdv).float())
                        self._cge.nodes[parent]["confidence"] = 0.9
                    try:
                        self._cge.add_edge(child, parent, edge_type)
                        count += 1
                    except:
                        pass
                log(f"CGE taxonomy: {len(self._cge.nodes)} nodes, {count} edges from taxa")
        except Exception as e:
            log(f"Taxonomy load error: {e}")
        # Carrega conceitos semanticos apos taxonomia
        self._load_semantic_concepts()

    def _load_semantic_concepts(self):
        """Carrega conceitos dos episodios como nos CGE (ate 5000)."""
        if self._cge is None or not self._store:
            return
        try:
            conn = __import__("sqlite3").connect(self._store.db_path)
            rows = conn.execute(
                "SELECT content FROM episodes WHERE type IN ('perception_text','perception_caption') LIMIT 2000"
            ).fetchall()
            conn.close()
            import json
            count = 0
            for row in rows:
                content = json.loads(row[0]) if row[0] else {}
                text = str(content.get("input", "") or content.get("caption", "") or "")
                for term in self._extract_terms(text):
                    if term not in self._cge.nodes and len(term) > 2:
                        if count >= 5000:
                            break
                        hdv = self._text_to_hdv(term)
                        vec = __import__("torch").from_numpy(hdv).float()
                        self._cge.create_node(term, semantic_vec=vec)
                        self._cge.nodes[term]["confidence"] = 0.3
                        count += 1
                if count >= 5000:
                    break
            if count > 0:
                log(f"CGE: +{count} conceitos semanticos")
        except Exception as e:
            log(f"Semantic concepts error: {e}")

    def _extract_knowledge(self):
        """Extrai conhecimento estruturado do input e integra no CGE."""
        if not self.ms.current_input or len(self.ms.current_input) < 10:
            return
        try:
            if self._ke is None:
                from .knowledge_extractor import KnowledgeExtractor
                self._ke = KnowledgeExtractor()
            knowledge = self._ke.process(self.ms.current_input)
            if knowledge["facts"] and self._cge:
                result = self._ke.integrate_into_cge(knowledge, self._cge)
                if result["nodes_created"] > 0 or result["edges_created"] > 0:
                    log(f"KE: +{result['nodes_created']} nos, +{result['edges_created']} arestas, "
                        f"{result['conflicts']} conflitos")
                    if result["conflicts"] > 0:
                        for c in result["conflict_details"][:3]:
                            log(f"  CONFLITO: {c['subject']}.{c['verb']}: "
                                f"'{c['existing']}' vs '{c['new']}'")
                    self.ms.register("knowledge", {
                        "facts": len(knowledge["facts"]),
                        "hypotheses": len(knowledge["hypotheses"]),
                        "exceptions": len(knowledge["exceptions"]),
                        "nodes_created": result["nodes_created"],
                        "edges_created": result["edges_created"],
                        "conflicts": result["conflicts"],
                    })
        except Exception as e:
            log(f"Knowledge extraction error: {e}")

    def _process_visual(self, image_path: str = None, region_hdvs: dict = None):
        """Processa input visual: HDVs de regioes -> memoria -> CGE.

        Chamado externamente quando uma imagem chega (camera ou render).
        Nao contem conhecimento especifico de dominio.
        Apenas:
          1. Converte imagem em HDVs de regiao (DINOv2)
          2. Armazena na memoria visual
          3. Busca experiencias visuais similares
          4. Ativa nos CGE baseado em similaridade visual
        """
        try:
            # Garante encoder visual
            if self._vis_enc is None and image_path:
                from .visual_encoder import VisualEncoder
                self._vis_enc = VisualEncoder("dinov2_vits14")
            if self._vis_mem is None:
                from .visual_memory import VisualMemory
                self._vis_mem = VisualMemory()

            # Obtem HDVs das regioes
            if image_path and self._vis_enc:
                from PIL import Image
                img = Image.open(image_path).convert("RGB")
                region_hdvs = self._vis_enc.get_region_hdvs(img)

            if not region_hdvs:
                return

            # Armazena na memoria
            self._vis_mem.store(region_hdvs, {
                "source": "camera" if not image_path else image_path,
            })

            # Busca experiencias visuais similares
            similar = self._vis_mem.search(region_hdvs, top_k=3)
            if similar and similar[0]["similarity"] > 0.7:
                # Atualiza prediction error se a cena e similar ao que ja viu
                self.ms.prediction_error = 0.3 + 0.7 * (1 - similar[0]["similarity"])
                self.ms.register("visual", {
                    "similar_count": len([s for s in similar if s["similarity"] > 0.7]),
                    "top_similarity": round(similar[0]["similarity"], 3),
                    "n_changed_regions": similar[0]["n_changed"],
                })
            else:
                # Cena nova: aumenta prediction error
                self.ms.prediction_error = min(1.0, self.ms.prediction_error + 0.2)
                self.ms.register("visual", {
                    "novel_scene": True,
                    "top_similarity": round(similar[0]["similarity"], 3) if similar else 0,
                })
        except Exception as e:
            log(f"Visual process error: {e}")

    def _worldmodel_simulate(self):
        """World Model: gera hipoteses e escreve de volta no CGE."""
        cge_state = self.ms.read("cge") or {}
        active_nodes = cge_state.get("active_nodes", [])
        if not active_nodes:
            return

        try:
            if "world_model" not in self.modules:
                from .world_model import WorldModel
                wm = WorldModel()
                self.register_module("world_model", wm)
            else:
                wm = self.modules["world_model"]

            # Para cada par de nos ativos, tenta encontrar relacao causal
            novos_links = 0
            for i in range(min(len(active_nodes), 8)):
                for j in range(i + 1, min(len(active_nodes), 8)):
                    a, b = active_nodes[i], active_nodes[j]
                    try:
                        assoc = wm.query_association(a, b)
                        if assoc and assoc.get("strength", 0) > 0.3 and self._cge:
                            # Verifica se o link ja existe no CGE
                            exists = any(e.get("from") == a and e.get("to") == b for e in self._cge.edges[-200:])
                            if not exists:
                                try:
                                    self._cge.add_edge(a, b, self._cge.EDGE_CAUSES)
                                    novos_links += 1
                                except:
                                    pass
                    except:
                        pass

            self.ms.register("world_model", {
                "active_inputs": active_nodes[:5],
                "hypotheses_count": novos_links,
                "new_edges": novos_links,
            })

            if novos_links > 0:
                log(f"WM: +{novos_links} arestas causais no CGE")

        except Exception as e:
            pass

    def _forja_observe(self):
        """Forja observa padroes temporais na propagacao do CGE.

        Mantem historico de ativacoes e detecta sequencias repetitivas.
        Se um padrao aparecer N vezes, cria um novo operador ou conceito.
        """
        cge_state = self.ms.read("cge") or {}
        active = cge_state.get("active_nodes", [])

        if not active:
            return

        # Historico de ativacoes (ultimos 20 ciclos)
        if not hasattr(self, "_activation_history"):
            self._activation_history = []
        if active:
            self._activation_history.append(set(active))
        if len(self._activation_history) > 50:
            self._activation_history = self._activation_history[-50:]

        # Detecta padroes repetitivos: mesmos 3 nos aparecendo juntos
        from collections import Counter
        pattern_counts = Counter()
        for i in range(len(self._activation_history) - 2):
            triple = tuple(sorted(list(self._activation_history[i] & self._activation_history[i+1] & self._activation_history[i+2]))[:3])
            if len(triple) >= 2:
                pattern_counts[triple] += 1

        # Se um padrao apareceu 5+ vezes, e um novo conceito?
        novos_conceitos = 0
        for pattern, count in pattern_counts.most_common(5):
            if count >= 5 and self._cge:
                # Cria um novo conceito que representa esse padrao
                concept_name = "+".join(pattern[:3])
                if concept_name not in self._cge.nodes:
                    # HDV combinado dos conceitos no padrao
                    import torch
                    hdvs = []
                    for term in pattern[:3]:
                        if term in self._cge.nodes:
                            hdvs.append(self._cge.nodes[term].get("semantic", torch.zeros(10000)))
                    if hdvs:
                        combined = torch.sign(torch.stack(hdvs).mean(dim=0))
                        combined[combined == 0] = 1
                        self._cge.create_node(concept_name, semantic_vec=combined)
                        self._cge.nodes[concept_name]["confidence"] = 0.5 + count * 0.05
                        novos_conceitos += 1

        if novos_conceitos > 0:
            log(f"Forja: +{novos_conceitos} conceitos de padroes (ciclo {self.ms.total_cycles})")
            self.ms.register("forja", {
                "new_concepts": novos_conceitos,
                "patterns_found": len(pattern_counts),
            })

        # ─── Criação de procedimentos ──────────────────────
        # Detecta SEQUENCIAS de ativacao (nao apenas co-ocorrencia)
        if not hasattr(self, "_activation_sequences"):
            self._activation_sequences = []
        
        # Registra a sequencia atual (foco -> ativos)
        if self.ms.focus and active:
            seq = (self.ms.focus, tuple(sorted(active[:5])))
            self._activation_sequences.append(seq)
        if len(self._activation_sequences) > 100:
            self._activation_sequences = self._activation_sequences[-100:]
        
        # Procura sequencias que aparecem 8+ vezes
        from collections import Counter as _Counter
        seq_counts = _Counter(self._activation_sequences)
        novos_procedimentos = 0
        for (focus, active_set), count in seq_counts.most_common(10):
            if count >= 8 and len(active_set) >= 3:
                # Cria um procedimento: se focus X, ativar Y
                proc_name = f"PROC_{focus}"
                if not hasattr(self, "_created_procs"):
                    self._created_procs = set()
                if proc_name not in self._created_procs and self._cge:
                    self._created_procs.add(proc_name)
                    
                    # Cria no CGE para o procedimento
                    hdv = self._text_to_hdv(proc_name)
                    import torch
                    vec = torch.from_numpy(hdv).float()
                    self._cge.create_node(proc_name, semantic_vec=vec)
                    self._cge.nodes[proc_name]["confidence"] = 0.7
                    
                    # Conecta procedimento aos nos ativos
                    for term in list(active_set)[:5]:
                        if term in self._cge.nodes:
                            try:
                                self._cge.add_edge(proc_name, term, self._cge.EDGE_CAUSES)
                            except:
                                pass
                    novos_procedimentos += 1
                    log(f"Forja: NOVO PROCEDIMENTO '{proc_name}' ({count}x repeticoes)")
        
        if novos_procedimentos > 0:
            forja_state = self.ms.read("forja") or {}
            forja_state["new_procedures"] = forja_state.get("new_procedures", 0) + novos_procedimentos
            self.ms.register("forja", forja_state)

    def _neurofield_update(self):
        """NeuroField atualiza DA/NA/ACH/5HT baseado no prediction_error."""
        nf = self.modules.get("neurofield")
        if nf:
            try:
                nf.update(self.ms.nt_context, self.ms.prediction_error)
                self.ms.da = getattr(nf, "da", self.ms.da)
                self.ms.na = getattr(nf, "na", self.ms.na)
                self.ms.ach = getattr(nf, "ach", self.ms.ach)
                self.ms.ht5 = getattr(nf, "ht5", self.ms.ht5)
            except Exception as e:
                log(f"NeuroField error: {e}")

    def _compete(self):
        """Competicao entre hipoteses com evidencias tipadas por categoria."""
        cge_state = self.ms.read("cge") or {}
        active = cge_state.get("active_nodes", [])
        seed = cge_state.get("seed", "")

        if not seed or not active:
            return

        try:
            from .hypothesis_competition import HypothesisCompetition
            hc = HypothesisCompetition()

            # Gera hipoteses separadas por categoria (is_a, has_property, related, similar)
            hc.generate_from_cge(seed, self._cge, self._store, max_per_category=5)

            audit_dict = hc.to_audit_dict()

            # Registra vencedoras por categoria no MentalState
            best_is_a = audit_dict["by_category"]["is_a"]
            best_prop = audit_dict["by_category"]["has_property"]

            competition_result = {
                "n_hypotheses": audit_dict["n_hypotheses"],
                "entropy": audit_dict["entropy"],
                "global_winner": audit_dict["global_winner"],
                "global_score": audit_dict["global_score"],
                "confidence_decomposition": audit_dict["confidence_decomposition"],
                "justification": audit_dict["justification"],
                "is_a_winner": best_is_a["winner"],
                "is_a_score": best_is_a["winner_score"],
                "property_winner": best_prop["winner"],
                "property_score": best_prop["winner_score"],
            }
            self.ms.register("competition", competition_result)

            # Audit
            audit = self._get_audit()
            if audit_dict["global_winner"]:
                audit.add_hypothesis(audit_dict["global_winner"],
                                    audit_dict["global_score"])
                audit.set_mental_state(
                    winner=audit_dict["global_winner"],
                    entropy=audit_dict["entropy"],
                    n_hypotheses=audit_dict["n_hypotheses"],
                )

            # Atualiza confianca com base no score da vencedora
            if audit_dict["global_score"] > 0.3:
                self.ms.confidence = audit_dict["global_score"]

        except Exception as e:
            log(f"Competition error: {e}")

    # ─── Helpers ────────────────────────────────────────────

    def _count_hypotheses(self) -> int:
        wm = self.ms.read("world_model")
        return len(wm.get("branches", [])) if wm and isinstance(wm, dict) else 0

    def _count_concepts(self) -> int:
        forja = self.ms.read("forja")
        return forja.get("count", 0) if forja and isinstance(forja, dict) else 0

    def get_output(self) -> str:
        recall = self.ms.read("recall") or {}
        episodes = recall.get("episodes", [])
        if episodes:
            best = episodes[0]
            inp = best.get("content", {}).get("input", "") if isinstance(best, dict) else str(best)
            return f"[recall] {inp[:120]}"
        cge = self.ms.read("cge") or {}
        active = cge.get("active_nodes", [])
        if active:
            return f"[cge] ativos: {', '.join(active[:5])}"
        if self.ms.current_input:
            return f"[sense] {self.ms.current_input[:120]}"
        return "[idle]"

    # ─── Sono (a cada 20 ticks): consolida + forja + decai ──

    def _sleep_cycle(self):
        self._sleep_counter = 0
        log(f"Sono (ciclo {self._tick_count})")

        # Decaimento de confianca em nos nao visitados
        if self._cge:
            for nid in list(self._cge.nodes.keys())[:500]:
                if self._cge.nodes[nid].get("access_count", 0) == 0:
                    c = self._cge.nodes[nid].get("confidence", 0.5)
                    if c > 0.3:
                        self._cge.nodes[nid]["confidence"] = c - 0.1

        # Forja com janela maior
        if hasattr(self, "_activation_history") and self._cge:
            from collections import Counter
            patterns = Counter()
            for i in range(len(self._activation_history) - 3):
                q = tuple(sorted(
                    self._activation_history[i] & self._activation_history[i+1] &
                    self._activation_history[i+2] & self._activation_history[i+3]
                ))[:4]
                if len(q) >= 2:
                    patterns[q] += 1
            for pattern, count in patterns.most_common(5):
                if count >= 10:
                    name = "PROC_" + "+".join(pattern[:3])
                    if not hasattr(self, "_created_procs"):
                        self._created_procs = set()
                    if name not in self._created_procs:
                        self._created_procs.add(name)
                        hdv = self._text_to_hdv(name)
                        import torch
                        self._cge.create_node(name, semantic_vec=torch.from_numpy(hdv).float())
                        self._cge.nodes[name]["confidence"] = 0.8
                        for t in list(pattern)[:5]:
                            if t in self._cge.nodes:
                                try: self._cge.add_edge(name, t, self._cge.EDGE_CAUSES)
                                except: pass
                        log(f"  +procedimento: {name}")

        # Replay de memoria
        mem = self.modules.get("memory")
        if mem and hasattr(mem, "trigger_sleep_cycle"):
            try: mem.trigger_sleep_cycle()
            except: pass

    # ─── Curiosidade (quando prediction error alto) ─────────

    def _generate_curiosity(self):
        if not self._cge:
            return
        candidates = [n for n, nd in self._cge.nodes.items()
                     if nd.get("confidence", 0.5) < 0.4 and nd.get("access_count", 0) < 5]
        if not candidates:
            return
        target = min(candidates, key=lambda n: self._cge.nodes[n].get("confidence", 0.5))
        self._curiosity_queue.append(f"what is {target}")
        log(f"Curiosidade: o que e '{target}'?")

        if self._store:
            try:
                res = self._store.query_text(target, top_k=1)
                if res:
                    txt = res[0].get("content", {}).get("input", "")
                    self._curiosity_queue.append(txt)
            except:
                pass
