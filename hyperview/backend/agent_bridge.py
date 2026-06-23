"""
AgentBridge — Ponte AGI -> WebSocket.
Emite eventos ricos em tempo real para o HyperView.
"""

import sys, os, json, asyncio, time, traceback, hashlib
from typing import Optional, Callable
from pathlib import Path

NOVAMIND_DIR = str(Path(__file__).parent.parent.parent / 'novamind')
if NOVAMIND_DIR not in sys.path:
    sys.path.insert(0, NOVAMIND_DIR)

import torch
import numpy as np
from sklearn.decomposition import PCA

from core.grounding_engine import GroundingEngine
from core.hdc_memory import HDCMemory
from core.autoregressive_generator import AutoregressiveGenerator
from core.positional_binder import PositionalBinder
from core.broca_decoder import BrocaDecoder
from core.temporal_buffer import TemporalBuffer
from core.code_extractor import sanitize_code, desanitize_code
from core.cognitive_answerer import CognitiveAnswerer


SOURCE_COLORS = {
    "code_brain":  [0.0, 0.8, 1.0],
    "python_brain":[0.0, 0.6, 0.9],
    "msvd_brain": [0.2, 0.9, 0.5],
    "text_library":[0.8, 0.4, 1.0],
}


class AgentBridge:
    def __init__(self, emit_callback: Optional[Callable] = None):
        self.emit = emit_callback or (lambda e: None)
        self.engine = GroundingEngine(device='cuda')
        # Move modelos não-usados (visão/áudio) para CPU para liberar VRAM
        self.engine.resnet = self.engine.resnet.cpu()
        self.engine.encodec_model = self.engine.encodec_model.cpu()
        self.engine.proj_audio = self.engine.proj_audio.cpu()
        self.engine.mel_transform = self.engine.mel_transform.cpu()
        torch.cuda.empty_cache()

        self.binder = PositionalBinder()
        self.decoder = BrocaDecoder(self.engine)
        self.buffer = TemporalBuffer(self.engine)
        self.memory = HDCMemory()
        self.generator = AutoregressiveGenerator(
            self.engine, self.binder, self.decoder,
            hdc_memory=self.memory, temporal_buffer=self.buffer)

        self.answerer = CognitiveAnswerer(
            self.engine, self.binder, self.decoder, self.memory,
            device=self.engine.device)

        self.pca = PCA(n_components=3)
        self.pca_fitted = False
        self.pca_points = []
        self.all_vectors_np = None
        self.all_labels_full = []
        self.all_sources = []

        self.current_intention = ""
        self.output = ""
        self.residual_history = []

        # Inicializa componentes de incorporação para a UI
        try:
            from core.neuromodulator_state import NeuromodulatorState
            from core.simulated_body import SimulatedBody
            from core.body_encoder import BodyEncoder
            from core.cerebellum import Cerebellum
            
            self.nt_state = NeuromodulatorState()
            self.body = SimulatedBody()
            self.body_encoder = BodyEncoder(seed=88)
            self.cerebellum = Cerebellum()
            self._body_concept = "stable"
        except ImportError:
            self.nt_state = None
            self.body = None
            self.body_encoder = None
            self.cerebellum = None
            self._body_concept = "unknown"

    async def emit_event(self, event_type: str, **data):
        event = {"type": event_type, "timestamp": time.time(), **data}
        if self.emit:
            await self.emit(event)

    def compute_similarity_graph(self, points, top_k=3):
        """Compute real similarity edges between PCA points."""
        if len(points) < 2:
            return []
        pos = np.array([[p['x'], p['y'], p['z']] for p in points])
        norms = np.linalg.norm(pos, axis=1, keepdims=True) + 1e-8
        pos_n = pos / norms
        sims = pos_n @ pos_n.T
        edges = []
        for i in range(len(points)):
            idxs = np.argsort(-sims[i])[1:top_k+1]
            for j in idxs:
                if sims[i, j] > 0.5:
                    edges.append({
                        "source": i,
                        "target": int(j),
                        "similarity": float(sims[i, j]),
                    })
        return edges

    def build_pca_map(self, query_text=None, max_vectors=5000):
        all_vectors = []
        self.all_labels_full = []
        self.all_sources = []

        for key, matrix in self.memory.matrices.items():
            index = self.memory.indices[key]
            n = len(index)
            step = max(1, n // (max_vectors // max(1, len(self.memory.matrices))))
            for i in range(0, n, step):
                vec = matrix[i].cpu().numpy()
                all_vectors.append(vec)
                meta = index[i]
                label = meta.get('text', meta.get('id', ''))[:60]
                self.all_labels_full.append(label)
                self.all_sources.append(key)

        if len(all_vectors) < 3:
            return []

        X = np.array(all_vectors)
        self.all_vectors_np = X

        if not self.pca_fitted:
            projected = self.pca.fit_transform(X)
            self.pca_fitted = True
        else:
            projected = self.pca.transform(X)

        self.pca_points = []
        query_labels = set()
        if query_text:
            results = self.memory.query(query_text, top_k=10)
            for sim, meta, src in results:
                lbl = meta.get('text', meta.get('id', ''))[:60]
                query_labels.add(lbl)

        for i, (x, y, z) in enumerate(projected):
            src = self.all_sources[i] if i < len(self.all_sources) else "unknown"
            color = SOURCE_COLORS.get(src, [0.3, 0.7, 0.9])
            is_active = self.all_labels_full[i] in query_labels if i < len(self.all_labels_full) else False
            self.pca_points.append({
                "x": float(x), "y": float(y), "z": float(z),
                "label": self.all_labels_full[i] if i < len(self.all_labels_full) else f"item_{i}",
                "active": is_active,
                "source": src,
                "color": color,
                "index": i,
            })

        return self.pca_points

    def project_vector(self, vec: torch.Tensor) -> dict:
        if not self.pca_fitted:
            return {"x": 0, "y": 0, "z": 0}
        v = vec.cpu().numpy().reshape(1, -1)
        x, y, z = self.pca.transform(v)[0]
        return {"x": float(x), "y": float(y), "z": float(z)}

    def get_subconcepts(self, label, top_k=6):
        """Find sub-concepts/similar items to a given label."""
        if self.all_vectors_np is None:
            return []
        idx = None
        for i, lbl in enumerate(self.all_labels_full):
            if lbl == label:
                idx = i
                break
        if idx is None:
            return []
        vec = self.all_vectors_np[idx]
        sims = self.all_vectors_np @ vec
        top = np.argsort(-sims)[1:top_k+1]
        subs = []
        for j in top:
            if j < len(self.all_labels_full):
                pt = self.pca_points[j] if j < len(self.pca_points) else None
                subs.append({
                    "label": self.all_labels_full[j][:40],
                    "similarity": float(sims[j]),
                    "position": {"x": float(pt['x']), "y": float(pt['y']), "z": float(pt['z'])} if pt else {"x": 0, "y": 0, "z": 0},
                    "source": self.all_sources[j] if j < len(self.all_sources) else "unknown",
                })
        return subs

    async def query(self, text: str, max_chunks: int = 12):
        self.output = ""
        self.current_intention = text
        self.residual_history = []

        await self.emit_event("intention", text=text)

        pca_points = self.build_pca_map(query_text=text)
        edges = self.compute_similarity_graph(pca_points, top_k=3)
        await self.emit_event("pca_update", points=pca_points, edges=edges)

        results = self.memory.query(text, top_k=8)
        activated = []
        for sim, meta, source in results:
            label = meta.get('text', meta.get('id', ''))[:60]
            activated.append({"label": label, "similarity": sim, "source": source})
            for pt in self.pca_points:
                if pt["label"] == label:
                    pt["active"] = True

        await self.emit_event("memory_activate", concepts=activated)
        edges = self.compute_similarity_graph(self.pca_points, top_k=3)
        await self.emit_event("pca_update", points=self.pca_points, edges=edges)

        result = await self._generate_with_events(text, max_chunks)

        self.output = result
        await self.emit_event("complete", output=result)

        concepts = list(self.generator.working_hdv.keys())[-30:]
        await self.emit_event("working_hdv", items=concepts, count=len(self.generator.working_hdv))

        return result

    async def _generate_with_events(self, target_text: str, max_chunks: int):
        from core.autoregressive_generator import IntentionDrivenGenerator

        gen = IntentionDrivenGenerator(
            self.engine, self.binder, self.decoder,
            self.memory, self.buffer,
            self.generator.working_hdv,
            self.generator.working_hdv_keys)

        sanitized = sanitize_code(target_text)
        intention_mf = self.engine.encode_text(sanitized)
        intention_vec = intention_mf.tensor.float()
        spoken_vec = torch.zeros(10000, device=self.engine.device)
        generated = []
        generated_keys = []
        repetition_count = 0

        intention_3d = self.project_vector(intention_vec)
        await self.emit_event("fe_update", fe=1.0, dopamine=0.0, noradrenaline=0.0, acetylcholine=0.0)
        self.residual_history.append(1.0)

        for step in range(max_chunks):
            residual_vec = intention_vec - spoken_vec
            residual_vec = torch.sign(residual_vec)

            residual_energy = float(torch.nn.functional.cosine_similarity(
                residual_vec.unsqueeze(0), intention_vec.unsqueeze(0)))
            self.residual_history.append(residual_energy)

            dopamine = max(0, 1.0 - residual_energy)
            noradrenaline = max(0, residual_energy - 0.5) * 2
            acetylcholine = 0.0

            await self.emit_event("fe_update",
                fe=residual_energy,
                dopamine=dopamine,
                noradrenaline=noradrenaline,
                acetylcholine=acetylcholine)

            if residual_energy < 0.15:
                break

            chunk_text = gen._search(residual_vec, 4, 0.0)

            if not chunk_text:
                break

            if chunk_text.lower() in generated_keys:
                repetition_count += 1
                noise = torch.randn(10000, device=self.engine.device) * 0.1 * repetition_count
                residual_vec = torch.sign(residual_vec + noise)
                await self.emit_event("forge_activate", residual=float(residual_energy))
                acetylcholine = 0.8 + repetition_count * 0.1
                await self.emit_event("fe_update",
                    fe=residual_energy, dopamine=dopamine,
                    noradrenaline=noradrenaline, acetylcholine=acetylcholine)
                chunk_text = gen._search(residual_vec, 4, 0.0)
                if not chunk_text or chunk_text.lower() in generated_keys:
                    break
                repetition_count = 0

            if generated:
                await self.emit_event("bind",
                    concept_a=generated[-1][:50],
                    concept_b=chunk_text[:50],
                    residual=float(residual_energy))

            gen._add_to_working_hdv(chunk_text)
            generated.append(chunk_text)
            generated_keys.append(chunk_text.lower())

            # Atualiza o estado físico/químico da AGI a cada chunk
            if hasattr(self, 'body') and self.body:
                try:
                    from core.simulated_body import Actuator
                    act = Actuator()
                    if "fire" in chunk_text.lower() or "danger" in chunk_text.lower() or "pain" in chunk_text.lower():
                        # Movimento brusco para trás (evitação)
                        act.move = [-1.0, 0.0, -1.0]
                    else:
                        # Exploração leve
                        act.move = [0.2, 0.0, 0.2]

                    reading = self.body.apply_actuator(act, dt=0.1)
                    
                    # Identifica contextos
                    contexts = []
                    if "fire" in chunk_text.lower():
                        contexts.append("fireplace" if "home" in chunk_text.lower() else "gas_leak")
                    if "social" in chunk_text.lower() or "friend" in chunk_text.lower():
                        contexts.append("social")
                        
                    if self.nt_state:
                        self.nt_state.set_contexts(contexts)
                        self.nt_state.update_from_body(reading.to_dict())
                        
                        # Ativação voluntária baseada em palavras
                        for concept in ["fire", "danger", "pain", "success", "reward", "social", "unknown"]:
                            if concept in chunk_text.lower():
                                self.nt_state.activate_concept(concept, contexts, strength=0.3)
                        
                        self._body_concept = self.body_encoder.get_body_concept(reading) if self.body_encoder else "unknown"

                    # Cerebelo
                    if self.cerebellum:
                        state_vec = reading.to_array()[:15]
                        action_vec = np.array(act.move + [act.grasp], dtype=np.float32)
                        next_reading = self.body.read_sensors()
                        next_state_vec = next_reading.to_array()[:15]
                        self.cerebellum.observe_result(state_vec, action_vec, next_state_vec)
                except Exception as e:
                    print(f"[BRIDGE] Erro na simulação de incorporação do chunk: {e}")

            await self.emit_event("chunk",
                text=chunk_text,
                residual_energy=float(residual_energy),
                step=step + 1)

            chunk_mf = self.engine.encode_text(chunk_text)
            proj = self.project_vector(chunk_mf.tensor.float())
            await self.emit_event("chunk_3d", position=proj, text=chunk_text[:30])

            spoken_vec = spoken_vec + self.engine.encode_text(
                " ".join(chunk_text.split()[:4])).tensor.float()
            spoken_vec = torch.sign(spoken_vec)

        if not generated:
            return "[...]"
        sep = "\n" if ('def ' in target_text or 'class ' in target_text) else " "
        result = sep.join(generated)
        return desanitize_code(result)

    async def query_compositional(self, text: str):
        """Query usando CognitiveAnswerer (composicional, não preditivo)."""
        self.output = ""
        self.current_intention = text
        await self.emit_event("intention", text=text)

        intention_mf = self.engine.encode_text(text)
        intention_vec = intention_mf.tensor.float()

        result = self.answerer.answer_compositional(text, top_k=5)

        result_mf = self.engine.encode_text(result[:200])
        fe = float(torch.nn.functional.cosine_similarity(
            result_mf.tensor.unsqueeze(0), intention_vec.unsqueeze(0)))
        dopamine = max(0, 0.5 - abs(fe - 0.3))
        noradrenaline = max(0, fe - 0.6) * 2
        acetylcholine = max(0, 0.3 - fe) * 2
        await self.emit_event("fe_update", fe=fe, dopamine=dopamine,
                              noradrenaline=noradrenaline, acetylcholine=acetylcholine)

        chunks = result.split('. ')
        for i, chunk in enumerate(chunks[:3]):
            await self.emit_event("chunk", text=chunk, step=i+1)
            await self.emit_event("fe_update", fe=fe * (1 - i/10),
                                  dopamine=dopamine, noradrenaline=noradrenaline,
                                  acetylcholine=acetylcholine * (1 + i/5))

        self.output = result
        self.residual_history.append(fe)
        await self.emit_event("complete", output=result)
        return result

    async def query_analogical(self, concept_a: str, concept_b: str):
        """Transferência analógica: a ⊕ b → interseção."""
        text = f"{concept_a} similar to {concept_b}"
        self.output = ""
        self.current_intention = text
        await self.emit_event("intention", text=text)

        a_mf = self.engine.encode_text(concept_a)
        b_mf = self.engine.encode_text(concept_b)
        bundle_vec = torch.sign(a_mf.tensor.float() + b_mf.tensor.float())
        fe = float(torch.nn.functional.cosine_similarity(
            bundle_vec.unsqueeze(0), a_mf.tensor.unsqueeze(0)))
        dopamine = max(0, 0.5 - abs(fe - 0.3))
        noradrenaline = max(0, 0.4 - fe) * 2
        acetylcholine = max(0, 0.2 - fe) * 2
        await self.emit_event("fe_update", fe=fe, dopamine=dopamine,
                              noradrenaline=noradrenaline, acetylcholine=acetylcholine)

        result = self.answerer.analogical_transfer(concept_a, concept_b, top_k=5)
        chunks = result.split('. ')
        for i, chunk in enumerate(chunks[:3]):
            await self.emit_event("chunk", text=chunk, step=i+1)
            await self.emit_event("fe_update", fe=fe * (1 - i/10),
                                  dopamine=dopamine, noradrenaline=noradrenaline,
                                  acetylcholine=acetylcholine * (1 + i/5))

        self.output = result
        self.residual_history.append(fe)
        await self.emit_event("complete", output=result)
        return result

    def get_status(self) -> dict:
        stats = self.memory.get_stats() if hasattr(self.memory, 'get_stats') else {}
        out = {
            "intention": self.current_intention,
            "output": self.output,
            "memory_items": stats.get('total_items', 0),
            "memory_sources": stats.get('sources', []),
            "working_hdv_size": len(self.generator.working_hdv),
            "residual_history": self.residual_history[-50:],
            "pca_points": len(self.pca_points),
        }

        # Injeta dados de incorporação se disponíveis
        if hasattr(self, 'body') and self.body:
            out["body_concept"] = self._body_concept
            out["body_state"] = self.body.get_state()
        if hasattr(self, 'nt_state') and self.nt_state:
            out["nt_state_6d"] = self.nt_state.as_dict()
            out["nt_mood"] = self.nt_state.mood
            out["nt_gains"] = {
                "learning_rate":   round(self.nt_state.get_gain("learning_rate"), 3),
                "attention_width": round(self.nt_state.get_gain("attention_width"), 3),
                "threat_bias":     round(self.nt_state.get_gain("threat_bias"), 3),
                "trust_level":     round(self.nt_state.get_gain("trust_level"), 3),
            }
        if hasattr(self, 'cerebellum') and self.cerebellum:
            out["cerebellum_error"] = round(self.cerebellum.get_prediction_error(), 4)
            
        return out
