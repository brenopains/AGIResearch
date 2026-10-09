"""
FlightRecorder — Registro estruturado de cada ciclo cognitivo.

Similar a um "flight recorder" de aviao: cada ciclo gera um registro
completo que pode ser reproduzido depois como um filme do raciocinio.

Uso:
    recorder = FlightRecorder()
    recorder.start_cycle(input_text)
    recorder.record_recall(results)
    recorder.record_cge(state)
    recorder.record_worldmodel(hypotheses)
    recorder.record_forja(actions)
    recorder.record_neurofield(da, na, ach, ht5)
    recorder.end_cycle(output, justification)

    # Replay:
    for entry in recorder.replay(episode_id):
        print(entry)
"""
import time, json, os, copy
from typing import Dict, List, Optional, Any
from collections import defaultdict

RECORDER_DIR = r"D:\AGI2\flight_recorder"
os.makedirs(RECORDER_DIR, exist_ok=True)


class FlightRecorder:
    """Registrador de voo cognitivo. Cada ciclo vira um episodio reproduzivel."""

    def __init__(self):
        self.current: Dict[str, Any] = {}
        self.history: List[Dict] = []
        self._pe_trajectory = []
        self._nf_trajectory = []

    def start_cycle(self, input_text: str, seed_hdv: str = "", modality: str = "text"):
        """Inicia um novo ciclo de registro."""
        self.current = {
            "timestamp": time.time(),
            "cycle_id": f"cycle_{int(time.time()*1000)}_{len(self.history)}",
            "input": input_text,
            "modality": modality,
            "seed_hdv": seed_hdv,
            "recall": [],
            "cge_activation": [],
            "cge_activation_order": [],
            "hypotheses": [],
            "hypotheses_discarded": [],
            "world_model_interventions": [],
            "forja_actions": [],
            "pe_trajectory": [],
            "neurofield_trajectory": [],
            "attention": {},
            "output": "",
            "justification": "",
            "iterations": 0,
            "duration": 0.0,
        }
        self._pe_trajectory = []
        self._nf_trajectory = []

    def record_recall(self, episodes: list, method: str = ""):
        """Registra resultados do recall."""
        self.current["recall"] = [
            {
                "id": e.get("id", ""),
                "score": e.get("score", 0),
                "text": str(e.get("content", {}).get("input", ""))[:80],
                "type": e.get("type", ""),
            }
            for e in episodes[:5]
        ]
        self.current["recall_method"] = method

    def record_cge_activation(self, seed: str, active_nodes: list, active_edges: list,
                               total_nodes: int, total_edges: int):
        """Registra estado do CGE apos ativacao."""
        self.current["cge_seed"] = seed
        self.current["cge_total_nodes"] = total_nodes
        self.current["cge_total_edges"] = total_edges
        # Ordem de ativacao (mais relevante primeiro)
        self.current["cge_activation"] = active_nodes[:15]
        self.current["cge_activation_order"] = active_nodes[:10]
        self.current["cge_active_edges"] = [
            {"from": e.get("from", ""), "to": e.get("to", "")}
            for e in active_edges[:10]
        ]

    def record_hypothesis(self, hypothesis: str, confidence: float, source: str = "cge"):
        """Registra uma hipotese gerada."""
        self.current["hypotheses"].append({
            "hypothesis": hypothesis,
            "confidence": round(confidence, 3),
            "source": source,
            "timestamp": time.time(),
        })

    def discard_hypothesis(self, hypothesis: str, reason: str):
        """Registra uma hipotese descartada."""
        self.current["hypotheses_discarded"].append({
            "hypothesis": hypothesis,
            "reason": reason,
            "timestamp": time.time(),
        })

    def record_worldmodel(self, intervention: str, result: str, strength: float = 0.0):
        """Registra intervencao do World Model."""
        self.current["world_model_interventions"].append({
            "intervention": intervention,
            "result": result,
            "strength": round(strength, 3),
        })

    def record_forja(self, action_type: str, description: str, accepted: bool = True):
        """Registra acao da Forja (criacao de conceito/procedimento)."""
        self.current["forja_actions"].append({
            "type": action_type,
            "description": description,
            "accepted": accepted,
        })

    def record_pe(self, prediction_error: float):
        """Registra prediction error (chamado a cada iteracao)."""
        self._pe_trajectory.append(round(prediction_error, 4))

    def record_neurofield(self, da: float, na: float, ach: float, ht5: float):
        """Registra estado do NeuroField (chamado a cada iteracao)."""
        self._nf_trajectory.append({
            "da": round(da, 3), "na": round(na, 3),
            "ach": round(ach, 3), "ht5": round(ht5, 3),
        })

    def record_attention(self, attention: Dict[str, float]):
        """Registra distribuicao de atencao."""
        self.current["attention"] = {
            k: round(v, 3) for k, v in attention.items()
        }

    def end_cycle(self, output: str, justification: str = "", iterations: int = 0):
        """Finaliza o ciclo e salva o registro."""
        self.current["output"] = output
        self.current["justification"] = justification
        self.current["iterations"] = iterations
        self.current["duration"] = round(time.time() - self.current["timestamp"], 3)
        self.current["pe_trajectory"] = self._pe_trajectory
        self.current["neurofield_trajectory"] = self._nf_trajectory

        # Salva em disco
        entry = copy.deepcopy(self.current)
        self.history.append(entry)

        # Salva como JSON individual
        path = os.path.join(RECORDER_DIR, f"{entry['cycle_id']}.json")
        with open(path, "w") as f:
            json.dump(entry, f, indent=1, default=str)

        return entry["cycle_id"]

    def get_last(self) -> Optional[Dict]:
        """Retorna o ultimo ciclo registrado."""
        return self.history[-1] if self.history else None

    def get_by_id(self, cycle_id: str) -> Optional[Dict]:
        """Busca um ciclo por ID."""
        path = os.path.join(RECORDER_DIR, f"{cycle_id}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        for entry in self.history:
            if entry["cycle_id"] == cycle_id:
                return entry
        return None

    def list_cycles(self, n: int = 20) -> List[Dict]:
        """Lista os ultimos N ciclos."""
        files = sorted(os.listdir(RECORDER_DIR), reverse=True)[:n]
        cycles = []
        for fname in files:
            if fname.endswith(".json"):
                path = os.path.join(RECORDER_DIR, fname)
                try:
                    with open(path) as f:
                        data = json.load(f)
                        cycles.append({
                            "cycle_id": data["cycle_id"],
                            "input": data["input"][:60],
                            "output": data["output"][:60],
                            "iterations": data.get("iterations", 0),
                            "duration": data.get("duration", 0),
                            "timestamp": data.get("timestamp", 0),
                        })
                except:
                    pass
        return cycles


class CognitiveReplay:
    """Reproduz um ciclo cognitivo como 'filme' do raciocinio.

    Uso:
        replay = CognitiveReplay(recorder)
        replay.show("cycle_1234567890")
        # ou
        for frame in replay.stream("cycle_1234567890"):
            print(frame)
    """

    def __init__(self, recorder: FlightRecorder):
        self.recorder = recorder

    def show(self, cycle_id: str) -> str:
        """Retorna representacao textual completa de um ciclo."""
        entry = self.recorder.get_by_id(cycle_id)
        if not entry:
            return f"[Ciclo {cycle_id} nao encontrado]"

        lines = []
        lines.append(f"{'='*60}")
        lines.append(f"  REPLAY COGNITIVO - Ciclo {entry['cycle_id']}")
        lines.append(f"{'='*60}")

        # Input
        lines.append(f"\nINPUT: {entry['input']}")
        if entry.get("seed_hdv"):
            lines.append(f"SEED HDV: {entry['seed_hdv'][:50]}")
        lines.append(f"MODALITY: {entry.get('modality', 'text')}")

        # Recall
        recall = entry.get("recall", [])
        if recall:
            lines.append(f"\nRECALL ({entry.get('recall_method', '?')}):")
            for r in recall[:3]:
                lines.append(f"  [{r.get('score',0):.3f}] {r.get('text','')[:60]}")

        # CGE
        active = entry.get("cge_activation", [])
        if active:
            lines.append(f"\nCGE ATIVACAO (seed={entry.get('cge_seed','')}):")
            lines.append(f"  Ordem: {' -> '.join(active[:8])}")
            lines.append(f"  Total: {entry.get('cge_total_nodes',0)} nos / {entry.get('cge_total_edges',0)} arestas")
            edges = entry.get("cge_active_edges", [])
            if edges:
                edge_strs = []
                for edge in edges[:5]:
                    frm = edge.get("from", "")
                    to = edge.get("to", "")
                    edge_strs.append(f"{frm}->{to}")
                lines.append(f"  Arestas: {'; '.join(edge_strs)}")

        # Hipotese
        hyps = entry.get("hypotheses", [])
        if hyps:
            lines.append(f"\nHIPOTESES:")
            for h in hyps:
                lines.append(f"  [{h.get('confidence',0):.2f}] {h.get('hypothesis','')}")
        discarded = entry.get("hypotheses_discarded", [])
        if discarded:
            lines.append(f"  (descartadas: {len(discarded)})")

        # World Model
        wm = entry.get("world_model_interventions", [])
        if wm:
            lines.append(f"\nWORLD MODEL:")
            for w in wm[-3:]:
                lines.append(f"  {w.get('intervention','')} -> {w.get('result','')[:40]}")

        # Forja
        forja = entry.get("forja_actions", [])
        if forja:
            lines.append(f"\nFORJA:")
            for f_a in forja:
                lines.append(f"  [{f_a.get('type','')}] {f_a.get('description','')}")

        # PE trajectory
        pe = entry.get("pe_trajectory", [])
        if pe:
            lines.append(f"\nPREDICTION ERROR trajectory:")
            for i, p in enumerate(pe):
                marker = " <- estagnou" if i == len(pe) - 1 and len(pe) < 5 else ""
                lines.append(f"  Iter {i+1}: PE={p:.4f}{marker}")

        # NeuroField
        nf = entry.get("neurofield_trajectory", [])
        if nf:
            lines.append(f"\nNEUROFIELD trajectory:")
            for i, n in enumerate(nf):
                lines.append(f"  Iter {i+1}: DA={n.get('da','?'):.3f} NA={n.get('na','?'):.3f} "
                            f"ACh={n.get('ach','?'):.3f} 5HT={n.get('ht5','?'):.3f}")

        # Output
        lines.append(f"\nRESPOSTA: {entry.get('output','')}")
        if entry.get("justification"):
            lines.append(f"JUSTIFICATIVA: {entry['justification']}")

        lines.append(f"\nDuracao: {entry.get('duration',0):.1f}s | Iteracoes: {entry.get('iterations',0)}")
        lines.append(f"{'='*60}")

        return "\n".join(lines)

    def summary(self, n: int = 10) -> str:
        """Resumo dos ultimos N ciclos."""
        cycles = self.recorder.list_cycles(n)
        if not cycles:
            return "(nenhum ciclo registrado)"

        lines = [f"{'CICLO':30s} {'INPUT':30s} {'OUTPUT':30s} {'ITERS':5s} {'DURACAO':8s}"]
        lines.append("-" * 105)
        for c in cycles:
            inp = c.get("input", "")[:28]
            out = c.get("output", "")[:28]
            iters = c.get("iterations", "?")
            dur = f"{c.get('duration',0):.1f}s"
            cid = c.get("cycle_id", "?")[:28]
            lines.append(f"{cid:30s} {inp:30s} {out:30s} {str(iters):5s} {dur:8s}")
        return "\n".join(lines)
