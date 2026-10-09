"""
AGI Live — Runner principal da AGI com tradutor MentalState + Explain Mode.

Uso:
  python agi_live.py                    # modo interativo
  python agi_live.py --input "texto"    # input unico
  python agi_live.py --explain --input "texto"  # modo explicacao
"""
import sys, os, time, json, argparse, threading
sys.path.insert(0, os.path.dirname(__file__))
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from core.mental_state import get_mental_state, reset_mental_state
from core.cognitive_cycle import CognitiveCycle


class Translator:
    """Tradutor que le do MentalState completo (nao apenas recall).

    Nao faz parte do ciclo cognitivo.
    Verbaliza o estado mental apos processamento.
    """

    def translate(self, ms, mode="human") -> str:
        if mode == "explain":
            return self._explain(ms)
        return self._human(ms)

    def _human(self, ms) -> str:
        """Modo humano: resposta natural a partir do MentalState."""
        # Prioridade 0: Competicao de hipoteses (vencedoras por categoria)
        comp = ms.read("competition") or {}
        
        is_a_winner = comp.get("is_a_winner", "")
        is_a_score = comp.get("is_a_score", 0)
        prop_winner = comp.get("property_winner", "")
        prop_score = comp.get("property_score", 0)
        global_winner = comp.get("global_winner", "")
        justification = comp.get("justification", "")
        
        # Monta resposta a partir das vencedoras por categoria
        parts = []
        if is_a_winner and is_a_score > 0.3:
            seed = is_a_winner.split("→")[0] if "→" in is_a_winner else ""
            target = is_a_winner.split("→")[1] if "→" in is_a_winner else is_a_winner
            if seed and target:
                parts.append(f"{seed} is a {target}")
        if prop_winner and prop_score > 0.3:
            prop_name = prop_winner.split("→")[-1] if "→" in prop_winner else prop_winner
            parts.append(prop_name)
        
        if parts:
            base = " → ".join(parts)
            if global_winner and comp.get("global_score", 0) > 0.3:
                return f"[{comp['global_score']:.2f}] {base}"
            return base
        
        # Fallback: CGE ativo
        cge = ms.read("cge") or {}
        active = cge.get("active_nodes", [])
        active_edges = cge.get("active_edges", [])
        seed = cge.get("seed", "")

        # Prioridade 1: Inferencia via arestas CGE
        if seed and active_edges:
            path = self._build_cge_path(seed, active, active_edges)
            if len(path) >= 2:
                return " → ".join(path)

        # Prioridade 2: CGE ativo
        if active:
            # Mostra apenas nos relevantes (remove seed se ja incluso)
            display = [n for n in active if n != seed][:6]
            if seed:
                return f"{seed} → {' → '.join(display)}" if display else seed
            return ", ".join(active[:8])

        # Prioridade 3: Recall (fallback)
        recall = ms.read("recall") or {}
        episodes = recall.get("episodes", [])
        if episodes:
            best = episodes[0]
            inp = best.get("content", {}).get("input", "") if isinstance(best, dict) else str(best)
            score = best.get("score", 0) if isinstance(best, dict) else 0
            return f"[{score:.2f}] {inp[:120]}"

        return f"[sense] {ms.current_input[:80]}" if ms.current_input else "[idle]"

    def _build_cge_path(self, seed: str, active: list, edges: list) -> list:
        """Constroi caminho de inferencia a partir das arestas CGE ativas."""
        if not edges:
            return [seed]
        # BFS simples: do seed, segue as arestas ativas
        visited = {seed}
        queue = [seed]
        parent = {seed: None}
        # Limita a busca
        for _ in range(len(active)):
            current = queue.pop(0) if queue else None
            if current is None:
                break
            for e in edges:
                if e.get("from") == current and e["to"] not in visited:
                    visited.add(e["to"])
                    parent[e["to"]] = current
                    queue.append(e["to"])
                elif e.get("to") == current and e["from"] not in visited:
                    visited.add(e["from"])
                    parent[e["from"]] = current
                    queue.append(e["from"])
        # Converte o caminho do seed ate o no mais distante
        order = list(visited)
        if len(order) <= 1:
            return [seed]
        return order

    def _explain(self, ms) -> str:
        """Modo explicacao: trajetoria completa do raciocinio."""
        lines = []

        # Input
        lines.append(f"INPUT: {ms.current_input}")

        # Recall
        recall = ms.read("recall") or {}
        eps = recall.get("episodes", [])
        if eps:
            lines.append(f"RECALL: [{eps[0].get('score',0):.2f}] {eps[0].get('content',{}).get('input','')[:60]}")
        else:
            lines.append("RECALL: (none)")

        # CGE
        cge = ms.read("cge") or {}
        active = cge.get("active_nodes", [])
        lines.append(f"CGE: {cge.get('total_nodes',0)} nos/{cge.get('total_edges',0)} arestas")
        if active:
            lines.append(f"  ATIVOS: {', '.join(active[:10])}")
        if cge.get("seed"):
            lines.append(f"  SEED: {cge['seed']}")

        # WorldModel
        wm = ms.read("world_model") or {}
        h = wm.get("hypotheses_count", 0)
        if h:
            lines.append(f"WORLD MODEL: {h} hipoteses")

        # Forja
        forja = ms.read("forja") or {}
        nc = forja.get("new_concepts", 0)
        if nc:
            lines.append(f"FORJA: +{nc} conceitos novos")

        # Competicao de hipoteses (por categoria)
        comp = ms.read("competition") or {}
        if comp:
            lines.append(f"COMPETICAO: {comp.get('n_hypotheses',0)} hipoteses, "
                        f"entropy={comp.get('entropy',0):.3f}")
            lines.append(f"  IS_A: {comp.get('is_a_winner','?')} ({comp.get('is_a_score',0):.3f})")
            lines.append(f"  PROP: {comp.get('property_winner','?')} ({comp.get('property_score',0):.3f})")
            decomp = comp.get("confidence_decomposition", {})
            if decomp:
                lines.append("  CONFIDENCE DECOMPOSITION:")
                for k, v in sorted(decomp.items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"    {k}: {v:.3f}")
            if comp.get("justification"):
                lines.append(f"  JUST: {comp['justification'][:100]}")

        # NeuroField
        lines.append(f"NEURO: DA={ms.da:.3f} NA={ms.na:.3f} ACh={ms.ach:.3f} 5HT={ms.ht5:.3f}")

        # Ciclo
        lines.append(f"CICLO: {ms.cycle_iterations} iters, PE={ms.prediction_error:.4f}, {ms.last_cycle_duration:.1f}s")

        # Resposta
        response = self._human(ms)
        lines.append(f"\nRESPOSTA: {response}")

        return "\n".join(lines)


class AGI:
    """AGI completa com ciclo cognitivo continuo + workspace sensorial."""

    def __init__(self, load_backup: bool = True, sensory: bool = False):
        self.ms = get_mental_state()
        self.cycle = CognitiveCycle()
        self.translator = Translator()

        self.neurofield = SimpleNeuroField()
        self.cycle.register_module("neurofield", self.neurofield)

        # Workspace sensorial (audio + video)
        self.sensory_ws = None
        if sensory:
            self._init_sensory()

        if load_backup:
            print("[AGI] EpisodeStore sera carregado sob demanda")

    def _init_sensory(self):
        """Inicia captura continua de audio e video."""
        try:
            from core.sensory_workspace import SensoryWorkspace
            self.sensory_ws = SensoryWorkspace(self.ms)
            if self.sensory_ws.start():
                print("[AGI] Workspace sensorial ativo (audio+video)")
            else:
                self.sensory_ws = None
        except Exception as e:
            print(f"[AGI] Sensory init error: {e}")
            self.sensory_ws = None

    def process(self, text_input: str, modality: str = "text", mode: str = "human") -> str:
        self.cycle.tick(text_input, modality)
        return self.translator.translate(self.ms, mode=mode)

    def get_state(self) -> dict:
        store_stats = {}
        try:
            if hasattr(self.cycle, "_store") and self.cycle._store:
                store_stats = self.cycle._store.get_stats()
        except:
            pass
        
        # Estado sensorial
        sensory_state = {}
        if self.sensory_ws:
            try:
                sensory_state = self.sensory_ws.get_state()
            except:
                pass
        
        return {
            "mental": self.ms.to_dict(),
            "store": store_stats,
            "cycle": {
                "total": self.ms.total_cycles,
                "inner_iterations": self.ms.cycle_iterations,
                "last_duration": round(self.ms.last_cycle_duration, 3),
            },
            "sensory": sensory_state,
        }

    def save(self):
        if hasattr(self.cycle, "_store") and self.cycle._store:
            store = self.cycle._store
            print(f"[AGI] Estado mantido em EpisodeStore ({store.n_episodes} episodios)")
        else:
            print("[AGI] (sem memoria para salvar)")


class SimpleNeuroField:
    def __init__(self):
        self.da = self.na = self.ach = self.ht5 = 0.5

    def update(self, nt_context: dict = None, prediction_error: float = 0.0):
        surprise = min(1.0, prediction_error * 2)
        self.na = 0.3 + 0.7 * surprise
        self.da = max(0.1, self.da - surprise * 0.1)
        self.ach = 0.3 + 0.7 * (1 - surprise)
        self.ht5 = 0.4 + 0.3 * (1 - surprise)
        decay = 0.99
        self.da = self.da * decay + 0.5 * (1 - decay)
        self.na = self.na * decay + 0.5 * (1 - decay)
        self.ach = self.ach * decay + 0.5 * (1 - decay)
        self.ht5 = self.ht5 * decay + 0.5 * (1 - decay)


def interactive_mode(agi: AGI):
    print("\n" + "=" * 60)
    print("  NOVAMIND AGI — Modo Interativo")
    print("=" * 60)
    print("  /explain  /state  /save  /help  /quit")
    print()

    mode = "human"
    while True:
        try:
            inp = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break

        if not inp:
            continue
        if inp == "/quit":
            break
        elif inp == "/save":
            agi.save(); continue
        elif inp == "/state":
            s = agi.get_state()
            print(json.dumps(s["mental"], indent=2, default=str)); continue
        elif inp == "/explain":
            mode = "explain" if mode == "human" else "human"
            print(f"Modo: {mode}"); continue
        elif inp == "/replay":
            cycles = agi.cycle.list_recorded_cycles(10)
            if not cycles:
                print("  Nenhum ciclo registrado")
                continue
            print("  Ultimos ciclos:")
            for c in cycles:
                print(f"    {c['cycle_id'][:30]} | {c['input'][:40]}")
            print("  Use /replay <cycle_id> para ver detalhes")
            continue
        elif inp.startswith("/replay "):
            cid = inp[8:].strip()
            replay = agi.cycle.get_replay(cid)
            print(replay.show(cid))
            continue
        elif inp == "/audit":
            report = agi.cycle.get_audit_report()
            print(report)
            continue
        elif inp.startswith("/audit "):
            aid = inp[7:].strip()
            report = agi.cycle.get_audit_report(aid)
            print(report)
            continue
        elif inp == "/help":
            print("  /explain  - alterna modo explicacao")
            print("  /state    - estado mental completo")
            print("  /replay   - lista ciclos registrados")
            print("  /replay <id> - reproduz ciclo")
            print("  /audit    - auditoria completa do ultimo ciclo")
            print("  /audit <id> - auditoria de ciclo especifico")
            print("  /save     - salvar estado")
            print("  /quit     - sair")
            continue

        output = agi.process(inp, mode=mode)
        if mode == "explain":
            print(output)
        else:
            print(f"  {output}")

    agi.save()
    print("[AGI] Fim")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input unico")
    parser.add_argument("--fresh", action="store_true", help="Sem carregar backup")
    parser.add_argument("--explain", action="store_true", help="Modo explicacao")
    parser.add_argument("--sensory", action="store_true", help="Workspace sensorial (cam+mic)")
    args = parser.parse_args()

    agi = AGI(load_backup=not args.fresh, sensory=args.sensory)
    mode = "explain" if args.explain else "human"

    if args.input:
        output = agi.process(args.input, mode=mode)
        print(output)
        if mode == "human":
            s = agi.get_state()
            print(json.dumps(s["mental"], indent=2, default=str))
        agi.save()
    else:
        interactive_mode(agi)
