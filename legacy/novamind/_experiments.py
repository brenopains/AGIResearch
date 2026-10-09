"""
Bateria de Experimentos Cognitivos — Mede evolucao da AGI ao longo do tempo.

Cada experimento e executado e registrado em log. Resultados sao comparaveis
entre dias/semanas para medir se a AGI esta evoluindo cognitivamente.

Uso:
    python _experiments.py                      # roda todos os experimentos
    python _experiments.py --list               # lista experimentos
    python _experiments.py --only retention     # roda apenas um

Metricas salvas em: D:\AGI2\experiments\results.json
"""
import sys, os, json, time, datetime, random, hashlib
sys.path.insert(0, os.path.dirname(__file__))

from core.mental_state import reset_mental_state
from core.cognitive_cycle import CognitiveCycle

EXPERIMENTS_DIR = r"D:\AGI2\experiments"
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(EXPERIMENTS_DIR, "results.json")
HISTORY_FILE = os.path.join(EXPERIMENTS_DIR, "history.jsonl")


def log_result(name, metrics):
    """Salva resultado de um experimento."""
    entry = {
        "timestamp": time.time(),
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "experiment": name,
        "metrics": metrics,
    }
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    
    # Atualiza results.json (ultimo resultado de cada experimento)
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    results[name] = entry
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=1, default=str)
    
    print(f"  [LOG] {name}: salvos {len(metrics)} metricas")


def show_history(name=None, days=7):
    """Mostra historico de um experimento."""
    if not os.path.exists(HISTORY_FILE):
        print("  (sem historico)")
        return
    
    cutoff = time.time() - days * 86400
    entries = []
    with open(HISTORY_FILE) as f:
        for line in f:
            try:
                e = json.loads(line)
                if e["timestamp"] >= cutoff and (name is None or e["experiment"] == name):
                    entries.append(e)
            except:
                pass
    
    if not entries:
        print(f"  (nenhum resultado nos ultimos {days} dias)")
        return
    
    print(f"  Historico ({len(entries)} entradas):")
    for e in entries[-10:]:
        print(f"    {e['date']} | {e['experiment']}: {json.dumps(e['metrics'], default=str)[:100]}")
    return entries


class ExperimentRunner:
    """Executa bateria de experimentos cognitivos."""

    def __init__(self):
        self.ciclo = CognitiveCycle()
        self.ciclo.register_module("neurofield", type("NF",(),{
            "update":lambda s,c,e: None,
            "da":.5,"na":.5,"ach":.5,"ht5":.5
        })())
        self.results = {}
        self.start_time = time.time()

    # ─── Experimento 1: Retencao ───────────────────────────

    def retention(self):
        """Testa se a AGI lembra de informacao apos ciclos.

        Ensina um fato novo, depois pergunta sobre ele.
        Mede: recall score, CGE nodes ativos, prediction error.
        """
        # Fato raro (hash-based para ser unico)
        fact_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        fact = f"A {fact_id} is a rare mineral found only in volcanoes."
        question = f"What is a {fact_id}?"

        # Ensina
        self.ciclo.tick(fact, "text")
        
        # Pergunta imediatamente
        ms1 = self.ciclo.tick(question, "text")
        cge1 = ms1.read("cge") or {}
        recall1 = ms1.read("recall") or {}
        
        # Pergunta novamente apos 10 ciclos (sem o fato)
        for _ in range(10):
            self.ciclo.tick(f"tell me about {fact_id}", "text")
        ms2 = self.ciclo.tick(question, "text")
        cge2 = ms2.read("cge") or {}
        
        # Metricas
        has_node_immediate = fact_id in cge1.get("active_nodes", [])
        has_node_delayed = fact_id in cge2.get("active_nodes", [])
        
        metrics = {
            "fact": fact,
            "immediate_cge": has_node_immediate,
            "delayed_cge": has_node_delayed,
            "cge_nodes_created": cge1.get("total_nodes", 0),
            "retained": has_node_delayed,
            "retention_score": 1.0 if has_node_delayed else 0.0,
        }
        log_result("retention", metrics)
        return metrics

    # ─── Experimento 2: Generalizacao ──────────────────────

    def generalization(self):
        """Testa se a AGI generaliza para combinacoes novas.

        Ensina "All X are Y" e "Z is X", depois pergunta "Is Z Y?"
        """
        concepts = ["dax", "wug", "blick", "glim", "florp"]
        random.shuffle(concepts)
        a, b, c = concepts[:3]

        # Ensina relacoes
        self.ciclo.tick(f"Every {a} is a {b}.", "text")
        self.ciclo.tick(f"A {c} is a {a}.", "text")

        # Pergunta
        ms = self.ciclo.tick(f"Is a {c} a {b}?", "text")
        cge = ms.read("cge") or {}
        active = cge.get("active_nodes", [])
        
        # Verifica se os conceitos foram ativados
        has_a = a in active
        has_b = b in active
        has_c = c in active
        all_three = has_a and has_b and has_c
        
        metrics = {
            f"{a}_activated": has_a,
            f"{b}_activated": has_b,
            f"{c}_activated": has_c,
            "all_three_activated": all_three,
            "iterations": ms.cycle_iterations,
            "pe": round(ms.prediction_error, 4),
            "generalization_score": 1.0 if all_three else 0.0,
        }
        log_result("generalization", metrics)
        return metrics

    # ─── Experimento 3: Transferencia ──────────────────────

    def transfer(self):
        """Testa se conhecimento de um dominio transfere para outro.

        Ensina sobre "golpes" (blows), depois pergunta sobre "socos" (punches).
        """
        # Ensina sobre "hitting" (dominio A)
        self.ciclo.tick("A punch is a strike with a fist.", "text")
        
        # Pergunta sobre "slap" (dominio B - relacionado)
        ms = self.ciclo.tick("Is a slap similar to a punch?", "text")
        cge = ms.read("cge") or {}
        active = cge.get("active_nodes", [])
        
        punch_in_cge = "punch" in active or "strike" in active
        
        metrics = {
            "punch_in_cge": punch_in_cge,
            "active_nodes": active[:8],
            "iterations": ms.cycle_iterations,
            "transfer_score": 1.0 if punch_in_cge else 0.0,
        }
        log_result("transfer", metrics)
        return metrics

    # ─── Experimento 4: Consolidacao ───────────────────────

    def consolidation(self):
        """Testa se episodios viram conceitos apos sono.

        Alimenta varios episodios similares, depois verifica se
        um conceito foi criado pela Forja.
        """
        # Ensina variacoes de um mesmo tema
        tema = "quantum" + hashlib.md5(str(time.time()).encode()).hexdigest()[:4]
        for i in range(5):
            self.ciclo.tick(f"A {tema}_{i} exhibits quantum entanglement.", "text")
        
        # Forca sono
        self.ciclo._sleep_cycle()
        
        # Verifica se um procedimento foi criado
        if hasattr(self.ciclo, "_created_procs"):
            procs = list(self.ciclo._created_procs)
        else:
            procs = []
        
        metrics = {
            "tema": tema,
            "procedures_created": len(procs),
            "procedures": procs[:5],
            "consolidation_score": min(1.0, len(procs) / 3),
        }
        log_result("consolidation", metrics)
        return metrics

    # ─── Experimento 5: Esquecimento ───────────────────────

    def forgetting(self):
        """Testa se informacao irrelevante e esquecida."""
        # Adiciona fato propositalmente irrelevante
        noise_id = "noise_" + hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        self.ciclo.tick(f"The {noise_id} is irrelevant.", "text")
        
        # Muitos ciclos sem mencionar
        for _ in range(30):
            self.ciclo.tick("The sky is blue. Water is wet. Fire is hot.", "text")
        
        # Pergunta sobre o fato irrelevante
        ms = self.ciclo.tick(f"What is a {noise_id}?", "text")
        cge = ms.read("cge") or {}
        active = cge.get("active_nodes", [])
        forgot = noise_id not in active
        
        metrics = {
            "noise_concept": noise_id,
            "forgotten": forgot,
            "forgetting_score": 1.0 if forgot else 0.0,
        }
        log_result("forgetting", metrics)
        return metrics

    # ─── Experimento 6: Autonomia ──────────────────────────

    def autonomy(self):
        """Testa se a AGI gera curiosidade autonomamente."""
        # Verifica se a fila de curiosidade tem itens
        queue_size = len(self.ciclo._curiosity_queue)
        
        # Alimenta um conceito de baixa confianca
        self.ciclo.tick("Explain quantum chromodynamics.", "text")
        
        # Verifica se gerou curiosidade
        queue_after = len(self.ciclo._curiosity_queue)
        generated = queue_after > queue_size
        
        metrics = {
            "curiosity_queue_before": queue_size,
            "curiosity_queue_after": queue_after,
            "generated_curiosity": generated,
            "autonomy_score": 1.0 if generated else 0.0,
        }
        log_result("autonomy", metrics)
        return metrics

    # ─── Runner ─────────────────────────────────────────────

    ALL_EXPERIMENTS = {
        "retention": retention,
        "generalization": generalization,
        "transfer": transfer,
        "consolidation": consolidation,
        "forgetting": forgetting,
        "autonomy": autonomy,
    }

    def run_all(self):
        print(f"\n{'='*60}")
        print(f"  EXPERIMENTOS COGNITIVOS — NovaMind AGI")
        print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}")
        
        for name, method in self.ALL_EXPERIMENTS.items():
            print(f"\n  >>> {name.upper()}")
            try:
                method(self)
            except Exception as e:
                print(f"  [ERRO] {e}")
                import traceback
                traceback.print_exc()
        
        self._summary()

    def run_one(self, name):
        if name in self.ALL_EXPERIMENTS:
            print(f"\n  >>> {name.upper()}")
            self.ALL_EXPERIMENTS[name](self)
        else:
            print(f"  Experimento '{name}' nao encontrado. Use --list")

    def _summary(self):
        duration = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"  EXPERIMENTOS CONCLUIDOS")
        print(f"  Duracao: {duration:.0f}s")
        print(f"  Resultados: {RESULTS_FILE}")
        print(f"  Historico: {HISTORY_FILE}")
        print(f"{'='*60}")
        
        # Mostra ultimos resultados
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE) as f:
                results = json.load(f)
            print()
            for name, entry in results.items():
                m = entry.get("metrics", {})
                scores = {k: v for k, v in m.items() if "score" in k}
                vs = ", ".join(f"{k}={v}" for k, v in scores.items()) if scores else ""
                print(f"  {name:20s}: {vs}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="Listar experimentos")
    parser.add_argument("--only", type=str, help="Rodar apenas um experimento")
    parser.add_argument("--history", type=str, nargs="?", const=None,
                       help="Mostrar historico de um experimento")
    parser.add_argument("--days", type=int, default=7, help="Dias de historico")
    args = parser.parse_args()

    if args.list:
        print("Experimentes disponiveis:")
        for name in ExperimentRunner.ALL_EXPERIMENTS:
            print(f"  - {name}")
    elif args.history is not None:
        show_history(args.history, args.days)
    else:
        runner = ExperimentRunner()
        if args.only:
            runner.run_one(args.only)
        else:
            runner.run_all()
