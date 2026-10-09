"""Benchmark Suite — Testes estruturados de raciocinio.

Executa baterias de problemas em 6 categorias e registra acertos/erros.

Uso:
    python _benchmark.py              # roda todos os testes
    python _benchmark.py --category analogia  # so uma categoria
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))

from core.mental_state import reset_mental_state
from core.cognitive_cycle import CognitiveCycle


class Benchmark:
    """Executa bateria de testes e mede desempenho."""

    def __init__(self):
        self.ciclo = CognitiveCycle()
        self.ciclo.register_module("neurofield", type("NF",(),{
            "update":lambda s,c,e: None,
            "da":.5,"na":.5,"ach":.5,"ht5":.5
        })())
        self.results = []
        self.start_time = time.time()

    def test(self, category, name, query, expected_hint=""):
        """Roda um teste e registra resultado."""
        ms = self.ciclo.tick(query, "text")
        cge = ms.read("cge") or {}
        recall = ms.read("recall") or {}
        active = cge.get("active_nodes", [])
        seed = cge.get("seed", "")

        # Avalia: CGE tem nos relevantes?
        relevant = expected_hint and any(h in n for n in active for h in expected_hint.split())
        score = 1.0 if relevant else 0.0

        result = {
            "category": category, "name": name,
            "query": query, "expected": expected_hint,
            "seed": seed, "cge_active": active[:8],
            "recall_score": recall.get("episodes", [{}])[0].get("score", 0) if recall.get("episodes") else 0,
            "iterations": ms.cycle_iterations,
            "pe": round(ms.prediction_error, 4),
            "duration": round(ms.last_cycle_duration, 2),
            "cge_nodes": cge.get("total_nodes", 0),
            "score": score,
        }
        self.results.append(result)
        self._print(result)
        return result

    def _print(self, r):
        status = "PASS" if r["score"] > 0 else "FAIL"
        print(f"  [{status}] {r['name']:30s} seed={r['seed']:15s} PE={r['pe']:.3f} CGE={r['cge_active'][:4]}")

    def run_all(self):
        print(f"\n{'='*60}")
        print(f"  BENCHMARK SUITE — NOVAMIND AGI")
        print(f"{'='*60}")

        for cat, tests in self._tests().items():
            print(f"\n  --- {cat.upper()} ---")
            for t in tests:
                self.test(cat, *t)

        self._summary()

    def _tests(self):
        return {
            "grounding": [
                ("cat -> animal", "is a cat an animal?", "cat animal"),
                ("dog -> mammal", "is a dog a mammal?", "dog mammal"),
                ("whale -> mammal", "is a whale a mammal?", "whale mammal"),
                ("penguin -> bird", "is a penguin a bird?", "penguin bird"),
                ("water -> liquid", "is water a liquid?", "water liquid"),
                ("gold -> metal", "is gold a metal?", "gold metal"),
                ("rose -> flower", "is a rose a flower?", "rose flower"),
                ("snake -> reptile", "is a snake a reptile?", "snake reptile"),
            ],
            "causality": [
                ("gravity -> fall", "what happens when you drop something?", "gravity fall"),
                ("fire -> heat", "does fire produce heat?", "fire heat"),
                ("water -> boil", "what happens when you heat water?", "water boil"),
                ("cold -> freeze", "what happens when water gets very cold?", "cold freeze"),
                ("salt dissolve", "does salt dissolve in water?", "salt dissolve"),
                ("metal conduct", "does metal conduct electricity?", "metal conduct"),
            ],
            "analogy": [
                ("bird:wing", "birds use wings to fly. what do fish use?", "fish fin"),
                ("dog:bark", "dogs bark. what sound do cats make?", "cat meow"),
                ("human:hand", "humans use hands. what do birds use?", "bird wing"),
            ],
            "composition": [
                ("cat+dog", "what do cats and dogs have in common?", "mammal animal"),
                ("bird+fish", "are birds and fish both animals?", "animal"),
                ("whale+dolphin", "what do whales and dolphins have in common?", "mammal"),
            ],
            "test_knowledge": [
                ("mammal breathe", "do all mammals breathe air?", "mammal breathe"),
                ("bird wings", "do birds have wings?", "bird wing"),
                ("fish gills", "do fish have gills?", "fish gill"),
                ("plant sunlight", "do plants need sunlight?", "plant sunlight"),
                ("gravity orbit", "why do planets orbit the sun?", "gravity orbit"),
            ],
        }

    def _summary(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r["score"] > 0)
        duration = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"  RESULTADOS: {passed}/{total} passed ({passed/total*100:.0f}%)")
        print(f"  Duracao: {duration:.0f}s")
        print(f"  CGE final: {self.results[-1]['cge_nodes']} nodes" if self.results else "")
        print(f"{'='*60}")
        
        # Por categoria
        cats = {}
        for r in self.results:
            cats.setdefault(r["category"], {"t": 0, "p": 0})
            cats[r["category"]]["t"] += 1
            if r["score"] > 0:
                cats[r["category"]]["p"] += 1
        for cat, stats in cats.items():
            print(f"  {cat:15s}: {stats['p']}/{stats['t']} ({stats['p']/stats['t']*100:.0f}%)")

        # Salvar
        log_file = f"D:/AGI2/memories/benchmark_{int(time.time())}.json"
        with open(log_file, "w") as f:
            json.dump(self.results, f, indent=1, default=str)
        print(f"\n  Log salvo: {log_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, help="Filtrar por categoria")
    args = parser.parse_args()

    b = Benchmark()
    b.run_all()
