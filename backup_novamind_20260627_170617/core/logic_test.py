"""
Teste de Lógica Autônoma com Rastreabilidade HDC
==================================================
Problemas que LLMs tipicamente FALHAM:

  1. RELAÇÕES ANALÓGICAS: "A está para B assim como C está para ?"
  2. RACIOCÍNIO TRANSITIVO: "A > B, B > C, então ?"
  3. NEGAÇÃO: "Todos A são B. X não é B. Logo ?"
  4. CONTRAFACTUAIS: monóxido de dinitrogênio (N₂O) → "gas hilariante"

Cada teste mostra:
  - TEXTO do pensamento (o que a AGI "disse" que pensou)
  - VETOR HDC do pensamento (primeiros 20 elementos + norma + soma)
  - INGREDIENTES semânticos extraídos
  - SIMILARIDADES entre etapas do raciocínio

Isso permite RASTREABILIDADE COMPLETA: se a resposta estiver errada,
sabemos EXATAMENTE em que passo a geometria falhou.
"""

import os, sys, json, time, torch
sys.path.insert(0, ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))

from core.semantic_composition import SemanticCompositionEngine
from core.hdc_memory import HDCMemory
from core.grounding_engine import GroundingEngine
from core.executive_planner import ExecutivePlanner


class LogicTest:
    """
    Executa teste lógico com rastreabilidade completa.

    Cada teste tem:
      - premisse: descrição do problema (em linguagem natural)
      - expected: None (aberto) ou resposta esperada
      - domain: área de conhecimento
    """

    def __init__(self):
        self.engine = GroundingEngine()
        self.memory = HDCMemory()
        self.composer = SemanticCompositionEngine(
            hdc_memory=self.memory, engine=self.engine, device='cuda')
        self.composer.build_word_cache()

        # Cache de rastreabilidade
        self.trace_log = []

    def trace(self, label: str, vec: torch.Tensor, text: str = ""):
        """Registra um passo do raciocínio com vetor HDC para auditoria."""
        entry = {
            "step": len(self.trace_log),
            "label": label,
            "text": text,
            "vector_first10": vec.flatten()[:10].tolist(),
            "vector_norm": float(vec.flatten().float().norm()),
            "vector_sum": float(vec.flatten().float().sum()),
            "cosine_similarity_to_prev": 0.0,
        }
        if self.trace_log:
            prev = self.trace_log[-1]["vector"]
            entry["cosine_similarity_to_prev"] = float(torch.cosine_similarity(
                prev.flatten().float().unsqueeze(0),
                vec.flatten().float().unsqueeze(0)))
        self.trace_log.append({"vector": vec.clone(), **entry})
        return entry

    def run(self, premise: str, expected: str = "", domain: str = "logic"):
        """Executa um teste lógico completo."""
        print(f"\n{'='*70}")
        print(f"TESTE LOGICO: {domain}")
        print(f"{'='*70}")
        print(f"Premissa: {premise}")
        print(f"Esperado: {expected or '[aberto — sem gabarito]'}")
        print()

        self.trace_log = []
        result = {}

        # 1. Codifica a premissa como pensamento HDC
        premise_vec = self.engine.encode_text(premise).tensor
        self.trace("PREMISSA", premise_vec, premise)

        # 2. Decompõe em ingredientes semânticos
        ingredients = self.composer._extract_ingredients(premise_vec)
        ing_text = ", ".join([w for w, _, _ in ingredients])
        self.trace("INGREDIENTES", premise_vec,
                   f"Palavras-chave: {ing_text}")

        # 3. Gera resposta textual via compositor
        answer = self.composer.translate_thought(premise_vec)
        answer_vec = self.engine.encode_text(answer).tensor
        self.trace("RESPOSTA", answer_vec, answer)

        # 4. Mede similaridade entre premissa e resposta
        cos_pa = float(torch.cosine_similarity(
            premise_vec.flatten().unsqueeze(0),
            answer_vec.flatten().unsqueeze(0)))

        result = {
            "premise": premise,
            "expected": expected,
            "answer": answer,
            "cos_premise_answer": cos_pa,
            "n_ingredients": len(ingredients),
            "ingredients": [{"word": w, "sim": round(s, 4)}
                            for w, _, s in ingredients],
            "trace": [e for e in self.trace_log if "vector" not in e],
        }

        print(f"Resposta da AGI: {answer}")
        print(f"Cosseno premissa->resposta: {cos_pa:.4f}")
        print(f"Ingredientes ({len(ingredients)}):")
        for w, _, s in ingredients[:5]:
            print(f"   {w:20s} sim={s:.4f}")
        print()

        return result

    def run_reasoning_chain(self, premises: list, expected: str = ""):
        """
        Executa uma CADEIA DE RACIOCÍNIO: múltiplas premissas encadeadas.
        Cada passo usa o resultado do anterior como nova premissa.
        """
        print(f"\n{'='*70}")
        print(f"CADEIA DE RACIOCINIO ({len(premises)} etapas)")
        print(f"{'='*70}")

        chain_results = []
        current_vec = None
        self.trace_log = []

        for i, premise in enumerate(premises):
            print(f"\n── Etapa {i+1}: {premise[:80]}")

            premise_vec = self.engine.encode_text(premise).tensor
            self.trace(f"ETAPA_{i+1}_PREMISSA", premise_vec, premise)

            if current_vec is not None:
                # Quanto do raciocínio anterior está preservado?
                cos_prev = float(torch.cosine_similarity(
                    premise_vec.flatten().unsqueeze(0),
                    current_vec.flatten().unsqueeze(0)))
                print(f"   Continuidade com etapa anterior: {cos_prev:.4f}")

            answer = self.composer.translate_thought(premise_vec)
            answer_vec = self.engine.encode_text(answer).tensor
            self.trace(f"ETAPA_{i+1}_RESPOSTA", answer_vec, answer)

            ingredients = self.composer._extract_ingredients(premise_vec)

            chain_results.append({
                "step": i + 1,
                "premise": premise,
                "answer": answer,
                "ingredients": [{"word": w, "sim": round(s, 4)}
                                for w, _, s in ingredients[:5]],
            })

            current_vec = answer_vec
            print(f"   Resposta: {answer}")

        return chain_results


# ─── Testes ──────────────────────────────────────────────────

def run_all_tests():
    tester = LogicTest()

    results = {}

    # TESTE 1: Relação Analógica (LLMs frequentemente erram)
    r1 = tester.run(
        premise="up is to down as left is to",
        expected="right",
        domain="analogia"
    )
    results["analogia"] = r1

    # TESTE 2: Transitividade
    r2 = tester.run(
        premise="all humans are mortal, Socrates is human, therefore",
        expected="Socrates is mortal",
        domain="silogismo"
    )
    results["silogismo"] = r2

    # TESTE 3: Negação
    r3 = tester.run(
        premise="no bird is a mammal, a bat is not a bird, a bat nurses its young, therefore",
        expected="bat is a mammal",
        domain="negacao"
    )
    results["negacao"] = r3

    # TESTE 4: Problema aberto (o dataset decide)
    r4 = tester.run(
        premise="if force equals mass times acceleration, and mass increases while force stays constant",
        expected="acceleration decreases",
        domain="fisica"
    )
    results["fisica"] = r4

    # TESTE 5: Cadeia de raciocínio
    chain = tester.run_reasoning_chain([
        "every cell contains DNA",
        "DNA is made of nucleotides",
        "nucleotides encode genetic information",
        "this information determines protein structure",
    ], expected="genetic information determines protein structure")
    results["cadeia_dna"] = chain

    return results


def print_report(results):
    """Imprime relatório formatado dos testes."""
    print(f"\n\n{'='*70}")
    print("RELATÓRIO DE TESTES LÓGICOS — RASTREABILIDADE HDC")
    print(f"{'='*70}")

    for name, r in results.items():
        if isinstance(r, dict):
            print(f"\n── {name.upper()} ──")
            print(f"  Premissa: {r['premise'][:80]}")
            print(f"  Esperado: {r['expected'][:60]}")
            print(f"  Resposta: {r['answer']}")
            print(f"  Cosseno P→R: {r['cos_premise_answer']:.4f}")
            print(f"  Ingredientes: {r['n_ingredients']}")
            if r['trace']:
                last = r['trace'][-1]
                print(f"  Traço final: {last['label']} → {last['text'][:60]}")
        elif isinstance(r, list):
            print(f"\n── {name.upper()} (CADEIA) ──")
            for step in r:
                print(f"  Etapa {step['step']}:")
                print(f"    Premissa: {step['premise'][:60]}")
                print(f"    Resposta: {step['answer']}")
                ings = ", ".join([x["word"] for x in step["ingredients"]])
                print(f"    Ingredientes: {ings}")


if __name__ == "__main__":
    results = run_all_tests()
    print_report(results)
