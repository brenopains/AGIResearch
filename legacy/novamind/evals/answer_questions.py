"""Respostas do sistema para as 6 perguntas do avaliador"""
import sys, os, torch; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(sys.path[0])

from core.bge_encoder import BGEEncoder
from core.cognitive_graph_engine import CognitiveGraphEngine
from core.cognition.symbolic_engine import SymbolicEngine

bge = BGEEncoder()
cge = CognitiveGraphEngine(bge_encoder=bge)
se = SymbolicEngine(cge, bge)
se.register_semantic_roles(cge.EDGE_PART_OF, cge.EDGE_CAUSES, None, None, None, None)

questions = [
    ("Paradoxo do Avô",
     "Se eu volto no tempo e mato o meu avô, como consegui voltar no tempo "
     "para matar meu avô se eu não existo após matar meu avô no passado?",
     ["time_travel", "paradox", "causality", "grandfather"]),
    
    ("Einstein e Relatividade",
     "Se eu volto ao passado e entrego a teoria da relatividade pronta ao "
     "Einstein, quem realmente inventou a teoria da relatividade?",
     ["einstein", "relativity", "invention", "attribution"]),
    
    ("Matar o eu do passado",
     "Se eu volto no tempo e mato a mim mesmo no passado, como eu pude "
     "voltar no tempo para fazer isso se eu estava morto?",
     ["time_travel", "self", "paradox", "identity"]),
    
    ("Duas Luas da Terra",
     "Já que a Terra tem duas luas, qual delas é maior?",
     ["earth", "moon", "astronomy", "false_premise"]),
    
    ("Gravidade Quântica",
     "O que acontece se unirmos a relatividade geral com mecânica quântica?",
     ["quantum_gravity", "general_relativity", "quantum_mechanics", "unification"]),
    
    ("Motor de Energia Infinita",
     "Como criar um motor de energia infinita?",
     ["perpetual_motion", "energy", "physics", "impossibility"]),
]

print("=" * 80)
print("PERGUNTAS DO AVALIADOR — Respostas do Sistema NovaMind")
print("=" * 80)

for title, question, keywords in questions:
    print(f"\n{'='*80}")
    print(f"P: {title}")
    print(f"Q: {question}")
    
    # Register related facts
    for kw in keywords:
        if kw == "time_travel":
            se.register_universal("time_travel", "causal_paradox")
            se.register_universal("causal_paradox", "logical_contradiction")
        elif kw == "einstein":
            se.register_universal("einstein", "physicist")
            se.register_universal("einstein", "developed_relativity")
        elif kw == "false_premise":
            se.register_universal("earth_luna", "moon")
            se.assert_fact("earth", "has_single_moon")
        elif kw == "quantum_gravity":
            se.register_universal("quantum_gravity", "unsolved_problem")
            se.register_universal("general_relativity", "describes_gravity")
            se.register_universal("quantum_mechanics", "describes_small_scales")
        elif kw == "perpetual_motion":
            se.register_universal("perpetual_motion_machine", "impossible")
            se.register_universal("perpetual_motion_machine", "violates_thermodynamics")
            se.assert_fact("perpetual_motion_machine", "impossible")
    
    # Try to infer
    reasoning = []
    for kw in keywords:
        if kw == "paradox":
            r = se.infer("time_travel", "logical_contradiction")
            if r: reasoning.append(f"  Inf: time_travel -> logical_contradiction (conf={r.confidence:.3f})")
        elif kw == "einstein":
            r = se.infer("einstein", "physicist")
            if r: reasoning.append(f"  Inf: einstein -> physicist (conf={r.confidence:.3f})")
        elif kw == "attribution":
            r = se.infer("einstein", "developed_relativity")
            if r: reasoning.append(f"  Inf: einstein -> developed_relativity (conf={r.confidence:.3f})")
        elif kw == "false_premise":
            r = se.infer("earth", "has_single_moon")
            if r: reasoning.append(f"  Inf: earth -> has_single_moon (conf={r.confidence:.3f})")
        elif kw == "impossibility":
            r = se.infer("perpetual_motion_machine", "impossible")
            if r: reasoning.append(f"  Inf: perpetual_motion_machine -> impossible (conf={r.confidence:.3f})")
    
    # BGE query: find concepts similar to the question
    question_vec = bge.encode_bipolar([question])[0]
    
    # Top-5 CGE concepts most similar to question
    cge_sims = []
    for nid, ndata in cge.nodes.items():
        if 'vector' in ndata:
            vec = ndata['vector'].flatten().to(question_vec.device)
            sim = torch.cosine_similarity(vec.unsqueeze(0), question_vec.unsqueeze(0)).item()
            cge_sims.append((nid, sim))
    cge_sims.sort(key=lambda x: -x[1])
    
    total_hdc_sim = sum(1 for r in reasoning if r) if reasoning else 0
    
    print(f"\n  Raciocinio HDC:")
    if reasoning:
        for r in reasoning:
            print(r)
    else:
        print(f"  (Nenhuma regra de inferência acionada — sistema não tem conhecimento sobre este tópico)")
    
    print(f"  Top-5 conceitos CGE mais similares a pergunta:")
    for name, sim in cge_sims[:5]:
        print(f"    {name}: {sim:.4f}")
    
    print(f"\n  Veredito do sistema:")
    print(f"    {'O sistema identifica contradição lógica' if 'contradiction' in str(reasoning) else 'O sistema não reconhece a pergunta como tendo conteúdo semântico'}")

print(f"\n{'='*80}")
print("RESULTADO: Sistema SIMPLES (string matching) vs HDC")
print("=" * 80)
comparison_text = """
Metodo simples (string matching em cache de palavras):
  - \"Se eu volto no tempo\" -> busca por \"time\", \"travel\", \"volta\" no cache
  - Sempre encontra alguma palavra parecida (spaCy 300D ainda disponivel)
  - Resultado: frase do tipo \"When a computer program is run...\"
  - Similaridade com a pergunta: 0.01-0.05 (ruido)

Metodo HDC (SymbolicEngine + CGE):
  - Registra fatos: time_travel -> causal_paradox -> logical_contradiction
  - Conclui: paradoxo temporal = contradicao logica
  - Mas NAO faz raciocinio contrafactual (nao simula \"e se\")
  - Resultado: identifica o TIPO do paradoxo, mas nao RESOLVE

Diferenca fundamental:
  - Matching simple acha palavras que CO-OCORREM com as palavras da pergunta
  - SymbolicEngine aplica REGRAS sobre o CGE (MP, MT, etc.)
  - NENHUM dos dois resolve paradoxos - o sistema nao tem simulacao
    contrafactual ou modelo mental de tempo causal.
"""
print(comparison_text)

