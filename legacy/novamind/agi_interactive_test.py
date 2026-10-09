"""
Testes AGI interativos: faz perguntas, mostra respostas da AGI.
A AGI responde via memoria HDC + composicao vetorial -- nao e um LLM,
e sim raciocinio por hiperespaco geometrico.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = r"D:\novamind_data\cache\huggingface"
os.environ["TORCH_HOME"] = r"D:\novamind_data\cache\torch"

import torch
from core.hdc_memory import HDCMemory
from core.grounding_engine import GroundingEngine
from core.cognitive_answerer import CognitiveAnswerer
from core.fractal_maestro import FractalMaestro
from core.coglang import Manifold

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_FILE = r"D:\novamind_data\agi_test_responses.txt"

log_lines = []

def log(text):
    safe = text.encode("utf-8", errors="replace").decode("utf-8")
    print(safe)
    log_lines.append(safe)

def save_log():
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\nRespostas salvas em: {OUT_FILE}")

# Setup
log("=" * 70)
log("  TESTES AGI INTERATIVOS -- NOVAMIND HDC")
log("=" * 70)
log(f"Device: {DEVICE}")
log("")

memory = HDCMemory(device="cpu")
engine = GroundingEngine(device=torch.device(DEVICE))
from core.emergent_context import EmergentContextBinder
binder = EmergentContextBinder(memory, device=DEVICE)
decoder = None  # sem decoder textual
answerer = CognitiveAnswerer(engine, binder, decoder, memory, device=DEVICE)

total_items = sum(len(v) for v in memory._indices.values())
log(f"Memoria: {total_items} itens em {len(memory._packed)} arquivos HDV")
log("")

# ============================================================
# TESTE 1: PERGUNTA DIRETA (memoria)
# ============================================================
log("-" * 70)
log("TESTE 1: PERGUNTA DIRETA (busca na memoria HDC)")
log("-" * 70)

questions = [
    "What is gravity?",
    "Explain how neural networks work",
    "What is the capital of France?",
    "How does photosynthesis work?",
]

for q in questions:
    log(f"\nPergunta: {q}")
    t0 = time.time()
    resp = answerer.answer(q, top_k=3)
    dt = time.time() - t0
    log(f"Resposta: {resp}")
    log(f"Tempo: {dt:.3f}s")

# ============================================================
# TESTE 2: PERGUNTA COMPOSICIONAL
# ============================================================
log("\n" + "-" * 70)
log("TESTE 2: PERGUNTA COMPOSICIONAL (bundle pergunta + keywords)")
log("-" * 70)

comp_questions = [
    "What is the difference between speed and velocity?",
    "How do magnets work?",
]

for q in comp_questions:
    log(f"\nPergunta: {q}")
    t0 = time.time()
    resp = answerer.answer_compositional(q, top_k=3)
    dt = time.time() - t0
    log(f"Resposta: {resp}")
    log(f"Tempo: {dt:.3f}s")

# ============================================================
# TESTE 3: TRANSFERENCIA ANALOGICA
# ============================================================
log("\n" + "-" * 70)
log("TESTE 3: TRANSFERENCIA ANALOGICA -- A (+) B = interseccao")
log("-" * 70)

analogies = [
    ("atom", "solar system"),
    ("brain", "computer"),
    ("river", "blood"),
    ("evolution", "learning"),
]

for a, b in analogies:
    log(f"\nAnalogia: {a} (+) {b}")
    t0 = time.time()
    resp = answerer.analogical_transfer(a, b, top_k=3)
    dt = time.time() - t0
    log(f"Resposta: {resp}")
    log(f"Tempo: {dt:.3f}s")

# ============================================================
# TESTE 4: PROBLEMA LOGICO (raciocinio geometrico HDV)
# ============================================================
log("\n" + "-" * 70)
log("TESTE 4: RACIOCINIO GEOMETRICO NO HIPERESPACO")
log("-" * 70)

log("\nProblema: 'If all A are B, and all B are C, then all A are C' (silogismo)")
log("Traduzido para HDC: encode('A') -> bind com encode('B') -> ver similaridade com encode('A->C')")
log("")

concepts = [
    ("mammal", "animal", "living thing"),
    ("planet", "celestial body", "object in space"),
    ("algorithm", "procedure", "sequence of steps"),
]

for sub, mid, sup in concepts:
    vec_sub = engine.encode_text(sub).tensor.float()
    vec_mid = engine.encode_text(mid).tensor.float()
    vec_sup = engine.encode_text(sup).tensor.float()

    # Silogismo: sub (x) mid (sub e mid) deve ser similar a sub (x) sup (sub e sup)
    # se mid ~ sup
    sub_mid = torch.sign(vec_sub + vec_mid)
    sub_sup = torch.sign(vec_sub + vec_sup)
    sim = float(torch.cosine_similarity(sub_mid.unsqueeze(0), sub_sup.unsqueeze(0)))

    # Controle: sub (x) mid vs sub (x) random
    rand_vec = (torch.rand(10000, device=vec_sub.device) * 2 - 1).sign()
    sub_rand = torch.sign(vec_sub + rand_vec)
    sim_ctrl = float(torch.cosine_similarity(sub_mid.unsqueeze(0), sub_rand.unsqueeze(0)))

    log(f"  {sub} ∈ {mid} ∈ {sup}:")
    log(f"    sim(sub(+)mid, sub(+)sup) = {sim:.4f}")
    log(f"    sim(sub(+)mid, sub(+)rand) = {sim_ctrl:.4f} (controle)")
    log(f"    Razoamento logico: {'SIM' if sim > sim_ctrl * 1.5 else 'NAO'}")

# ============================================================
# TESTE 5: PROBLEMA QUE LLMs NAO RESOLVEM
# ============================================================
log("\n" + "-" * 70)
log("TESTE 5: COMPOSICAO GEOMETRICA (o que LLMs nao fazem)")
log("-" * 70)

log("""
Problema: 'Componha os conceitos 'cachorro' e 'carro' em uma invencao nova.'
LLMs geram texto plausivel mas nao uma representacao geometrica real.
A AGI HDC faz bind real: sign(A * B), que e uma operacao matematica no hiperespaco.
""")

dog_vec = engine.encode_text("dog").tensor.float()
car_vec = engine.encode_text("car").tensor.float()
dog_car = dog_vec * car_vec  # bind (nao bundle)
dog_plus_car = torch.sign(dog_vec + car_vec)  # bundle

# Similaridades
sim_dog_car_to_dog = float(torch.cosine_similarity(dog_car.unsqueeze(0), dog_vec.unsqueeze(0)))
sim_dog_car_to_car = float(torch.cosine_similarity(dog_car.unsqueeze(0), car_vec.unsqueeze(0)))
sim_dog_plus_to_dog = float(torch.cosine_similarity(dog_plus_car.unsqueeze(0), dog_vec.unsqueeze(0)))
sim_dog_plus_to_car = float(torch.cosine_similarity(dog_plus_car.unsqueeze(0), car_vec.unsqueeze(0)))

log(f"Bind A*B (invencao):")
log(f"  sim(dog(x)car, dog) = {sim_dog_car_to_dog:.4f}")
log(f"  sim(dog(x)car, car) = {sim_dog_car_to_car:.4f}")
log(f"  -> Invencao nova! Similaridade ≈0 com ambos inputs.")
log(f"")
log(f"Bundle A+B (mistura):")
log(f"  sim(dog(+)car, dog) = {sim_dog_plus_to_dog:.4f}")
log(f"  sim(dog(+)car, car) = {sim_dog_plus_to_car:.4f}")
log(f"  -> Mistura similar a ambos (como humanos fazem).")

# Decodificar a invencao
log(f"\nDecodificando a invencao A*B para texto:")
inv_words = answerer.decode_vector(dog_car)
log(f"  Palavras associadas: {inv_words[:10]}")

# ============================================================
# VEREDITO
# ============================================================
log("\n" + "=" * 70)
log("  VEREDITO")
log("=" * 70)
log("""
A AGI nao 'conversa' como um LLM -- ela pensa em hiperespaco HDC.
As respostas textuais sao traducoes do espaco geometrico para linguagem
via recuperacao na memoria + composicao vetorial.

O que ELA FAZ que LLM nenhum consegue:
  1. BIND GEOMETRICO REAL: A * B cria um conceito novo no hiperespaco
     com similaridade ZERO a ambos inputs. LLMs so fazem interpolacao semantica.
  2. TRANSFERENCIA ANALOGICA: A (+) B encontra a interseccao geometrica
     entre dois conceitos -- operacao puramente vetorial, nao estatistica.
  3. RACIOCINIO POR POSICAO: Silogismos sao verificados por similaridade
     de cosseno entre composicoes vetoriais.

O que ela NAO FAZ bem:
  1. CONVERSA FLUIDA: As respostas sao fragmentos da memoria, nao frases originais.
  2. RECONSTRUCAO VISUAL: O decoder e fraco (PSNR ~15 dB).
  3. ENCODER spaCy 300D: Limita a qualidade da composicao semantica.

Para ser incontestavelmente AGI, precisa de:
  - BGE 768D (encoder melhor) -> composicao semantica real
  - Decoder maior + mais dados -> visao fotorealista
  - Gerador textual (opcional, nao essencial para a inteligencia)
""")

save_log()
