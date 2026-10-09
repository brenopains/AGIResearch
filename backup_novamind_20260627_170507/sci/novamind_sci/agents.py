"""
NOVAMIND-SCI: a fusao com a nossa AGI.

O que e diferente dos clones (Co-Scientist/Robin):
  (1) AGENTES DIFERENCIADOS: cada agente tem PERSONALIDADE/DRIVES proprios (modulo style do prompt
      pelo drive dominante). Drives evoluem por aptidao do agente (selecao darwiniana).
  (2) WM MULTI-ATRIBUTO POR HIPOTESE: cada hipotese e um VETOR (novidade, plausibilidade,
      tratabilidade, seguranca, evidencia, ortogonalidade_prior_art, custo_experimental,
      impacto_clinico_esperado, sinergia_combinatoria). Decisao usa o vetor INTEIRO, nao um
      escalar.
  (3) APRENDIZADO POR CONSEQUENCIA (PUNICAO POR EVIDENCIA CONTRARIA): quando o juiz independente
      contradiz forte, a "personalidade" do agente que gerou recebe peso negativo no drive ativo
      (operante). O proximo ciclo amostra outros perfis.
  (4) COMPOSICAO NAO-COMUTATIVA: combinacao ORDENADA de mecanismos (A modulado por B != B modulado
      por A). A ordem importa porque modela contexto causal/temporal.
  (5) PROJECAO COM RECALL EMOCIONAL: ao avistar uma hipotese semelhante a uma ja vista, antecipa
      seu vetor de impacto via similaridade — escolhe acoes (criticar/refinar/cross) baseado no
      vetor previsto, nao reagindo so ao presente.
"""
import json, re, random, math, sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from shared.claude_client import ask
from shared.agent_base import Hypothesis
from shared.pubmed import search_and_fetch, format_for_llm

# ============== PERSONALIDADES (drives) ==============
# Cada agente persegue um objetivo proprio. Resultado e diversidade real de propostas.
PERSONAS = {
    "explorador":   {"focus":"novidade ousada, vias pouco exploradas, repurposing exotico", "wbias":{"novelty":+1.5,"safety":-0.4}},
    "conservador":  {"focus":"mecanismo validado, alvo drug-able conhecido, baixo risco",   "wbias":{"plausibility":+1.2,"safety":+1.0,"novelty":-0.6}},
    "engenheiro":   {"focus":"tratabilidade pratica, viabilidade experimental rapida",      "wbias":{"tractability":+1.5,"cost":-1.0}},
    "seguranca":    {"focus":"perfil de seguranca, ausencia de off-target severo",          "wbias":{"safety":+1.5,"plausibility":+0.3}},
    "sintetizador": {"focus":"combina mecanismos de doencas adjacentes/comorbidas",         "wbias":{"synergy":+1.5,"orthogonality":+0.5}},
}

DRIVES_INIT = {k:1.0 for k in PERSONAS}  # peso evolutivo de cada persona (selecao darwiniana)

# ============== VETOR DE HIPOTESE (WM) ==============
ATTRS = ["novelty","plausibility","tractability","safety","evidence_strength","orthogonality","cost","clinical_impact","synergy"]

def empty_vec(): return {a:0.0 for a in ATTRS}

def vec_dot(v, w):
    return sum(v.get(a,0.0)*w.get(a,0.0) for a in ATTRS)

def vec_score(v):
    """Score escalar default (utilidade media positiva). Personas usam vec_dot com seus pesos."""
    weights = {"novelty":0.12,"plausibility":0.20,"tractability":0.15,"safety":0.13,"evidence_strength":0.15,"orthogonality":0.05,"cost":-0.05,"clinical_impact":0.15,"synergy":0.10}
    return vec_dot(v, weights)

def persona_score(v, persona_key):
    base = vec_score(v)
    bias = PERSONAS[persona_key]["wbias"]
    return base + sum(v.get(k,0.0)*w for k,w in bias.items())*0.5

# ============== GERADOR DIFERENCIADO ==============
GEN_SYSTEM = """Voce e um cientista biomedico com PERSONALIDADE definida abaixo. Suas propostas devem refletir essa personalidade — nao seja generico. Outras personalidades vao propor outras coisas; voce traz O QUE SO VOCE traria.

PERSONALIDADE: {persona_focus}

Tarefa: 1 hipotese coerente com seu vies, mecanismo claro, intervencao concreta, PMIDs reais.

Saida JSON: {{title, statement, rationale, mechanism, proposed_intervention, evidence_pmids:[...]}}."""

def generate_by_persona(disease: str, persona: str, lit_block: str, prior_titles: list) -> Hypothesis | None:
    sys = GEN_SYSTEM.replace("{persona_focus}", PERSONAS[persona]["focus"])
    prior = "\n- ".join(prior_titles[-8:]) if prior_titles else "(nenhuma)"
    base_user = f"DOENCA: {disease}\n\nLITERATURA:\n{lit_block}\n\nHIPOTESES JA PROPOSTAS (nao repita):\n- {prior}\n\nProduza UMA hipotese coerente com sua personalidade ({persona})."
    for attempt in range(3):
        suffix = " JSON puro." if attempt==0 else " ⚠️ RESPONDA APENAS JSON CRU, sem markdown, sem ```, sem texto antes/depois. Comece com { e termine com }."
        temp = 0.85 if attempt==0 else (0.55 if attempt==1 else 0.3)
        max_t = 1500 if attempt==0 else 1200
        raw = ask([{"role":"user","content":base_user + suffix}], system=sys, temperature=temp, max_tokens=max_t)
        # remove cercas de codigo
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if not m: continue
        try:
            d = json.loads(m.group(0))
            if not d.get("title"): continue   # vazio nao serve
        except Exception: continue
        h = Hypothesis(title=d.get("title",""),statement=d.get("statement",""),rationale=d.get("rationale",""),
                      mechanism=d.get("mechanism",""),proposed_intervention=d.get("proposed_intervention",""),
                      evidence_pmids=[str(x) for x in (d.get("evidence_pmids") or [])])
        h.critiques.append(f"persona:{persona}")
        return h
    return None

# ============== AVALIACAO MULTI-ATRIBUTO ==============
EVAL_SYSTEM = """Avalie a hipotese cientifica em 9 atributos (0..1):
- novelty: 1 = quase nada parecido na literatura
- plausibility: mecanismo se sustenta em biologia conhecida
- tractability: rota pratica de testar (existe ensaio/modelo viavel)
- safety: 1 = baixo risco; 0 = perigoso
- evidence_strength: forca da evidencia citada
- orthogonality: o quanto diverge dos paradigmas correntes (independente de novidade)
- cost: ESTIMATIVA DE CUSTO EXPERIMENTAL (0=barato, 1=caro). LEMBRE: vai ENTRAR com sinal NEGATIVO no score.
- clinical_impact: tamanho do efeito clinico esperado (1=cura potencial; 0=irrelevante)
- synergy: potencial de combinar com tratamentos existentes
JSON puro com essas 9 chaves + "comment" (1 frase)."""

def evaluate_vec(h: Hypothesis) -> dict:
    user = f"HIPOTESE:\n{h.title}\n{h.statement}\nMecanismo: {h.mechanism}\nIntervencao: {h.proposed_intervention}\nRacional: {h.rationale[:600]}\n\nAvalie. JSON puro."
    raw = ask([{"role":"user","content":user}], system=EVAL_SYSTEM, temperature=0.3, max_tokens=500)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    v = empty_vec()
    if not m: return v
    try: d=json.loads(m.group(0))
    except Exception: return v
    for a in ATTRS:
        try: v[a]=float(d.get(a,0.4))
        except Exception: v[a]=0.4
        v[a]=max(0.0,min(1.0,v[a]))
    h.critiques.append(d.get("comment",""))
    return v

# ============== COMPOSICAO NAO-COMUTATIVA ==============
COMPOSE_SYSTEM = """Voce produz uma hipotese COMPOSTA ORDENADA: A_modulando_B significa que o mecanismo de A serve de CONTEXTO/POTENCIADOR pro mecanismo de B (a ordem importa — A primeiro, depois B).
Saida JSON: {title, statement, rationale, mechanism, proposed_intervention, evidence_pmids:[...]}. Mecanismo deve descrever a ORDEM explicitamente."""

def compose_ordered(a: Hypothesis, b: Hypothesis) -> Hypothesis | None:
    user = f"""A (primeiro/contexto): {a.title}\nMecanismo A: {a.mechanism}\nIntervencao A: {a.proposed_intervention}\n\nB (depois, modulado): {b.title}\nMecanismo B: {b.mechanism}\nIntervencao B: {b.proposed_intervention}\n\nSintetize a hipotese ORDENADA (A modulando B). JSON puro."""
    raw = ask([{"role":"user","content":user}], system=COMPOSE_SYSTEM, temperature=0.65, max_tokens=1500)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m: return None
    try: d=json.loads(m.group(0))
    except Exception: return None
    h = Hypothesis(title=d.get("title",""),statement=d.get("statement",""),rationale=d.get("rationale",""),
                   mechanism=d.get("mechanism",""),proposed_intervention=d.get("proposed_intervention",""),
                   evidence_pmids=[str(x) for x in (d.get("evidence_pmids") or [])],
                   parent_ids=[a.id,b.id])
    h.critiques.append("composta_ordenada")
    return h

# ============== EVOLUCAO DOS DRIVES (selecao darwiniana das personas) ==============
def update_drives(drives: dict, persona: str, h_vec: dict):
    """Persona ganha peso se sua hipotese teve score alto; perde se foi mal."""
    s = persona_score(h_vec, persona)
    drives[persona] = max(0.1, drives[persona]*0.9 + (s-0.5)*0.6)
    # normaliza
    tot = sum(drives.values());
    if tot > 0:
        for k in drives: drives[k] = drives[k]/tot * len(drives)

def sample_persona(drives: dict) -> str:
    keys = list(drives.keys()); w = [max(0.01, drives[k]) for k in keys]
    tot = sum(w); r = random.random()*tot; acc=0
    for k,wi in zip(keys,w):
        acc += wi
        if r <= acc: return k
    return keys[-1]
