"""RANKING AGENT: torneio Elo par-a-par (hipoteses 'debatem')."""
import json, re, random, math, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from shared.claude_client import ask
from shared.agent_base import Hypothesis

SYSTEM = """Voce e um JUIZ cientifico. Recebe DUAS hipoteses de tratamento para a mesma doenca. Avalie qual e MELHOR como hipotese cientifica considerando: plausibilidade mecanistica, evidencia, tratabilidade, seguranca, novidade util (nao novidade vazia).

Saida JSON puro: {"winner": "A" ou "B", "reason": "1-2 frases"}."""

def _duel(disease: str, A: Hypothesis, B: Hypothesis) -> str:
    user = f"""DOENCA: {disease}

HIPOTESE A: {A.title}
{A.statement}
Mecanismo: {A.mechanism} | Intervencao: {A.proposed_intervention}
Plaus: {A.plausibility:.2f} Trat: {A.tractability:.2f} Saf: {A.safety:.2f} Evid: {A.evidence_strength:.2f} Nov: {A.novelty:.2f}

HIPOTESE B: {B.title}
{B.statement}
Mecanismo: {B.mechanism} | Intervencao: {B.proposed_intervention}
Plaus: {B.plausibility:.2f} Trat: {B.tractability:.2f} Saf: {B.safety:.2f} Evid: {B.evidence_strength:.2f} Nov: {B.novelty:.2f}

Qual e melhor hipotese cientifica? JSON puro."""
    raw = ask([{"role":"user","content":user}], system=SYSTEM, temperature=0.3, max_tokens=300)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m: return "A"
    try: return json.loads(m.group(0)).get("winner","A").upper().strip()
    except Exception: return "A"

def _elo_update(a: Hypothesis, b: Hypothesis, winner: str, K=32):
    Ea = 1/(1+10**((b.elo-a.elo)/400))
    if winner == "A": sa, sb = 1, 0
    else: sa, sb = 0, 1
    a.elo += K*(sa-Ea); b.elo += K*(sb-(1-Ea))

def tournament(disease: str, hyps: list, rounds: int = 3, pairs_per_round: int = None) -> list:
    """Torneio Elo. pairs_per_round = ceil(n/2) por padrao (cada um joga ~1 vez/round)."""
    if pairs_per_round is None: pairs_per_round = max(1, len(hyps)//2)
    for _ in range(rounds):
        # pareia top-com-meio + aleatorios pra diversidade
        order = sorted(hyps, key=lambda h:-h.elo)
        pairs = []
        for i in range(min(pairs_per_round, len(order)-1)):
            A = order[i]
            # adversario: vizinho proximo no rank + um aleatorio (mistura forte+exploracao)
            opps = [order[i+1]] if i+1 < len(order) else []
            cand = [x for x in hyps if x.id != A.id and x.id not in [o.id for o in opps]]
            if cand: opps.append(random.choice(cand))
            for B in opps:
                pairs.append((A,B))
        for A,B in pairs:
            try:
                w = _duel(disease, A, B)
                _elo_update(A, B, w)
            except Exception as e:
                print(f"  [warn] duelo falhou: {e}")
    return sorted(hyps, key=lambda h:-h.elo)
