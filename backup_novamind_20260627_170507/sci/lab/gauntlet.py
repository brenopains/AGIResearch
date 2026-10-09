"""
lab/gauntlet.py — LABORATORIO IN-SILICO: checa o OUTPUT (hipoteses) contra EVIDENCIA REAL e reduz MUITAS->POUCAS.
Cada hipotese gerada: (1) LLM extrai {alvo, droga/classe}; (2) PubMed = ancoragem (ha literatura?);
(3) ClinicalTrials.gov v2 = maturidade clinica (ja em ensaio? fase? terminado=falhou?); (4) score + veredito.
Filtra: ANCORADA + NAO-ESGOTADA + DROGAVEL = as poucas acionaveis. Triagem COMPUTACIONAL, NAO wet-lab.
"""
import os, json, time, re, requests
os.environ.setdefault('LLM_MODEL','qwen3-235b-a22b-instruct-2507')
from dotenv import load_dotenv; load_dotenv(os.path.join(os.path.dirname(__file__),'..','.env'))
import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))
from shared.llm_client import _openai_compat_post, VENICE_URL, VENICE_KEY
from shared.pubmed import search_pmids
M="qwen3-235b-a22b-instruct-2507"
def llm(sysp,usr,mx=400):
    for _ in range(3):
        try: return _openai_compat_post(VENICE_URL,VENICE_KEY,M,[{"role":"system","content":sysp},{"role":"user","content":usr}],0.4,mx,is_venice=True)
        except: time.sleep(2)
    return ""
def ct_trials(cond, intr):
    try:
        r=requests.get('https://clinicaltrials.gov/api/v2/studies', params={'query.cond':cond,'query.intr':intr,'countTotal':'true','pageSize':20,'fields':'NCTId,Phase,OverallStatus'}, timeout=20)
        d=r.json(); total=d.get('totalCount',0); studies=d.get('studies',[])
        phases=set(); statuses=[]
        for s in studies:
            ps=s.get('protocolSection',{}); ph=ps.get('designModule',{}).get('phases',[]); st=ps.get('statusModule',{}).get('overallStatus','')
            phases.update(ph); statuses.append(st)
        return total, sorted(phases), statuses
    except Exception as e: return 0,[],[]
def pubmed_count(q):
    try: return len(search_pmids(q, retmax=50))
    except: return 0
def phase_rank(phases):
    order={'EARLY_PHASE1':1,'PHASE1':2,'PHASE2':3,'PHASE3':4,'PHASE4':5}
    return max([order.get(p,0) for p in phases], default=0)

def gauntlet(disease, hypotheses):
    print(f"\n{'='*66}\nLABORATORIO IN-SILICO — {disease} | {len(hypotheses)} hipoteses -> filtrar p/ poucas\n{'='*66}")
    scored=[]
    for i,h in enumerate(hypotheses):
        ext=llm("Extraia de uma hipotese terapeutica. Responda SO JSON: {\"alvo\":\"gene/proteina\",\"droga\":\"droga ou classe\"}",
                f"Hipotese:\n{h}\n\nJSON:")
        try: m=json.loads(re.search(r'\{.*\}', ext, re.S).group(0))
        except: m={"alvo":"?","droga":"?"}
        alvo=m.get('alvo','?')[:40]; droga=m.get('droga','?')[:40]
        pm = pubmed_count(f"{alvo} {disease} therapy")            # ancoragem
        tot_t, phases, stats = ct_trials(disease, f"{alvo} OR {droga}")  # maturidade clinica
        pr=phase_rank(phases); n_term=sum(1 for s in stats if s in ('TERMINATED','WITHDRAWN','SUSPENDED'))
        n_active=sum(1 for s in stats if s in ('RECRUITING','ACTIVE_NOT_RECRUITING','NOT_YET_RECRUITING','ENROLLING_BY_INVITATION'))
        # SCORE: ancoragem(log) + drogabilidade(ja existe droga) - esgotamento(falhou em fase alta) + headroom(pouco/nao testado)
        import math
        grounding=min(3, math.log1p(pm))
        druggable=1.0 if (tot_t>0 or droga not in ('?','')) else 0.3
        exhausted = 1.0 if (pr>=4 and n_active==0) else 0.0   # ja em fase 3+ e parado = direcao esgotada/falha
        headroom = 1.0 if tot_t==0 else (0.7 if n_active>0 else 0.4)
        score = grounding*1.0 + druggable*1.0 + headroom*1.2 - exhausted*1.5
        if pm==0: verdict="SEM ancoragem (descartar?)"
        elif exhausted: verdict="direcao ESGOTADA/falhou (rever ou combinar)"
        elif tot_t==0 and pm>0: verdict="NOVA + ancorada (alta prioridade)"
        elif n_active>0: verdict="direcao VALIDADA e ativa (forte, mas concorrida)"
        else: verdict="ancorada, testes parados"
        scored.append({"i":i,"alvo":alvo,"droga":droga,"pubmed":pm,"trials":tot_t,"fase_max":pr,
                       "term":n_term,"ativos":n_active,"score":round(score,2),"verdict":verdict,"hyp":h[:90]})
        print(f"[{i+1}] alvo={alvo} | droga={droga}\n    PubMed={pm} | trials={tot_t} fase_max={pr} ativos={n_active} term={n_term} | SCORE={score:.2f} -> {verdict}")
    scored.sort(key=lambda x:-x['score'])
    print(f"\n{'='*66}\n>>> AS POUCAS (top por evidencia):")
    for s in scored[:3]:
        print(f"  #{s['i']+1} [{s['score']}] {s['alvo']} ({s['verdict']}) — PubMed {s['pubmed']}, {s['trials']} trials")
    json.dump(scored, open(os.path.join(os.path.dirname(__file__),'gauntlet_result.json'),'w'), ensure_ascii=False, indent=1)
    print(f"\n[salvo lab/gauntlet_result.json] — {len(hypotheses)} -> {min(3,len(scored))} priorizadas")
    return scored

if __name__=="__main__":
    disease="amyotrophic lateral sclerosis"
    # gera VARIAS hipoteses diversas (o "muitas") p/ o gauntlet filtrar
    sysp="Voce gera UMA hipotese terapeutica concisa (alvo + droga/classe + mecanismo, 2 frases) para a doenca. Seja diverso e especifico."
    angles=["estresse oxidativo mitocondrial","neuroinflamacao microglial","excitotoxicidade glutamatergica",
            "agregacao de TDP-43 / splicing","autofagia/proteostase","disfuncao de STMN2 / regeneracao axonal"]
    hyps=[]
    for a in angles:
        h=llm(sysp, f"Doenca: {disease}\nAngulo: {a}\nHipotese:", 300).strip()
        if h: hyps.append(h); print(f"[gerada/{a}] {h[:80]}...")
    gauntlet(disease, hyps)
