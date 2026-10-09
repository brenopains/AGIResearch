"""
agent.py — BRACOS AGENTICOS GERAIS (a AGI ganha "maos" no universo digital).
Filosofia do usuario: nao dar MCPs especificos engessados, mas dar FERRAMENTAS gerais + ENSINAR O PROPOSITO de
cada uma (como humanos sabem instintivamente pra que serve abrir/fechar a mao). A AGI entao DECIDE sozinha qual
ferramenta usar pra atingir uma meta — loop ReAct (raciocina -> age -> observa -> repete). O LLM e o raciocinador;
as ferramentas sao os bracos; nossa arquitetura e o loop que escolhe e encadeia.
"""
import os, json, re, time, subprocess, requests, sys
try: sys.stdout.reconfigure(encoding='utf-8'); sys.stderr.reconfigure(encoding='utf-8')
except: pass
os.environ.setdefault('LLM_MODEL','qwen3-235b-a22b-instruct-2507')
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE,'..'))
from dotenv import load_dotenv; load_dotenv(os.path.join(HERE,'..','.env'))
from shared.llm_client import _openai_compat_post, VENICE_URL, VENICE_KEY
from shared.pubmed import search_pmids, search_and_fetch, format_for_llm
M="qwen3-235b-a22b-instruct-2507"
def llm(sysp,usr,mx=700):
    for _ in range(3):
        try: return _openai_compat_post(VENICE_URL,VENICE_KEY,M,[{"role":"system","content":sysp},{"role":"user","content":usr}],0.3,mx,is_venice=True)
        except: time.sleep(2)
    return ""

# ===== FERRAMENTAS (cada uma com PROPOSITO ensinado — a "afordancia") =====
def t_pubmed(q):  # PROPOSITO: buscar evidencia cientifica real (ancoragem)
    pmids=search_pmids(q,retmax=8); return f"{len(pmids)} artigos PubMed p/ '{q}'. PMIDs: {pmids[:8]}"
def t_pubmed_read(q):  # PROPOSITO: LER resumos da literatura p/ extrair conhecimento
    return format_for_llm(search_and_fetch(q,retmax=4))[:2200]
def t_trials(args):  # PROPOSITO: checar maturidade clinica (ja em ensaio? fase? falhou?)
    p=args.split('|'); cond=p[0].strip(); intr=p[1].strip() if len(p)>1 else ''
    try:
        r=requests.get('https://clinicaltrials.gov/api/v2/studies',params={'query.cond':cond,'query.intr':intr,'countTotal':'true','pageSize':10,'fields':'BriefTitle,Phase,OverallStatus'},timeout=20).json()
        tot=r.get('totalCount',0);ex=[]
        for s in r.get('studies',[])[:5]:
            ps=s.get('protocolSection',{});ex.append(f"{ps.get('statusModule',{}).get('overallStatus','')}/{','.join(ps.get('designModule',{}).get('phases',[]))}")
        return f"{tot} ensaios p/ {cond}+{intr}. amostra: {ex}"
    except Exception as e: return f"erro trials: {e}"
def t_calc(expr):  # PROPOSITO: calcular/raciocinar numericamente
    try: return f"= {eval(expr,{'__builtins__':{}},{ 'min':min,'max':max,'sum':sum,'len':len,'abs':abs,'round':round})}"
    except Exception as e: return f"erro calc: {e}"
def t_write(args):  # PROPOSITO: salvar um resultado/artefato em disco (memoria persistente)
    p=args.split('|',1); fn=p[0].strip(); txt=p[1] if len(p)>1 else ''
    path=os.path.join(HERE,'agent_out',re.sub(r'[^a-z0-9_.-]','_',fn.lower()))
    os.makedirs(os.path.dirname(path),exist_ok=True); open(path,'w',encoding='utf-8').write(txt)
    return f"salvo em agent_out/{os.path.basename(path)} ({len(txt)} chars)"
def t_lab(disease):  # PROPOSITO: rodar o laboratorio in-silico completo (gerar hipoteses + verificar) p/ uma doenca
    try:
        import lab_server as L; L.pipeline(disease); r=L.JOB.get('result')
        if r: return "TOP: "+ "; ".join(f"{h['alvo']}({h['label']},score{h['score']})" for h in r['hypotheses'][:4])
    except Exception as e: return f"erro lab: {e}"
    return "lab sem resultado"

TOOLS={
 "pubmed":      {"fn":t_pubmed,      "purpose":"buscar EVIDENCIA cientifica (quantos artigos existem). arg: termo de busca"},
 "pubmed_read": {"fn":t_pubmed_read, "purpose":"LER resumos da literatura p/ extrair conhecimento. arg: termo de busca"},
 "trials":      {"fn":t_trials,      "purpose":"checar ensaios clinicos (maturidade/falha). arg: 'doenca | intervencao'"},
 "calc":        {"fn":t_calc,        "purpose":"calcular numericamente. arg: expressao python"},
 "lab":         {"fn":t_lab,         "purpose":"rodar o LABORATORIO in-silico (gera hipoteses+verifica) p/ uma doenca. arg: nome da doenca"},
 "write":       {"fn":t_write,       "purpose":"SALVAR um artefato/resultado em disco. arg: 'nome_arquivo | conteudo'"},
}
def tools_doc(): return "\n".join(f"- {k}: {v['purpose']}" for k,v in TOOLS.items())

def agent(goal, max_steps=8, emit=print):
    emit(f"\n🤖 META: {goal}\n{'='*60}")
    history=[]
    sysp=("Voce e um agente autonomo com BRACOS (ferramentas). A cada passo, RACIOCINE e escolha UMA acao.\n"
      "FERRAMENTAS disponiveis (com proposito):\n"+tools_doc()+
      "\nResponda SO JSON: {\"pensamento\":\"...\",\"ferramenta\":\"nome\",\"arg\":\"...\"} para agir, "
      "ou {\"pensamento\":\"...\",\"final\":\"resposta final\"} quando tiver a resposta.")
    for step in range(max_steps):
        ctx=f"META: {goal}\n\nHISTORICO:\n"+("\n".join(history) if history else "(inicio)")+"\n\nProxima acao (JSON):"
        raw=llm(sysp, ctx, 600)
        try: act=json.loads(re.search(r'\{.*\}',raw,re.S).group(0))
        except: emit("  (resposta nao-JSON, encerrando)"); break
        if act.get('final'):
            emit(f"\n✅ RESPOSTA FINAL:\n{act['final']}"); return act['final']
        tool=act.get('ferramenta'); arg=str(act.get('arg','')); th=act.get('pensamento','')[:120]
        emit(f"\n[passo {step+1}] 💭 {th}\n   🔧 {tool}({arg[:70]})")
        if tool not in TOOLS: emit("   ⚠ ferramenta inexistente"); history.append(f"passo{step+1}: ferramenta '{tool}' nao existe"); continue
        obs=str(TOOLS[tool]['fn'](arg))[:600]
        emit(f"   👁 {obs[:200]}")
        history.append(f"passo{step+1}: {tool}({arg[:50]}) -> {obs[:300]}")
    emit("\n(limite de passos atingido)"); return None

if __name__=="__main__":
    goal=" ".join(sys.argv[1:]) or "Encontre um alvo terapeutico promissor e POUCO explorado para a doenca de Huntington, verifique a evidencia (PubMed) e a maturidade clinica (ensaios), e salve um resumo."
    agent(goal)
