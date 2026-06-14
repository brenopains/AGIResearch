"""
cross_disease.py — ENGINE CROSS-DOENCA (busca de NOS-MECANISMO compartilhados / alvos pan-doenca).
Filosofia honesta: NAO existe "uma cura pra tudo". Mas mecanismos CONVERGEM entre doencas
(proteostase, neuroinflamacao, disfuncao mitocondrial, autofagia...). Esta engine:
  1. gera hipoteses (alvo+mecanismo+sistema) p/ um PAINEL de doencas (reuso do gerador por angulos);
  2. agrupa os mecanismos recorrentes ENTRE doencas (clustering via LLM);
  3. p/ cada cluster: conta em quantas doencas aparece (cobertura) + lista alvos candidatos;
  4. verifica os alvos pan-doenca top na literatura real (PubMed) p/ ancorar;
  5. retorna nos-mecanismo ranqueados + alvos pan-doenca, com framing HONESTO (no a montante, nao cura universal).
Stdlib + shared.* apenas. Testavel sozinho: `python cross_disease.py`
"""
import os, json, re, time, math, sys
try: sys.stdout.reconfigure(encoding='utf-8'); sys.stderr.reconfigure(encoding='utf-8')
except: pass
os.environ.setdefault('LLM_MODEL','qwen3-235b-a22b-instruct-2507')
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE,'..'))
from dotenv import load_dotenv; load_dotenv(os.path.join(HERE,'..','.env'))
from shared.llm_client import _openai_compat_post, VENICE_URL, VENICE_KEY
from shared.pubmed import search_pmids
M="qwen3-235b-a22b-instruct-2507"

# paineis pre-definidos: cada doenca = (nome PT p/ gerar hipotese, nome EN p/ buscar no PubMed)
PANELS={
 "neurodegenerativas":[("esclerose lateral amiotrofica","amyotrophic lateral sclerosis"),("doenca de Alzheimer","Alzheimer disease"),("doenca de Parkinson","Parkinson disease"),("doenca de Huntington","Huntington disease")],
 "autoimunes":[("artrite reumatoide","rheumatoid arthritis"),("lupus eritematoso sistemico","systemic lupus erythematosus"),("esclerose multipla","multiple sclerosis"),("doenca de Crohn","Crohn disease")],
 "metabolicas":[("diabetes tipo 2","type 2 diabetes"),("esteatose hepatica (NAFLD)","non-alcoholic fatty liver disease"),("aterosclerose","atherosclerosis"),("sindrome metabolica","metabolic syndrome")],
 "cancer":[("cancer de pancreas","pancreatic cancer"),("glioblastoma","glioblastoma"),("cancer de mama triplo-negativo","triple negative breast cancer"),("cancer de pulmao","lung cancer")],
}
def _norm_panel(items):
    """aceita lista de tuplas (pt,en) OU lista de strings livres -> normaliza p/ dicts {pt,en}."""
    out=[]
    for it in items:
        if isinstance(it,(list,tuple)): out.append({"pt":it[0],"en":it[1] if len(it)>1 else it[0]})
        else: out.append({"pt":it,"en":it})
    return out

def llm(sysp,usr,mx=420,temp=0.4):
    for _ in range(3):
        try: return _openai_compat_post(VENICE_URL,VENICE_KEY,M,[{"role":"system","content":sysp},{"role":"user","content":usr}],temp,mx,is_venice=True)
        except: time.sleep(2)
    return ""
def pm_count(q):
    try: return len(search_pmids(q,retmax=40))
    except: return 0
def pm_count_co(target,diseases):
    """em quantas doencas do painel ha literatura ligando ESTE alvo (co-ocorrencia real, busca em INGLES)."""
    hits=[]
    for d in diseases:
        n=pm_count(f"{target} AND {d['en']}")
        if n>0: hits.append((d['pt'],n))
    return hits

ANGLES=["proteostase / agregacao proteica","neuroinflamacao / sistema imune","disfuncao mitocondrial / estresse oxidativo",
        "via de sinalizacao principal","autofagia / clearance celular","reposicionamento de farmaco existente"]

def gen_targets(disease, emit=print):
    """gera alvos candidatos (alvo+mecanismo+sistema) p/ uma doenca. disease={pt,en}."""
    emit(f"  · gerando hipoteses p/ {disease['pt']}...")
    out=[]
    for a in ANGLES:
        r=llm("Voce e farmacologista. Responda SO JSON, conciso.",
            f"Doenca: {disease['pt']}\nAngulo mecanistico: {a}\n"
            'JSON: {"alvo":"simbolo de GENE ou proteina especifica (ex: SNCA, LRRK2, NLRP3)","mecanismo":"2-5 palavras","sistema":"sistema corporal primario"}', 200)
        try:
            m=json.loads(re.search(r'\{.*\}',r,re.S).group(0))
            out.append({"alvo":str(m.get('alvo','?'))[:48],"mecanismo":str(m.get('mecanismo','?'))[:60],
                        "sistema":str(m.get('sistema','?'))[:24],"angulo":a,"disease":disease['pt']})
        except: pass
    return out

def extract_targets(node, diseases):
    """fallback: se o no veio sem alvos concretos, pede 1-3 simbolos de gene/proteina ACIONAVEIS."""
    dz=', '.join(d['pt'] for d in diseases)
    r=llm("Voce e farmacologista molecular. Responda SO JSON.",
        f"Mecanismo compartilhado: {node.get('nome','')} — {node.get('mecanismo','')}\n"
        f"Doencas: {dz}\nDe 1-3 ALVOS moleculares acionaveis (simbolo de gene HGNC ou proteina/complexo especifico) representativos deste mecanismo.\n"
        'JSON: {"alvos":["SIMBOLO1","SIMBOLO2"]}', 160)
    try: return [str(x)[:40] for x in json.loads(re.search(r'\{.*\}',r,re.S).group(0)).get('alvos',[])][:3]
    except: return []

def cluster_mechanisms(all_targets, diseases, emit=print):
    """agrupa os alvos em NOS-MECANISMO recorrentes ENTRE doencas (LLM). Robusto: terso + retry."""
    emit("  · agrupando mecanismos convergentes entre as doencas...")
    listing="\n".join(f"- [{t['disease']}] {t['alvo']} ({t['mecanismo']})" for t in all_targets)
    sysp=("Voce identifica MECANISMOS CONVERGENTES entre doencas (nos a montante compartilhados). "
          "Agrupe os alvos abaixo em ATE 4 NOS-MECANISMO presentes em MULTIPLAS doencas (>=2). "
          "Cada no: 'nome' curto, 'mecanismo' (<=8 palavras), 'doencas' que cobre, 'alvos' = 1-3 SIMBOLOS DE GENE/proteina "
          "ACIONAVEIS (ex: SNCA, LRRK2, NLRP3, mTOR, HDAC6) NUNCA vazio, 'racional' (<=12 palavras). Responda SO JSON compacto.")
    usr=(f"Painel: {', '.join(d['pt'] for d in diseases)}\n\nAlvos:\n{listing}\n\n"
         'JSON: {"nos":[{"nome":"","mecanismo":"","doencas":[""],"alvos":["GENE"],"racional":""}]}')
    for mx in (900, 650):  # tenta terso; se falhar, ainda mais terso
        r=llm(sysp, usr, mx, 0.3)
        try:
            nos=json.loads(re.search(r'\{.*\}',r,re.S).group(0)).get('nos',[])
            if nos: return nos
        except: pass
        emit("  · (cluster vazio, retry mais compacto...)")
    return []

def cross_analyze(diseases, emit=print):
    diseases=_norm_panel(diseases)
    emit(f"PAINEL: {', '.join(d['pt'] for d in diseases)}")
    all_targets=[]
    for d in diseases: all_targets+=gen_targets(d, emit)
    emit(f"{len(all_targets)} alvos candidatos gerados ({len(diseases)} doencas).")
    nodes=cluster_mechanisms(all_targets, diseases, emit)
    emit(f"{len(nodes)} nos-mecanismo convergentes. ancorando alvos pan-doenca na literatura...")
    enriched=[]
    for nd in nodes:
        cov=[d for d in (nd.get('doencas') or []) if d]  # cobertura declarada
        alvos=[a for a in (nd.get('alvos') or []) if a and str(a).strip() and str(a)!='?'][:3]
        if not alvos:  # fallback: extrai alvos concretos
            emit(f"  · (fallback) extraindo alvos p/ {nd.get('nome','?')[:28]}...")
            alvos=extract_targets(nd, diseases)
        # VERIFICACAO REAL: p/ cada alvo, em quantas doencas do painel ha literatura co-ocorrente
        best_alvo=None; best_hits=[]
        for alvo in alvos:
            hits=pm_count_co(alvo, diseases)
            if len(hits)>len(best_hits): best_hits=hits; best_alvo=alvo
        real_cov=len(best_hits)  # cobertura VERIFICADA (nao so declarada pelo LLM)
        total_pm=sum(n for _,n in best_hits)
        # score pan-doenca: cobertura real verificada domina; literatura ancora
        score=round(real_cov*2.0 + min(3,math.log1p(total_pm)), 2)
        if real_cov>=3: tier='pan'; label='NO PAN-DOENCA'
        elif real_cov==2: tier='dual'; label='COMPARTILHADO (2 doencas)'
        elif real_cov==1: tier='single'; label='ANCORADO (1 doenca)'
        else: tier='unverified'; label='NAO VERIFICADO'
        enriched.append({"nome":nd.get('nome','?'),"mecanismo":nd.get('mecanismo','?'),
            "racional":nd.get('racional',''),"alvos":alvos,"alvo_top":best_alvo,
            "cobertura_declarada":cov,"cobertura_verificada":[d for d,_ in best_hits],
            "n_verificada":real_cov,"pubmed_total":total_pm,"score":score,"tier":tier,"label":label})
        emit(f"  · {nd.get('nome','?')[:32]}: alvo {best_alvo} verificado em {real_cov}/{len(diseases)} doencas (PubMed {total_pm})")
    enriched.sort(key=lambda x:-x['score'])
    return {"panel":[d['pt'] for d in diseases],"nodes":enriched,"generated":"cross-disease-in-silico"}

if __name__=="__main__":
    panel=sys.argv[1] if len(sys.argv)>1 else "neurodegenerativas"
    diseases=PANELS.get(panel, [s.strip() for s in panel.split(',') if s.strip()])
    res=cross_analyze(diseases)
    print("\n"+"="*64+"\nNOS-MECANISMO PAN-DOENCA (ranqueados):\n"+"="*64)
    for n in res['nodes']:
        print(f"\n[{n['label']}] {n['nome']}  (score {n['score']})")
        print(f"   mecanismo: {n['mecanismo']}")
        print(f"   alvo top: {n['alvo_top']}  | verificado em {n['n_verificada']} doencas: {n['cobertura_verificada']}")
        print(f"   PubMed total: {n['pubmed_total']}  | racional: {n['racional'][:100]}")
    out=os.path.join(HERE,'cross_result.json'); open(out,'w',encoding='utf-8').write(json.dumps(res,ensure_ascii=False,indent=1))
    print(f"\nsalvo: {out}")
