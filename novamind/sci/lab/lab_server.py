"""
lab_server.py — SERVIDOR DO LABORATORIO IN-SILICO (interativo).
Serve o frontend Stark + endpoint /analyze?disease=X que roda o pipeline ao vivo:
  gerar hipoteses (personas) -> extrair alvo/droga + mapear corpo -> verificar PubMed + ClinicalTrials -> score -> JSON.
Job em thread de fundo + /result para polling (com log de progresso estilo HUD). Stdlib only (sem Flask).
"""
import os, json, re, time, threading, urllib.parse, http.server, requests, math
os.environ.setdefault('LLM_MODEL','qwen3-235b-a22b-instruct-2507')
import sys; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE,'..'))
from dotenv import load_dotenv; load_dotenv(os.path.join(HERE,'..','.env'))
from shared.llm_client import _openai_compat_post, VENICE_URL, VENICE_KEY
from shared.pubmed import search_pmids
import cross_disease as CD
M="qwen3-235b-a22b-instruct-2507"
SYSTEMS=["nervoso","muscular","imune","cardiovascular","respiratorio","digestivo","hepatico","renal","endocrino","esqueletico"]
JOB={"status":"idle","disease":"","log":[],"result":None}
JOB2={"status":"idle","panel":"","log":[],"result":None}  # cross-doenca
def emit2(msg): JOB2["log"].append(msg); JOB2["log"]=JOB2["log"][-14:]; print("[cross]",msg)
def cross_pipeline(panel):
    diseases=CD.PANELS.get(panel,[s.strip() for s in panel.split(',') if s.strip()])
    JOB2.update(status="running",panel=panel,log=[],result=None)
    try:
        res=CD.cross_analyze(diseases, emit2)
        JOB2["result"]=res; JOB2["status"]="done"; emit2(f"concluido: {len(res['nodes'])} nos-mecanismo")
    except Exception as e:
        JOB2["status"]="error"; emit2(f"erro: {e}")
def emit(msg): JOB["log"].append(msg); JOB["log"]=JOB["log"][-12:]; print("[lab]",msg)
def llm(sysp,usr,mx=420):
    for _ in range(3):
        try: return _openai_compat_post(VENICE_URL,VENICE_KEY,M,[{"role":"system","content":sysp},{"role":"user","content":usr}],0.4,mx,is_venice=True)
        except: time.sleep(2)
    return ""
def ct_trials(cond,intr):
    try:
        r=requests.get('https://clinicaltrials.gov/api/v2/studies',params={'query.cond':cond,'query.intr':intr,'countTotal':'true','pageSize':20,'fields':'Phase,OverallStatus'},timeout=20)
        d=r.json(); tot=d.get('totalCount',0); ph=set(); st=[]
        for s in d.get('studies',[]):
            ps=s.get('protocolSection',{}); ph.update(ps.get('designModule',{}).get('phases',[])); st.append(ps.get('statusModule',{}).get('overallStatus',''))
        order={'EARLY_PHASE1':1,'PHASE1':2,'PHASE2':3,'PHASE3':4,'PHASE4':5}
        pr=max([order.get(p,0) for p in ph],default=0)
        return tot,pr,sum(1 for x in st if x in('TERMINATED','WITHDRAWN','SUSPENDED')),sum(1 for x in st if x in('RECRUITING','ACTIVE_NOT_RECRUITING','NOT_YET_RECRUITING','ENROLLING_BY_INVITATION'))
    except: return 0,0,0,0
def pm_count(q):
    try: return len(search_pmids(q,retmax=50))
    except: return 0
_EN_CACHE={}
def disease_en(pt):
    """traduz nome da doenca p/ termo medico EN padrao (PubMed/ClinicalTrials sao indexados em ingles). cacheado."""
    k=pt.strip().lower()
    if k in _EN_CACHE: return _EN_CACHE[k]
    r=llm("Traduza o nome da doenca para o termo medico EM INGLES padrao (sem explicacao). So o termo.",
          f"Doenca: {pt}\nTermo em ingles:",40).strip().strip('."').splitlines()[0][:60] or pt
    _EN_CACHE[k]=r; return r

# ===================== HISTORICO PERSISTENTE (arquivo de tudo que foi testado) =====================
import glob
HISTDIR=os.path.join(HERE,'history'); os.makedirs(HISTDIR,exist_ok=True)
_HSEQ=[0]
def _mode_of(r):
    if r.get('crispr_modo'): return 'genetico'
    if r.get('refino'): return 'refino'
    return 'analise'
def save_history(r):
    """salva CADA analise concluida em disco (history/*.json) p/ arquivo browsavel + export."""
    if not r or not r.get('hypotheses'): return
    try:
        ts=time.strftime('%Y%m%d_%H%M%S'); _HSEQ[0]+=1
        slug=re.sub(r'[^a-z0-9]+','_',str(r.get('disease','x')).lower()).strip('_')[:40] or 'x'
        hid=f"{ts}_{_HSEQ[0]}_{slug}"
        rec=dict(r); rec['_id']=hid; rec['_mode']=_mode_of(r); rec['_when']=time.strftime('%d/%m/%Y %H:%M')
        open(os.path.join(HISTDIR,hid+'.json'),'w',encoding='utf-8').write(json.dumps(rec,ensure_ascii=False))
        print('[hist] salvo',hid)
    except Exception as e: print('[hist] erro',e)
def list_history():
    out=[]
    for f in sorted(glob.glob(os.path.join(HISTDIR,'*.json')),reverse=True):
        try:
            d=json.load(open(f,encoding='utf-8'))
            out.append({"id":d.get('_id'),"disease":d.get('disease'),"mode":d.get('_mode'),"when":d.get('_when'),"n":len(d.get('hypotheses',[]))})
        except: pass
    return out
def get_history(hid):
    p=os.path.join(HISTDIR,os.path.basename(str(hid))+'.json')
    if os.path.exists(p):
        try: return json.load(open(p,encoding='utf-8'))
        except: return None
    return None

# ===================== DOSSIE CIENTIFICO (HTML imprimivel + CSV) =====================
import html as _html, csv as _csv, io as _io, urllib.parse as _up
PHASE_LBL=['—','Pré-clínico','Fase 1','Fase 2','Fase 3','Fase 4']
def _esc(x): return _html.escape(str(x if x is not None else ''))
def _pubmed_link(alvo,den): return 'https://pubmed.ncbi.nlm.nih.gov/?term='+_up.quote(f"{alvo} {den}")
def _ct_link(alvo,den): return 'https://clinicaltrials.gov/search?cond='+_up.quote(den)+'&term='+_up.quote(alvo)
REPORT_CSS="""
*{box-sizing:border-box}body{font-family:'Segoe UI',Arial,sans-serif;color:#1a2230;margin:0;background:#eef2f6;line-height:1.55}
.page{max-width:920px;margin:24px auto;background:#fff;box-shadow:0 2px 18px rgba(0,0,0,.12);padding:0 0 48px}
.cover{background:linear-gradient(135deg,#0b2e4f,#114b86);color:#fff;padding:38px 48px 30px}
.cover .brand{font-size:13px;letter-spacing:4px;opacity:.85;font-weight:700}.cover .brand b{color:#5fd0ff}
.cover h1{font-size:27px;margin:10px 0 4px;font-weight:700}.cover .sub{font-size:14px;opacity:.9}
.cover .meta{margin-top:16px;font-size:12px;opacity:.85;display:flex;gap:22px;flex-wrap:wrap}
.cover .meta b{color:#9fe0ff}
.warn{background:#fff6e5;border-left:4px solid #e0a020;color:#6b4e10;padding:12px 18px;margin:0 48px;margin-top:-16px;position:relative;font-size:12.5px;border-radius:0 6px 6px 0}
.sec{padding:22px 48px 4px}.sec h2{font-size:17px;color:#0b3a66;border-bottom:2px solid #e2e8f0;padding-bottom:7px;margin:18px 0 12px}
.sec h3{font-size:13px;color:#114b86;margin:14px 0 4px;letter-spacing:1px;text-transform:uppercase}
.sum{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:6px 0 4px}
.sum .b{background:#f4f8fc;border:1px solid #e2ecf5;border-radius:9px;padding:12px;text-align:center}
.sum .b .n{font-size:24px;font-weight:800;color:#0b3a66}.sum .b .l{font-size:10px;color:#5a7186;letter-spacing:.5px;text-transform:uppercase}
.method ol{margin:6px 0;padding-left:20px}.method li{margin:6px 0;font-size:13px}
.method .k{color:#0b3a66;font-weight:700}
table.hyp{width:100%;border-collapse:collapse;margin:8px 0;font-size:12px}
table.hyp th{background:#0b3a66;color:#fff;text-align:left;padding:7px 9px;font-size:10.5px;letter-spacing:.5px}
table.hyp td{border-bottom:1px solid #e6edf3;padding:7px 9px;vertical-align:top}
table.hyp tr:nth-child(even) td{background:#f7fafc}
.pill{display:inline-block;font-size:9px;font-weight:800;padding:2px 8px;border-radius:9px;color:#fff}
.card{border:1px solid #e2ecf5;border-left:4px solid var(--c,#114b86);border-radius:9px;padding:14px 16px;margin:12px 0;break-inside:avoid}
.card h4{margin:0 0 3px;font-size:15px;color:#0b2e4f}
.card .modal{font-size:12px;color:#3a5168;margin-bottom:8px}
.kv{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;margin:8px 0}
.kv div b{color:#0b3a66}
.card .blk{font-size:12px;margin:6px 0;color:#33485e}.card .blk b{color:#0b3a66}
.card a{color:#1167c0;text-decoration:none}.card a:hover{text-decoration:underline}
.crispr{background:#eef6ff;border:1px solid #cfe2f5;border-radius:7px;padding:8px 11px;font-size:11.5px;margin:8px 0}
.risk{background:#fff3f3;border:1px solid #f3d3d3;border-radius:7px;padding:7px 11px;font-size:11.5px;color:#8a3030;margin-top:6px}
.foot{padding:18px 48px;color:#6a7d90;font-size:11px;border-top:1px solid #e2e8f0;margin-top:18px}
.conv{font-size:12px;color:#33485e}.conv li{margin:4px 0}
@media print{body{background:#fff}.page{box-shadow:none;margin:0;max-width:100%}.warn{margin-top:8px}.card,.sec{break-inside:avoid}@page{margin:14mm}}
"""
def _status_color(st): return {'verde':'#1f9d57','dourado':'#c79100','vermelho':'#c0392b','cinza':'#7f8c99','azul':'#1167c0'}.get(st,'#1167c0')
def build_report_html(r):
    dz=r.get('disease','—'); den=disease_en(dz); hyps=r.get('hypotheses',[])
    mode={'crispr-harmonia':'Modo Genético / CRISPR (harmonia)','auto-refino':'Auto-refino iterativo'}.get(r.get('generated'),'Análise in-silico')
    when=r.get('_when',time.strftime('%d/%m/%Y %H:%M'))
    nverde=sum(1 for h in hyps if h['status']=='verde'); ndour=sum(1 for h in hyps if h['status']=='dourado')
    ncrispr=any(h.get('crispr') for h in hyps)
    top=hyps[0] if hyps else None
    H=[]
    H.append(f"<!doctype html><html lang='pt-br'><head><meta charset='utf-8'><title>Dossiê NovaMind · {_esc(dz)}</title><style>{REPORT_CSS}</style></head><body><div class='page'>")
    # capa
    H.append("<div class='cover'><div class='brand'>NOVA<b>MIND</b> · IN-SILICO LAB</div>")
    H.append(f"<h1>Dossiê de Triagem Terapêutica</h1><div class='sub'>Alvo clínico: <b>{_esc(dz)}</b></div>")
    H.append(f"<div class='meta'><span>Modo: <b>{_esc(mode)}</b></span><span>Termo indexado: <b>{_esc(den)}</b></span><span>Data: <b>{_esc(when)}</b></span><span>Hipóteses: <b>{len(hyps)}</b></span></div></div>")
    H.append("<div class='warn'><b>⚠ Natureza do documento:</b> triagem <b>computacional</b> de hipóteses, ancorada em PubMed e ClinicalTrials.gov. NÃO constitui cura, diagnóstico ou recomendação clínica validada. Toda candidata exige validação experimental (in vitro, modelos animais, ensaios de off-target/imunogenicidade/farmacocinética) antes de qualquer aplicação.</div>")
    # sumario executivo
    H.append("<div class='sec'><h2>Sumário executivo</h2>")
    H.append(f"<div class='sum'><div class='b'><div class='n'>{len(hyps)}</div><div class='l'>Hipóteses priorizadas</div></div><div class='b'><div class='n'>{nverde}</div><div class='l'>Validadas ativas</div></div><div class='b'><div class='n'>{ndour}</div><div class='l'>Novas promissoras</div></div><div class='b'><div class='n'>{PHASE_LBL[top['fase_max']] if top else '—'}</div><div class='l'>Maturidade do líder</div></div></div>")
    if top:
        H.append(f"<p style='font-size:13px'>Candidata líder: <b>{_esc(top['alvo'])}</b> ({_esc(top['droga'])}) — <b>{_esc(top['label'])}</b>, ancorada em <b>{top['pubmed']}</b> publicações e <b>{top['trials']}</b> ensaios clínicos registrados. As candidatas abaixo estão ordenadas por prioridade computacional.</p>")
    H.append("</div>")
    # metodologia & arquitetura
    H.append("<div class='sec method'><h2>Metodologia &amp; arquitetura</h2>")
    H.append("<p style='font-size:13px'>O NovaMind é um sistema de IA que une uma <b>arquitetura cognitiva de composição e raciocínio</b> a uma <b>verificação ancorada em bases científicas reais</b>. Diferente de um modelo de linguagem isolado, ele não apenas propõe ideias — submete cada uma a um <i>gauntlet</i> de verificação e refina iterativamente. Etapas:</p>")
    H.append("<ol>")
    H.append("<li><span class='k'>Geração multi-ângulo.</span> Múltiplas personas mecanísticas geram hipóteses diversas (proteostase, neuroinflamação, disfunção mitocondrial, vias de sinalização, reposicionamento de fármacos, regeneração).</li>")
    H.append("<li><span class='k'>Gauntlet de verificação in-silico.</span> Cada hipótese é mapeada a um alvo molecular e modalidade, e então ancorada: volume de <b>publicações no PubMed</b> (evidência acumulada) e <b>ensaios no ClinicalTrials.gov</b> com fase máxima (maturidade e saturação clínica). Buscas feitas com o termo médico em inglês para máxima cobertura.</li>")
    H.append("<li><span class='k'>Pontuação quantitativa.</span> O score combina ancoragem na literatura, <i>druggability</i>, espaço clínico em aberto (<i>headroom</i>) e penaliza direções já esgotadas ou que falharam clinicamente.</li>")
    H.append("<li><span class='k'>Auto-refino.</span> Um loop mantém as hipóteses sobreviventes, aprofunda e combina as mais promissoras, descarta os becos sem saída, e converge num <i>shortlist</i> priorizado.</li>")
    if ncrispr: H.append("<li><span class='k'>Modo genético / CRISPR.</span> Avalia a <b>harmonia</b> de cada edição: correção da causa-raiz (durabilidade — a doença não recorre), especificidade (off-target) e colateral multissistêmico previsto.</li>")
    H.append("</ol>")
    H.append("<h3>Por que é útil ao laboratório</h3><p style='font-size:13px'>Reduz um espaço enorme de hipóteses a poucas candidatas priorizadas, cada uma ancorada em evidência <b>rastreável</b> (links diretos abaixo), encurtando o tempo de triagem inicial e direcionando o investimento experimental.</p>")
    H.append("<h3>Escopo &amp; limitações</h3><p style='font-size:12.5px;color:#5a4a20;background:#fffaf0;border:1px solid #f0e3c0;border-radius:7px;padding:10px 13px'>As contagens de PubMed/ClinicalTrials são <b>indicadores de ancoragem, não prova de eficácia</b>. O sistema não executa simulação de genoma, célula ou organismo. As hipóteses são pontos de partida para investigação, não conclusões. A decisão clínica e experimental permanece integralmente com a equipe científica humana.</p></div>")
    # tabela
    H.append("<div class='sec'><h2>Hipóteses priorizadas</h2><table class='hyp'><tr><th>#</th><th>Alvo</th><th>Abordagem</th><th>Status</th><th>PubMed</th><th>Ensaios</th><th>Fase</th><th>Score</th></tr>")
    for i,h in enumerate(hyps):
        c=_status_color(h['status'])
        H.append(f"<tr><td>{i+1}</td><td><b>{_esc(h['alvo'])}</b></td><td>{_esc(h['droga'])}</td><td><span class='pill' style='background:{c}'>{_esc(h['label'])}</span></td><td>{h['pubmed']}</td><td>{h['trials']}</td><td>{PHASE_LBL[h['fase_max']]}</td><td><b>{h['score']}</b></td></tr>")
    H.append("</table></div>")
    # cards detalhados
    H.append("<div class='sec'><h2>Detalhamento das candidatas</h2>")
    for i,h in enumerate(hyps):
        c=_status_color(h['status'])
        H.append(f"<div class='card' style='--c:{c}'><h4>{i+1}. {_esc(h['alvo'])} <span class='pill' style='background:{c}'>{_esc(h['label'])}</span></h4><div class='modal'>{_esc(h['droga'])}</div>")
        if h.get('hyp'): H.append(f"<div class='blk'><b>Hipótese gerada:</b> {_esc(h['hyp'])}</div>")
        H.append(f"<div class='kv'><div><b>Evidência (PubMed):</b> {h['pubmed']} pubs · <a href='{_pubmed_link(h['alvo'],den)}' target='_blank'>verificar ↗</a></div>"
                 f"<div><b>Ensaios clínicos:</b> {h['trials']} · {PHASE_LBL[h['fase_max']]} · <a href='{_ct_link(h['alvo'],den)}' target='_blank'>verificar ↗</a></div>"
                 f"<div><b>Score:</b> {h['score']}</div></div>")
        H.append(f"<div class='blk'><b>Onde age:</b> {_esc(h.get('orgao','—'))} · sistema {_esc(h.get('alvo_sistema','—'))} &nbsp;|&nbsp; <b>Sistemas afetados:</b> {_esc(' · '.join(h.get('sistemas_afetados') or []))}</div>")
        x=h.get('crispr')
        if x:
            col=', '.join(x.get('sistemas_colaterais') or []) or 'nenhum colateral multissistêmico previsto'
            H.append(f"<div class='crispr'>🧬 <b>Edição:</b> {_esc(x.get('modalidade'))} via {_esc(x.get('entrega'))} · {'<b>corrige a causa-raiz (não recorre)</b>' if x.get('corrige_causa_raiz') else 'paliativa (não corrige a causa)'}{' · <b style=color:#c0392b>germinativa</b>' if x.get('germinativa') else ''}<br>Off-target: <b>{_esc(x.get('off_target'))}</b> · Harmonia sistêmica: <b>{round((x.get('harmonia_sistemica') or 0)*100)}%</b> · Colateral previsto: {_esc(col)}</div>")
        if h.get('riscos'): H.append(f"<div class='risk'><b>⚠ Riscos:</b> {_esc('; '.join(h.get('riscos') or []))}</div>")
        H.append("</div>")
    H.append("</div>")
    # convergencia
    rf=r.get('refino')
    if rf:
        H.append("<div class='sec'><h2>Convergência do auto-refino</h2><div class='conv'>")
        H.append(f"<p>Becos descartados: <b>{_esc(', '.join(rf.get('descartados') or []) or 'nenhum')}</b></p><ul>")
        for c in rf.get('convergencia') or []:
            H.append(f"<li>Ciclo {c['ciclo']}: {c['gerados']} gerados → {c['sobreviventes']} sobreviventes · líderes: {_esc(', '.join(t['alvo']+' ('+str(t['score'])+')' for t in c['top']))}</li>")
        H.append("</ul></div></div>")
    # rodape
    H.append(f"<div class='foot'><b>Proveniência:</b> hipóteses geradas e verificadas pelo NovaMind In-Silico Lab. Fontes: PubMed (NCBI E-utilities) e ClinicalTrials.gov API v2. Termo de busca indexado: <b>{_esc(den)}</b>. Gerado em {_esc(when)}. Documento de triagem — não substitui revisão por pares nem validação experimental.</div>")
    H.append("</div></body></html>")
    return ''.join(H)
def build_report_csv(r):
    den=disease_en(r.get('disease','')); buf=_io.StringIO(); w=_csv.writer(buf)
    w.writerow(['rank','alvo','abordagem','status','label','pubmed','trials','fase_max','score','alvo_sistema','orgao','sistemas_afetados','riscos','pubmed_link','clinicaltrials_link','crispr_modalidade','crispr_causa_raiz','crispr_off_target','crispr_colateral','hipotese'])
    for i,h in enumerate(r.get('hypotheses',[])):
        x=h.get('crispr') or {}
        w.writerow([i+1,h['alvo'],h['droga'],h['status'],h['label'],h['pubmed'],h['trials'],PHASE_LBL[h['fase_max']],h['score'],
            h.get('alvo_sistema',''),h.get('orgao',''),' | '.join(h.get('sistemas_afetados') or []),' | '.join(h.get('riscos') or []),
            _pubmed_link(h['alvo'],den),_ct_link(h['alvo'],den),x.get('modalidade',''),x.get('corrige_causa_raiz',''),x.get('off_target',''),
            ' | '.join(x.get('sistemas_colaterais') or []),(h.get('hyp') or '').replace('\n',' ')])
    return buf.getvalue()

def pipeline(disease):
    JOB.update(status="running",disease=disease,log=[],result=None)
    emit(f"alvo: {disease}")
    angles=["estresse oxidativo / mitocondria","neuroinflamacao","via de sinalizacao principal da doenca",
            "agregacao proteica / proteostase","alvo molecular reposicionavel","mecanismo de regeneracao/reparo"]
    emit("gerando hipoteses (personas)...")
    hyps=[gen_hyp(disease,a) for a in angles]; hyps=[h for h in hyps if h]
    den=disease_en(disease); emit(f"{len(hyps)} hipoteses geradas (termo EN: {den}). verificando...")
    out=[]
    for i,h in enumerate(hyps):
        emit(f"[{i+1}/{len(hyps)}] verificando (PubMed + ClinicalTrials)...")
        out.append(verify_hyp(disease,h,den))
    out.sort(key=lambda x:-x['score'])
    JOB["result"]={"disease":disease,"systems":SYSTEMS,"hypotheses":out,"generated":"in-silico"}
    save_history(JOB["result"]); JOB["status"]="done"; emit(f"concluido: {len(out)} hipoteses priorizadas")

def gen_hyp(disease, focus):
    """gera UMA hipotese terapeutica concisa (alvo+droga+mecanismo) p/ uma doenca, dado um foco/angulo."""
    return llm("Voce gera UMA hipotese terapeutica concisa (alvo + droga/classe + mecanismo, 2 frases). Especifica e diversa.",
               f"Doenca: {disease}\nAngulo/foco: {focus}\nHipotese:",300).strip()

def verify_hyp(disease, h, den=None):
    """extrai alvo/droga/sistemas + verifica na literatura real (PubMed) e maturidade clinica (ClinicalTrials) -> dict scored.
    den = nome da doenca EM INGLES p/ as queries (PubMed/ClinicalTrials sao indexados em ingles)."""
    den=den or disease_en(disease)
    ext=llm(f"Farmacologista. Sistemas validos: {', '.join(SYSTEMS)}. Responda SO JSON.",
        f"Hipotese:\n{h}\n\nJSON: {{\"alvo\":\"gene/proteina\",\"droga\":\"droga/classe\",\"alvo_sistema\":\"sistema primario\",\"orgao\":\"orgao\",\"sistemas_afetados\":[\"1-3 sistemas\"],\"riscos\":[\"1-2 riscos multi-sistema concretos\"]}}")
    try: m=json.loads(re.search(r'\{.*\}',ext,re.S).group(0))
    except: m={"alvo":"?","droga":"?","alvo_sistema":"nervoso","orgao":"?","sistemas_afetados":["nervoso"],"riscos":[]}
    alvo=str(m.get('alvo','?'))[:40]; droga=str(m.get('droga','?'))[:40]
    pm=pm_count(f"{alvo} {den} therapy")
    tot,pr,nterm,nact=ct_trials(den,f"{alvo} OR {droga}")
    grounding=min(3,math.log1p(pm)); druggable=1.0 if(tot>0 or droga not in('?',''))else 0.3
    exhausted=1.0 if(pr>=4 and nact==0)else 0.0; headroom=1.0 if tot==0 else(0.7 if nact>0 else 0.4)
    score=round(grounding+druggable+headroom*1.2-exhausted*1.5,2)
    if pm==0 and tot==0: status='cinza'; label='SEM EVIDENCIA'
    elif exhausted: status='vermelho'; label='FALHOU/ESGOTADA'
    elif tot==0 and pm>0: status='dourado'; label='NOVA PROMISSORA'
    elif tot>0 and nact>0: status='verde'; label='VALIDADA ATIVA'
    else: status='azul'; label='ANCORADA'
    return {"alvo":alvo,"droga":droga,"status":status,"label":label,"pubmed":pm,"trials":tot,"fase_max":pr,
        "score":score,"alvo_sistema":m.get('alvo_sistema','nervoso'),"orgao":m.get('orgao','?'),
        "sistemas_afetados":(m.get('sistemas_afetados') or ['nervoso'])[:3],"riscos":(m.get('riscos') or [])[:2],"hyp":h[:700]}

# ===================== MODO GENETICO / CRISPR =====================
CRISPR_ANGLES=[
 "silenciar/knockout do gene CAUSAL da doenca (nuclease Cas9 ou CRISPRi)",
 "CORRIGIR a mutacao patogenica por BASE EDITING (sem corte de fita dupla)",
 "CORRIGIR a mutacao por PRIME EDITING (edicao precisa generica)",
 "knock-in de alelo PROTETOR ou copia saudavel do gene",
 "ativacao CRISPRa de um gene COMPENSATORIO/neuroprotetor",
 "editar um MODIFICADOR genetico (nao o gene causal) p/ resiliencia, evitando gene essencial",
]
def crispr_assess(disease, den, h):
    """extrai gene+modalidade+entrega e AVALIA HARMONIA: durabilidade(causa-raiz), off-target, colateral multi-sistemico."""
    sysp=f"Voce e engenheiro de terapia genica. Sistemas validos: {', '.join(SYSTEMS)}. Responda SO JSON compacto e completo."
    usr=(f"Estrategia CRISPR:\n{h}\n\nJSON: {{\"gene\":\"simbolo HGNC (ex: SOD1)\",\"modalidade\":\"knockout/base-editing/prime/CRISPRa/CRISPRi/knock-in\","
        "\"entrega\":\"AAV/LNP/ex-vivo\",\"alvo_sistema\":\"sistema primario\",\"orgao\":\"orgao\","
        "\"corrige_causa_raiz\":true,\"germinativa\":false,\"off_target\":\"baixo/medio/alto\","
        "\"essencial_outros_sistemas\":\"baixo/medio/alto\",\"sistemas_colaterais\":[\"sistema\"],\"riscos\":[\"risco\"]}}")
    m={}
    for mx in (600,650):
        try:
            m=json.loads(re.search(r'\{.*\}',llm(sysp,usr,mx),re.S).group(0))
            if str(m.get('gene','?')) not in('?','',None): break
        except: m={}
    gene=str(m.get('gene','?'))[:24]; modal=str(m.get('modalidade','?'))[:20]; entrega=str(m.get('entrega','?'))[:18]
    pm=pm_count(f"CRISPR {gene} {den}")
    tot,pr,nterm,nact=ct_trials(den,f"gene therapy OR {gene} OR CRISPR")
    lvl={"baixo":1.0,"medio":0.5,"alto":0.0}
    durab=1.0 if m.get('corrige_causa_raiz') else 0.35           # corrige causa -> nao volta
    espec=lvl.get(str(m.get('off_target','medio')).lower(),0.5)  # off-target baixo = especifico
    harm_sis=lvl.get(str(m.get('essencial_outros_sistemas','medio')).lower(),0.5)  # gene nao essencial alhures
    ncol=len(m.get('sistemas_colaterais') or [])
    germ_pen=0.6 if m.get('germinativa') else 0.0               # germinativa = problema etico/colateral hereditario
    grounding=min(2,math.log1p(pm))
    harmonia=round(durab*2.5 + espec*1.5 + harm_sis*1.5 + grounding - ncol*0.4 - germ_pen, 2)
    # status pela harmonia + ancoragem
    if pm==0 and tot==0: status='cinza'; label='SEM EVIDENCIA'
    elif durab>=1 and espec>=1 and harm_sis>=1 and ncol<=1: status='verde'; label='HARMONICA'
    elif durab>=1: status='dourado'; label='CAUSAL (refinar colateral)'
    else: status='azul'; label='PALIATIVA/RISCO'
    return {"alvo":gene,"droga":f"{modal} · {entrega}","status":status,"label":label,"pubmed":pm,"trials":tot,"fase_max":pr,
        "score":harmonia,"alvo_sistema":m.get('alvo_sistema','nervoso'),"orgao":m.get('orgao','?'),
        "sistemas_afetados":(m.get('sistemas_colaterais') or [m.get('alvo_sistema','nervoso')])[:3],
        "riscos":(m.get('riscos') or [])[:2],"hyp":h[:700],
        "crispr":{"modalidade":modal,"entrega":entrega,"durabilidade":round(durab,2),"off_target":m.get('off_target','?'),
            "harmonia_sistemica":round(harm_sis,2),"sistemas_colaterais":(m.get('sistemas_colaterais') or [])[:4],
            "corrige_causa_raiz":bool(m.get('corrige_causa_raiz')),"germinativa":bool(m.get('germinativa'))}}

def crispr_pipeline(disease, cycles=2):
    """MODO GENETICO: gera estrategias CRISPR -> verifica -> avalia HARMONIA (cura causa-raiz, sem colateral, nao volta)
    -> auto-refina empurrando p/ a edicao mais harmonica (menos colateral + duravel)."""
    JOB.update(status="running",disease=disease,log=[],result=None)
    den=disease_en(disease); emit(f"MODO GENETICO/CRISPR: {disease} (EN: {den}) — buscando edicao harmonica")
    survivors=[]; dead=set(); cyc_log=[]
    for c in range(cycles):
        if c==0 or not survivors:
            emit(f"ciclo {c+1}: gerando estrategias CRISPR ({len(CRISPR_ANGLES)} modalidades)...")
            cand=[gen_hyp(disease,a) for a in CRISPR_ANGLES]
        else:
            top=survivors[:3]; mortos=", ".join(list(dead)[:6]) or "nenhum"
            emit(f"ciclo {c+1}: refinando p/ MAIS harmonia (menos colateral, mais durabilidade)...")
            cand=[gen_hyp(disease, f"REFINE esta edicao genetica p/ MAIOR HARMONIA: {s['alvo']} via {s['crispr']['modalidade']}. "
                  f"Reduza colateral multi-sistemico e off-target, garanta correcao da CAUSA-RAIZ (doenca nao volta), "
                  f"sem tocar genes essenciais. Evite becos: {mortos}") for s in top]
        cand=[h for h in cand if h]
        scored=[]
        for i,h in enumerate(cand):
            emit(f"  ciclo {c+1} [{i+1}/{len(cand)}] avaliando harmonia (off-target+colateral+durabilidade)...")
            scored.append(crispr_assess(disease,den,h))
        pool={}
        for s in survivors+scored:
            k=s['alvo'].lower()
            if k in('?','',None): continue  # descarta extracoes falhas
            if k not in pool or s['score']>pool[k]['score']: pool[k]=s
        ranked=sorted(pool.values(),key=lambda x:-x['score'])
        for s in scored:
            if s['status']=='cinza': dead.add(s['alvo'])
        survivors=[s for s in ranked if s['status']!='cinza'][:5] or ranked[:3]
        cyc_log.append({"ciclo":c+1,"gerados":len(scored),"sobreviventes":len(survivors),
            "top":[{"alvo":s['alvo'],"score":s['score'],"label":s['label']} for s in survivors[:4]]})
        emit(f"ciclo {c+1}: {len(survivors)} edicoes (top harmonia: {', '.join(s['alvo'] for s in survivors[:3])})")
    survivors.sort(key=lambda x:-x['score'])
    JOB["result"]={"disease":disease,"systems":SYSTEMS,"hypotheses":survivors,"generated":"crispr-harmonia",
        "crispr_modo":True,"refino":{"ciclos":cycles,"descartados":sorted(dead),"convergencia":cyc_log}}
    save_history(JOB["result"]); JOB["status"]="done"; emit(f"MODO GENETICO concluido: {len(survivors)} edicoes harmonicas priorizadas")

def refine_pipeline(disease, cycles=3):
    """LOOP AUTO-REFINANTE: gera -> verifica (gauntlet) -> mantem sobreviventes -> RE-GERA focando neles
    (aprofunda/combina/adjacentes) e descarta os becos sem saida -> converge num shortlist refinado."""
    JOB.update(status="running",disease=disease,log=[],result=None)
    den=disease_en(disease); emit(f"REFINO de {disease} (EN: {den}) — {cycles} ciclos")
    angles=["via de sinalizacao principal","agregacao proteica / proteostase","neuroinflamacao",
            "alvo reposicionavel","mecanismo de regeneracao","estresse oxidativo / mitocondria"]
    survivors=[]; dead=set(); cyc_log=[]
    for c in range(cycles):
        if c==0 or not survivors:
            emit(f"ciclo {c+1}: gerando amplo ({len(angles)} angulos)...")
            cand=[gen_hyp(disease,a) for a in angles]
        else:
            top=survivors[:3]; foco="; ".join(f"{s['alvo']} ({s['droga']})" for s in top)
            mortos=", ".join(list(dead)[:6]) or "nenhum"
            emit(f"ciclo {c+1}: aprofundando os {len(top)} melhores, descartando becos...")
            cand=[]
            for s in top:  # aprofunda cada sobrevivente (refinar/combinar/adjacente)
                cand.append(gen_hyp(disease, f"APROFUNDE/REFINE esta direcao promissora: {s['alvo']} via {s['droga']}. "
                    f"Proponha uma versao MELHOR (alvo a montante/jusante, combinacao sinergica, ou modalidade superior). "
                    f"NAO repita becos sem saida ja descartados: {mortos}"))
            cand.append(gen_hyp(disease, f"alvo NOVO e pouco explorado, distinto de [{foco}] e evitando [{mortos}]"))
        cand=[h for h in cand if h]
        scored=[]
        for i,h in enumerate(cand):
            emit(f"  ciclo {c+1} [{i+1}/{len(cand)}] verificando...")
            scored.append(verify_hyp(disease,h,den))
        # funde com sobreviventes anteriores, dedup por alvo, mantem melhores
        pool={}
        for s in survivors+scored:
            k=s['alvo'].lower()
            if k not in pool or s['score']>pool[k]['score']: pool[k]=s
        ranked=sorted(pool.values(),key=lambda x:-x['score'])
        # becos sem saida (a EVITAR) = so os 'vermelho' (falharam/esgotaram clinicamente)
        for s in scored:
            if s['status']=='vermelho': dead.add(s['alvo'])
        # sobreviventes = melhores direcoes p/ aprofundar (exclui vermelho); SEMPRE mantem sementes
        survivors=[s for s in ranked if s['status']!='vermelho'][:5] or ranked[:3]
        cyc_log.append({"ciclo":c+1,"gerados":len(scored),"sobreviventes":len(survivors),
            "top":[{"alvo":s['alvo'],"score":s['score'],"label":s['label']} for s in survivors[:4]]})
        emit(f"ciclo {c+1}: {len(survivors)} sobreviventes (top: {', '.join(s['alvo'] for s in survivors[:3])})")
    survivors.sort(key=lambda x:-x['score'])
    JOB["result"]={"disease":disease,"systems":SYSTEMS,"hypotheses":survivors,"generated":"auto-refino",
        "refino":{"ciclos":cycles,"descartados":sorted(dead),"convergencia":cyc_log}}
    save_history(JOB["result"]); JOB["status"]="done"; emit(f"REFINO concluido: {len(survivors)} hipoteses convergidas em {cycles} ciclos")

class H(http.server.SimpleHTTPRequestHandler):
    def __init__(s,*a,**k): super().__init__(*a,directory=HERE,**k)
    def _json(s,obj,code=200): s.send_response(code); s.send_header('Content-Type','application/json; charset=utf-8'); s.send_header('Access-Control-Allow-Origin','*'); s.end_headers(); s.wfile.write(json.dumps(obj,ensure_ascii=False).encode('utf-8'))
    def _raw(s,text,ctype='text/html; charset=utf-8',download=None,code=200):
        s.send_response(code); s.send_header('Content-Type',ctype); s.send_header('Access-Control-Allow-Origin','*')
        if download: s.send_header('Content-Disposition',f'attachment; filename="{download}"')
        s.end_headers(); s.wfile.write(text.encode('utf-8'))
    def do_GET(s):
        u=urllib.parse.urlparse(s.path); q=urllib.parse.parse_qs(u.query)
        if u.path=='/analyze':
            dz=(q.get('disease',['']) or [''])[0].strip()
            if not dz: return s._json({"error":"sem doenca"},400)
            if JOB["status"]=="running": return s._json({"status":"running","msg":"ja rodando"})
            threading.Thread(target=pipeline,args=(dz,),daemon=True).start(); return s._json({"status":"started","disease":dz})
        if u.path=='/refine':
            dz=(q.get('disease',['']) or [''])[0].strip()
            if not dz: return s._json({"error":"sem doenca"},400)
            if JOB["status"]=="running": return s._json({"status":"running","msg":"ja rodando"})
            cyc=int((q.get('cycles',['3']) or ['3'])[0] or 3)
            threading.Thread(target=refine_pipeline,args=(dz,cyc),daemon=True).start(); return s._json({"status":"started","disease":dz,"mode":"refino"})
        if u.path=='/crispr':
            dz=(q.get('disease',['']) or [''])[0].strip()
            if not dz: return s._json({"error":"sem doenca"},400)
            if JOB["status"]=="running": return s._json({"status":"running","msg":"ja rodando"})
            cyc=int((q.get('cycles',['2']) or ['2'])[0] or 2)
            threading.Thread(target=crispr_pipeline,args=(dz,cyc),daemon=True).start(); return s._json({"status":"started","disease":dz,"mode":"crispr"})
        if u.path=='/result': return s._json({"status":JOB["status"],"disease":JOB["disease"],"log":JOB["log"],"result":JOB["result"]})
        if u.path=='/cross':
            pn=(q.get('panel',['']) or [''])[0].strip()
            if not pn: return s._json({"error":"sem painel"},400)
            if JOB2["status"]=="running": return s._json({"status":"running","msg":"ja rodando"})
            threading.Thread(target=cross_pipeline,args=(pn,),daemon=True).start(); return s._json({"status":"started","panel":pn})
        if u.path=='/crossresult': return s._json({"status":JOB2["status"],"panel":JOB2["panel"],"log":JOB2["log"],"result":JOB2["result"]})
        if u.path=='/history': return s._json({"items":list_history()})
        if u.path=='/history/get':
            r=get_history((q.get('id',['']) or [''])[0])
            return s._json({"result":r}) if r else s._json({"error":"nao encontrado"},404)
        if u.path in('/report','/report.csv'):
            hid=(q.get('id',['']) or [''])[0]
            r=get_history(hid) if hid else (JOB.get('result') or (get_history((list_history() or [{}])[0].get('id')) if list_history() else None))
            if not r or not r.get('hypotheses'): return s._raw("<h2>Nenhum resultado para gerar relatório.</h2>",code=404)
            slug=re.sub(r'[^a-z0-9]+','_',str(r.get('disease','x')).lower()).strip('_')[:40] or 'relatorio'
            if u.path=='/report.csv': return s._raw(build_report_csv(r),'text/csv; charset=utf-8',download=f'novamind_{slug}.csv')
            return s._raw(build_report_html(r))
        if u.path=='/' : s.path='/stark.html'
        return super().do_GET()
    def log_message(s,*a): pass

if __name__=="__main__":
    port=int(sys.argv[1]) if len(sys.argv)>1 else 8772
    print(f"Laboratorio in-silico em http://localhost:{port}/  (stark.html)")
    http.server.ThreadingHTTPServer(('127.0.0.1',port),H).serve_forever()
