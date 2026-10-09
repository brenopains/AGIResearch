"""
Servidor web: FastAPI + SSE p/ ver os agentes trabalhando em tempo real.
Frontend single-page em frontend/index.html.

Rotas:
  POST /api/run                   -> dispara um run (system, disease, params)
  GET  /api/stream/{run_id}       -> SSE com eventos em tempo real
  GET  /api/result/{run_id}       -> resultado final
  GET  /api/runs                  -> lista runs salvos em disco
  GET  /                          -> frontend
"""
import os, sys, json, time, uuid, threading, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
# Forca UTF-8 em stdout/stderr (Windows default cp1252 quebra com beta, alpha etc) — conserta charmap codec
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace"); sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
os.environ["PYTHONIOENCODING"] = "utf-8"
# carrega .env (VENICE_API_KEY etc)
try:
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).parent / ".env")
except Exception: pass
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from shared.events import BUS, bus_emitter
from co_scientist.run import run as run_coscientist
from robin.run import run as run_robin
from novamind_sci.run import run as run_novamindsci
from unified.run import run as run_unified

ROOT = pathlib.Path(__file__).parent
RUNS_DIR = ROOT / "runs"

app = FastAPI(title="NovaMind-Sci")

class RunReq(BaseModel):
    system: str           # 'co_scientist' | 'robin' | 'novamind_sci' | 'both'
    disease: str
    n: int = 6
    rounds: int = 2
    tourn: int = 2
    cycles: int = 12
    pop: int = 8

def _runner(run_id: str, req: RunReq):
    emit = bus_emitter(run_id)
    try:
        if req.system == "co_scientist":
            result = run_coscientist(req.disease, n_initial=req.n, rounds=req.rounds, tourn_rounds=req.tourn, emit=emit)
        elif req.system == "robin":
            result = run_robin(req.disease, n_hypotheses=req.n, emit=emit)
        elif req.system == "novamind_sci":
            result = run_novamindsci(req.disease, cycles=req.cycles, pop=req.pop, emit=emit)
        elif req.system == "unified":
            result = run_unified(req.disease, generation_cycles=req.cycles, top_for_tournament=req.pop, tourn_rounds=req.tourn, deep_dive_top=3, emit=emit)
        elif req.system == "both":
            # roda os dois EM SEQUENCIA (compartilhando o mesmo bus, com prefixo de sistema)
            emit_a = lambda et, p=None: BUS.emit(run_id, "A:"+et, p)
            emit_b = lambda et, p=None: BUS.emit(run_id, "B:"+et, p)
            BUS.emit(run_id, "side_by_side_start", {"a":"co_scientist","b":"novamind_sci"})
            res_a = run_coscientist(req.disease, n_initial=req.n, rounds=req.rounds, tourn_rounds=req.tourn, emit=emit_a)
            res_b = run_novamindsci(req.disease, cycles=req.cycles, pop=req.pop, emit=emit_b)
            result = {"a":res_a, "b":res_b, "disease":req.disease}
        else:
            BUS.emit(run_id, "error", {"msg": f"system desconhecido: {req.system}"})
            BUS.finish(run_id, {"error": True}); return
        BUS.finish(run_id, result)
    except Exception as e:
        import traceback
        BUS.emit(run_id, "error", {"msg": str(e), "trace": traceback.format_exc()[-1500:]})
        BUS.finish(run_id, {"error": str(e)})

@app.post("/api/run")
def start_run(req: RunReq):
    run_id = uuid.uuid4().hex[:12]
    BUS.new_run(run_id)
    t = threading.Thread(target=_runner, args=(run_id, req), daemon=True)
    t.start()
    return {"run_id": run_id}

@app.get("/api/stream/{run_id}")
async def stream(run_id: str, request: Request):
    def gen():
        # se o run_id nao existe mais (ex: server reiniciou), encerra cleanly em vez de inundar
        with BUS.lock:
            known = run_id in BUS.queues
        if not known:
            yield f"event: end\ndata: {json.dumps({'type':'__END__','payload':{'reason':'run_lost_on_restart','run_id':run_id}})}\n\n"
            return
        for ev in BUS.stream(run_id):
            if ev.get("type") == "__END__":
                yield f"event: end\ndata: {json.dumps(ev)}\n\n"; break
            if ev.get("type") == "error" and "run_id" in (ev.get("payload",{}).get("msg","")):
                # mesma protecao se o bus perdeu o id
                yield f"event: end\ndata: {json.dumps({'type':'__END__','payload':ev.get('payload',{})})}\n\n"; break
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.get("/api/result/{run_id}")
def result(run_id: str):
    r = BUS.get_result(run_id)
    if not r: return JSONResponse({"error":"ainda nao concluido"}, status_code=404)
    return r

@app.get("/api/runs")
def list_runs():
    out = []
    for p in sorted(RUNS_DIR.glob("*.json"), key=lambda x:-x.stat().st_mtime):
        if p.name.startswith("_"): continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out.append({
                "name": p.stem, "disease": d.get("disease",""), "system": d.get("system",""),
                "duration_s": d.get("duration_s",0), "n_hypotheses": len(d.get("hypotheses",[])),
                "partial": d.get("__partial__"),    # se != None, e parcial (run nao terminou)
                "ts": p.stat().st_mtime,
            })
        except Exception: pass
        if len(out) >= 60: break
    return out

@app.get("/api/run_detail/{name}")
def run_detail(name: str):
    p = RUNS_DIR / f"{name}.json"
    if not p.exists(): return JSONResponse({"error":"nao encontrado"}, status_code=404)
    return json.loads(p.read_text(encoding="utf-8"))

def _export_html(d: dict) -> str:
    from html import escape as e
    disease = d.get("disease","?"); system = d.get("system","?")
    hyps = d.get("hypotheses",[])
    personas = d.get("personas",{}) or {}; vectors = d.get("vectors",{}) or {}
    promise = d.get("promise") or {}
    review = d.get("meta_review") or ""
    drives = d.get("drives_final") or {}
    deep = d.get("deep_dive") or []
    partial = d.get("__partial__")
    ATTRS = ["novelty","plausibility","tractability","safety","evidence_strength","orthogonality","cost","clinical_impact","synergy"]
    # explicacoes legiveis p/ leigo
    ATTR_INFO = {
        "novelty":        ("Novidade",          "1.0 = ideia rara/inédita. 0.0 = já é o tratamento padrão."),
        "plausibility":   ("Plausibilidade",    "1.0 = mecanismo se sustenta com biologia conhecida. 0.0 = especulativo."),
        "tractability":   ("Praticabilidade",   "1.0 = existe forma realista de testar. 0.0 = inviável."),
        "safety":         ("Segurança",         "1.0 = baixo risco de efeito colateral grave. 0.0 = perigoso."),
        "evidence_strength":("Força da evidência","1.0 = papers fortes citados. 0.0 = só hipótese solta."),
        "orthogonality":  ("Ortogonalidade",    "1.0 = sai do paradigma dominante. 0.0 = repete o que todo mundo já faz."),
        "cost":           ("Custo experimental","1.0 = caro de testar. 0.0 = barato. (Entra com sinal negativo no score)"),
        "clinical_impact":("Impacto clínico",   "1.0 = potencial de cura. 0.0 = irrelevante."),
        "synergy":        ("Sinergia",          "1.0 = combina bem com tratamentos atuais. 0.0 = conflita ou redundante."),
    }
    PERSONA_INFO = {
        "explorador":   ("🟣 Explorador",   "Busca novidade ousada — vias pouco exploradas, repurposing exótico. Aceita risco maior."),
        "conservador":  ("🔵 Conservador",  "Prefere alvos drug-able já validados, baixo risco. Replica o que funciona."),
        "engenheiro":   ("🟠 Engenheiro",   "Foca em praticabilidade: o que dá pra testar rápido, com modelo experimental claro."),
        "seguranca":    ("🟢 Segurança",    "Prioriza ausência de efeitos colaterais graves. Conservador em risco off-target."),
        "sintetizador": ("🩷 Sintetizador", "Combina mecanismos de doenças adjacentes ou comorbidades — busca pontes."),
        "composto":     ("🟡 Composto A→B", "Hipótese sintetizada: mecanismo A serve de contexto para B. Ordem importa."),
        "composto_rev": ("🟣 Composto B→A", "Mesma sintese mas em ordem inversa. Comparada com A→B revela assimetria causal."),
    }
    css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    :root{--bg:#fafbfc;--panel:#fff;--ink:#1a1f2c;--ink2:#5a6573;--ink3:#9ba5b1;--line:#e4e7ec;--accent:#0066cc;--accent2:#cc7a00;--good:#1aa860;--warn:#b87800;--bad:#c0334a}
    *{box-sizing:border-box}
    body{font-family:'Inter',-apple-system,sans-serif;background:var(--bg);color:var(--ink);margin:0;padding:0;line-height:1.6;font-size:14px}
    .wrap{max-width:920px;margin:0 auto;padding:32px 28px 64px}
    header.report{border-bottom:3px solid var(--accent);padding-bottom:18px;margin-bottom:24px}
    header.report .tag{display:inline-block;background:var(--accent);color:#fff;padding:3px 10px;border-radius:4px;font-size:11px;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:8px}
    header.report h1{margin:0;font-size:30px;font-weight:700;color:var(--ink);letter-spacing:-0.5px}
    header.report .meta{color:var(--ink2);font-size:13px;margin-top:6px}
    header.report .meta b{color:var(--ink)}
    .intro{background:#fff7eb;border-left:4px solid #f0a500;padding:14px 18px;border-radius:6px;margin-bottom:24px;font-size:13px;line-height:1.6}
    .intro strong{color:#a06600}
    .partial{background:#fef3c7;border:1px solid #f0a500;color:#7a4a00;padding:10px 16px;border-radius:6px;margin:14px 0;font-size:13px}
    h2{font-size:18px;font-weight:600;color:var(--ink);margin:36px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line)}
    h2 .num{color:var(--ink3);font-weight:400;font-size:14px;margin-left:8px}
    .explainer{background:#f0f5ff;border-left:3px solid var(--accent);padding:10px 14px;border-radius:0 6px 6px 0;font-size:12.5px;color:var(--ink2);margin-bottom:14px;line-height:1.6}
    .explainer b{color:var(--accent)}
    details.legend{background:#fff;border:1px solid var(--line);border-radius:8px;padding:0;margin-bottom:28px;box-shadow:0 1px 3px rgba(0,0,0,0.04)}
    details.legend summary{padding:14px 18px;cursor:pointer;font-weight:600;color:var(--ink);font-size:15px;list-style:none;display:flex;justify-content:space-between;align-items:center}
    details.legend summary::after{content:'▼';color:var(--ink3);font-size:11px;transition:transform 0.2s}
    details.legend[open] summary::after{transform:rotate(180deg)}
    details.legend summary:hover{background:#f7f8fa}
    .legend-body{padding:0 18px 18px;border-top:1px solid var(--line)}
    .legend-body h4{font-size:13px;color:var(--ink);text-transform:uppercase;letter-spacing:0.5px;margin:18px 0 8px}
    .legend-body .item{margin:6px 0;font-size:13px;color:var(--ink2);padding-left:14px;border-left:2px solid var(--line)}
    .legend-body .item b{color:var(--ink)}
    .legend-body .item .name{display:inline-block;min-width:170px;color:var(--ink);font-weight:600}
    .promise{background:linear-gradient(135deg,#fff,#f7f9fb);border:1px solid var(--line);border-radius:10px;padding:22px;display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:22px;margin-bottom:18px;box-shadow:0 2px 6px rgba(0,0,0,0.04)}
    .promise .light{font-size:68px;line-height:1}
    .promise h3{margin:0 0 4px;font-size:18px;color:var(--ink)}
    .promise .lab{color:var(--ink);font-weight:500;font-size:14px}
    .promise .score-big{font-size:48px;font-weight:700;color:var(--accent);font-family:'JetBrains Mono',monospace;line-height:1}
    .promise .score-big small{display:block;font-size:11px;font-weight:400;color:var(--ink3);text-align:right;font-family:'Inter',sans-serif;margin-top:4px}
    .signals{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:14px}
    .signal{background:#fff;border:1px solid var(--line);padding:12px;border-radius:6px}
    .signal .lab{font-size:10.5px;color:var(--ink2);text-transform:uppercase;letter-spacing:0.5px;font-weight:500}
    .signal .v{font-size:22px;font-weight:600;color:var(--ink);margin-top:4px;font-family:'JetBrains Mono',monospace}
    .signal .bar{height:5px;background:var(--line);border-radius:3px;margin-top:8px;overflow:hidden}
    .signal .bar > div{height:100%;background:linear-gradient(90deg,var(--accent),var(--good));transition:width .4s}
    .signal .sub{font-size:10px;color:var(--ink3);margin-top:6px;line-height:1.4}
    .summary-box{background:#f7f8fa;border:1px solid var(--line);border-radius:6px;padding:12px 16px;font-family:'JetBrains Mono',monospace;font-size:12.5px;color:var(--ink);white-space:pre-wrap;line-height:1.7;margin-top:14px}
    .drives{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 20px}
    .drive{background:#fff;border:1px solid var(--line);padding:8px 14px;border-radius:6px;font-size:12px;display:flex;flex-direction:column;border-left:3px solid var(--ink3)}
    .drive[data-p=explorador]{border-left-color:#a855f7}
    .drive[data-p=conservador]{border-left-color:#3b82f6}
    .drive[data-p=engenheiro]{border-left-color:#f59e0b}
    .drive[data-p=seguranca]{border-left-color:#10b981}
    .drive[data-p=sintetizador]{border-left-color:#ec4899}
    .drive .n{font-size:10px;color:var(--ink2);text-transform:uppercase;letter-spacing:0.4px}
    .drive .v{font-size:16px;font-weight:600;color:var(--ink);font-family:'JetBrains Mono',monospace}
    .card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px 20px;margin-bottom:14px;border-left:4px solid var(--ink3);box-shadow:0 1px 3px rgba(0,0,0,0.04);page-break-inside:avoid}
    .card.explorador{border-left-color:#a855f7}
    .card.conservador{border-left-color:#3b82f6}
    .card.engenheiro{border-left-color:#f59e0b}
    .card.seguranca{border-left-color:#10b981}
    .card.sintetizador{border-left-color:#ec4899}
    .card.composto{border-left-color:#eab308;border-left-width:5px}
    .card.composto_rev{border-left-color:#a855f7;border-left-width:5px;border-left-style:dashed}
    .card .head{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:8px;flex-wrap:wrap}
    .card .num{color:var(--ink3);font-weight:600;font-size:13px;font-family:'JetBrains Mono',monospace}
    .card h3{margin:0;font-size:16px;font-weight:600;color:var(--ink);flex:1}
    .badges{display:flex;gap:6px;flex-wrap:wrap}
    .badge{font-size:10.5px;padding:3px 9px;border-radius:11px;background:#f0f2f5;color:var(--ink2);font-weight:500;border:1px solid var(--line);text-transform:uppercase;letter-spacing:0.3px}
    .badge.score{background:#ecfdf5;color:#065f46;border-color:#a7f3d0}
    .badge.elo{background:#eff6ff;color:#1e40af;border-color:#bfdbfe}
    .badge.persona{background:#fffbeb;color:#854d0e;border-color:#fde68a}
    .field{margin:8px 0;font-size:13.5px;color:var(--ink);line-height:1.6}
    .field .label{font-weight:600;color:var(--ink);font-size:12px;text-transform:uppercase;letter-spacing:0.4px;margin-bottom:2px;color:var(--ink2)}
    .pmids{font-family:'JetBrains Mono',monospace;font-size:11.5px;color:var(--ink2)}
    .attrs{display:grid;grid-template-columns:repeat(9,1fr);gap:6px;margin-top:14px;padding-top:12px;border-top:1px dashed var(--line)}
    .attr{text-align:center}
    .attr .nm{font-size:9.5px;color:var(--ink2);text-transform:uppercase;letter-spacing:0.3px;font-weight:500}
    .attr .bar{height:6px;background:var(--line);border-radius:3px;margin-top:4px;overflow:hidden}
    .attr .bar > div{height:100%;background:linear-gradient(90deg,#3b82f6,#10b981)}
    .attr.cost .bar > div{background:linear-gradient(90deg,#10b981,#ef4444)}
    .attr .pct{font-size:9.5px;color:var(--ink3);margin-top:2px;font-family:'JetBrains Mono',monospace}
    .deep{background:#fafbfc;border:1px solid var(--line);border-radius:8px;padding:16px 20px;margin-bottom:16px}
    .deep h4{margin:0 0 10px;color:var(--accent2);font-size:15px}
    .deep .pre-wrap{background:#fff;border:1px solid var(--line);border-radius:6px;padding:12px 14px;font-family:'JetBrains Mono',monospace;font-size:11.5px;color:var(--ink);white-space:pre-wrap;line-height:1.6;max-height:340px;overflow:auto;margin-top:6px}
    .meta-review{background:#fff;border:1px solid var(--line);border-radius:8px;padding:18px 22px;line-height:1.7;font-size:14px;color:var(--ink);white-space:pre-wrap}
    .disclaimer{margin-top:36px;background:#fff8e1;border-left:4px solid #f0a500;color:#5a4400;padding:14px 18px;border-radius:0 6px 6px 0;font-size:12.5px;line-height:1.6}
    .disclaimer b{color:#7a5500}
    footer{margin-top:32px;padding-top:16px;border-top:1px solid var(--line);text-align:center;color:var(--ink3);font-size:11px}
    @media print{
      body{background:#fff}.wrap{padding:18px;max-width:none}
      details.legend[open] summary::after{display:none}
      .card,.deep,.meta-review,.promise,.signal{box-shadow:none!important;break-inside:avoid}
      header.report{border-bottom-color:#000}
      .disclaimer{background:#fff;border-color:#999;color:#000}
    }
    """
    # ----- HTML BUILD -----
    parts = [f"<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><title>{e(disease)} — Relatório NovaMind-Sci</title><style>{css}</style></head><body><div class='wrap'>"]
    # HEADER
    parts.append(f"<header class='report'><div class='tag'>NovaMind-Sci · Relatório de Hipóteses</div>")
    parts.append(f"<h1>{e(disease.title())}</h1>")
    parts.append(f"<div class='meta'>Sistema: <b>{e(system)}</b> · {len(hyps)} hipóteses geradas · Duração: {round(d.get('duration_s',0))}s</div></header>")

    # INTRO em portugues simples
    parts.append("""<div class='intro'><strong>O que é este relatório?</strong> O sistema NovaMind-Sci roda múltiplos agentes de IA com personalidades diferentes (cada um com viés próprio), faz eles proporem tratamentos para a doença pesquisada, criticarem-se mutuamente num torneio, e combinarem ideias em sínteses ordenadas. Ao final, um "Indicador de Promessa" diz, em linguagem de leigo, se vale levar pra laboratório. <strong>Nada aqui substitui validação experimental real</strong> — são <em>hipóteses qualificadas</em>, não tratamentos.</div>""")

    if partial:
        parts.append(f"<div class='partial'>⏸ <b>Esta execução foi interrompida</b> antes de terminar (parou em <code>{e(str(partial))}</code>). Os dados abaixo são o que foi gerado até esse ponto.</div>")

    # LEGENDA — expandida por padrao
    parts.append("<details class='legend' open><summary>📖 Legenda — o que significa cada coisa</summary><div class='legend-body'>")
    parts.append("<h4>🧬 Personas (cores das hipóteses)</h4>")
    parts.append("<p style='font-size:12px;color:var(--ink2);margin:0 0 8px'>Cada hipótese foi gerada por um \"agente\" com viés próprio — a cor da borda esquerda indica qual:</p>")
    for k,(name,desc) in PERSONA_INFO.items():
        parts.append(f"<div class='item'><span class='name'>{e(name)}</span> — {e(desc)}</div>")
    parts.append("<h4>🎯 Atributos (as 9 barrinhas em cada hipótese)</h4>")
    parts.append("<p style='font-size:12px;color:var(--ink2);margin:0 0 8px'>Cada hipótese é avaliada num vetor de 9 atributos (barra cheia = 1.0):</p>")
    for k,(name,desc) in ATTR_INFO.items():
        parts.append(f"<div class='item'><span class='name'>{e(name)}</span> — {e(desc)}</div>")
    parts.append("<h4>📊 Score e ELO (cabeçalho da hipótese)</h4>")
    parts.append("<div class='item'><span class='name'>Score</span> — combinação ponderada dos 9 atributos. Vai de 0 a 1. Acima de 0.7 = forte.</div>")
    parts.append("<div class='item'><span class='name'>ELO</span> — ranking de torneio (como em xadrez). 1200 = neutro. Acima = venceu mais duelos com outras hipóteses.</div>")
    parts.append("<h4>🚦 Indicador de Promessa (semáforo)</h4>")
    parts.append("<div class='item'><span class='name'>🟢 ≥ 0.65</span> — Alta promessa. Vale considerar wet-lab.</div>")
    parts.append("<div class='item'><span class='name'>🟡 0.40–0.65</span> — Média. Vale revisão manual antes de seguir.</div>")
    parts.append("<div class='item'><span class='name'>🔴 < 0.40</span> — Baixa. Provavelmente é ruído.</div>")
    parts.append("</div></details>")

    # PROMISE PANEL
    if promise:
        sig = promise.get("signals",{})
        parts.append("<h2>🚦 Indicador de Promessa <span class='num'>— veredito para leigo</span></h2>")
        parts.append("<div class='explainer'>O <b>Indicador de Promessa</b> combina 6 sinais objetivos. Cada sinal pode ser conferido independentemente. Score alto = sistema recuperou direções reais de pesquisa, várias personas concordaram, e há novidade verificável.</div>")
        parts.append(f"<div class='promise'><div class='light'>{e(promise.get('emoji','🟡'))}</div>")
        parts.append(f"<div><h3>Promessa: {e(promise.get('label','—'))}</h3>")
        parts.append(f"<div class='lab'>Combinação dos 6 sinais abaixo, peso por relevância.</div></div>")
        parts.append(f"<div><div class='score-big'>{promise.get('promise',0):.2f}<small>de 0 a 1</small></div></div></div>")
        sig_meta = [
            ("Recuperação de lead","lead_recovery_rate","%","% das top-K com tratamento já em literatura clínica — 'puxou da realidade'"),
            ("Convergência","convergence","%","% das top-K mirando o mesmo alvo molecular — 'várias rotas mesma ideia'"),
            ("Concordância (juiz)","judge_top3_avg","%","Média do juiz independente sobre top-3 — 'segunda opinião concorda'"),
            ("Ganho composicional","composition_gain_max","","|A→B − B→A| — assimetria causal descoberta"),
            ("Diversidade","diversity","","vias mecanísticas distintas no top-K (nem mono nem disperso)"),
            ("Novidade real","novelty_real","","top-1 NÃO é o primeiro hit óbvio do PubMed — não é cópia"),
        ]
        parts.append("<div class='signals'>")
        for lbl,key,unit,sub in sig_meta:
            v = sig.get(key,0); display = f"{(v*100):.0f}%" if unit=='%' else f"{v:.2f}"
            parts.append(f"<div class='signal'><div class='lab'>{e(lbl)}</div><div class='v'>{e(display)}</div><div class='bar'><div style='width:{v*100:.0f}%'></div></div><div class='sub'>{e(sub)}</div></div>")
        parts.append("</div>")
        if promise.get("summary"): parts.append(f"<div class='summary-box'>{e(promise['summary'])}</div>")

    # DRIVES
    if drives:
        parts.append("<h2>🧬 Drives das Personas <span class='num'>— quem ganhou peso ao longo da execução</span></h2>")
        parts.append("<div class='explainer'>Cada persona começou com peso 1.0. À medida que produzia hipóteses com score alto, ganhava peso (e era amostrada mais vezes). À medida que produzia hipóteses ruins, perdia peso. <b>É seleção darwiniana das personas</b>: o sistema descobre sozinho qual viés serve melhor para a doença em questão.</div>")
        parts.append("<div class='drives'>")
        for k,v in sorted(drives.items(), key=lambda x:-x[1]):
            parts.append(f"<div class='drive' data-p='{e(k)}'><span class='n'>{e(k)}</span><span class='v'>{v:.2f}</span></div>")
        parts.append("</div>")

    # HIPOTESES
    parts.append(f"<h2>💡 Hipóteses <span class='num'>— {len(hyps)} ranqueadas</span></h2>")
    parts.append("<div class='explainer'>Cada cartão é uma hipótese de tratamento gerada pelo sistema. A <b>cor da borda esquerda</b> indica qual persona a gerou. Os 9 atributos no rodapé são a avaliação multidimensional. <b>Hipóteses 'composto' / 'composto_rev'</b> são sínteses ordenadas de duas hipóteses anteriores — a diferença de score entre A→B e B→A revela se a ordem importa (causalidade descoberta).</div>")
    for i,h in enumerate(hyps,1):
        persona = personas.get(h.get("id","")) or ""
        vec = vectors.get(h.get("id","")) or {a: h.get(a,0) for a in ATTRS}
        score = h.get("overall",0); elo = h.get("elo",1200)
        cls = persona or ""
        parts.append(f"<div class='card {e(cls)}'>")
        parts.append(f"<div class='head'><div style='display:flex;gap:10px;align-items:baseline;flex:1'><span class='num'>#{i}</span><h3>{e(h.get('title','(sem titulo)'))}</h3></div>")
        parts.append("<div class='badges'>")
        if persona: parts.append(f"<span class='badge persona'>{e(persona)}</span>")
        parts.append(f"<span class='badge score'>score {score:.2f}</span>")
        if elo and abs(elo-1200) > 1: parts.append(f"<span class='badge elo'>elo {elo:.0f}</span>")
        parts.append("</div></div>")
        if h.get("statement"):
            parts.append(f"<div class='field'>{e(h['statement'])}</div>")
        if h.get("mechanism"):
            parts.append(f"<div class='field'><div class='label'>Mecanismo molecular</div>{e(h['mechanism'])}</div>")
        if h.get("proposed_intervention"):
            parts.append(f"<div class='field'><div class='label'>Intervenção proposta</div>{e(h['proposed_intervention'])}</div>")
        if h.get("rationale"):
            parts.append(f"<div class='field'><div class='label'>Racional</div>{e(h['rationale'][:1500])}</div>")
        if h.get("evidence_pmids"):
            parts.append(f"<div class='field'><div class='label'>Evidência — PubMed IDs</div><span class='pmids'>{e(', '.join(h['evidence_pmids'][:10]))}</span></div>")
        if h.get("critiques"):
            for c in h["critiques"][-2:]:
                if c and not str(c).startswith("persona:"):
                    parts.append(f"<div class='field'><div class='label'>Crítica recebida</div>{e(str(c))}</div>")
        # vetor com numeros
        parts.append("<div class='attrs'>")
        short_names = {"novelty":"Novid.","plausibility":"Plaus.","tractability":"Pratic.","safety":"Seg.","evidence_strength":"Evid.","orthogonality":"Ortog.","cost":"Custo","clinical_impact":"Impac.","synergy":"Sinerg."}
        for a in ATTRS:
            v = vec.get(a,0) or 0; cls = " cost" if a=="cost" else ""
            parts.append(f"<div class='attr{cls}' title='{e(ATTR_INFO[a][1])}'><div class='nm'>{e(short_names[a])}</div><div class='bar'><div style='width:{v*100:.0f}%'></div></div><div class='pct'>{v:.2f}</div></div>")
        parts.append("</div></div>")

    # DEEP-DIVE
    if deep:
        parts.append(f"<h2>🔬 Deep-Dive (Robin) <span class='num'>— análise profunda das {len(deep)} mais promissoras</span></h2>")
        parts.append("<div class='explainer'>Para as top hipóteses do torneio, o agente <b>Falcon</b> mergulha na literatura específica do alvo molecular, o <b>Finch</b> propõe um plano concreto de análise de dados (dataset GEO, pipeline RNA-seq, etc.), e o agente de código gera o script Python executável. É a parte do sistema inspirada no Robin do FutureHouse.</div>")
        for x in deep:
            hid = x.get("hypothesis_id","?")
            target = next((h for h in hyps if h.get("id")==hid), {})
            parts.append(f"<div class='deep'><h4>{e(target.get('title','(?)'))}</h4>")
            if x.get("falcon"): parts.append(f"<div class='field'><div class='label'>Falcon — Revisão profunda da literatura</div><div class='pre-wrap'>{e(x['falcon'][:6000])}</div></div>")
            if x.get("finch"):  parts.append(f"<div class='field'><div class='label'>Finch — Plano de análise de dados</div><div class='pre-wrap'>{e(x['finch'][:6000])}</div></div>")
            if x.get("code"):   parts.append(f"<div class='field'><div class='label'>Código Python gerado (executável em ambiente apropriado)</div><div class='pre-wrap'>{e(x['code'][:10000])}</div></div>")
            parts.append("</div>")

    # META-REVIEW
    if review:
        parts.append("<h2>🧠 Meta-Review <span class='num'>— síntese final do sistema</span></h2>")
        parts.append("<div class='explainer'>O agente Meta-Review olha as top hipóteses do torneio e produz uma síntese executiva: quais alvos convergiram, qual é a top-1, qual experimento próximo testá-la, e sinais de alerta.</div>")
        parts.append(f"<div class='meta-review'>{e(review)}</div>")

    # DISCLAIMER
    parts.append("<div class='disclaimer'>⚠️ <b>Aviso honesto e necessário:</b> nenhum sistema de inteligência artificial pode afirmar 'isto cura uma doença' apenas analisando literatura. Este relatório combina <em>proxies</em> validáveis: recuperação de leads citados em ensaios reais, convergência entre personas independentes, ganho composicional (descoberta de assimetria causal), concordância de juiz com prompt diferente, e novidade contextual. <b>Promessa alta significa que vale a pena alguém de verdade investigar — não significa que é cura.</b> O caminho honesto para validar qualquer hipótese aqui é: literatura focada → ensaios pré-clínicos → ensaios clínicos.</div>")
    parts.append(f"<footer>Gerado por NovaMind-Sci · {len(hyps)} hipóteses · sistema <b>{e(system)}</b></footer>")
    parts.append("</div></body></html>")
    return "".join(parts)

def _export_md(d: dict) -> str:
    lines = []
    lines.append(f"# {d.get('disease','?')}")
    lines.append(f"_Sistema: **{d.get('system','?')}** · Hipóteses: {len(d.get('hypotheses',[]))} · Duração: {round(d.get('duration_s',0))}s_")
    if d.get("__partial__"): lines.append(f"\n> ⏸ Run parcial (interrompida em `{d['__partial__']}`).\n")
    p = d.get("promise") or {}
    if p:
        lines.append(f"\n## {p.get('emoji','🟡')} Promessa: {p.get('label','—')} ({p.get('promise',0):.2f})\n")
        if p.get("summary"): lines.append("```\n"+p["summary"]+"\n```")
    if d.get("drives_final"):
        lines.append("\n## Drives finais")
        for k,v in d["drives_final"].items(): lines.append(f"- **{k}**: {v:.2f}")
    lines.append(f"\n## Hipóteses ({len(d.get('hypotheses',[]))})")
    personas = d.get("personas",{}) or {}
    for i,h in enumerate(d.get("hypotheses",[]),1):
        persona = personas.get(h.get("id","")) or ""
        score = h.get("overall",0); elo = h.get("elo",1200)
        lines.append(f"\n### {i}. {h.get('title','(sem titulo)')}")
        lines.append(f"_persona: **{persona}** · score: **{score:.2f}** · elo: **{elo:.0f}**_\n")
        if h.get("statement"): lines.append(h["statement"])
        if h.get("mechanism"): lines.append(f"\n**Mecanismo**: {h['mechanism']}")
        if h.get("proposed_intervention"): lines.append(f"\n**Intervenção**: {h['proposed_intervention']}")
        if h.get("rationale"): lines.append(f"\n**Racional**: {h['rationale'][:1500]}")
        if h.get("evidence_pmids"): lines.append(f"\n**PMIDs**: {', '.join(h['evidence_pmids'][:8])}")
    if d.get("meta_review"): lines.append(f"\n## Meta-Review\n\n{d['meta_review']}")
    lines.append("\n---\n_⚠️ Indicadores são proxies. Não substituem validação experimental._")
    return "\n".join(lines)

@app.get("/api/export/{name}.html", response_class=None)
def export_html(name: str):
    p = RUNS_DIR / f"{name}.json"
    if not p.exists(): return JSONResponse({"error":"nao encontrado"}, status_code=404)
    d = json.loads(p.read_text(encoding="utf-8"))
    from fastapi.responses import HTMLResponse, Response
    return HTMLResponse(_export_html(d), headers={"Content-Disposition": f'inline; filename="{name}.html"'})

@app.get("/api/export/{name}.md")
def export_md(name: str):
    p = RUNS_DIR / f"{name}.json"
    if not p.exists(): return JSONResponse({"error":"nao encontrado"}, status_code=404)
    d = json.loads(p.read_text(encoding="utf-8"))
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(_export_md(d), headers={"Content-Disposition": f'attachment; filename="{name}.md"'})

@app.get("/api/export/{name}.json")
def export_json(name: str):
    from fastapi.responses import FileResponse
    p = RUNS_DIR / f"{name}.json"
    if not p.exists(): return JSONResponse({"error":"nao encontrado"}, status_code=404)
    return FileResponse(str(p), media_type="application/json", filename=f"{name}.json")

# ---- frontend estatico ----
FRONT = ROOT / "frontend"
@app.get("/")
def root(): return FileResponse(FRONT / "index.html")
@app.get("/app.js")
def js(): return FileResponse(FRONT / "app.js", media_type="application/javascript")
@app.get("/style.css")
def css(): return FileResponse(FRONT / "style.css", media_type="text/css")

if __name__ == "__main__":
    import uvicorn
    print("\n=== NovaMind-Sci Server ===\nVENICE_API_KEY:", "SET" if os.environ.get('VENICE_API_KEY') or os.environ.get('VENICE_ADMIN_KEY') else "MISSING")
    print("LM Studio Nemotron fallback: http://localhost:1234/v1")
    print("Abra:  http://localhost:8765\n")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
