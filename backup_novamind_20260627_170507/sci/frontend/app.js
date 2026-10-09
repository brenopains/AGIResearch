// NovaMind-Sci frontend client
const $ = sel => document.querySelector(sel);
const $$ = sel => Array.from(document.querySelectorAll(sel));

let currentEs = null;
let currentRunId = null;

const ATTRS = ["novelty","plausibility","tractability","safety","evidence_strength","orthogonality","cost","clinical_impact","synergy"];
const ATTR_SHORT = {novelty:"nov",plausibility:"plau",tractability:"trat",safety:"saf",evidence_strength:"evid",orthogonality:"ort",cost:"$",clinical_impact:"imp",synergy:"syn"};

const PHASE_LABELS = {
  generation:"Gerar", reflection:"Criticar", ranking:"Ranquear", proximity:"Dedup",
  evolution:"Evoluir", meta_review:"Meta-Review",
  literature:"Literatura", crow:"Crow", falcon:"Falcon", finch:"Finch", data_analyst:"Codigo",
  generation_agi:"Personas-AGI", reflection_tournament:"Torneio", deep_dive:"Deep-Dive", promise:"Promessa"
};

function setConn(state){
  const dot=$('#conn-dot'), txt=$('#conn-text');
  dot.className='dot '+(state==='on'?'dot-on':state==='err'?'dot-err':'dot-off');
  txt.textContent={on:'rodando',err:'erro',off:'offline'}[state]||state;
}

function paneInit(side, title){
  const pane = $(`[data-side="${side}"]`);
  pane.style.display='';
  $(`#pane-${side}-title`).textContent = title;
  $(`#cards-${side}`).innerHTML='';
  $(`#log-${side}`).innerHTML='';
  $(`#hcount-${side}`).textContent='0';
  $(`#evcount-${side}`).textContent='0';
  $(`#phase-${side}`).innerHTML='';
  const drives = $(`#drives-${side}`); if(drives) drives.innerHTML='';
}

function logLine(side, type, payload){
  const log = $(`#log-${side}`);
  const ev = document.createElement('div');
  ev.className='ev '+(type==='error'?'error':type==='warn'?'warn':'');
  const t = new Date().toLocaleTimeString();
  let body = '';
  if(payload){
    if(payload.line) body = ' '+payload.line.replace(/\n/g,' ').substring(0,140);
    else body = ' '+JSON.stringify(payload).substring(0,120);
  }
  ev.innerHTML = `<span class="et">[${type}]</span><span class="ts" style="color:#444"> ${t}</span>${body}`;
  log.appendChild(ev);
  log.scrollTop = log.scrollHeight;
  const c = $(`#evcount-${side}`); c.textContent = parseInt(c.textContent||0)+1;
}

function attrsHtml(vec){
  if(!vec) return '';
  let html='<div class="attrs">';
  for(const a of ATTRS){
    const v = (vec[a]||0);
    html += `<div class="attr ${a==='cost'?'cost':''}"><span>${ATTR_SHORT[a]}</span><div class="bar"><div style="width:${(v*100).toFixed(0)}%"></div></div></div>`;
  }
  html+='</div>';
  return html;
}

function renderCard(side, h){
  const cards = $(`#cards-${side}`);
  let el = cards.querySelector(`[data-hid="${h.id}"]`);
  const exists = !!el;
  if(!el){
    el = document.createElement('div');
    el.className='card new';
    el.dataset.hid = h.id;
    cards.prepend(el);
    setTimeout(()=>el.classList.remove('new'), 400);
  }
  if(h.persona) el.dataset.persona = h.persona;
  const persona = h.persona || (h.critiques||[]).find(c=>String(c).startsWith('persona:'))?.replace('persona:','') || '';
  const score = h.overall ?? h.default_score ?? 0;
  const elo = h.elo ?? 1200;
  const pmidStr = (h.evidence_pmids||[]).slice(0,5).join(', ') || '—';
  const vec = h.vector || {novelty:h.novelty,plausibility:h.plausibility,tractability:h.tractability,safety:h.safety,evidence_strength:h.evidence_strength,clinical_impact:h.clinical_impact,synergy:h.synergy,cost:h.cost,orthogonality:h.orthogonality};
  el.innerHTML = `
    <div class="h-title">
      <div>${escapeHtml(h.title||'(sem titulo)')}</div>
      <div class="badges">
        ${persona?`<span class="badge persona">${persona}</span>`:''}
        <span class="badge score">score ${score.toFixed(2)}</span>
        ${elo!==1200?`<span class="badge elo">elo ${elo.toFixed(0)}</span>`:''}
      </div>
    </div>
    <div class="h-statement">${escapeHtml(h.statement||'')}</div>
    <div class="h-meta">
      <span><b>intervenção:</b> ${escapeHtml((h.proposed_intervention||'').substring(0,140))}</span>
    </div>
    ${attrsHtml(vec)}
    <span class="expand">+ detalhes</span>
    <div class="expanded">
      <div class="label">Mecanismo</div><div>${escapeHtml(h.mechanism||'—')}</div>
      <div class="label">Racional</div><div>${escapeHtml((h.rationale||'').substring(0,800))}</div>
      <div class="label">PMIDs</div><div>${pmidStr}</div>
      ${(h.critiques&&h.critiques.length)?`<div class="label">Crítica</div><div>${escapeHtml(h.critiques.slice(-1)[0])}</div>`:''}
    </div>
  `;
  el.querySelector('.expand').onclick = e=>{
    const exp = el.querySelector('.expanded');
    exp.classList.toggle('show');
    e.target.textContent = exp.classList.contains('show')?'– fechar':'+ detalhes';
  };
  if(!exists){
    const c = $(`#hcount-${side}`); c.textContent = cards.children.length;
  }
}

function escapeHtml(s){if(s==null)return '';return String(s).replace(/[&<>"']/g, ch=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[ch]))}

function setPhase(side, phase){
  const strip = $(`#phase-${side}`);
  let steps = strip.dataset.steps;
  if(!steps){
    // monta tracks com base no sistema
    const sys = strip.dataset.system||'co_scientist';
    if(sys==='novamind_sci') steps='literature,generation,evolution,meta_review';
    else if(sys==='robin') steps='crow,falcon,finch,data_analyst';
    else if(sys==='unified') steps='literature,generation_agi,reflection_tournament,deep_dive,meta_review,promise';
    else steps='generation,reflection,ranking,proximity,evolution,meta_review';
    strip.dataset.steps = steps;
    strip.innerHTML = steps.split(',').map(s=>`<span class="step" data-phase="${s}">${PHASE_LABELS[s]||s}</span>`).join('');
  }
  strip.querySelectorAll('.step').forEach(el=>{
    el.classList.toggle('on', el.dataset.phase===phase);
  });
}

function renderDrives(side, drives){
  const wrap = $(`#drives-${side}`); if(!wrap) return;
  wrap.innerHTML='';
  for(const [p,v] of Object.entries(drives)){
    const d = document.createElement('div');
    d.className='drive'; d.dataset.p=p;
    d.innerHTML = `<span class="name">${p}</span><span class="val">${v.toFixed(2)}</span>`;
    wrap.appendChild(d);
  }
}

function handleEvent(ev, side){
  const type = ev.type;
  const payload = ev.payload;
  if(type==='ping') return;
  if(type!=='log') logLine(side, type, payload);
  switch(type){
    case 'start':
      $(`#phase-${side}`).dataset.system = payload.system;
      $(`#phase-${side}`).dataset.steps = '';
      break;
    case 'phase':
      setPhase(side, payload.name);
      break;
    case 'hypothesis_new':
    case 'hypothesis_update':
      renderCard(side, payload);
      break;
    case 'drives':
      renderDrives(side, payload.drives);
      break;
    case 'asymmetry':
      logLine(side, 'asymmetry', {line: `A>>B=${payload.a_to_b.toFixed(2)} vs B>>A=${payload.b_to_a.toFixed(2)} Δ=${payload.delta.toFixed(2)}`});
      break;
    case 'meta_review':
      $('#meta-review-panel').style.display='';
      $('#meta-review-content').textContent = payload.markdown;
      break;
    case 'promise':
      renderPromise(payload);
      break;
    case 'neuromod':
      renderNeuromod(side, payload);
      break;
    case 'log':
      logLine(side, 'log', payload);
      break;
    case 'saved':
      logLine(side, 'saved', {line: 'salvo: '+payload.name});
      loadRuns();
      break;
    case 'error':
      logLine(side, 'error', payload);
      setConn('err');
      break;
  }
}

function renderPromise(p){
  $('#promise-panel').style.display='';
  $('#promise-light').textContent = p.emoji || '🟡';
  $('#promise-label').textContent = p.label || '—';
  $('#promise-score').textContent = (p.promise!=null) ? p.promise.toFixed(2) : '—';
  const sig = p.signals || {};
  const items = [
    ['Recuperação de lead','lead_recovery_rate','%','dos top citados em literatura clínica'],
    ['Convergência','convergence','%','para o mesmo alvo molecular'],
    ['Concordância (juiz)','judge_top3_avg','%','média do juiz independente'],
    ['Ganho composicional','composition_gain_max','','|A>>B − B>>A|'],
    ['Diversidade','diversity','','vias mecanísticas distintas'],
    ['Novidade real','novelty_real','','não é o 1º hit óbvio do PubMed'],
  ];
  $('#promise-signals').innerHTML = items.map(([lbl,key,unit,sub])=>{
    const v = sig[key] ?? 0;
    const display = unit==='%' ? `${(v*100).toFixed(0)}%` : v.toFixed(2);
    return `<div class="signal"><div class="label">${lbl}</div><div class="value">${display}</div>
      <div class="bar"><div style="width:${(v*100).toFixed(0)}%"></div></div>
      <div class="label" style="margin-top:4px">${sub}</div></div>`;
  }).join('');
  $('#promise-summary').textContent = p.summary || '';
  // scroll para ele
  $('#promise-panel').scrollIntoView({behavior:'smooth',block:'nearest'});
}
function renderNeuromod(side, nm){
  // pequena adicao ao topo do log com estado neuromodular global
  const log = $(`#log-${side}`); if(!log) return;
  const ev = document.createElement('div');
  ev.className='ev';
  ev.innerHTML = `<span class="et" style="color:#ffd866">[nm]</span> da=${nm.da} ne=${nm.ne} se=${nm.se}`;
  log.appendChild(ev); log.scrollTop=log.scrollHeight;
}

async function startRun(){
  const disease = $('#disease').value.trim();
  if(!disease){ alert('digite uma doença'); return; }
  const system = document.querySelector('input[name="sys"]:checked').value;
  const params = {
    system, disease,
    n: +$('#p-n').value, rounds: +$('#p-rounds').value, tourn: +$('#p-tourn').value,
    cycles: +$('#p-cycles').value, pop: +$('#p-pop').value,
  };

  // reset UI
  $('#meta-review-panel').style.display='none';
  $('#meta-review-content').textContent='';
  $('#promise-panel').style.display='none';
  $('#panes').classList.toggle('side-by-side', system==='both');
  if(system==='both'){
    paneInit('A','Co-Scientist (Normal)');
    paneInit('B','NovaMind-Sci (AGI)');
  } else {
    paneInit('A', {co_scientist:'Co-Scientist',novamind_sci:'NovaMind-Sci (AGI)',robin:'Robin',unified:'🌟 Fusão (AGI+Co-Sci+Robin)'}[system] || system);
    $('[data-side="B"]').style.display='none';
  }

  $('#run-btn').style.display='none';
  $('#abort-btn').style.display='';
  setConn('on');

  const resp = await fetch('/api/run', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(params)});
  if(!resp.ok){ alert('erro ao iniciar'); resetButtons(); return; }
  const {run_id} = await resp.json();
  currentRunId = run_id;
  if(currentEs){ currentEs.close(); }
  const es = new EventSource(`/api/stream/${run_id}`);
  currentEs = es;
  es.onmessage = e=>{
    try {
      const ev = JSON.parse(e.data);
      if(system==='both'){
        if(ev.type.startsWith('A:')){ ev.type = ev.type.slice(2); handleEvent(ev,'A'); }
        else if(ev.type.startsWith('B:')){ ev.type = ev.type.slice(2); handleEvent(ev,'B'); }
        else { logLine('A', ev.type, ev.payload); logLine('B', ev.type, ev.payload); }
      } else {
        handleEvent(ev, 'A');
      }
    } catch(err){ console.error(err); }
  };
  es.addEventListener('end', e=>{
    try { const d = JSON.parse(e.data); if(d.payload?.reason === 'run_lost_on_restart'){
      logLine('A','warn',{line:'run perdido (servidor reiniciou) — feche e rode de novo.'});
    } } catch(_){}
    setConn('off'); resetButtons();
    es.close();
    currentEs = null;
  });
  es.onerror = ()=>{
    setConn('err');
    // se a stream ja fechou, nao tenta reconectar (browser tenta por padrao em EventSource)
    if(es.readyState === 2){ es.close(); currentEs=null; resetButtons(); setConn('off'); }
  };
}

function resetButtons(){
  $('#run-btn').style.display='';
  $('#abort-btn').style.display='none';
}

async function loadRuns(){
  try {
    const r = await fetch('/api/runs'); if(!r.ok) return;
    const list = await r.json();
    $('#runs-list').innerHTML = list.map(x=>{
      const partial = x.partial ? ` <span style="color:#ffd866">⏸ ${x.partial}</span>` : '';
      return `<li data-name="${x.name}">
        <div class="dx">${escapeHtml(x.disease)}${partial}</div>
        <div class="sx">${x.system} · ${x.n_hypotheses}h · ${Math.round(x.duration_s)}s</div>
        <div class="row-exports">
          <a class="mini-exp" href="/api/export/${x.name}.html" target="_blank" title="abrir relatorio HTML">HTML</a>
          <a class="mini-exp" href="/api/export/${x.name}.md" download title="baixar Markdown">MD</a>
        </div>
      </li>`;
    }).join('');
    $$('#runs-list li').forEach(li=>li.onclick=(ev)=>{
      // ignora clique se foi nos botoes de export
      if(ev.target.classList.contains('mini-exp')) return;
      openHistoryModal(li.dataset.name);
    });
  } catch(e){}
}

async function openHistoryModal(name){
  const r = await fetch('/api/run_detail/'+name);
  if(!r.ok){ alert('falha ao carregar run'); return; }
  const d = await r.json();
  $('#modal-title').textContent = d.disease || name;
  const partial = d.__partial__ ? ` · ⏸ parcial (${d.__partial__})` : '';
  $('#modal-subtitle').textContent = `${d.system||'?'} · ${(d.hypotheses||[]).length} hipóteses · ${Math.round(d.duration_s||0)}s${partial}`;
  $('#export-html').href = `/api/export/${name}.html`;
  $('#export-md').href = `/api/export/${name}.md`;
  $('#export-json').href = `/api/export/${name}.json`;
  $('#modal-frame').src = `/api/export/${name}.html`;
  $('#history-modal').style.display = 'flex';
}
function closeHistoryModal(){ $('#history-modal').style.display='none'; $('#modal-frame').src='about:blank'; }
document.addEventListener('keydown', e=>{ if(e.key==='Escape') closeHistoryModal(); });

$('#run-btn').onclick = startRun;
$('#abort-btn').onclick = ()=>{ if(currentEs){ currentEs.close(); } resetButtons(); setConn('off'); };
$('#modal-close').onclick = closeHistoryModal;
$('#history-modal').onclick = (e)=>{ if(e.target.id==='history-modal') closeHistoryModal(); };
loadRuns();
setInterval(loadRuns, 8000);
