let pollTimer = null;

document.addEventListener('DOMContentLoaded', () => {
  const input = document.getElementById('inputText');
  const btn = document.getElementById('btnSend');
  const chat = document.getElementById('chatMessages');

  function send() {
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    addMessage('user', text);
    btn.disabled = true;
    btn.textContent = '...';

    fetch('/api/think', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text}),
    })
    .then(r => r.json())
    .then(data => {
      if (data.error) return addMessage('nova', `Erro: ${data.error}`);
      let html = data.response;
      if (data.contradiction) {
        html = `<span class="contradiction">${html}</span>`;
      } else if (data.inferences && data.inferences.length) {
        html += ` <span class="inference">⟹ ${data.inferences[0]}</span>`;
      } else if (data.goal_satisfied) {
        html = `<span class="goal">${html}</span>`;
      }
      const traceParts = [];
      if (data.trace.concepts && data.trace.concepts.length) {
        traceParts.push(`conceitos=[${data.trace.concepts.slice(0,6).join(', ')}]`);
      }
      traceParts.push(`confiança=${data.trace.confidence}`);
      traceParts.push(data.trace.novel ? 'NOVO' : 'conhecido');
      if (data.trace.recall_count > 0) {
        traceParts.push(`recall=${data.trace.recall_count} ep(s)`);
      }
      if (data.cycle_time_ms) {
        traceParts.push(`${data.cycle_time_ms.toFixed(0)}ms`);
      }
      addMessage('nova', html, traceParts.join(' | '));
      updateState();
    })
    .catch(err => addMessage('nova', `Erro: ${err.message}`))
    .finally(() => { btn.disabled = false; btn.textContent = 'Enviar'; input.focus(); });
  }

  btn.addEventListener('click', send);
  input.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });
  input.focus();

  // Initial state
  updateState();
  // Poll state every 3s
  pollTimer = setInterval(updateState, 3000);
});

function addMessage(type, text, trace) {
  const chat = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.className = `msg ${type}`;
  div.innerHTML = text;
  if (trace) {
    const t = document.createElement('div');
    t.className = 'trace';
    t.textContent = trace;
    div.appendChild(t);
  }
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function updateState() {
  fetch('/api/state')
    .then(r => r.json())
    .then(s => {
      // Header
      document.getElementById('statCycle').textContent = `Cycle: ${s.cycle}`;
      document.getElementById('statUptime').textContent = `Uptime: ${Math.floor(s.uptime)}s`;
      document.getElementById('statVRAM').textContent = `VRAM: ${s.system.vram_mb.toFixed(0)}MB / ${s.system.vram_total}MB`;

      // Metacognition
      const m = s.metacognition;
      setGauge('gConfidence', 'vConfidence', m.confidence);
      setGauge('gCoherence', 'vCoherence', m.coherence);
      setGauge('gCuriosity', 'vCuriosity', m.curiosity);
      setGauge('gConfusion', 'vConfusion', m.confusion);
      setGauge('gSurprise', 'vSurprise', m.surprise);
      setGauge('gCertainty', 'vCertainty', m.certainty);

      // Goals
      document.getElementById('goalsText').textContent = `${s.goals.active} ativos, ${s.goals.total} total`;
      document.getElementById('unansweredText').textContent = `${s.goals.unanswered} perguntas sem resposta`;

      // World Model
      document.getElementById('wmBadge').textContent = `${s.world_model.nodes} nós, ${s.world_model.edges} arestas`;
      const chainsDiv = document.getElementById('chainsList');
      if (s.world_model.chains && s.world_model.chains.length) {
        chainsDiv.innerHTML = s.world_model.chains.map(c =>
          `<div class="chain">${c.map((n, i) => `<span class="node">${n}</span>${i < c.length - 1 ? ' <span class="arrow">→</span> ' : ''}`).join('')}</div>`
        ).join('');
      } else {
        chainsDiv.innerHTML = '<em>Nenhuma cadeia causal ainda.</em>';
      }

      // Memory
      document.getElementById('memBadge').textContent = `${s.memory.episodic_count} episódios`;
      const epDiv = document.getElementById('episodesList');
      if (s.memory.recent_episodes && s.memory.recent_episodes.length) {
        epDiv.innerHTML = s.memory.recent_episodes.map(ep =>
          `<div class="ep-item">${ep.input}</div>`
        ).join('');
      } else {
        epDiv.innerHTML = '<em>Nenhum episódio ainda.</em>';
      }

      // Semantic Rules
      document.getElementById('semBadge').textContent = `${s.semantic_rules.length}`;
      const rulesDiv = document.getElementById('rulesList');
      if (s.semantic_rules && s.semantic_rules.length) {
        rulesDiv.innerHTML = s.semantic_rules.map(r =>
          `<div class="rule-item"><span class="rel">${r.relation}</span> ${r.display} <span class="src">${r.source}</span></div>`
        ).join('');
      } else {
        rulesDiv.innerHTML = '<em>Nenhuma regra ainda.</em>';
      }
    });
}

function setGauge(fillId, textId, value) {
  const pct = Math.min(100, Math.max(0, Math.round(value * 100)));
  const fill = document.getElementById(fillId);
  const text = document.getElementById(textId);
  fill.style.width = `${pct}%`;
  text.textContent = `${pct}%`;
  // Color gradient
  if (pct > 70) fill.style.background = '#3fb950';
  else if (pct > 40) fill.style.background = '#d29922';
  else fill.style.background = '#f85149';
}
