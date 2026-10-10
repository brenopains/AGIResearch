/* Renderer de pontos de superficie em coordenadas 3D. Nao gera geometria. */
(() => {
  'use strict';
  const el = id => document.getElementById(id), cv = el('neuro-canvas');
  const gl = cv.getContext('webgl', {antialias: true, alpha: false});
  let scene = null, selected = null, revision = null, evidence = false, yaw = 0, pitch = 0;
  let camera = [0, 0, 0], unit = 1, count = 0, drag = null, last = performance.now(), loading = false;
  const keys = new Set();
  const escape = t => String(t).replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
  let program, posBuffer, colorBuffer, supportBuffer, triangles = false;
  if (gl) {
    function shader(type, src) {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    }
    program = gl.createProgram();
    gl.attachShader(program, shader(gl.VERTEX_SHADER, `
      attribute vec3 p; attribute vec3 color; attribute float support;
      uniform mat4 view; uniform vec3 eye; uniform vec2 screen; uniform float scale;
      varying vec3 rgb; varying float score;
      void main() {
        vec3 v=(view*vec4(p-eye,1.)).xyz;
        float aspect=screen.x/screen.y, near=.002*scale, far=200.*scale;
        gl_Position=vec4(v.x/aspect, v.y, ((far+near)*v.z-2.*far*near)/(far-near), v.z);
        gl_PointSize=clamp(2.6*scale/max(v.z,.001),1.4,8.);
        rgb=color; score=support;
      }`));
    gl.attachShader(program, shader(gl.FRAGMENT_SHADER, `
      precision mediump float; varying vec3 rgb; varying float score; uniform float evidence; uniform float ispoint;
      void main() {
        if(ispoint>.5 && length(gl_PointCoord-vec2(.5))>.5) discard;
        vec3 confidence=mix(vec3(.96,.51,.16),vec3(.12,.82,.89),score);
        gl_FragColor=vec4(mix(rgb,confidence,evidence),1.);
      }`));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    posBuffer=gl.createBuffer(); colorBuffer=gl.createBuffer(); supportBuffer=gl.createBuffer();
    gl.enable(gl.DEPTH_TEST);
  }
  function upload(buffer, attribute, array, size) {
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(array), gl.STATIC_DRAW);
    const loc=gl.getAttribLocation(program,attribute); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc,size,gl.FLOAT,false,0,0);
  }
  function reset(index=0) {
    if (!scene) return;
    const p=scene.cameras[index] || scene.cameras[0], R=p.map(r=>r.slice(0,3)), t=p.map(r=>r[3]);
    camera=[0,1,2].map(i=>-R.reduce((v,row,j)=>v+row[i]*t[j],0));
    const forward=R[2];
    yaw=Math.atan2(forward[0],forward[2]); pitch=Math.asin(Math.max(-1,Math.min(1,-forward[1])));
  }
  async function select(card) {
    const id=card.id, version=card.revisao;
    const response=await fetch('/api/espacial/'+encodeURIComponent(id));
    if (!response.ok) throw new Error('Nao foi possivel abrir a cena');
    const data=await response.json();
    if (selected!==id || !data.pontos) return;
    const changed=revision!==id+'@'+version;
    scene=data; revision=id+'@'+version; count=data.pontos.length; unit=data.raio;
    if (gl) {
      gl.useProgram(program);
      triangles=!!data.facetas?.length;
      const ids=triangles?data.facetas.flat():data.pontos.map((_,i)=>i);
      count=ids.length;
      upload(posBuffer,'p',ids.flatMap(i=>data.pontos[i]),3);
      upload(colorBuffer,'color',ids.flatMap(i=>data.cores[i]).map(c=>c/255),3);
      // Indicador geometricamente calculado, sem apresentar probabilidade calibrada.
      upload(supportBuffer,'support',ids.map(i=>Math.max(0,Math.min(1,(1-data.erro_px[i]/1.8)*Math.min(data.paralaxe_graus[i]/4,1)))),1);
    }
    el('neuro-vazio').hidden=!!gl;
    if (!gl) el('neuro-vazio').textContent='Este navegador não disponibilizou WebGL para abrir a cena.';
    el('neuro-tempo').max=Math.max(0,data.cameras.length-1);
    if (changed) { el('neuro-tempo').value=0; reset(); }
    const source=data.urls.map(u=>/^https?:\/\//.test(u) ? `<a href="${escape(u)}" target="_blank" rel="noopener">${escape(u)}</a>` : escape(u)).join('<br>');
    el('neuro-info').innerHTML=`<p><strong>${data.pontos.length.toLocaleString('pt-BR')} amostras de superfície</strong><br>${data.cameras.length} vistas · revisão ${version}</p>
      <p>As cores vêm dos pixels recebidos. A profundidade foi <strong>inferida por correspondência entre vistas</strong>.</p>
      <p>${data.calibracao==='fornecida'?'Câmera calibrada fornecida.':'Distância focal aproximada.'} Escala relativa; as distâncias não são metros.</p>
      <p>Erro de projeção em pontos reservados no ajuste da última câmera: <strong>${data.erro_reserva_px.toFixed(2)} px</strong>. Isso não mede compreensão dos objetos.</p>
      <p>A superfície une amostras vizinhas com profundidade consistente. Buracos e partes escondidas permanecem desconhecidos. Ainda não há nomes de objetos, preenchimento aprendido de lacunas nem dinâmica 3D.</p>
      <p>VER SUPORTE colore as amostras pela consistência geométrica: ciano indica maior suporte e laranja menor suporte. Esse indicador não é uma probabilidade.</p>
      <p style="overflow-wrap:anywhere">Origem:<br>${source}</p>`;
  }
  async function poll() {
    if (el('neuro3d').hidden || loading) return;
    loading=true;
    try {
      const d=await (await fetch('/api/espacial')).json(), cards=(d.cenas || []).slice().reverse();
      if (!cards.some(c=>c.id===selected)) { selected=cards[0]?.id || null; scene=null; count=0; }
      el('neuro-lista').innerHTML=cards.map((c,i)=>`<button type="button" class="holo-item ${c.id===selected?'ativa':''}" data-id="${escape(c.id)}">CENA ${cards.length-i}<br><small>${c.pontos.toLocaleString('pt-BR')} pontos · ${c.vistas} vistas · revisão ${c.revisao}</small></button>`).join('') || '<p class="vazio">Aguardando evidência espacial.</p>';
      el('neuro-lista').querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{
        selected=b.dataset.id; revision=null; poll();
      }));
      const recent=(d.tentativas || []).slice(-3).reverse();
      el('neuro-tentativas').innerHTML=recent.length ? '<p>Observações sem profundidade identificada:</p>'+recent.map(t=>`<p>${escape(t.motivo)}</p>`).join('') : '';
      const card=cards.find(c=>c.id===selected);
      if (card && revision!==card.id+'@'+card.revisao) await select(card);
      if (!card) {
        el('neuro-vazio').hidden=false;
        el('neuro-info').textContent='Nenhuma cena foi preenchida artificialmente. A memória 3D começará pelas observações que permitirem estimar estrutura e câmera.';
      }
    } catch(e) { el('neuro-info').textContent='Não foi possível carregar a memória espacial: '+e.message; }
    finally { loading=false; }
  }
  cv.tabIndex=0;
  cv.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY]; cv.setPointerCapture(e.pointerId); cv.focus();});
  cv.addEventListener('pointermove',e=>{if(!drag)return; yaw+=(e.clientX-drag[0])*.004; pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-drag[1])*.004)); drag=[e.clientX,e.clientY];});
  cv.addEventListener('pointerup',()=>{drag=null;}); cv.addEventListener('pointercancel',()=>{drag=null;});
  cv.addEventListener('wheel',e=>{e.preventDefault();const step=-e.deltaY*.001*unit; camera[0]+=Math.sin(yaw)*step;camera[1]-=Math.sin(pitch)*step;camera[2]+=Math.cos(yaw)*step;},{passive:false});
  cv.addEventListener('keydown',e=>{const k=e.key.toLowerCase();if('wasdqe'.includes(k)&&k.length===1){keys.add(k);e.preventDefault();}});
  addEventListener('keyup',e=>keys.delete(e.key.toLowerCase())); cv.addEventListener('blur',()=>keys.clear());
  el('neuro-origem').addEventListener('click',()=>{el('neuro-tempo').value=0;reset();});
  el('neuro-evidencia').addEventListener('click',()=>{evidence=!evidence;el('neuro-evidencia').textContent=evidence?'VER APARÊNCIA':'VER SUPORTE';});
  el('neuro-tempo').addEventListener('input',e=>reset(Number(e.target.value)));
  function frame(now) {
    requestAnimationFrame(frame); const dt=Math.min((now-last)/1000,.05);last=now;
    if(el('neuro3d').hidden || !gl)return;
    const w=cv.clientWidth,h=cv.clientHeight,dpr=Math.min(devicePixelRatio,2);
    if(cv.width!==Math.round(w*dpr)||cv.height!==Math.round(h*dpr)){cv.width=Math.round(w*dpr);cv.height=Math.round(h*dpr);}
    gl.viewport(0,0,cv.width,cv.height);gl.clearColor(.012,.019,.029,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    if(!scene || !count)return;
    const speed=dt*unit*.7, fw=(keys.has('w')?1:0)-(keys.has('s')?1:0), side=(keys.has('d')?1:0)-(keys.has('a')?1:0);
    camera[0]+=(fw*Math.sin(yaw)+side*Math.cos(yaw))*speed;
    camera[1]+=((keys.has('q')?1:0)-(keys.has('e')?1:0))*speed;
    camera[2]+=(fw*Math.cos(yaw)-side*Math.sin(yaw))*speed;
    const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
    // World Y points down, as in source camera coordinates. The viewport Y points up.
    const view=[cy,-sy*sp,sy*cp,0, 0,-cp,-sp,0, -sy,-cy*sp,cy*cp,0, 0,0,0,1];
    gl.useProgram(program);gl.uniformMatrix4fv(gl.getUniformLocation(program,'view'),false,new Float32Array(view));
    gl.uniform3fv(gl.getUniformLocation(program,'eye'),new Float32Array(camera));
    gl.uniform2f(gl.getUniformLocation(program,'screen'),cv.width,cv.height);
    gl.uniform1f(gl.getUniformLocation(program,'scale'),unit);gl.uniform1f(gl.getUniformLocation(program,'evidence'),evidence?1:0);
    gl.uniform1f(gl.getUniformLocation(program,'ispoint'),triangles?0:1);
    gl.drawArrays(triangles?gl.TRIANGLES:gl.POINTS,0,count);
  }
  new MutationObserver(poll).observe(el('neuro3d'),{attributes:true,attributeFilter:['hidden']});
  setInterval(poll,6000);poll();requestAnimationFrame(frame);
})();
