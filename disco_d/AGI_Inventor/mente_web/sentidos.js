// mente_web — sentidos injetados em toda pagina ANTES dos scripts dela (corpo 2)
// 1) som acessivel: video/audio carregam pedindo permissao de origem cruzada (CORS);
//    se o site nao permite, o elemento recarrega sem pedir (ela ve, mas nao ouve).
// 2) OUVIDO: liga o elemento que esta tocando a um ouvido (Web Audio) que guarda o
//    som a 16 kHz para ela; a saida para a caixa de som e zero (e o navegador e mudo).
(() => {
  const marca = (el) => {
    if (!el || el.__semCors || el.__tentouCors || el.crossOrigin != null) return;
    el.__tentouCors = true;
    el.crossOrigin = "anonymous";
    el.addEventListener("error", () => {
      if (!el.__semCors) { el.__semCors = true; el.removeAttribute("crossorigin"); el.load(); }
    });
  };
  const P = HTMLMediaElement.prototype, d = Object.getOwnPropertyDescriptor(P, "src");
  Object.defineProperty(P, "src", {
    set(v) { marca(this); d.set.call(this, v); }, get() { return d.get.call(this); }, configurable: true,
  });
  const sa = Element.prototype.setAttribute;
  Element.prototype.setAttribute = function (k, v) {
    if (k === "src") {
      if (this instanceof HTMLMediaElement) marca(this);
      else if (this instanceof HTMLSourceElement && this.parentElement instanceof HTMLMediaElement) marca(this.parentElement);
    }
    return sa.call(this, k, v);
  };
  const ld = P.load;
  P.load = function () { marca(this); return ld.call(this); };
  new MutationObserver((ms) => {
    for (const m of ms) for (const n of m.addedNodes) {
      if (n instanceof HTMLMediaElement) marca(n);
      else if (n.querySelectorAll) n.querySelectorAll("video,audio").forEach(marca);
    }
  }).observe(document, {childList: true, subtree: true});

  window.__ouve = function (el) {
    if (!el) return null;
    if (el.__ouvido) return el.__ouvido.ctx.state;
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const src = ctx.createMediaElementSource(el);
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    const buf = [];
    proc.onaudioprocess = (e) => {
      const x = e.inputBuffer.getChannelData(0), r = Math.max(1, Math.round(ctx.sampleRate / 16000));
      for (let i = 0; i + r <= x.length; i += r) { let s = 0; for (let k = 0; k < r; k++) s += x[i + k]; buf.push(s / r); }
      if (buf.length > 16000 * 8) buf.splice(0, buf.length - 16000 * 8);
      e.outputBuffer.getChannelData(0).fill(0);
    };
    src.connect(proc);
    proc.connect(ctx.destination);
    ctx.resume();
    el.__ouvido = {ctx, buf};
    return ctx.state;
  };
  window.__escuta = function (el) {
    if (!el || !el.__ouvido) return null;
    const b = el.__ouvido.buf.splice(0);
    return {estado: el.__ouvido.ctx.state, n: b.length, amostras: Array.from(b, (v) => Math.round(v * 32767))};
  };
  // o elemento que esta de fato tocando (players trocam o <video> quando se clica)
  window.__vivo = () => [...document.querySelectorAll("video,audio")]
    .sort((a, b) => (b.currentTime - a.currentTime) || ((b.readyState || 0) - (a.readyState || 0)))[0];
})();
