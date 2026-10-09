"""
novaphys.render — ver o que ela imagina
=========================================

DE ONDE VEM
-------------
Do usuario: "Daria para transformar em video ou gif com algum tipo de decoder
em caso de em algum momento eu querer criar um frontend e visualizar isso
quando ela comecar a imaginar?"

A RESPOSTA TEM DUAS METADES, E SO UMA E BOA
---------------------------------------------
NAO da para decodificar um hipervetor de volta em pixels arbitrarios. O cleanup
do HDC recupera so combinacoes de atomos JA presentes no codebook — e por isso
que ele mede 2/400 como gerador (ver [[indice-ve-superficie]]). Um vetor nao
volta a ser um fluxo qualquer.

DA para renderizar, e nao precisa de decoder nenhum. O que o sistema produz ao
imaginar e um fluxo T x K com um alvo PREVISTO por instante — isso ja e uma
animacao. E o desenho de duas vias do usuario e o que torna isso possivel: uma
copia vai para o indice e outra fica BRUTA, entao o renderizador le o bruto
recuperado e sobrepoe o que foi imaginado. A imaginacao esta na regra aplicada,
nao em reconstruir pixels a partir do vetor.

Este modulo escreve um HTML autocontido (canvas, sem dependencia, sem rede) que
serve de base para o frontend: canais no tempo, cabeca de leitura, e duas
fitas — IMAGINADO e REAL — com os erros marcados.
"""

from __future__ import annotations

import html
import json
from typing import Sequence

import numpy as np

_PAGINA = """<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITULO__</title>
<style>
:root{--ceu:#0d1117;--tinta:#e6edf3;--fraco:#8b949e;--linha:#30363d;
      --mais:#2f81f7;--menos:#f78166;--erro:#f85149;--fundo:#161b22}
:root:not([data-theme="dark"]) @media (prefers-color-scheme: light){}
html,body{margin:0;background:var(--ceu);color:var(--tinta);
  font:14px/1.5 ui-monospace,"SF Mono",Menlo,Consolas,monospace}
.env{max-width:960px;margin:0 auto;padding:24px 16px}
h1{font-size:17px;font-weight:600;margin:0 0 2px;letter-spacing:.01em}
p.sub{color:var(--fraco);margin:0 0 18px;font-size:12.5px}
.painel{background:var(--fundo);border:1px solid var(--linha);
  border-radius:10px;padding:14px;margin-bottom:12px}
canvas{width:100%;height:auto;display:block}
.barra{display:flex;gap:10px;align-items:center;margin-top:10px;flex-wrap:wrap}
button{background:#21262d;color:var(--tinta);border:1px solid var(--linha);
  border-radius:6px;padding:6px 14px;cursor:pointer;font:inherit;font-size:13px}
button:hover{border-color:var(--fraco)}
button:focus-visible{outline:2px solid var(--mais);outline-offset:2px}
input[type=range]{flex:1;min-width:140px;accent-color:var(--mais)}
.leg{display:flex;gap:16px;color:var(--fraco);font-size:12px;margin-top:8px;
  flex-wrap:wrap}
.chip{display:inline-flex;align-items:center;gap:6px}
.pt{width:10px;height:10px;border-radius:2px;display:inline-block}
.num{font-variant-numeric:tabular-nums}
</style></head><body><div class="env">
<h1>__TITULO__</h1>
<p class="sub">__LEGENDA__</p>
<div class="painel"><canvas id="c" width="1840" height="820"></canvas>
<div class="barra">
  <button id="p">pausar</button>
  <input type="range" id="s" min="0" max="__TMAX__" value="0">
  <span class="num" id="r"></span>
</div>
<div class="leg">
  <span class="chip"><i class="pt" style="background:var(--mais)"></i>alvo +1</span>
  <span class="chip"><i class="pt" style="background:var(--menos)"></i>alvo -1</span>
  <span class="chip"><i class="pt" style="background:var(--erro)"></i>imaginou errado</span>
</div></div></div>
<script>
const D = __DADOS__;
const cv=document.getElementById('c'),cx=cv.getContext('2d');
const W=cv.width,H=cv.height,T=D.y.length,K=D.canais.length;
const ML=70,MR=24,MT=24,alturaFita=54,gap=14;
const areaC=H-MT-(alturaFita*2+gap*2+34);
const larg=W-ML-MR, px=t=>ML+larg*(t/(T-1));
const cor=v=>v>0?'#2f81f7':'#f78166';
function eixoY(k,v){const h=areaC/K, base=MT+h*k;
  return base+h*0.5-(v*h*0.42);}
function quadro(tt){
  cx.fillStyle='#0d1117';cx.fillRect(0,0,W,H);
  cx.font='22px ui-monospace,monospace';
  for(let k=0;k<K;k++){
    cx.strokeStyle='#30363d';cx.lineWidth=1;
    cx.beginPath();cx.moveTo(ML,eixoY(k,0));cx.lineTo(W-MR,eixoY(k,0));cx.stroke();
    cx.fillStyle='#8b949e';cx.textAlign='right';
    cx.fillText(D.canais[k].nome,ML-12,eixoY(k,0)+8);
    cx.strokeStyle=D.canais[k].cor;cx.lineWidth=2.4;cx.beginPath();
    for(let t=0;t<=tt;t++){const y=eixoY(k,D.canais[k].v[t]);
      t?cx.lineTo(px(t),y):cx.moveTo(px(t),y);}
    cx.stroke();
  }
  const y1=MT+areaC+gap, y2=y1+alturaFita+gap+18;
  cx.textAlign='left';cx.fillStyle='#8b949e';
  cx.fillText('imaginado',ML,y1-6); cx.fillText('real',ML,y2-6);
  for(let t=0;t<=tt;t++){
    const x=px(t), w=Math.max(1.5,larg/(T-1)+0.5);
    cx.fillStyle=cor(D.pred[t]);cx.fillRect(x,y1,w,alturaFita);
    cx.fillStyle=cor(D.y[t]);   cx.fillRect(x,y2,w,alturaFita);
    if(D.pred[t]!==D.y[t]){cx.fillStyle='#f85149';
      cx.fillRect(x,y1+alturaFita-8,w,8);}
  }
  cx.strokeStyle='#e6edf3';cx.lineWidth=1.5;cx.beginPath();
  cx.moveTo(px(tt),MT-6);cx.lineTo(px(tt),y2+alturaFita+6);cx.stroke();
}
let t=0,tocando=true;
const rot=document.getElementById('r'),sl=document.getElementById('s'),
      bt=document.getElementById('p');
function acerto(ate){let c=0;for(let i=0;i<=ate;i++)if(D.pred[i]===D.y[i])c++;
  return (100*c/(ate+1)).toFixed(1);}
function pinta(){quadro(t);sl.value=t;
  rot.textContent='t='+t+'/'+(T-1)+'   acerto ate aqui '+acerto(t)+'%';}
function passo(){if(tocando){t=(t+1)%T;pinta();}requestAnimationFrame(passo);}
bt.onclick=()=>{tocando=!tocando;bt.textContent=tocando?'pausar':'tocar';};
sl.oninput=()=>{tocando=false;bt.textContent='tocar';t=+sl.value;pinta();};
pinta();requestAnimationFrame(passo);
</script></body></html>
"""

_CORES = ("#7ee787", "#d2a8ff", "#ffa657", "#79c0ff", "#ff7b72", "#a5d6ff")


def para_html(caminho: str, fluxo: np.ndarray, y_real: Sequence[int],
              y_imaginado: Sequence[int], titulo: str, legenda: str,
              nomes: Sequence[str] | None = None) -> str:
    """
    Escreve um HTML autocontido com a animacao do episodio.

    fluxo        T x K, cru (e padronizado aqui so para desenhar)
    y_real       o que de fato aconteceu
    y_imaginado  o que a regra recuperada previu
    """
    A = np.asarray(fluxo, dtype=float)
    Z = (A - A.mean(axis=0)) / np.where(A.std(axis=0) > 1e-9, A.std(axis=0), 1.0)
    Z = np.clip(Z / 2.6, -1, 1)
    canais = [{"nome": (nomes[k] if nomes else f"canal {k}"),
               "cor": _CORES[k % len(_CORES)],
               "v": [round(float(x), 4) for x in Z[:, k]]}
              for k in range(Z.shape[1])]
    dados = {"canais": canais,
             "y": [int(v) for v in y_real],
             "pred": [int(v) for v in y_imaginado]}
    pag = (_PAGINA.replace("__DADOS__", json.dumps(dados))
           .replace("__TITULO__", html.escape(titulo))
           .replace("__LEGENDA__", html.escape(legenda))
           .replace("__TMAX__", str(len(dados["y"]) - 1)))
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(pag)
    return caminho
