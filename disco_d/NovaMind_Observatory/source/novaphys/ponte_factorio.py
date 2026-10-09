"""novaphys.ponte_factorio — o CANAL: Factorio vira `Fluxo` do substrato.py.

O substrato.py ja existe e ja faz o que o usuario pediu: constroi representacao a
partir de FLUXO CRU (T instantes x K canais, sem nome e sem semantica), minera o
que se repete em PRIMITIVA, e testa TRANSFERENCIA para canais nunca vistos.

Ele so nunca viu dado real — `gerar_fluxo()` e sintetico. Esta ponte troca o
sintetico pelo Factorio. NADA do nucleo e adaptado: o canal e o encoder, como o
usuario disse; o mecanismo e o mesmo.
"""
from __future__ import annotations
import sys, os, time, json, random
os.makedirs("D:/agi_memoria", exist_ok=True)
sys.path.insert(0, ('C:/Users/usuario/Documents/AGI 2.0').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))
from novaphys.fabrica import Fabrica

CANAIS = ["n_drill", "drill_ativa", "n_forno", "forno_ativo", "coal_drill",
          "coal_forno", "ore_no_forno", "plate_no_forno"]


def limpa(f):
    f.lua('''local s=game.surfaces[1]
for _,e in pairs(s.find_entities_filtered{force="player", type={"mining-drill","furnace","transport-belt"}}) do e.destroy() end''')


def monta(f, n, coal):
    return int(f.lua(f'''
local s=game.surfaces[1]
local p=s.find_entities_filtered{{name="iron-ore", limit=1}}[1]
if not p then rcon.print("0") return end
local x,y=math.floor(p.position.x),math.floor(p.position.y)
local k=0
for i=0,{n-1} do
  local d=s.create_entity{{name="burner-mining-drill", position={{x+3*i,y}}, force="player", direction=defines.direction.north}}
  if d then d.insert{{name="coal",count={coal}}}
    local fo=s.create_entity{{name="stone-furnace", position={{x+3*i,y-2}}, force="player"}}
    if fo then fo.insert{{name="coal",count={coal}}}; k=k+1 end end
end
rcon.print(tostring(k))''') or 0)


def coleta_fluxo(regimes=None, por_regime=14, dt=1.2, verbose=True):
    """Devolve (fluxo, alvo, K) no formato EXATO que o substrato.py espera.
    alvo[t] = +1 se a fabrica PRODUZIU placa naquele instante, -1 se nao."""
    f = Fabrica()
    regimes = regimes or [(1, 40), (3, 40), (5, 40), (2, 40),
                          (4, 3), (4, 1), (6, 40), (1, 2), (5, 5), (3, 20)]
    fluxo, alvo, prev = [], [], None
    for ri, (n, coal) in enumerate(regimes):
        limpa(f); k = monta(f, n, coal)
        if verbose:
            print(f"  regime {ri+1}/{len(regimes)}: {k} pares, carvao={coal}", flush=True)
        for _ in range(por_regime):
            o = f.observaveis()
            if o:
                fluxo.append([float(o[c]) for c in CANAIS])
                pt = o["plate_total"]
                alvo.append(1 if (prev is not None and pt > prev) else -1)
                prev = pt
            time.sleep(dt)
    return fluxo, alvo, len(CANAIS)


if __name__ == "__main__":
    from novaphys.substrato import resolver, minerar, Primitiva
    print("coletando FLUXO REAL do Factorio...", flush=True)
    fluxo, alvo, K = coleta_fluxo()
    json.dump({"fluxo": fluxo, "alvo": alvo, "canais": CANAIS},
              open("D:/agi_memoria/fluxo_factorio.json", "w"))
    pos = sum(1 for a in alvo if a > 0)
    print(f"\nfluxo: {len(fluxo)} instantes x {K} canais | alvo +1 em {pos}/{len(alvo)}")
    CONST = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0)
    print("\nrodando o SUBSTRATO (o mesmo codigo, dado real):")
    r = resolver(fluxo, alvo, K, CONST, [])
    print(f"  REAL     ajuste {r['ajuste']:.3f}  tamanho {r['tamanho']}  "
          f"programas avaliados {r['n_avaliados']}")
    rng = random.Random(7)
    emb = list(alvo); rng.shuffle(emb)
    rn = resolver(fluxo, emb, K, CONST, [])
    print(f"  NULO     ajuste {rn['ajuste']:.3f}  (alvo embaralhado)")
    print(f"\n  diferenca real-nulo: {r['ajuste']-rn['ajuste']:+.3f}")
    if r["programa"]:
        print(f"  programa encontrado: {r['programa'].esqueleto()}")
