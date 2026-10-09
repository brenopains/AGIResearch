"""Regime com VARIACAO: fabrica estatica da serie sem informacao.

Nas orbitas o dado variava (trajetorias diferentes) e foi dai que saiu
h = x*vy - y*vx. Fabrica parada da so contadores subindo em linha reta —
nao ha o que descobrir. Aqui o numero de perfuratrizes e a quantidade de
combustivel MUDAM ao longo da coleta, de proposito.
"""
import sys, json, time, random
sys.path.insert(0, ('C:/Users/usuario/Documents/AGI 2.0').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))
from novaphys.fabrica import Fabrica


def limpa(f):
    f.lua('''local s=game.surfaces[1]
for _,e in pairs(s.find_entities_filtered{force="player", type={"mining-drill","furnace","transport-belt"}}) do e.destroy() end''')


def monta(f, n, coal):
    r = f.lua(f'''
local s=game.surfaces[1]
local p=s.find_entities_filtered{{name="iron-ore", limit=1}}[1]
if not p then rcon.print("0") return end
local x,y = math.floor(p.position.x), math.floor(p.position.y)
local k=0
for i=0,{n-1} do
  local d=s.create_entity{{name="burner-mining-drill", position={{x+3*i, y}}, force="player", direction=defines.direction.north}}
  if d then
    d.insert{{name="coal", count={coal}}}
    local fo=s.create_entity{{name="stone-furnace", position={{x+3*i, y-2}}, force="player"}}
    if fo then fo.insert{{name="coal", count={coal}}}; k=k+1 end
  end
end
rcon.print(tostring(k))''')
    return int(r or 0)


def main(saida="D:/factorio_servidor/serie.json"):
    f = Fabrica()
    rng = random.Random(20260917)
    serie = []
    # REGIMES: n perfuratrizes x quantidade de carvao. Variar os dois e o que
    # da ao dado a informacao necessaria para separar as causas.
    regimes = [(1, 40), (3, 40), (5, 40), (2, 40),
               (4, 3), (4, 1), (6, 40), (1, 2), (5, 5), (3, 20)]
    for ri, (n, coal) in enumerate(regimes):
        limpa(f)
        k = monta(f, n, coal)
        print(f"regime {ri+1}/{len(regimes)}: {k} pares, carvao={coal}", flush=True)
        for _ in range(12):
            o = f.observaveis()
            if o:
                o["regime"] = ri
                serie.append(o)
            time.sleep(1.5)
    json.dump(serie, open(saida, "w"))
    print(f"\nserie salva: {len(serie)} pontos em {len(regimes)} regimes")
    if serie:
        import collections
        print("variacao de n_drill:", sorted({int(p['n_drill']) for p in serie}))
        print("variacao de drill_ativa:", sorted({int(p['drill_ativa']) for p in serie}))


if __name__ == "__main__":
    main()
