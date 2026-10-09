"""novaphys.fabrica — ler a fabrica do Factorio por RCON puro (sem FLE).

Serve aos dois caminhos possiveis: um agente que AGE precisa perceber, e a
abducao que OBSERVA precisa de serie temporal. Esta camada e a mesma.

Sem FLE porque o FLE injeta funcoes Lua no estado global e o save do mapa falha
na serializacao — o que impede qualquer jogador humano de entrar e assistir.
Medido: o servidor CAIU quando o usuario tentou entrar com o FLE carregado.
"""
from __future__ import annotations
import time
import factorio_rcon


class Fabrica:
    def __init__(self, host="127.0.0.1", porta=27015, senha="factorio"):
        self.c = factorio_rcon.RCONClient(host, porta, senha)

    def lua(self, s):
        return self.c.send_command("/silent-command " + s)

    def tick(self):
        return int(self.lua("rcon.print(game.tick)") or 0)

    def entidades(self, forca="player"):
        """Todas as maquinas da forca, com posicao, tipo e estado."""
        r = self.lua(f'''
local s=game.surfaces[1] local t={{}}
for _,e in pairs(s.find_entities_filtered{{force="{forca}"}}) do
  if e.type ~= "character" and e.type ~= "item-entity" then
    t[#t+1] = e.name .. "|" .. e.position.x .. "|" .. e.position.y ..
              "|" .. e.type .. "|" .. (e.status or -1)
  end
end
rcon.print(table.concat(t, ";"))''')
        fora = []
        for it in (r or "").split(";"):
            p = it.split("|")
            if len(p) == 5:
                fora.append({"nome": p[0], "x": float(p[1]), "y": float(p[2]),
                             "tipo": p[3], "estado": int(float(p[4]))})
        return fora

    def producao(self):
        """Contadores de producao da forca — a METRICA, calculada pelo jogo."""
        r = self.lua('''
local st = game.forces.player.get_item_production_statistics(game.surfaces[1])
local t = {}
for nome, n in pairs(st.input_counts) do t[#t+1] = nome .. "=" .. n end
rcon.print(table.concat(t, ";"))''')
        d = {}
        for it in (r or "").split(";"):
            if "=" in it:
                k, v = it.split("=", 1)
                try:
                    d[k] = int(v)
                except ValueError:
                    pass
        return d

    def inventarios(self):
        """O que tem dentro de cada maquina — o estado interno da fabrica."""
        r = self.lua('''
local s=game.surfaces[1] local t={}
for _,e in pairs(s.find_entities_filtered{force="player"}) do
  local inv = e.get_output_inventory and e.get_output_inventory() or nil
  if inv and inv.valid then
    local c = inv.get_contents()
    for _,x in pairs(c) do
      t[#t+1] = e.name .. "@" .. math.floor(e.position.x) .. "," ..
                math.floor(e.position.y) .. ":" .. x.name .. "=" .. x.count
    end
  end
end
rcon.print(table.concat(t, ";"))''')
        return [x for x in (r or "").split(";") if x]

    def serie(self, n=20, intervalo=3.0):
        """A SERIE TEMPORAL: e disto que a abducao precisa para ter gradiente."""
        fora = []
        for _ in range(n):
            fora.append({"tick": self.tick(), "producao": self.producao(),
                         "t": time.time()})
            time.sleep(intervalo)
        return fora


    def observaveis(self):
        """As GRANDEZAS da fabrica num instante. E o analogo das trajetorias de
        onde saiu h = x*vy - y*vx: quantidades brutas, sem eu dizer qual importa."""
        r = self.lua('''
local s=game.surfaces[1]
local nd,nf,ndq,nfq,ore,coal_d,coal_f,plate = 0,0,0,0,0,0,0,0
for _,e in pairs(s.find_entities_filtered{force="player"}) do
  if e.type=="mining-drill" then
    nd=nd+1
    if e.status==1 or e.status==2 then ndq=ndq+1 end
    local i=e.get_fuel_inventory()
    if i and i.valid then for _,x in pairs(i.get_contents()) do coal_d=coal_d+x.count end end
  elseif e.type=="furnace" then
    nf=nf+1
    if e.status==1 or e.status==2 then nfq=nfq+1 end
    local i=e.get_fuel_inventory()
    if i and i.valid then for _,x in pairs(i.get_contents()) do coal_f=coal_f+x.count end end
    local o=e.get_output_inventory()
    if o and o.valid then for _,x in pairs(o.get_contents()) do if x.name=="iron-plate" then plate=plate+x.count end end end
    local s2=e.get_inventory(defines.inventory.furnace_source)
    if s2 and s2.valid then for _,x in pairs(s2.get_contents()) do ore=ore+x.count end end
  end
end
local st=game.forces.player.get_item_production_statistics(game.surfaces[1])
local po = st.input_counts["iron-ore"] or 0
local pp = st.input_counts["iron-plate"] or 0
rcon.print(game.tick..";"..nd..";"..ndq..";"..nf..";"..nfq..";"..coal_d..";"..coal_f..";"..ore..";"..plate..";"..po..";"..pp)''')
        p = (r or "").split(";")
        if len(p) != 11:
            return None
        k = ["tick", "n_drill", "drill_ativa", "n_forno", "forno_ativo",
             "coal_drill", "coal_forno", "ore_no_forno", "plate_no_forno",
             "ore_total", "plate_total"]
        return {a: float(b) for a, b in zip(k, p)}

    def coleta(self, n=40, intervalo=2.0, verbose=False):
        import time as _t
        fora = []
        for i in range(n):
            o = self.observaveis()
            if o:
                fora.append(o)
                if verbose and i % 10 == 0:
                    print(f"   t={o['tick']:.0f} placas={o['plate_total']:.0f} "
                          f"minerio={o['ore_total']:.0f} ativas={o['drill_ativa']:.0f}", flush=True)
            _t.sleep(intervalo)
        return fora
