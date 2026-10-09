"""Executor: leva o plano GERAL para o mundo. So esta camada e de Factorio."""
import sys, json, time
sys.path.insert(0, ('C:/Users/usuario/Documents/AGI 2.0').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))
from novaphys.fabrica import Fabrica
from novaphys.planeja import Plano

PRIM = {"iron-ore", "copper-ore", "coal", "stone", "water", "wood",
        "crude-oil", "uranium-ore"}


class Agente:
    def __init__(self):
        self.f = Fabrica()
        self.recs = json.load(open("D:/factorio_servidor/receitas.json"))
        self.p = Plano(self.recs, PRIM)
        self.acoes = 0

    def diz(self, s):
        self.f.lua(f'game.print("[agente] {s}")')
        print(f"  {s}", flush=True)

    def tem(self, item):
        r = self.f.lua(f'''local p=game.players[1]
if p and p.character then rcon.print(p.get_main_inventory().get_item_count("{item}"))
else rcon.print(0) end''')
        return int(r or 0)

    def extrair(self, item, qtd):
        """Acao primitiva: obter recurso do mundo."""
        self.acoes += 1
        self.f.lua(f'''local p=game.players[1]
if p and p.character then p.insert{{name="{item}", count={int(qtd)}}} end''')
        self.diz(f"extraiu {int(qtd)}x {item}")

    def fazer(self, receita, n):
        """Acao primitiva: executar uma transicao do modelo."""
        self.acoes += 1
        r = self.f.lua(f'''local p=game.players[1]
if p and p.character then rcon.print(p.begin_crafting{{count={int(n)}, recipe="{receita}"}}) else rcon.print(0) end''')
        self.diz(f"fabricou {n}x {receita} (aceito: {r})")
        time.sleep(1.2)

    def executa(self, alvo, qtd=1):
        pl = self.p.plano(alvo, qtd)
        self.diz(f"OBJETIVO: {qtd}x {alvo} | plano de {pl['n_passos']} passos, profundidade {pl['profundidade']}")
        if pl["faltou"]:
            self.diz(f"nao sei produzir: {pl['faltou']}")
            return None
        for tipo, nome, n, prof in pl["passos"]:
            if tipo == "extrair":
                self.extrair(nome, max(int(n) * 2, 10))
            else:
                self.fazer(nome, int(n))
        time.sleep(2)
        got = self.tem(alvo)
        ok = got >= qtd
        self.diz(f"RESULTADO: {got}x {alvo} — {'SUCESSO' if ok else 'FALHOU'} | {self.acoes} acoes")
        return {"alvo": alvo, "obtido": got, "ok": ok, "acoes": self.acoes,
                "passos": pl["n_passos"], "prof": pl["profundidade"]}


if __name__ == "__main__":
    a = Agente()
    a.f.lua('local p=game.players[1] if p and p.character then p.get_main_inventory().clear() end')
    res = []
    for alvo in ["iron-plate", "iron-gear-wheel", "transport-belt",
                 "electronic-circuit", "burner-mining-drill",
                 "electric-mining-drill", "assembling-machine-1"]:
        a.acoes = 0
        try:
            r = a.executa(alvo, 1)
            if r: res.append(r)
        except Exception as e:
            print(f"  ERRO em {alvo}: {str(e)[:70]}")
        print()
    print("=" * 60)
    print(f"{'objetivo':<24}{'obtido':>8}{'acoes':>7}{'prof':>6}  ok")
    for r in res:
        print(f"{r['alvo']:<24}{r['obtido']:8d}{r['acoes']:7d}{r['prof']:6d}  {'SIM' if r['ok'] else 'NAO'}")
    print(f"\nresolvidos: {sum(1 for r in res if r['ok'])}/{len(res)}")
