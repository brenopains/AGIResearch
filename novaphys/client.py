"""
novaphys.client — acesso ao mundo físico simulado
==================================================

Cliente do "Simulador de Realidade" rodando como servidor de mundo.
Usa apenas a biblioteca padrão: nenhuma dependência nova entra no projeto
por causa disto.

O que este módulo dá à AGI que ela não tinha:

  Um mundo cujas leis ela NÃO conhece e não pode ler.

O `PhysicalSandbox` do core aprende archetypes observando pares
(antes, depois) — mas hoje quem produz esses pares é ele mesmo, então o
aprendizado é circular. O `ActiveExperimenter` escolhe experimentos por
ganho de informação — mas ajusta uma power-law contra dados que ele
próprio gerou com uma power-law. Nos dois casos falta a mesma coisa: um
mundo externo, com física de verdade, que não colabore.

Aqui a física é resolvida por métodos numéricos estabelecidos (split-step
de Fourier, velocity-Verlet, Barnes-Hut, geodésicas de Kerr, Bosch-Hale,
projeção de pressão) e validada contra resultados conhecidos. O agente
recebe medições com nome, unidade e descrição — a mesma semântica que um
humano recebe na interface — e manoplas para girar. As equações, não.

Uso mínimo:

    from novaphys import PhysicsWorld

    w = PhysicsWorld()
    w.reset(modulo="gravidade", semente=42)
    obs = w.act(ajustar={"darkMatter": 1.2}, passos=200)
    print(obs["escalares"]["g_virial"])
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

DEFAULT_URL = "http://localhost:8124/rpc"


class WorldError(RuntimeError):
    """Erro devolvido pelo servidor de mundo."""


class PhysicsWorld:
    """
    Ponte para o mundo físico.

    Cada método é uma chamada RPC. A latência local é de ~1 ms, o que é
    irrelevante para um laço de descoberta (o custo real está em integrar
    a física, não em falar com ela).
    """

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 600.0):
        self.url = url
        self.timeout = timeout
        self._caps: Optional[Dict[str, Any]] = None

    # ─────────────────────────────────────────────── transporte
    def _rpc(self, metodo: str, params: Optional[Dict[str, Any]] = None) -> Any:
        corpo = json.dumps({"metodo": metodo, "params": params or {}}).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=corpo,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                resp = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                resp = json.loads(e.read().decode("utf-8"))
            except Exception:
                raise WorldError(f"HTTP {e.code} em {metodo}") from e
        except urllib.error.URLError as e:
            raise WorldError(
                f"servidor de mundo inacessível em {self.url}. "
                f"Rode:  node bridge/server.mjs  no repo do simulador. ({e.reason})"
            ) from e
        if not resp.get("ok"):
            raise WorldError(f"{metodo}: {resp.get('erro')}")
        return resp["resultado"]

    # ─────────────────────────────────────────────── descoberta do ambiente
    def ping(self) -> Dict[str, Any]:
        return self._rpc("ping")

    def capacidades(self, recarregar: bool = False) -> Dict[str, Any]:
        """
        Catálogo completo: módulos, manoplas e medições, cada uma com nome
        legível, unidade e explicação em português.

        Esta é a única semântica que o agente recebe. É o material com que
        um sistema HDC pode ancorar símbolos em medições — o nome da série
        vira um conceito, a descrição vira o contexto, o valor vira o
        conteúdo.
        """
        if self._caps is None or recarregar:
            self._caps = self._rpc("capacidades")
        return self._caps

    def modulos(self) -> List[str]:
        return [m["id"] for m in self.capacidades()["modulos"]]

    def descrever(self, modulo: str) -> Dict[str, Any]:
        for m in self.capacidades()["modulos"]:
            if m["id"] == modulo:
                return m
        raise WorldError(f"módulo desconhecido: {modulo}")

    # ─────────────────────────────────────────────── ciclo percepção-ação
    def reset(self, modulo: str, semente: int = 12345,
              controles: Optional[Dict[str, Any]] = None,
              constantes: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Abre um episódio. Mesma semente + mesma sequência de ações produz
        exatamente a mesma trajetória — sem isso não há como comparar um
        agente com uma baseline de forma honesta.

        Guarda a sessão devolvida pelo servidor e a envia nas ações seguintes.
        O mundo é um só: sem isso, dois processos medindo ao mesmo tempo trocam
        o módulo um debaixo do outro e a medição sai errada em silêncio.
        """
        r = self._rpc("reset", {
            "modulo": modulo, "semente": semente,
            "controles": controles or {}, "constantes": constantes or {},
        })
        self._sessao = r.get("sessao")
        return r

    def observar(self, campos: bool = True, resolucao: int = 32,
                 narrativa: bool = False) -> Dict[str, Any]:
        """
        Percepção. Três canais:
          escalares — medições instrumentadas (proprioceptivo)
          campos    — grades 32×32 do que houver de campo 2D (visual)
          corpos    — estatística de nuvens de partículas, não posições cruas

        `narrativa=True` acrescenta a frase em português que um humano leria
        na interface. Útil para grounding linguístico; deve ficar desligada
        quando o ponto é descobrir a lei sozinho, senão vira cola.
        """
        return self._rpc("observar",
                         {"campos": campos, "resolucao": resolucao, "narrativa": narrativa,
                          "sessao": getattr(self, "_sessao", None)})

    def agir(self, ajustar: Optional[Dict[str, Any]] = None,
             constantes: Optional[Dict[str, float]] = None,
             comando: Optional[str] = None, passos: int = 1) -> Dict[str, Any]:
        """
        Uma ação: gira manoplas, aciona um comando do módulo, avança o tempo.
        É o análogo de mexer no aparelho e olhar o que aconteceu.
        """
        return self._rpc("agir", {
            "ajustar": ajustar or {}, "constantes": constantes or {},
            "comando": comando, "passos": passos,
            "sessao": getattr(self, "_sessao", None),
        })

    def passo(self, n: int = 1) -> Dict[str, Any]:
        return self._rpc("passo", {"n": n, "sessao": getattr(self, "_sessao", None)})

    def serie(self, chave: str, max_pts: int = 2000) -> Dict[str, Any]:
        """Série temporal completa de uma medição — o caderno de laboratório."""
        return self._rpc("serie", {"chave": chave, "maxPts": max_pts})

    def eventos(self, desde: int = 0) -> List[Dict[str, Any]]:
        return self._rpc("eventos", {"desde": desde})["eventos"]

    def snapshot(self) -> Dict[str, Any]:
        return self._rpc("snapshot")

    def restaurar(self, estado: Dict[str, Any]) -> Dict[str, Any]:
        return self._rpc("restaurar", {"estado": estado})

    # ─────────────────────────────────────────────── benchmark
    def tarefas(self) -> List[Dict[str, Any]]:
        return self._rpc("tarefas")

    def tarefa_iniciar(self, tarefa: str, semente: Optional[int] = None,
                       sondas: int = 8) -> Dict[str, Any]:
        p: Dict[str, Any] = {"tarefa": tarefa, "sondas": sondas}
        if semente is not None:
            p["semente"] = semente
        info = self._rpc("tarefaIniciar", p)
        # Guarda o token do episódio. Se outro cliente abrir uma tarefa neste
        # mundo, as chamadas seguintes falham em vez de medir o experimento
        # errado em silêncio — que foi exatamente o que aconteceu antes de
        # existir token, e produziu números irreprodutíveis.
        self._token = info.get("token")
        return info

    def tarefa_experimento(self, config: Dict[str, float]) -> Dict[str, Any]:
        return self._rpc("tarefaExperimento",
                         {"config": config, "token": getattr(self, "_token", None)})

    def tarefa_responder(self, predicoes: Optional[List[float]] = None,
                         otimo: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        return self._rpc("tarefaResponder",
                         {"predicoes": predicoes, "otimo": otimo,
                          "token": getattr(self, "_token", None)})

    def tarefa_verdades(self) -> Dict[str, Any]:
        return self._rpc("tarefaVerdades")
