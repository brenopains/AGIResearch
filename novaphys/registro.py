"""
novaphys.registro — o livro-razão que torna versão, fusão e regressão possíveis
================================================================================

A IDEIA DO USUARIO
--------------------
"imagina o proprio LLM e a propria AGI juntos detectando limitacoes e criando uma
versao 2.0, 2.1, as vezes regredindo manualmente pelo usuario ou automaticamente,
fundindo versoes, etc..."

E a consequencia da OUTRA restricao dele: "ler o proprio codigo nao pra
modifica-lo". Como nada escreve no nucleo, toda mudanca e obrigada a existir como
PROPOSTA ISOLADA + VEREDITO DE PORTAO. Isso torna cada versao reversivel por
construcao. Se o sistema pudesse reescrever o proprio nucleo, nao existiria
"versao" nenhuma — existiria um estado que ninguem sabe reconstruir. A trava e o
que da a capacidade.

O QUE E UMA VERSAO, ENTAO
---------------------------
    versao = nucleo (imutavel, em disco) + conjunto de propostas ADOTADAS

e cada proposta adotada carrega a evidencia que a justificou. Daqui saem as tres
operacoes que o usuario pediu, todas mecanicas:

    fundir(A, B)    uniao das propostas, re-testada em nao-regressao
    regredir(V, c)  remove uma proposta e re-mede
    comparar(A, B)  mesmo teto, mesmas tarefas, diferenca em tarefas e em bits

NAO E CONTROLE DE VERSAO DE CODIGO. Git guarda texto; isto guarda EVIDENCIA. A
pergunta que o livro-razao responde nao e "o que mudou" e sim "com que direito".
Uma proposta sem portao nao entra, e por isso uma versao nunca pode ficar melhor
no papel do que foi medida.

O QUE CADA ENTRADA GUARDA
---------------------------
  quem        o codec (nome da classe, arquivo, hash do fonte)
  de onde     a tarefa em que foi proposto, e o modo de falha que o motivou
  o portao    teto de avaliacoes, tarefas ganhas, tarefas perdidas, amostra de
              nao-regressao usada
  o custo     bits antes e depois, para a tarefa que o motivou
  quando      carimbo, e a versao em que entrou

Sem "melhorou bastante": numero, orcamento, e o controle contra o qual foi medido.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

PASTA = os.path.join(os.path.dirname(__file__), "versoes")
LIVRO = os.path.join(PASTA, "livro.json")


def _carrega():
    if not os.path.exists(LIVRO):
        return {"versao": "1.0", "entradas": []}
    with open(LIVRO, encoding="utf-8") as f:
        return json.load(f)


def _salva(d):
    os.makedirs(PASTA, exist_ok=True)
    with open(LIVRO, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def anota(classe, fonte, tarefa, modo, teto, ganhou, perdeu, guarda,
          bits_antes=None, bits_depois=None, arquivo=None):
    """Registra uma proposta ADOTADA com a evidencia que a justificou."""
    d = _carrega()
    h = hashlib.sha256(fonte.encode("utf-8")).hexdigest()[:12]
    if any(e["hash"] == h for e in d["entradas"]):
        return d["versao"]                     # ja registrada, nao duplica
    menor, maior = d["versao"].split(".")
    d["versao"] = f"{menor}.{int(maior) + 1}"
    d["entradas"].append({
        "classe": classe, "hash": h, "arquivo": arquivo,
        "tarefa_de_origem": tarefa, "modo_de_falha": modo,
        "portao": {"teto_avaliacoes": teto, "ganhou": sorted(ganhou),
                   "perdeu": sorted(perdeu), "nao_regressao": sorted(guarda)},
        "bits": {"antes": bits_antes, "depois": bits_depois},
        "versao": d["versao"],
        "quando": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    _salva(d)
    return d["versao"]


def carrega_versao(ate=None):
    """Devolve os fontes das propostas ate uma versao (None = todas). E isto que
    permite REGREDIR: basta pedir uma versao anterior."""
    d = _carrega()
    fora = []
    for e in d["entradas"]:
        if ate is not None and _maior(e["versao"], ate):
            continue
        cam = e.get("arquivo")
        if cam and os.path.exists(cam):
            fora.append((e["classe"], open(cam, encoding="utf-8").read(), e))
    return fora


def _maior(a, b):
    pa = [int(x) for x in a.split(".")]
    pb = [int(x) for x in b.split(".")]
    return pa > pb


def mostra():
    d = _carrega()
    print("=" * 78)
    print(f"LIVRO-RAZAO — versao atual {d['versao']}, "
          f"{len(d['entradas'])} propostas adotadas")
    print("=" * 78)
    if not d["entradas"]:
        print("  (vazio — nenhuma proposta passou no portao ainda)")
        print("\n  O nucleo em disco e a versao 1.0 e nao e tocado por nada disto.")
        return
    for e in d["entradas"]:
        g, p = e["portao"]["ganhou"], e["portao"]["perdeu"]
        print(f"\n  {e['versao']}  {e['classe']}  ({e['hash']})")
        print(f"      nasceu em {e['tarefa_de_origem']} — {e['modo_de_falha']}")
        print(f"      portao: teto {e['portao']['teto_avaliacoes']} avaliacoes, "
              f"ganhou {len(g)}, perdeu {len(p)}, "
              f"nao-regressao em {len(e['portao']['nao_regressao'])} tarefas")
        if e["bits"]["antes"] is not None:
            print(f"      bits: {e['bits']['antes']} -> {e['bits']['depois']}")
        if g:
            print(f"      ganhou: {g}")
        if p:
            print(f"      PERDEU: {p}")
    print(f"\n  fundir: uniao das propostas + re-teste de nao-regressao")
    print(f"  regredir: `carrega_versao(ate='1.2')` devolve o estado daquela versao")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ate", default=None, help="mostra o estado ate esta versao")
    a = ap.parse_args()
    if a.ate:
        for classe, _fonte, e in carrega_versao(a.ate):
            print(f"  {e['versao']}  {classe}")
    else:
        mostra()


if __name__ == "__main__":
    main()
