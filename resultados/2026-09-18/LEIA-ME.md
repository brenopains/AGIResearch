# Saídas brutas — 2026-09-18

Cada `.log` é a saída original, sem edição, do experimento correspondente.
A leitura, os pré-registros e as ressalvas estão na **seção 11** do
`ARQUITETURA.md`; os vereditos condensados estão em `memory/MEMORY.md`.

| log | módulo | veredito |
|---|---|---|
| `so4.log` | `segunda_ordem.py` | compor 82,5 % × aprender 50,3 % |
| `prim2.log` | `primitiva_do_dado.py` | versão com SELEÇÃO NO TESTE — **superada** |
| `prim_honesto.log` | `primitiva_do_dado.py` | corrigido: 85,1 % × controle justo 76,9 % ± 5,4 (Y2) |
| `explica5.log` | `explica.py` | leitura direta 6/7; três resumos discordam |
| `cena2.log` | `cena.py` | capacidade 800 objetos em D=40k, escala linear |
| `canto4.log` | `canto.py` | chunk × Markov: empate, chunk 8× menor |
| `variedade2.log` | `variedade.py` | precisão 1,8× → 3,0× o acaso |
| `acervo3.log` | `acervo.py` | extraído = bruto em acurácia, 3× menos variância |
| `matriz_voto.log` | `nuvem_matriz.py` | voto de pontes: célula de 5,6 % → 92,1 % |
| `alvo.log` | `alvo_proprio.py` | NULO — subdimensionado (folga 0,3 × ruído 0,7) |
| `har2.log` | `har.py` | **o fluxo era histograma**: real 0,528 × nulo 0,537 |
| `harlex.log` | `har_lexico.py` | **caminho da ordem: 0,778, ANDAR 94 % puro** |

Dado externo usado, não versionado aqui:
- `D:\agi_dados\har` — UCI Human Activity Recognition (61 MB)
- `D:\agi_dados\passaro` — Bengalese Finch song repository, só anotações (38,5 MB)

## Correção de 18/09 à noite

Uma revisão externa apontou **seleção no conjunto de teste** em
`primitiva_do_dado.py`, confirmada no código. O `prim2.log` fica preservado como
registro da versão superada; o número válido está em `prim_honesto.log`. A
análise completa das seis observações está na **seção 12** do `ARQUITETURA.md`.

---

## Madrugada de 19/09 — o piso de ruído (seção 13 do `ARQUITETURA.md`)

`13-abstrai-motivo.log` · `13-gap.py` · `13-triagem.py` · `13-triangulo.py` ·
`13-ruido.py` · `13-pareado.py` · `13-saidas.log`

Quatro medidas encadeadas. Abstrair pelo padrão de repetição é **nulo** e o
mecanismo foi medido (borra: sobe a concordância entre atividades diferentes mais
que dentro da mesma). Ao conferir um braço suspeito — uma **bijeção** de rótulos,
que por construção não pode carregar informação nova — descobriu-se que o
pipeline tem **±0,040 de ruído puro de sorteio de átomos**, nunca medido antes.
Isso reclassifica todo número absoluto de HDC anunciado de um sorteio só. O
pareamento cancela o ruído (diferença ±0,008) e confirma `motivo-e-pessoal`.
`har_pessoas.py` foi corrigido para reportar os dois separados.
