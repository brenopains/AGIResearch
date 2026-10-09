# Pré-registro — Vida 1 do organismo (2026-10-05)

Escrito ANTES de qualquer rodada final. Código congelado: hashes em
`MANIFESTO_congelado.json` (organismo.py `e5bc9f82…`, material `f499f7d1…`,
mundo `98db2059…`, vida `4bf38e9b…`, análise `b27f55f3…`).

## A pergunta

Organizar as MESMAS capacidades como organismo (estruturas de existência
graduada, mesoestado, orçamento autorregulado, coalizões, ecologia, regime,
procedência de dois eixos, sonho) paga mais do que organizá-las como pipeline
(booleano, tetos, ordem fixa, RAG)? E cada ideia, removida sozinha, faz falta?

## O fluxo

10 sementes finais (1001–1010), 120 problemas cada, idênticos entre braços
na mesma semente. Orçamento externo: 300 unidades por problema de lei, 120
por pergunta. Moldes lacrados: 3 sorteados da gramática `q:<grandeza>@<horizonte>`
(24 moldes), hash `07163fe9…` em `LACRE.json`; não foram impressos nem usados em
nenhuma rodada de desenvolvimento.

## Histórico de desenvolvimento (declarado)

- Sementes 1, 2, 3 usadas para depurar o ORGANISMO e checar folga.
- Todos os braços rodaram UMA vez na semente 1 só para pegar erro de execução;
  a taxa de acerto deles NÃO foi lida. Desses runs saíram dois consertos dos
  braços de comparação (antes de qualquer número): TETOS tinha um teto de
  "candidatas por decisão" que era invenção minha, não do projeto; SEM_META
  inundava o orçamento juntando dados antes de inventar.
- Os moldes `periodo_longo`, `rmax_grande`, `vmax_alta` foram expostos no
  desenvolvimento como lacrados de mentira: por isso o lacre verdadeiro saiu
  da gramática nova.
- Achado lateral do desenvolvimento: o guloso não ortogonal (o de
  `novaphys/orbita.py`) falha por BUSCA, não por dado (a forma verdadeira
  tem menos bits em 15–19 de 20 falhas); com busca ortogonal o einstein não
  precisa de h. Por isso o mundo que exige h é o spin-órbita.

## Medidas (só problemas percebidos; relatos não têm verificador)

`total` acerto; `rec` acerto em recorrências (família já vista); `h2` acerto nos
problemas 60–119; `t3` acerto e `custo_t3` custo por acerto em 80–119; `inv`
acerto nos que exigem estrutura inventada (lei: spin_orbita, magnetico,
ciclotron; pergunta: sentido, escapa, periapse_direita); `mag` magnético;
`lac` acerto nas lacradas e `lac_abl` acerto da MESMA mente, no mesmo
problema, com os conceitos (e leis/coalizões que dependem deles) removidos.

## Portões

Passa se a diferença média pareada atinge o limiar E o IC bootstrap de 90% fica
inteiro do lado certo do zero.

| Portão | Comparação | Medida | Limiar | Minha probabilidade |
|---|---|---|---|---|
| G1 | ORGANISMO − PIPELINE | total | +10 pts | 0,55 |
| G2a | ORGANISMO − SEM_MESO (o que pensou × o que aconteceu) | rec | +5 | 0,40 |
| G2b | ORGANISMO − PLACEBO_MESO (reativação por contexto sorteado) | rec | +5 | 0,40 |
| G3 | ORGANISMO − TETOS (números mágicos) | h2 | +5 | 0,45 |
| G4 | ORGANISMO ÷ SEM_COAL | custo_t3 | ≤ 0,80 com t3 ≥ −2 | 0,15 |
| G5 | ORGANISMO − SEM_REGIME | inv | +5 | 0,30 |
| G6 | ORGANISMO − SEM_PROC2 (ficção apagada) | mag | +5 | 0,25 |
| G7 | lacradas: com conceitos − mesma mente sem eles | lac − lac_abl | +15 | 0,35 |
| G8 | conceito de profundidade ≥ 2 NÃO trivial (mistura átomo de base com conceito) usado em acerto | vidas | ≥ 5/10 | 0,40 |
| G9 | ORGANISMO − SEM_META | total ≥ +5, ou total ≥ −2 com custo/acerto ≤ 0,85× | — | 0,50 |
| G10 | ORGANISMO − SEM_SONHO | rec | +3 | 0,25 |
| G11 | ORGANISMO − SEM_ECO (crédito míope) | inv | +5 | 0,50 |

Previsão de conjunto: espero que passem entre 3 e 5 dos 12. Onze portões
multiplicam a chance de algum passar por acaso; o IC de 90% com 10 sementes é
largo. Um portão que passe sozinho não é descoberta; o padrão é o que conta.

## Instrumento (G0, pré-condição)

Se o ORGANISMO ficar acima de 0,95 ou abaixo de 0,30 de acerto total, o fluxo
não discrimina e nenhum portão vale. Na checagem de folga (dev 1–3): 0,68.

## O que NÃO se pode concluir mesmo que tudo passe

Um domínio só (órbitas 2D), alfabeto do material escrito por mim, percepção de
contexto fixa, 120 problemas. Passar diz que a ORGANIZAÇÃO paga sobre este
material — não que é uma mente de nível humano, nem que transfere para outro
domínio (exige teste próprio), nem que funciona com a internet como mundo.

---
## ADENDO escrito DURANTE a rodada final, ANTES de ler qualquer resultado (16:52)

Abri o lacre depois de o código estar congelado e as vidas rodando:
`q:rmin@15`, `q:varre@40`, `q:vmax@5` (hash confere). Medi o teto do
instrumento nesses moldes (`dev_historico/_instrumento_lacre.py`, 8 problemas
por célula, h e E oráculos):

- `rmin@15` e `vmax@5`: 0/8 com base E com oráculo, nas três famílias —
  nada no vocabulário responde (dependem de fase/periapse fora de alcance).
- `varre@40`: base 3–4/8, oráculo 6–8/8 — o único molde onde conceito ajuda.

Consequência: o G7 sobre as 12 lacradas tem teto aproximado de +13 pontos,
abaixo do limiar +15. O G7 fica como registrado (vai ser relatado como está).
À parte, e rotulado PÓS-HOC, relato o mesmo contraste só em `varre@40`.
Este adendo não muda nenhum portão.
