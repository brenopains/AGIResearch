# Pré-registro — Vida 2: o alcance aprendido (2026-10-05)

Escrito ANTES da rodada final da vida 2. Hashes em `MANIFESTO_vida2.json`.

## De onde vem

Vida 1: o organismo empatou com o pipeline (0,640 × 0,641) trocando forças. O
defeito localizado: a crença por contexto tinha largura FIXA (`BETA0=8`, minha),
e com ela duas perguntas "sentido" em mundos diferentes (semelhança 0,5) se
enxergam com peso 0,018. O h, que nasce em problemas de lei, não chega às
perguntas. A proposta (do usuario e da outra IA): não declarar h global — cada
estrutura APRENDE o próprio alcance, de forma graduada e revisável.

## O mecanismo

Cada estrutura mantém uma distribuição sobre 7 larguras de generalização
(0 = vale em todo lugar … 16 = só no contexto exato), atualizada pela
verossimilhança PREDITIVA de cada registro novo (o quanto cada largura previu o
desfecho antes de vê-lo). Contraexemplo distante devolve peso às larguras
estreitas. Nada vira global=True; o alcance é uma distribuição.

## Mudanças de desenvolvimento achadas no caminho (valem para TODOS os braços da vida 2)

1. **Valor futuro da criação.** O metacontrole escolhia por P(ajudar ESTE
   problema)/custo; inventar quase nunca resolve o problema atual (3 invenções
   numa vida do v1). Agora inventar/derivar valem também pelos usos futuros
   bem-sucedidos dos produtos, aprendidos (a priori, 1 uso por criação).
2. **Crédito contrafactual do juntar.** Todo problema aberto juntado ganhava
   crédito sempre que a resposta final usava dado juntado — mesmo sem ter
   ajudado (o ajuste conjunto aceita a união das formas). Com alcance
   aprendido isso virou "juntar tudo em todo lugar" (~1.000 por vida, zero
   invenção). Agora só recebe crédito o problema aberto cujo dado OBEDECE à lei
   encontrada (reajuste de coeficientes dentro do ruído = mesmo mundo); dado de
   outro mundo que só passou porque o ajuste conjunto aceita a união das formas
   não recebe.

3. **Custo da invenção antes do 1º uso.** A fórmula multiplicava o trabalho por
   8 (calibração do AJUSTE); inventar parecia custar 264 em vez de ~40 e nunca
   era escolhido, logo o custo verdadeiro nunca era aprendido. Defeito meu de
   construção, presente no v1.
4. **Alfabeto da invenção escolhido pelo contexto.** Com o alfabeto = todo
   conceito que já pagou, 6 conceitos davam 117 mil subconjuntos (283 unidades).
   Agora entram só os conceitos que a crença acha relevantes AQUI — é onde o
   alcance aprendido decide se o h entra.
5. Desempenho: as 7 larguras calculadas de uma vez e memorizadas no problema
   (6× mais rápido); não muda decisões do v1 (verificado bit a bit).

ORGANISMO (v1) roda sem nenhuma dessas mudanças: reproduz a vida final 1001 do
v1 bit a bit (verificado), e serve de referência.

## Braços

ORGANISMO (v1), LOCAL2 (largura fixa 8 + mudanças 1–5), LARGURA_3 (fixa 2→3),
GLOBAL (fixa 0), ESCOPO (aprendida), PIPELINE. Sementes NOVAS 2001–2010;
mesmo mundo e mesmos lacrados da vida 1.

## Portões

| | Comparação | Medida | Limiar | Minha probabilidade |
|---|---|---|---|---|
| E1 | ESCOPO − LOCAL2 | total | +3, IC>0 | 0,30 |
| E2 | ESCOPO − MELHOR largura fixa (pela média) | total | +2, IC>0 | 0,15 |
| E3 | ESCOPO − PIPELINE (o G1 de novo) | total | +3, IC>0 | 0,30 |
| E4 | ESCOPO − LOCAL2 | spin-órbita + sentido (h) | +10, IC>0 | 0,25 |
| E5 | ESCOPO − LOCAL2 | magnético | ≥ −5 (média) | 0,60 |
| E6 | largura aprendida de quem pagou em ≥2 famílias < de quem pagou em 1 | Mann-Whitney | p < 0,05 | 0,30 |
| E7 | LOCAL2 − ORGANISMO (valor da criação + crédito) | exigem invenção | +5, IC>0 | 0,50 |

**Observação registrada ANTES (dev, semente 1):** o próprio h pagou em 7 mundos
e ficou com a largura MAIS estreita (15,8): "incluído e não usado" conta como
"não paga aqui", então o alcance mede ONDE É ÚTIL, não onde é verdadeiro. Por
isso a previsão do E6 é baixa. Não mudei a regra depois de ver isso.

Previsão de conjunto: 1 a 3 de 7.

E2 é o portão duro: se uma largura fixa mais generosa faz o mesmo, o ganho era
de uma constante mal escolhida, não de APRENDER o alcance.
