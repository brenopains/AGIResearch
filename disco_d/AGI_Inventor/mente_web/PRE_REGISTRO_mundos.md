# Pré-registro — a aba MUNDOS: o que ela mostra é verdade sobre o mundo? (transplante 12)

Escrito em 10/10/2026, **antes** de rodar as sementes registradas (20261320 a 20261329).
Código congelado neste commit para o teste: `laboratorio.py` (épocas da lei: `_marca_historia`,
`caminho_da_epoca`), `mundos.py` (`frases_da_lei` e seus limiares GIRO_MIN = 0,8, CENTRO_R2 = 0,5,
VEL_R = 0,5; `estados_imaginaveis` vem do transplante 11), `experimento_mundos.py`.

## De onde vem

Pedido do usuario (10/10): "clicar no modelo de mundo aprendido e ver o que ela extraiu de primitivas
e de regras; ver um ambiente quase 3D do que ela projetou; o modelo sendo refinado; sem simular dados
falsos". A aba MUNDOS (projeto por painel de juízes, três abordagens) mostra, para cada MUNDO FÍSICO
dela (uma teoria do laboratório = os vídeos que uma lei dela explica):
- as camadas (fundo < coisas < frente deduzida);
- as coisas com a aparência real;
- o caminho real e o que a lei dela imagina do mesmo começo;
- começos novos;
- as REGRAS em frases medidas na lei dela;
- as correntes da lei, com tracejado onde ela nunca viu nada;
- o REFINAMENTO (cada versão da lei imaginando o mesmo começo);
- o TEMPO como terceira dimensão (tubos x, y, t).

Duas perguntas diferentes:

1. **A tela mostra exatamente o que ela tem?** É fidelidade de código, e não é estatística:
   `teste_mundos.py` cobre W0 (cada caminho e cada pixel publicado refeito a partir do estado dela,
   bit a bit, e nenhum número de profundidade publicado) e M3 (publicar não muda nada nela). Mundos
   físicos: OK nas duas.
2. **O que ela tem, do jeito que a tela mostra, bate com o MUNDO?** É esta rodada.

## Desenho

Uma vida = 48 vídeos da arena (giro, mola_h, contragiro, mola; 12 de cada, sem aviso; oclusor em
~50%). Ao fim, para cada mundo dela com ≥ 6 vídeos (universo = o de onde vem a maioria deles):

- **W1** um mundo dela é um mundo: fração dos vídeos vinda do universo majoritário; e cada universo
  tem um mundo com ≥ 50% dos seus vídeos.
- **W2** as setas apontam certo onde ela viu. Grade de sondas: 8 × 8 lugares × 8 direções, na
  rapidez típica dela. "Viu" = ≥ 5 estados dela a menos de 0,2 do lugar. Mede-se o cosseno entre a
  aceleração da lei dela e a verdadeira (levada às coordenadas dela pela câmera), nas sondas com
  aceleração verdadeira acima de 10% da mediana. Por vida: a mediana dos mundos.
- **W3** a lei vale onde ela nunca viu: o mesmo cosseno nas sondas com 0 estados dela perto.
- **W4** as frases estão certas:
  - giro e contragiro: "gira" com o sentido VERDADEIRO na tela e sem "centro";
  - mola e mola_h: "centro" a ≤ 0,15 meia-largura do centro verdadeiro.
- **W5** imaginar daqui: 6 começos novos dela; a arena roda o filme real; a lei dela erra menos que a
  inércia aos 2 s.
- **W6** o refinamento é real: em mundos com ≥ 4 épocas, 12 vídeos NOVOS do universo (fluxo de
  sorteio separado); a lei da ÚLTIMA época prevê (1ª tentativa, 2 s) melhor que a da PRIMEIRA.

## Critérios (10 vidas)

| critério | passa se |
|---|---|
| **W1** | ≥ 80% dos mundos com pureza ≥ 0,8 E ≥ 8/10 vidas com todo universo coberto |
| **W2** | cosseno mediano ≥ 0,8 em ≥ 8/10 vidas |
| **W3** | cosseno mediano ≥ 0,8 em ≥ 80% das vidas avaliáveis (≥ 6 avaliáveis) |
| **W4** | ≥ 80% dos mundos com a frase certa |
| **W5** | lei < inércia em ≥ 8/10 vidas |
| **W6** | última < primeira em ≥ 70% das vidas avaliáveis (≥ 6 avaliáveis) |

## Dev (sementes 101–103) e a única mudança

W1 12/12 mundos puros, 3/3 vidas com cobertura · W2 cosseno ~1,0 · W4 12/12 frases certas (sentidos
opostos entre giro e contragiro; centro a ~1 px do verdadeiro) · W5 3/3 · W6 3/3 (a lei final erra
1,2 a 5,3 px menos em vídeos novos).

A hipótese original dos projetistas para W3 era "onde ela não viu, a seta erra mais" — o que
justificaria o tracejado como aviso de erro. **Caiu no dev**: lá a lei dela acerta igual (cosseno
~1,0), porque as leis dela são as leis do mundo e valem em todo lugar. W3 virou "a lei vale onde ela
nunca viu?" e o texto da tela mudou: o tracejado diz só "ela nunca viu nada aqui; a lei é extrapolação
dela", sem sugerir erro.

## Previsões

W1 passa (~80%) · W2 passa (~90%) · W3 passa (~85%) · W4 passa (~80%) · W5 passa (~90%) · W6 passa
(~80%).

Limite declarado, o mais importante: é a ARENA — leis simples e exatas, câmera parada, coisas
rígidas. Na internet a maioria dos vídeos nem vira mundo físico (câmera mexendo), e o que virar terá
leis aproximadas. O que estes testes mostram é que, quando ela tem um modelo de mundo, a aba o mostra
fielmente E ele está certo sobre o mundo nos pontos medidos. Não mostram que ela entende a internet.

## RESULTADO

(preenchido depois da rodada registrada, sem mudar nada acima)
