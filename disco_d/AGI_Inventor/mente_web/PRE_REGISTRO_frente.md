# Pré-registro — QUEM ESTÁ NA FRENTE: o primeiro degrau do 3D (transplante 12)

Escrito em 10/10/2026, **antes** de rodar as sementes registradas (20261310 a 20261319).
Código congelado neste commit: `laboratorio.py` (`frente`, `cena`, `desenha_cena(profundidade)`,
`imagina_video(profundidade)`), `arena.py` (`oclusor=` só para o avaliador), `experimento_frente.py`.

## De onde vem

Pedido do usuario (10/10): ver o modelo de mundo dela "quase 3D", sem dados inventados. Ela só
recebe quadros PLANOS. O jeito honesto de começar: profundidade como ORDEM — quem esconde quem.
Quando uma coisa some e reaparece (a permanência já religa isso), o caminho escondido passou por
trás de algo. Se esse caminho cai numa região do fundo que não é a cor dominante dele (onde as
coisas aparecem), essa região está NA FRENTE. Distância entre camadas ela não sabe; só a ordem.

## Regra (fixada no dev, sementes 101–103)

Pontos do caminho escondido: 7 por junção (interpolados entre o último ponto antes e o primeiro
depois). Regiões do fundo por cor (fundo = mediana no tempo, cor quantizada em passos de 24).
Frente = as regiões de UMA cor não-dominante que receberam ≥ 7 pontos e a maioria dos pontos
escondidos fora da cor dominante. Uma única mudança no dev: a primeira forma exigia maioria de
TODOS os pontos (inclusive junções de corpos que se cruzaram no fundo dominante) e achava a frente
em 5/11 episódios com oclusor; contando só os pontos fora da cor dominante, 25/31.

## Medidas e critérios (10 vidas)

Parte 1: 24 episódios por vida (os 4 mundos, armadilhas de sempre: oclusor em ~50%).
Parte 2: 8 vídeos do giro; filmes imaginados de começos dela cujo caminho (pela lei dela) cruza a
frente deduzida; o filme real roda com o MESMO oclusor; erro do que se move, 0–3 s; por vida, a
mediana dos filmes (vidas com ≥ 3 filmes).

| critério | passa se |
|---|---|
| **F1** detecção | frente deduzida em ≥ 60% dos episódios com oclusor |
| **F2** forma certa | IoU ≥ 0,8 com o oclusor verdadeiro em ≥ 90% das deduções |
| **F3** não inventa | frente deduzida em ≤ 5% dos episódios SEM oclusor |
| **F4** a imaginação respeita | erro com a frente < erro achatado em ≥ 80% das vidas avaliáveis, com ≥ 6 avaliáveis |

## Dev (101–103, a regra final)

F1 25/31 (0,81) · F2 25/25 (IoU mediano 1,0) · F3 0/41 · F4 3/3 vidas (avaliáveis 3).

## Previsões

F1 passa (~85%), F2 passa (~90%), F3 passa (~90%), F4 passa (~75%).
Limites declarados: a arena tem oclusor de cor lisa sobre fundo liso — o caso fácil. Em vídeo da
internet (fundo texturizado, câmera que mexe) a regra de "cor do fundo" vai falhar mais; o teste
de verdade lá é a vida. Isto é ordem de camadas (2,5D), não 3D.

## RESULTADO

(preenchido depois da rodada registrada, sem mudar nada acima)
