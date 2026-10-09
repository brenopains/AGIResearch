# Pré-registro — o VÍDEO MENTAL (transplante 11): ela imagina filmes com as leis que descobriu

Escrito em 09/10/2026, **antes** de rodar as sementes registradas (20261210 a 20261219).
Código congelado neste commit: `laboratorio.py` (seção 6: `cena`, `desenha_cena`,
`guarda_cena`, `estados_imaginaveis`, `imagina_video`, `video_mental`), `experimento_sonho.py`.

## De onde vem

Pedido do usuario (09/10): imaginação universal — "um vídeo mental imaginado", cenas com N
objetos, movimento, de forma multimodal. O passo honesto que cabe hoje: ela já tem leis de
movimento descobertas sozinha (laboratório) e guarda a APARÊNCIA real das coisas que viu
(fundo + recorte de cada coisa); juntando os dois, ela pode **imaginar um filme que nunca
viu**: pôr as coisas de uma cena em começos novos e movê-las pela lei dela, quadro a quadro.
Nada é animação feita à mão: posição = a mesma rolagem da previsão do laboratório;
aparência = pixels recortados de vídeos que ela assistiu.

## Duas perguntas

1. **O sonho ensina a visão?** Treinar o preditor de tempo do córtex (o que antecipa o próximo
   quadro, e cuja falha é a "surpresa" do vídeo na mente viva) com filmes imaginados pela lei
   dela ajuda a prever vídeos REAIS novos, mais do que filmes "tudo reto" ou rever o que viu?
2. **O filme imaginado acerta a realidade?** Ela imagina o filme ANTES; a arena roda o filme
   de verdade do mesmo começo. O que se move bate mais com a lei dela do que com os controles?

## Desenho (uma vida; câmera desconhecida; mundos giro e depois mola_h)

Em cada mundo: (1) TRONCO comum: 6 vídeos reais; o laboratório os observa e guarda a cena;
o córtex os assiste como na mente viva. (2) SONHO, cada braço numa CÓPIA do tronco, parte
que vê congelada, só o preditor de tempo treina, 60 passos de janelas de 12 quadros, mesmas
escolhas de janela em todos os braços:
- **REVER** — janelas dos 6 vídeos reais;
- **SONHO_LEI** — 10 filmes de 6 s imaginados pela teoria do último vídeo do mundo (3 coisas,
  começos = situações vistas mudadas como nas mãos, cena de um vídeo do mundo);
- **SONHO_INERCIA** — os mesmos começos e cenas, tudo reto (placebo);
- **SONHO_VERDADE** — os mesmos começos e cenas, lei VERDADEIRA (teto; ela nunca teria).
(3) TESTE: 4 vídeos reais novos; surpresa = erro médio da previsão do mapa do próximo quadro.
(4) FILME: 10 começos novos (dela); filme imaginado antes × filme real da arena (sem
armadilhas) do mesmo começo; compara-se o que SE MOVE (filme menos o próprio fundo, borrado
σ = 3 px), erro quadrático médio aos 1, 2 e 3 s; controles: filme "tudo reto" e, na mola_h,
filme com a lei do OUTRO mundo dela (a teoria do giro).

## Critérios (fixados agora; 10 vidas)

| critério | passa se |
|---|---|
| **V0** (o instrumento enxerga lei?) | surpresa SONHO_VERDADE < SONHO_INERCIA em ≥ 7/10 vidas |
| **S1** sonhar a lei dela vence sonhar reto | SONHO_LEI < SONHO_INERCIA em ≥ 7/10 — **só vale se V0 passar** |
| **S2** sonhar a lei dela vence rever | SONHO_LEI < REVER em ≥ 7/10 — **só vale se V0 passar** |
| **M1** o filme imaginado acerta mais que o reto | erro aos 2 s (média geométrica dos 2 mundos) lei < reto em ≥ 8/10 |
| **M2** é a lei CERTA, não qualquer curva | na mola_h, erro aos 2 s lei dela < lei do outro mundo em ≥ 8/10 |

Se V0 falhar, S1 e S2 ficam registrados como "sem valor": se nem a lei verdadeira ensina
mais que o placebo, o teste não consegue ver a diferença, e um "falhou" nele não diria nada
sobre a lei dela.

## Checagens dev (sementes 101–103)

1. **dev101**: só as perguntas do sonho. Todos os sonhos baixaram a surpresa igual (sem treino
   extra 0,054; REVER 0,0386; LEI 0,0384; RETO 0,0392; VERDADE 0,0388) e nem a lei verdadeira
   venceu o reto (1/3). Diagnóstico: o preditor do córtex olha UM quadro à frente (0,2 s) em
   células de 16 px; nessa escala curva e reta são o mesmo. Daí o V0 e a pergunta 2.
2. **dev101b**: + o filme imaginado × real. Lei < reto em 3/3 vidas (aos 2 s, mediana dos
   mundos: giro 0,00038 × 0,00060; mola_h 0,00062 × 0,00082); lei < lei do outro mundo na
   mola_h em 3/3. O sonho: igual ao dev101 (V0 1/3).

## Previsões

- **V0 provavelmente FALHA** (~80%), e S1/S2 ficam sem valor. A consequência para a mente viva:
  o modo de divagar "sonhar com o laboratório" NÃO entra neste transplante (não há evidência
  de que ensine a visão que ela tem hoje); o vídeo mental entra como IMAGINAÇÃO mostrada no
  painel e conferida contra o real.
- **M1 provavelmente PASSA** (~85%) e **M2 provavelmente PASSA** (~75%).

## RESULTADO

(preenchido depois da rodada registrada, sem mudar nada acima)
