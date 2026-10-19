# Memoria espacial multivista — primeira etapa

Pedido: explorar cenas mentais 3D persistentes, com previsoes, regras, transferencia
entre dominios e preenchimento de lacunas. Esta etapa implementa a base geometrica
e sua inspecao; nao declara o pedido universal concluido.

## Hipoteses e controles

- E1: recebe apenas imagens e intrinsecos opcionais, reconstrói pelo menos dois
  planos com profundidades relativas diferentes. O avaliador guarda a verdade,
  nunca repassada ao estimador. Aceitacao: razao entre 1,6 e 2,4 para verdade 2.
- E2: uma vista integralmente reservada pode ser localizada no modelo aprendido.
  Pose usa 80% das correspondencias; mediana de erro nos outros 20% < 1,8 px.
- E3: salvar/reabrir preserva o mapa; outro video da mesma cena conserva sua
  identidade por correspondencia geometrica, sem depender de URL ou de nomes.
- E4: camera fixa, rotacao pura e cortes entre cenas independentes nao devem
  ganhar profundidade inventada.
- E5: facetas conectam apenas vizinhos triangulados, sem saltos de profundidade.
- E6: video recebido pelo corpo chega ao modulo antes da reducao para 64x64.
- UI: navegação muda o ponto de vista; suporte e retorno a vistas funcionam;
  ausencia de mapa mostra vazio; sem erros JavaScript/WebGL.

## Representacao e limites

ORB, correspondencias reciprocas, matriz essencial, triangulacao, PnP e fluxo
bidirecional estimam pontos/cameras. Sao algoritmos geometricos programados;
nao pesos previamente treinados nem um decoder aprendido dos latentes do cortex.
As cores sao observadas; as coordenadas 3D sao inferidas. A triangulacao assume
camera de perspectiva e estrutura suficientemente rigida. A focal inicial de
video desconhecido e uma convencao (.9 * lado maior), declarada no painel.
A escala e relativa. Uma geometria com erro baixo pode ainda estar errada quando
essas hipoteses falham. Suporte geometrico nao e uma probabilidade calibrada.

Facetas fazem interpolacao local entre amostras observadas, nao completam objetos
escondidos. A memoria incorpora novas vistas/amostras; ainda nao faz ajuste global
conjunto de todas as cameras e superficies nem reconstrucao densa completa.
O renderer recebe XYZ/facetas e cores; nao cria palanques, mesas ou celulas.
Semantica de objetos, dinamica em 3D, aparencia temporal, superficies escondidas,
materiais, luz/sombra, audio espacial e composicao causal entre mundos nao foram
implementados nesta etapa. Os modelos dinamicos 2D existentes seguem separados.

Videos pequenos (ex.: corpo HTTP antigo de 32x32) nao permitem esta reconstrucao.
O corpo navegador envia recortes RGB maiores e preservamos ate 32 vistas a cada
0,8 s. Sem pistas suficientes, a tentativa e registrada como sem geometria.

Indice, arquivos NumPy e JSON ficam em `vidas/<vida>/espacial`. A classe de memoria
e excluida do pickle principal; fotos da vida incluem copia da memoria espacial.
Reconhecimento compara ate 20 cenas recentes mais cenas anteriores da mesma URL.
O indice conserva todas as cenas. Cada cena limita amostras de superficie a 24 mil,
referencias de localizacao a 4 mil e historico de cameras a 120 vistas.

## Correcoes confirmadas na auditoria dos anexos

`cortex.surpresa` existe e `mundos.py` ja esta integrado na versao do Claude.
A antiga primeira previsao escolhia retrospectivamente o menor erro do episodio.
Agora usa a teoria anterior da mesma fonte, escolhida sem usar o desfecho novo.
Classificacao posterior continua servindo para aprender e revisar teorias.
O progresso compara apenas episodios com o mesmo criterio de medicao; a historia
antiga e preservada sem ser misturada com a nova metrica causal.

A estrategia ativa por discordancia tem limites registrados anteriormente.
Nao declaramos que esta mudanca a fez superar observacao passiva. A avaliacao
de ganho posterior em episodios independentes continua uma etapa de pesquisa.

## Validacao em fotografias reais

Par `aloeL.jpg`/`aloeR.jpg`, exemplos publicos do OpenCV, mantido em vida de teste.
Nenhum modelo 3D ou mapa de profundidade verdadeiro entra no estimador. Este
ensaio demonstra entrada fotografica e geometria navegavel; nao tem terceira vista
nova independente nem prova precisao 3D ou generalidade em videos da internet.
Repetir a primeira fotografia na sequencia nao equivale a evidência nova.

Fontes dos algoritmos: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
e https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
Dados: https://github.com/opencv/opencv/tree/4.x/samples/data
