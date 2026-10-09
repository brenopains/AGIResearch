# PRE-REGISTRO — o residuo do farmaco nomeia o alvo que ninguem anotou

Escrito ANTES de baixar/olhar qualquer medida. 16/09/2026.

## Pergunta

Todo farmaco tem alvo anotado. Depois de escrever a parte do efeito que o
alvo anotado explica, o RESIDUO ainda carrega estrutura? E essa estrutura
nomeia um segundo gene, reprodutivelmente, em linhagens que nao foram usadas
para achar?

## Dados

- TAHOE-100M (Arc/Vevo, fev/2025): 1.100 farmacos x 50 linhagens, single-cell.
  Uso PSEUDOBULK por (farmaco, dose, linhagem). Assinatura = log-fold vs DMSO
  da mesma linhagem/placa.
- X-ATLAS/ORION (Xaira, jun/2025): Perturb-seq genoma inteiro. Assinatura de
  knockdown por gene. Uso como DICIONARIO de assinaturas genicas.
- Alvos anotados: ChEMBL / DrugBank (o "ja sabido", que vira first-writer).

## Arquitetura (é a do nucleo, nao um metodo novo)

  camada 0 (first-writer)  eixo de estresse/ciclo celular — o que TODO farmaco
                           em dose alta faz. NAO residualizo: escrevo primeiro.
  camada 1 (first-writer)  assinatura do(s) ALVO ANOTADO(S) do farmaco, tirada
                           do X-Atlas. Tambem escrita primeiro.
  residuo                  o que sobrou.
  camada 2 (o achado)      qual gene do X-Atlas comprime o residuo, em bits.

Um par (farmaco, gene) so e ACEITO se a camada 2 paga bits acima do limiar
depois que 0 e 1 ja escreveram.

## Portoes, numerados, com previsao ANTES

- **G0 — o dado existe e alinha.** >= 300 farmacos com alvo anotado presentes no
  X-Atlas, e >= 15 linhagens em comum. *Previsao: passa (~85%).*
- **G1 — controle positivo (o gabarito).** Farmacos de polifarmacologia CONHECIDA
  (inibidores de quinase multi-alvo: dasatinibe, sunitinibe, sorafenibe,
  ponatinibe, midostaurina...). Para esses, escondo o 2o alvo da anotacao e
  exijo que a camada 2 o RECUPERE. *Previsao: >= 4/10. Se der 0, o metodo nao
  ve mecanismo e eu paro aqui — sem consertar o estrato ate passar.*
- **G2 — nulo permutado.** Embaralhar rotulo de farmaco DENTRO de estrato
  (linhagem x faixa de dose). Falsas descobertas esperadas <= 1.
  *Previsao: passa (~80%) — o nulo nunca nos deu bits positivos.*
- **G3 — nulo do duto.** Rodar TODO o pipeline sobre substrato aleatorio que
  preserva a estrutura de correlacao do transcriptoma. Separa relacao-do-dado
  de relacao-da-derivacao. *Previsao: mata parte dos aceitos (~40% dos
  aceitos caem). Se matar tudo, o achado era do duto.*
- **G4 — extrapolacao.** Achar em metade das linhagens, replicar na METADE
  ESCONDIDA. E no nivel do EVENTO (familia de farmaco / complexo genico), nao
  do simbolo — licao do 9p21. *Previsao: >= 50% dos aceitos replicam.*
- **G5 — placebo pareado.** Farmaco com alvo anotado EMBARALHADO (camada 1 fica
  errada de proposito). Se o placebo produzir tantos aceitos quanto o real, o
  que estou medindo e tamanho de vocabulario, nao mecanismo — foi o que matou
  o `vocabulario-grande-piora`. *Previsao: placebo <= 25% do real.*
- **G6 — NOVIDADE.** Aceito sobrevivente que NAO esta em ChEMBL/DrugBank como
  alvo conhecido, e que nao aparece na literatura no nivel do evento. Busca de
  literatura e o portao final, nao a base de dados (licao do SynLethDB).
  *Previsao honesta: 20% de chance de sair >= 1 par genuinamente inedito que
  sobreviva a todos os portoes anteriores.*

## Apostas erradas que eu declaro agora

- Se G1 falhar, NAO vou afrouxar o estrato ate passar. Calibrar no gabarito foi
  o que eu me recusei a fazer na hipermutacao e me recuso aqui.
- Se G5 empatar com o real, o resultado e NULO e eu digo isso.
- Se sobrar so polifarmacologia ja anotada, o resultado e "o duto funciona,
  achado nao e inedito" — igual ao DepMap. Eu digo isso e nao enfeito.

## O que conta como PULAR DA CADEIRA

Par (farmaco aprovado, gene) que: passa G2-G5, nao esta em nenhuma base de
alvo, nao esta na literatura, e o gene tem ligacao com doenca sem tratamento.
Isso e uma hipotese de reposicionamento com mecanismo nomeado — testavel em
bancada por qualquer laboratorio em semanas.
