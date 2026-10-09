
---
## PODER MEDIDO EM VERDADE PLANTADA (antes de ver o dado real)

Tres defeitos do meu proprio duto, achados no sintetico:

1. **Dupla-imersao na camada 0.** A base de estresse era estimada no mesmo dado
   de onde e removida. Vazamento medido: razao sd(<r,s>)real/permutado = 1,292
   contra 0,994 com base independente. Conserto: ajuste cruzado por dobras.
2. **Espacos diferentes.** `<r,s>` vivia no subespaco do residuo e `<s,s>` no
   espaco inteiro. Conserto: o dicionario passa pela MESMA projecao.
3. **Nulo no lugar errado.** Permutava depois da projecao. Conserto: nulo do
   duto — permuta o dado CRU e re-roda tudo, estimativa de estresse inclusive.

Com os tres consertos, sob a regra real de aceite (o mesmo gene ser o topo em
>= 40% das linhagens, com bits acima do limiar do nulo do duto):

```
forca do 2o alvo    n   aceito  acertou  espurio
    0,80           24      24       24        0
    0,40           24      24       24        0
    0,20           24      24       24        0
    0,10           24       0        0        0   <- cego
    nenhum         24       0        -        0   <- ZERO falsos
```

**Limite declarado:** o duto ve segundo alvo com amplitude >= 1/10 do alvo
anotado e >= 1/15 do estresse. Abaixo disso nao ve. Falso-positivo zero sob
replicacao; com limiar numa condicao so, era 14/80 — a replicacao entre
linhagens e que carrega a calibracao, nao o limiar.

---
## G7 — PREVISAO CEGA ENTRE MODALIDADES (escrito antes do dado cair)

Correcao aceita: convergencia entre canais mata artefato do INSTRUMENTO, nao
confundidor do MUNDO. Tipo celular causa expressao, dependencia e morfologia ao
mesmo tempo. Os 9/11 validam o instrumento, nao identificacao causal.

### Desenho
O duto de expressao recebe o farmaco D SEM o alvo anotado (camada 1 desligada)
e emite um gene G* por conta propria. O alvo verdadeiro fica escondido da
proposta E da avaliacao. Depois, e so depois, o canal de viabilidade pergunta:
  corr(dependencia CRISPR de G*, lfc do PRISM de D) > 0 ?

### Dois desfechos, medidos separadamente
  (a) G* == alvo anotado          -> validacao de instrumento (recupera o sabido)
  (b) G* != alvo anotado E preve  -> candidato com apoio entre modalidades

### Nulo pareado
G* trocado por gene sorteado CASADO em media e variancia de dependencia (senao
o nulo compara contra genes sem variancia, que nao podem correlacionar com nada).

### Controles que discriminam confundidor de linhagem (a correcao do usuario)
  C1 congenerico  : D1 e D2, alvo anotado igual, quimica diferente -> mesmo G*?
                    linhagem nao sabe qual molecula foi usada.
  C2 dose-resposta: o par (D,G*) tem que ficar mais forte de 0,05 -> 0,5 -> 5 uM.
                    confundidor de linhagem e CONSTANTE na dose.
  C3 dentro do tecido: sobrevive como PORTAO, nunca como pre-processamento
                    (foi residualizar que derrubou o v2.1 de 9/11 para 1/5).

### Previsao honesta, antes de rodar
40% de que a proposta cega preveja viabilidade acima do nulo casado.
Razao da duvida, declarada: casar assinatura acha POSICAO NA VIA, nao parceiro
de ligacao. O gene cuja assinatura de knockdown casa melhor pode ser efetor a
jusante, nao o alvo fisico. Viabilidade depende do alvo fisico. Se (a) der alto
e (b) baixo, a leitura e "o duto acha a via, nao a molecula" — e isso e um
achado, nao um fracasso.

### Correcoes do usuario, incorporadas (mudam codigo, nao so texto)

**C1 congenerico vira PREVISAO, nao observacao.** Do jeito que eu tinha escrito,
eu escolheria a semelhanca depois de ver D2. Agora: o duto roda em D1, emite G*,
e eu REGISTRO "prevejo que D2 (mesmo alvo anotado, quimica diferente) tambem
emitira G*" ANTES de rodar D2. Acerto/erro contam como previsao.

**C2 dose: especificidade, nao magnitude.** Toxicidade geral tambem cresce com
dose. `dose x magnitude do residuo` NAO prova alvo. O teste e a progressao na
COMPONENTE DE G especificamente:
    alpha(D,G,dose) = <residuo(D,dose), s_G> / <s_G,s_G>
tem que crescer 0,05 -> 0,5 -> 5 uM. E a norma do residuo inteiro NAO pode
crescer na mesma proporcao — se crescer, e toxicidade e o par cai.

### MATRIZ DE INTERPRETACAO (para eu nao ser premiado por qualquer resultado)

| expressao | viabilidade | dose | congenerico | leitura |
|---|---|---|---|---|
| sim | sim | sim | sim | mecanismo forte — candidato de bancada |
| sim | sim | sim | nao | mecanismo provavel, especifico da molecula |
| sim | sim | nao | sim | associacao funcional; alvo pouco especifico |
| sim | sim | nao | nao | associacao funcional fraca; nao emitir |
| sim | nao | sim | - | duto pega VIA/ESTADO, nao efeito funcional |
| sim | nao | nao | - | hipotese nao atravessa modalidade. NULO. |
| nao | sim | - | - | mecanismo fora do transcriptoma; duto cego a ele |
| nenhum replica | | | | hipotese nao generaliza. NULO. |

Casos "NULO" sao resultado e eu os reporto como resultado, nao como "quase".

### Limite de dominio que eu preciso declarar

O dado de farmaco e OBSERVACIONAL. O sistema nao pode agir para expandir o
alcancavel — as condicoes (farmaco, linhagem, dose) sao as que existem. Pelo
resultado de `alcance.py` (informatividade e monotona nao-crescente sob
eliminacao; a unica fonte de valor prospectivo e ALCANCE), este dominio
exercita as fases 1, 2, 4 e 6 do ciclo e NAO exercita a fase 3 no sentido que
importa. Fase 3 precisa de um mundo onde agir muda o alcancavel.

---
## RESOLUCAO DO DICIONARIO (medida depois de baixar, antes de qualquer associacao)

QC meio-a-meio (lotes impares x pares). Primeira leitura enganosa: mediana 0,785,
100% acima de 0,3 — bom demais. O teste contra o GENE ERRADO desmontou:

```
k562 bruto     r_proprio 0,800   r_OUTRO 0,749   diferenca 0,051
k562 -8 PCs    r_proprio 0,455   r_OUTRO 0,397   diferenca 0,059
```

~94% da "confiabilidade" e estrutura COMPARTILHADA entre knockdowns, nao
identidade do gene. Derrubar X e derrubar Y da assinaturas que correlacionam
0,75 entre si.

Recuperacao rank-1 (metade impar de X -> metade par de X e o topo entre ~1.440):
```
k=0  11,1% | k=8  13,1% | k=24  21,4% | k=64  21,4% | k=128  21,3%   (acaso 0,07%)
top10: 22,8% / 26,7% / 39,9% / 37,9% / 37,2%
```

**Consequencias, todas obrigatorias:**
1. `camada0` usa k=24 (medido, satura). Eu estava usando 8, chutado.
2. O duto NAO PODE emitir um gene. Teto de resolucao top-1 = 21%. Emite
   CONJUNTO ordenado (top-10, 40%), e o canal de viabilidade passa de
   confirmacao para DESAMBIGUACAO dentro do conjunto.
3. **Aposta G6 revisada de 20% para ~8%**, por resolucao do instrumento medida.
   Revisao para BAIXO por instrumento mais fraco e legitima; para cima depois
   do resultado seria trapaca. Registrado antes de rodar a associacao.
4. Teto conjunto declarado: se expressao poe o gene certo no top-10 em 40% e a
   viabilidade discrimina dentro do conjunto, o teto do sistema e ~40%. Acima
   disso e artefato.

## RESTRICOES DO ESPACO DE CANDIDATOS (medidas antes da associacao)

- Alvo anotado presente no dicionario: so 45 de 264 farmacos (28 aprovados).
  Para os outros 219 o alvo fisico NAO esta no espaco de candidatos, logo o duto
  nao pode achar "segundo alvo" deles — so pode nomear NO DE VIA essencial.
- Dicionario arbitravel pela viabilidade: 786 de 1.970 (40%). 831 (42%) tem
  dependencia universal (>90%) e sao cegos ao segundo canal.
- Regra do canal, medida: funciona com dependencia CONDICIONAL, falha com
  UNIVERSAL (PARP1 2% ate CDK4 74% pegaram; PSMB5 95% falhou).

**A reivindicacao que este desenho PODE fazer, e so ela:** para farmaco cujo
resto, depois de escrito o estresse e o alvo anotado, e melhor explicado por um
gene do conjunto ARBITRAVEL, e cuja dependencia preve sensibilidade no PRISM.
Nao e "segundo alvo fisico" — e "no de via essencial condicional". Vou reportar
com esse nome.
