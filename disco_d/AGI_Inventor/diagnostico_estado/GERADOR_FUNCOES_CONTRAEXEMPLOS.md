# Crescer funções a partir de distinções que falham

Objetivo da etapa: gerar funções executáveis de dados crus, adicionar as que
ajudam a decisão fora do episódio de origem e compô-las pelos operadores de
ação. É um mecanismo limitado de pesquisa; mente humana e descoberta científica
autônoma continuam como objetivo maior não demonstrado.

A experiência anterior mudou a ação seguinte: ajustando só os pesos com
representação e transporte fixos, as decisões pioraram nos dois mundos.
As funções aqui propostas devem acrescentar uma leitura fora do espaço
linear das funções atuais, medida numericamente no treino, não apenas girar
ou reponderar esse espaço. Não é prova de independência funcional universal.

## Construção, seleção e leitura

Base nova:33funções (constante e32RBF de observações) treinadas exclusivamente
nos episódios0–3 do banco97. A cabeça de recompensa é reajustada deixando
cada episódio de construção fora, para revelar resíduos fora do próprio
ajuste. As funções propostas são:

- comparação não linear: `tanh(flatten(história)*direção - limiar)`; direção
  e limiar vêm da diferença e ponto médio de dois registros de episódios
  diferentes com a mesma ação e resultados diferentes;
- produto de duas dessas comparações. Os pais não precisam ter passado
  isoladamente para o produto ser avaliado.

Nenhum objeto, região, cor, atraso ou regra do jogo é listado no gerador.
Ainda existe uma linguagem matemática escolhida por nós: não é invenção
irrestrita. Gerar unidades e congelá-las tem precedentes, por exemplo
[Cascade-Correlation, Fahlman e Lebiere](https://proceedings.neurips.cc/paper/1989/file/69adc1e107f7f7d035d7baf04342e1ca-Paper.pdf).
Não afirmar que a técnica, ou sua integração aqui, seja inovação mundial.

No máximo24comparações e8produtos:25% dos candidatos são composições.
No máximo4funções adicionadas. Quatro posições, preenchidas comzero quando
não usadas, mantêm a dimensão37 em todos os corpos e controles. A geração
pode produzir menos funções; registrar o déficit, sem fabricar candidatos.
Custos reais e postos também são reportados; dimensão igual não significa
orçamento ou dificuldade de ajuste exatamente iguais.

Para cada candidata, reajustar recompensa e transporte só nos episódios0–3.
Seleção nos episódios4–5: menor arrependimento E menorMSE das diferenças de
retorno em16blocos, com melhora em pelo menos duas raízes distintas. As mesmas
raizes são sempre consideradas; não escolher pelos retornos. Usar novidade
numérica `MSE de projeção na base atual / variância > 1e-4`, apenas no treino.
Como a seleção depende de validação, ganho nessa parte é desenvolvimento.
Registrar falta de margem quando o arrependimento do corpo já forzero.

## Controles e avaliação

Mesmos limites, linguagem, seletor e dados para quatro geradores: pares por
contraexemplo; pares por resultados embaralhados por ação; pares aleatórios;
direções aleatórias. Os modelos finais de todos são ajustados com consequências
verdadeiras nos episódios0–3; o embaralhamento muda apenas a geração.
Referências: nenhuma função nova; retirar integralmente as coordenadas novas
do corpo escolhido, mantendo o bloco antigo de seus operadores; retirar só
contexto inicial; previsãozero com desempate uniforme.

Congelar fontes antes da construção, base e dicionários antes da seleção,
corpos escolhidos antes da avaliação6–7 em97 e avaliação103. Dados desses
mundos/sementes já foram vistos em trabalhos anteriores; são separados para
este novo mecanismo, não confirmação independente. Palavras inteiras não
são alvos de ajuste: só transições locais. Mesma regra nos dois jogos, pesos
separados: não é transferência de uma função aprendida entre domínios.

Portão de desenvolvimento fixo, sem mudar depois: em103/16blocos, primário
com ao menos uma função escolhida e menor arrependimento E menorMSE que
TODOS os controles em AMBOS os jogos. Mesmo passar não demonstraAGI.
Avaliar também4/8blocos e97/6–7, sem trocar o horizonte/banco principal.
Nenhum resultado aproximado recebe rótulo de prova exata.

Auditor reexecuta funções por distância aos pares (forma diferente da projeção),
corpos por matrizes aumentadas, registros de seleção e controles. Hashes de
todos os parâmetros, dados e fontes; eventos de geração, rejeição e escolha
reais emJSONL. Nenhum Python arbitrário aprendido é executado.

Teto principal120s, CPU/BLASdois threads, nenhum passoALE novo, até4funções.
Não ampliar linguagem/budget/critério para salvar resultado ruim. O piloto
não instala serviços nem altera o projeto original.

```powershell
cd D:\AGI_Inventor\diagnostico_estado
py -3.11 -B testes_funcoes_contraexemplos.py
py -3.11 -B piloto_funcoes_contraexemplos.py --congelar
py -3.11 -B piloto_funcoes_contraexemplos.py --executar
py -3.11 -B auditar_funcoes_contraexemplos.py
```
