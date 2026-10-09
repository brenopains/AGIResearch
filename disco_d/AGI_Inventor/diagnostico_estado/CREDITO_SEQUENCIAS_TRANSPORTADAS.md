# Consequências inteiras ensinam pesos de retorno

A decomposição mostrou grande erro na leitura local de recompensa. Isso não
prova que as33funções sejam suficientes nem que aprender nova representação
seja desnecessário. Primeiro intervir no objetivo de aprendizado, mantendo
representação e transporte congelados, para separar essas possibilidades.

Para uma sequênciaP, construir uma linha de créditoX_P: partir dephi(H0),
somar essa função na coluna da ação executada, transportar porM_a e repetir.
Então `retorno_previsto=X_P*w`, com132pesosw (quatro ações vezes33funções).
Com corpos/operadores fixos, ajustarwé um subproblema linear. Isso NÃO dá
receita geral para qualquer busca discreta/ARC. Não usamos medida de promessa.

Treino exclusivamente em todas as32raízes mistos97:512pares de prefixos1..16,
espelhados por antissimetria. Retornos vêm das recompensas já observadas;
não são512experiências independentes nem há novas chamadasALE. Normalizar
matriz de crédito por uma escala escalar de treino e ridge0,05 sobre a
correção dos pesos locais originais. Reportar também resíduo de mínimos
quadrados sem regularização,rcond1e-10, apenas no treino. Não interpretar
esse cálculo numérico como impossibilidade matemática ou garantia de teste.

Corpos novos congelados antes de ler103 neste ajuste. Esse banco já foi visto
em pesquisas anteriores: desenvolvimento, não confirmação independente.
32pares por jogo/horizontes4/8/16;16 é principal. Mesmo código nos dois jogos,
corpos separados. Controles ajustados com os mesmos512alvos prefixais:
alvos embaralhados por comprimento/identidade, contexto médio, transporte
identidade, histórias embaralhadas por raiz. Referências: pesos locais
congelados e previsãozero com desempate uniforme. Escala computacional e
tempo de seleção/ajuste entram no relatório; família numérica não é inédita.

Portão fixado: no horizonte16, menor arrependimento eMSE que todos os controles
em AMBOS os jogos. Se falhar, preservar negativo; sem selecionar outro
primário/horizonte. Mesmo passar seria um filtro de desenvolvimento, nãoAGI.
Reportar acurácia local da nova cabeça no predecessor real: melhorar retorno
pode vir de compensação de erro de dinâmica. Não chamar os novos pesos de
recompensa causal correta sem evidência adicional.

Fontes, dados e corpos anteriores comSHA256 antes do ajuste; novos corpos
antes do teste. Executor congelado reexecuta todas as comparações. Auditor
independente soma crédito mediante matrizes aumentadas e confirma a equação
normal do sistema resolvido. Não executar código arbitrário aprendido.
Teto principal120s, CPU com até dois threadsBLAS, sem coleta ou promoção.

```powershell
cd D:\AGI_Inventor\diagnostico_estado
py -3.11 -B testes_credito_sequencias.py
py -3.11 -B piloto_credito_sequencias.py --congelar
py -3.11 -B piloto_credito_sequencias.py --executar
py -3.11 -B auditar_credito_sequencias.py
```
