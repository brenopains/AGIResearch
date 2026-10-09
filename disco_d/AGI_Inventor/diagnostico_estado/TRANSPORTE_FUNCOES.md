# Executar programas transportando funções, sem criar imagens médias

Hipótese local, não receita deAGI: o fracasso da leitura direta de palavras
pode ser atacado ligando concatenação à execução aprendida. Para uma função
observávelf e uma açãoa, aprender `M_a f(H)` a partir de
`continuação*f(H_depois_a)`. Aprender a recompensar_a separadamente. Um programa
passa a ter corpo `v_vazio=0; v_aQ=r_a+M_a*v_Q`. Não estimar a próxima imagem
nem pedir que a interpolação de palavras descubra sozinha essa lei.

Implementação inicial delimitada:32suportes escolhidos por cobertura nas
histórias cruas do treino; funçõesRBF desses suportes mais constante. Regressão
ridge0,05, quatro ações anônimas, transições válidas do banco misto97. O alvo
conserva recompensa e término real do bloco. Não é invenção irrestrita de
representações: família e regra de geração foram escolhidas por nós.

Somente o mesmo código nos dois jogos, com ajuste próprio em cada um. Isso
não é reutilização de um corpo aprendido entre domínios nem aprendizagem de
mundos humanos. Não aumentar suportes/grau/budget depois de resultados ruins.

Piloto de desenvolvimento, sem coleta:32pares mistos103 por jogo, horizontes
4/8/16blocos. Ambos os bancos já vistos; não chamar esse piloto de confirmação.
Corpos congelados antes de ler alvos103 neste ajuste. Controles:

- transporte com alvos de recompensa/continuação/sucessor embaralhados juntos
  dentro de cada ação, preservando seus marginais;
- corpo primário com contexto inicial substituído por sua média de treino;
- corpo primário sem transporte, acumulando só recompensas lidas no início;
- operadores do primário com identificação de ações permutada[2,3,0,1];
- previsãozero, com desempate uniforme.

Portão de desenvolvimento conjunto, fixado antes do ajuste: no horizonte16,
arrependimento com empates uniformes eMSE das diferenças de retorno menores
que todos os controles em ambos os jogos. É apenas um filtro local; mesmo
passar não demonstraráAGI. Se não passar, registrar negativo e preservar.
Não escolher outro primário/horizonte. Nenhuma estatística de significância
ou promessa nova a partir desse banco já explorado.

Reportar também resíduos locais em transições103 com predecessor real, custo,
norma dos operadores, corpos completos eJSONL de cada par/horizonte. A lei
algébrica dos produtos é exata por construção, mas não certifica o mundo:
o erro físico `phi(H0)*M_a - continuidade*phi(H1)` é medido separadamente.
Erros locais pequenos não substituem decisão final. Nenhuma sonda assinada
ou estado interno como entrada. Não executarPython arbitrário aprendido.

Comandos no diretórioD:\AGI_Inventor\diagnostico_estado:

```powershell
py -3.11 -B testes_transporte_funcoes.py
py -3.11 -B piloto_transporte_funcoes.py --congelar
py -3.11 -B piloto_transporte_funcoes.py --executar
py -3.11 -B auditar_transporte_funcoes.py
```

Limite deste piloto:120segundos de execução principal, CPU/BLASdois threads,
sem novas chamadasALE. Hashes de código, dados e corpos preservados. Ausência
de um resultado positivo não identifica perfeitamente o gargalo: podem faltar
variáveis observáveis, cobertura, forma não linear de transporte ou crédito.
