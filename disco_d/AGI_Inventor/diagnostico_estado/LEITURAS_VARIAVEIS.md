# Piloto: leitura com posição variável e relações de fragmentos

Este protocolo é escrito antes de executar a extração e seleção desta rodada.
O objetivo maior de inteligência humana permanece não alcançado. Este teste
examina a linguagem das perguntas produzidas a partir de falhas, usando os
mesmos programas e experiências dos ensaios anteriores.

De cada resíduo de par gerado apenas em 97 episódios 0–3, extrair o componente
conexo de maior energia em cada sinal, com limiar de 25% do máximo daquele
sinal. Cada componente vira um fragmento normalizado, com seu corpo completo
armazenado. Procurar a correspondência mais forte desse fragmento em qualquer
posição dos pixels do efeito observado sob o programa. A leitura fornece
quatro números: duas forças de correspondência e deslocamento relativo x/y
entre os dois fragmentos. Não afirmar identidade, objetos ou velocidade.
Empates numéricos usam média das posições a 10⁻⁷ do máximo. Observação sem
correspondência produz zero. Não há limiar ou região específica de jogo.

Aprender a prever essas leituras dos dois quadros iniciais, usando as mesmas
experiências intermediárias de 97 e limites de 512/256 amostras do extrator
anterior. Verificar as leituras não lineares dos rótulos compostos contra os
ramos executados; erro acima de 10⁻⁶ impede o treino. A base preditiva de 32
dimensões e os programas são os mesmos, sem mais intervenções ALE.

103 é desenvolvimento e já foi visto anteriormente. Selecionar conjuntos de
leituras pelo arrependimento de retorno em 128 passos, com **oito** episódios.
Exigir ganho em pelo menos dois episódios e nenhuma piora nos demais. Um
episódio que já tem erro zero pode empatar, evitando o portão impossível
identificado antes. Escolher menor arrependimento médio, depois menor número
de leituras. As propostas de conjuntos são idênticas para a leitura variável,
fragmentos aleatórios do mesmo tamanho e associação de treinamento embaralhada.
Comparar também com a leitura linear fixa e respostas reais como oráculo.

Só abrir confirmação em uma semente intacta se o braço previsto superar
**simultaneamente** base de estado, melhor ação fixa escolhida em 97, leitura
aleatória selecionada e associação embaralhada selecionada. Nenhuma melhora
em desenvolvimento será chamada de confirmação ou de descoberta científica.
Em caso negativo, registrar o limite testado sem anunciar impossibilidade
de inventar representações. Tempo de CPU, hashes e margens serão publicados.

Controles de execução:

```powershell
py -3.11 -B testes_leituras_variaveis.py
py -3.11 -B testes_portao_leituras.py
```

Piloto, por jogo:

```powershell
py -3.11 -B extrator_leituras_variaveis.py breakout
py -3.11 -B avaliar_leituras_variaveis.py breakout
```

Repetir com `pong`. A busca e as definições ainda usam primitivas humanas de
espaço, correlação, sinal e redução; o que está sendo testado é a extração
automática e utilidade de suas composições, não aprendizado a partir de zero.

SHA-256 registrado antes da extração:

| Fonte | SHA-256 |
|---|---|
| `leituras_variaveis.py` | `b6d99cd4b8f237e4fd818f959fcb13682b6b1437c42f6e4136a8bcbc69916b4b` |
| `extrator_leituras_variaveis.py` | `4aa64d7a6be9dbbafe9d01ca05373290236c8e395fa12aa395dd302f342456a2` |
| `avaliar_leituras_variaveis.py` | `600e78132e1b1bb104f3c3e5eed419c052adad021b8abe071f6cb21199be7d84` |
