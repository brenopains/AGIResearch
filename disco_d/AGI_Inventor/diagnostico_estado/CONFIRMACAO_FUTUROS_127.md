# Confirmação independente: composição distribuída em Breakout

Antes deste pré-registro, não existe banco de Breakout na semente 127. O
mesmo número foi usado em um mundo de órbitas; isso não expôs estes episódios
de Breakout. Treino somente 97; desenvolvimento já visto 103. Não ajustar
hiperparâmetros, estado, crédito ou controles na confirmação.

O ganho em 103 foi 0,296875 unidade de arrependimento contra ação constante
em 128 passos, com sete episódios melhores e nenhum pior. Esse número não
é confirmação. Pong não mostrou ganho contra ação constante, portanto este
ensaio não confirma generalidade entre jogos.

Congelar o manifesto antes da coleta. Incluir fontes transitivas locais,
treino de estados/retornos, compressão, definições usadas pelo extrator,
predições de treino do estimador direto e o resultado de desenvolvimento.
O manifesto deve verificar os hashes antes e depois de executar; o resultado
terminado não pode ser sobrescrito.

Modelo: o mesmo kernel de oito vizinhos e os mesmos corpos de distribuição,
colapso à média e controles de `futuros_distribuidos.py`. A política recebe
somente os dois quadros iniciais. Respostas dos ramos futuros e seus retornos
ficam exclusivamente no avaliador. Comparar também com regressão direta de
preferências no estado previsto, ajustada em 97.

Medida primária: arrependimento médio em 128 passos, 64 raízes, oito episódios.
Exigir vantagem média e limite inferior bootstrap de 95% acima de zero contra
cada braço: ação constante escolhida em 97, estado médio, sucessores
embaralhados, recompensas embaralhadas, bancos de ação trocados e regressão
direta. Bootstrap de episódios completos, 10.000 amostras, semente 9757.
Exigir melhora em pelo menos dois episódios contra cada controle e nenhuma
piora por episódio contra ação constante. Todos os controles entram no portão.

Se não houver contraste ou margem contra a ação constante, inconclusivo.
Caso contrário, qualquer condição ausente significa falha do pré-registro.
Reportar também 64 passos como resultado secundário, sem mudar o veredito.
Uma confirmação provaria somente ganho entre quatro programas constantes,
neste jogo e nesta distribuição de episódios. Não prova novidade científica,
resolução de problemas abertos, invenção de representação ou inteligência humana.

Coleta curta com replay de 64 raízes e horizonte com certificação de 256
ramos curtos. Teto agregado 65.000 passos ALE; registrar passos e tempo.
Intervenções de treino custam experiência: este não é um ensaio de aprendizado
espontâneo a partir de apenas dois quadros. O modelo final recebe dois quadros.

```powershell
py -3.11 -B testes_confirmacao_futuros.py
py -3.11 -B confirmar_futuros_distribuidos.py --freeze
py -3.11 -B confirmar_futuros_distribuidos.py
```

Artefatos em `resultados/breakout_confirmacao_futuros_127.*`; bancos
`breakout_efeitos_locais_127.*` e `breakout_horizontes_127.*`.
