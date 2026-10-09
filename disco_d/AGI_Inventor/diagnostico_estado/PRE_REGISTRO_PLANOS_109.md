# Confirmação independente dos planos mistos

Registro escrito antes de coletar qualquer arquivo `*_109.*` desta rodada.
A semente 97 treina extrator, base de 32 dimensões e quatro operadores de
transição. A semente 103 foi usada somente para desenvolvimento e inspeção.
A semente 109 será avaliada uma vez com o mesmo código, sem ajuste por jogo.

Cada jogo tem oito episódios, oito raízes por episódio e três planos de oito
blocos de oito ações, fixados em `coletar_planos_mistos.py`. O avaliador vê
somente os dois quadros iniciais e o plano. As assinaturas dos estados futuros
ficam no avaliador. Um estado terminado fica fora da medida desde a primeira
terminação; a contagem de estados válidos será publicada.

A medida primária é a média por episódio da razão de erro do contraste entre
as quatro ações de sondagem; menor é melhor, e um preditor cego vale 1. Para
ambos os jogos, no horizonte de 64 passos, exigir **simultaneamente**:

1. razão do estado inicial previsto menor que 0,75;
2. razão menor que assinatura inicial persistente e que operadores de ação
   ciclicamente permutados;
3. vantagem pareada em pelo menos 6 dos 8 episódios sobre **cada** controle;
4. limite inferior do intervalo bootstrap percentil de 95% da diferença
   pareada controle menos proposta maior que zero para cada controle. A
   reamostragem usa episódios inteiros, 10.000 amostras e RNG 7701.

Se qualquer condição falhar, a confirmação falhou; não adaptar o limiar à
semente 109. Resultados intermediários de 8–56 passos serão descritivos. O
teste avalia composição causal local em dois jogos com pixels reduzidos,
ações discretas e planos curtos; sucesso não prova planejamento, descoberta
autônoma, transferência entre domínios nem inteligência de nível humano.

SHA-256 dos scripts congelados:

| Script | SHA-256 |
|---|---|
| `coletar_efeitos_locais.py` | `1afbc809e3e5b14ae1d823db236cb6e06eb1171ab95ca9f3c8446c371dda0392` |
| `coletar_horizontes.py` | `8d89851cdbe7d070510b96a9a3f41c7a312f2bb34effb82960ad0c7fb9c6d631` |
| `coletar_planos_mistos.py` | `e06f4786f12435c2a9822d29d6d69c7686d11a522e2c12f189b26b5b18d24daa` |
| `avaliar_planos_mistos.py` | `74ead3119eca78bffbfa8845243aa7a85e6196cbc5ac9ce2d531a5c62de4d9fd` |
| `avaliar_estado_preditivo.py` | `fdcde6494a1c97eba5453700b72ba9a079d43dc327b70da69f9c955b3b6a8d8f` |

Comandos, por jogo:

```powershell
py -3.11 -B coletar_efeitos_locais.py breakout --seed 109
py -3.11 -B coletar_horizontes.py breakout --seed 109
py -3.11 -B coletar_planos_mistos.py breakout --seed 109
py -3.11 -B avaliar_planos_mistos.py breakout --seed 109
```

Repetir `breakout` com `pong`. Não usar os arquivos 109 no treino.
