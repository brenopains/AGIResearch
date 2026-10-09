# Executar e inspecionar o teste de memória

Pasta de trabalho: `D:\AGI_Inventor\diagnostico_estado`. O projeto original
não foi alterado. Os quatro bancos e12corpos desta rodada já existem; scripts
recusam sobrescrita para preservar evidências. Não remover resultados para
procurar rodada favorável. Reproduções devem usar uma cópia separada com as
mesmas dependências e bancos antigos exigidos pelo manifesto.

Verificar implementação e auditar evidências existentes:

```powershell
py -3.11 -B testes_memoria_residual.py
py -3.11 -B auditar_memoria_residual.py
```

Sequência usada na cópia de trabalho, antes de existirem os artefatos:

```powershell
py -3.11 -B coletar_transicoes_mistas.py breakout --seed 97
py -3.11 -B coletar_transicoes_mistas.py pong --seed 97
py -3.11 -B coletar_transicoes_mistas.py breakout --seed 103
py -3.11 -B coletar_transicoes_mistas.py pong --seed 103
py -3.11 -B avaliar_memoria_residual.py breakout --ajustar
py -3.11 -B avaliar_memoria_residual.py pong --ajustar
py -3.11 -B avaliar_memoria_residual.py breakout --avaliar
py -3.11 -B avaliar_memoria_residual.py pong --avaliar
```

Coleta guarda história passada real, sondas custeadas e eventosJSONL.
Ajuste recebe somente97 e congela corpos NPZ/JSON e hashes. Avaliação usa103
depois desse congelamento; rollout recebe apenas observações iniciais reais
e seu programa. Os futuros só medem erro/retorno no avaliador.

Saídas `resultados/{jogo}_transicoes_mistas_{semente}*` e
`resultados/{jogo}_memoria_residual*`. A primeira tentativa de coleta97,
interrompida por problema de imagem após restauração, permanece separada;
`CORRECAO_COLETA_MISTA.md` explica código, planos e custos parciais preservados.

Resultado: ambos os portões negativos, sem promoção. Aprendizagem de projeções
numéricas da história foi implementada; melhoria de decisão geral ou mente
humana não foi demonstrada.103 é desenvolvimento, não confirmação nova.
