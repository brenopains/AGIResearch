# Seleção de experiências: vantagem artificial da arena

Em 09/10/2026, depois de atualizar a mente com o laboratório, o diário registrou
uma visita a vídeo da internet seguida de 231 escolhas consecutivas da arena.
O painel mostrava custo de viagem de aproximadamente 0,5 s para a arena e de
5–14 s para portas da internet, com pontuação da arena cerca de dez vezes maior.

O retorno das ações de exploração já é aprendido em `passo()` como sucesso de
`ganho / duração total`, comparado ao retorno típico da vida. `escolhe()` ainda
dividia esse posterior pelo custo de viagem. Uma probabilidade limitada a 1
dividida por 5 segundos não conseguia competir com aproximadamente 0,5 dividida
por 0,5 segundo. A incerteza de Thompson não removia essa barreira.

A seleção de exploração agora compara o posterior de retorno por segundo
vezes a probabilidade de a ferramenta funcionar. A medição dos tempos e sua
exibição no painel continuam disponíveis. A busca dirigida por lacunas mantém
sua regra anterior. Não foram apagados registros, pesos, teorias ou experiências,
nem inseridas cotas para internet, bloqueios da arena ou alternância por horário.

Comparação no checkpoint real de 10.848 passos, mesma página, 500 sementes
pareadas, sem executar ações nem treinar:

| Escolha | Antes | Correção |
| --- | ---: | ---: |
| Arena | 500 | 45 |
| Vídeo da internet | 0 | 396 |
| Experimento ativo | 0 | 54 |
| Pequena web | 0 | 3 |
| Wikipédia inglesa | 0 | 2 |

Essa é uma comparação das decisões no checkpoint fixo, não uma previsão da
distribuição futura durante aprendizagem. Uma atividade que continue pagando
mais pode ser preferida; a correção remove a vantagem extra da viagem barata.

Regressões em `teste_selecao.py`: retornos iguais competem mesmo com custos
diferentes; a opção lenta com melhor retorno vence; ferramentas que falham
continuam sendo penalizadas. O teste de integração `teste_laboratorio.py vidas/viva`
também verifica a atualização em uma cópia da vida real.

Backup anterior à correção:
`D:\AGI_Inventor\backups\antes_corrigir_selecao_2026-10-09_230113`.
