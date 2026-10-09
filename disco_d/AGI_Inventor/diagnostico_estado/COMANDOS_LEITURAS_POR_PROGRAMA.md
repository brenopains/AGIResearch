# Corpos de leitura visual executáveis

Tudo nesta pasta isolada do disco D. Projeto original preservado. Nenhuma
mente de nível humano, transferência geral ou descoberta de fronteira demonstrada.

Protocolo pré-ajuste: `LEITURAS_POR_PROGRAMA.md`. Gerador e funções completas:
`leituras_por_programa.py`. Ajustador/avaliador: `avaliar_leituras_por_programa.py`.
O primário recebe apenas história visual e palavras; não recebe assinatura
de sondas, estado oculto nem respostas reservadas. Pixels do banco são
cinza, reduzidos por médias10×10 a21×16 e normalizados por255. São16observações
passadas com frame_skip4 do ALE; não são imagens RGB integrais. Essa compressão
pode perder informação e faz parte do limite do ensaio.

Na pasta `D:\AGI_Inventor\diagnostico_estado`:

```powershell
py -3.11 -B testes_leituras_por_programa.py
py -3.11 -B testes_identificabilidade_leituras.py
py -3.11 -B auditar_leituras_por_programa.py
py -3.11 -B diagnosticar_identificabilidade_leituras.py
```

Ajuste e avaliação **já concluídos**. Estes comandos geraram os resultados;
os arquivos existentes fazem o programa recusar sobrescrita, de propósito:

```powershell
py -3.11 -B avaliar_leituras_por_programa.py breakout --ajustar
py -3.11 -B avaliar_leituras_por_programa.py pong --ajustar
py -3.11 -B avaliar_leituras_por_programa.py breakout --avaliar
py -3.11 -B avaliar_leituras_por_programa.py pong --avaliar
```

Não apagar bancos/corpos para conseguir repetir; conferir pela auditoria.
Para modificar algoritmo, usar nova versão/pasta com novo protocolo e hashes.
O banco103/RNG9811 já foi visto; qualquer uso dele permanece desenvolvimento.

Resultados por raiz/horizonte: `resultados/{jogo}_leituras_por_programa_novas_103.json`.
Eventos reais: arquivoJSONL correspondente. Corpos, exemplos de leituras e
origem congelados em `resultados/{jogo}_leituras_por_programa_97_manifesto.json`.
As funções exportadas contêm viés, termo linear, templates, termos quadráticos
e normalização. A execução dessas funções coincide com o cálculo dual e com
um espaço de características fixo; isso não é invenção irrestrita de conceitos.

Auditoria passou:58condições por jogo,1.388eventos de avaliação no total,
64programas completos novos por jogo; reexecução dos corpos reproduziu as
previsões salvas sem erro observado. Portão primário falhou em ambos os jogos.
Nenhum corpo promovido. Ajustes paralelos40,61segundos de parede por jogo,
203,53/212,59segundos CPU. Avaliações1,03/1,02segundos; zero passos ALE novos.

O diagnóstico posterior `resultados/identificabilidade_leituras_por_programa.json`
não muda portão ou primário. Quantifica repetição de comparações em contextos
diferentes e consistência do grafo finito de valores de palavras. A possibilidade
de interpolar essa amostra só com palavras não significa prever palavras nunca
vistas, nem que o contexto seja inútil. Limita a identificação do experimento.
