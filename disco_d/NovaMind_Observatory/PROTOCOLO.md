# Protocolo do NovaMind Observatory

Versão do envelope: `1.0`. Documento baseado no backend e SDK implementados nesta pasta. O observatório registra código, execuções e declarações de produtores; não infere estados mentais que o sistema não tenha exposto.

## Proveniência e cobertura

| Origem | Significado |
|---|---|
| Catálogo AST | Arquivos, funções, classes, docstrings e imports, sem executar os módulos para descobri-los |
| Adaptador local | Funções permitidas executadas na cópia em `source/novaphys` |
| Histórico importado | Conteúdo de arquivo preservado; não é execução ao vivo |
| SDK externo | Eventos relatados pelo produtor, não verificados independentemente |

Um módulo catalogado não está necessariamente integrado a um agente. `static_import` é dependência declarada; uma chamada observada depende de registro da execução. Posição e distância no desenho não provam relações cognitivas.

`source.sha256` identifica os bytes do arquivo. `source.path`, `source.function` e `source.line`, quando disponíveis, localizam a observação. O adaptador de Acervo também usa `source.symbol` e `source.adapter`. Hash identifica versão, não demonstra correção. Fontes de produtores externos são declaradas por eles.

## Envelope persistido

| Campo | Significado |
|---|---|
| `protocol_version` | `1.0` |
| `id` | ID do observatório: `run_id:seq` |
| `run_id` | Execução a que o evento pertence |
| `seq` | Ordem crescente desde 1 |
| `timestamp` | UTC do registro, ou `null` em histórico ordinal |
| `elapsed_ms` | Tempo no relógio do observatório, ou `null` em histórico ordinal |
| `time_basis` | `clock` ou `ordinal` |
| `parent_id` | Evento pai já registrado, ou `null` |
| `module_id` | Identificador lógico do módulo |
| `kind` | Tipo extensível de evento |
| `title`, `summary` | Rótulo e explicação para humanos |
| `data` | Conteúdo estruturado |
| `metrics` | Medidas reais, com nomes que esclareçam unidade e escala |
| `source` | Proveniência, modo e produtor quando aplicável |

Não substitua medida ausente por zero. Não publique confiança probabilística quando o algoritmo só produziu similaridade, custo ou pontuação.

Cada execução fica em `data/runs/<run_id>/`: `run.json` contém metadados, entradas, parâmetros, estado e resultado; `events.jsonl` contém um evento por linha. Estados em andamento encontrados após reiniciar o servidor passam para `interrupted`.

Replay reordena/apresenta os registros por `seq`: não executa novamente o algoritmo. `parent_id` fornece hierarquia, sem inferir causalidade apenas pela proximidade temporal. Em uma importação ordinal, a ordem é conhecida e a duração é desconhecida. Velocidade de reprodução é velocidade de apresentação, não aceleração de uma duração histórica medida. Os eventos de início/fim do processo atual de importação podem ter relógio; isso não fornece timestamps da busca antiga.

## Capacidades e eventos existentes

Consulte `GET /api/capabilities` para obter os esquemas e limites atuais. A resposta contém `protocol_version` e `capabilities`. Cada capacidade declara ID, título, descrição, módulo, esquema de entrada e, quando aplicável, esquema dos parâmetros.

### `lingua.segmenta`

Entrada `{text}`; parâmetros `rodadas`, `max_len`, `cand`. O adaptador preserva o texto original e remove espaços para formar a sequência. Executa a função original `novaphys.lingua.segmenta`, com observação por `sys.settrace`.

Tipos e dados relevantes:

- `input`: entrada original, sequência e comprimento.
- `function.enter` / `function.exit`: função, origem, hierarquia; a entrada pode informar `caller_module`.
- `hypothesis.proposed`: candidato, par de unidades, frequência, rodada e custo anterior.
- `hypothesis.evaluated`: `cost_before_bits`, `cost_after_bits`, tokens candidatos e resultado da comparação.
- `hypothesis.accepted`: léxico/tokens depois da atribuição original de aceitação.
- `hypothesis.rejected`: candidato que não reduziu o custo.
- `memory.snapshot`: léxico, unidades e resultado final.
- `telemetry.coverage`: módulos, chamadas registradas/omitidas e limite.

Limites: texto entre 2 e 12.000 caracteres e dois símbolos após remover espaços; 1–100 rodadas; `max_len` entre 2 e 24; 1–16 candidatos. Padrões: 40, 12 e 8. Tokens intermediários guardam até 300 valores com `tokens_total` e `tokens_sampled`. O resultado final guarda os tokens completos dentro do limite de entrada.

Há até 1.200 chamadas rastreadas, além dos eventos semânticos. A cobertura declara omissões; bibliotecas externas não são inteiramente rastreadas. A instrumentação acrescenta custo de execução. Unidades descobertas por compressão não demonstram compreensão semântica.

### `cena.hdc`

Entrada `{pairs: [{role, object}], query_role, codebook?}`; parâmetro `dim`.

Executa `Espaco`, `monta_cena` e `consulta` originais. Codebook ausente/vazio usa os objetos dos pares. O retorno vem de `consulta`; os scores completos são expostos recalculando a mesma função original `sim`.

`scene.snapshot` contém pares, dimensão, número de átomos e amostra do vetor. `query.result` contém `selected`, `query_role`, `scores`, `codebook` e objetos existentes no papel consultado. `memory.snapshot` contém o cache de átomos desse espaço.

Limites: 1–32 pares, até 80 caracteres por papel/objeto, até 128 candidatos, dimensão 128–16.384, padrão 2.048. A amostra possui até 96 coordenadas, índices, tamanho total e hash dos bytes do vetor completo. O arquivo conserva a amostra e o hash, não todas as dimensões.

Coordenadas da amostra não são uma projeção 3D. A disposição visual dos pares não é uma cena fotográfica produzida pelo algoritmo. Similaridade não é probabilidade, e recuperar objeto do codebook não é gerar objeto novo.

### `acervo.memory`

Entrada `items` com uma memória por linha no formato `rótulo | conteúdo`, e `query`. Usa `Acervo.guarda`, `Acervo.ordena` e HDC reais.

Eventos: `memory.started`, `memory.stored`, `memory.query`, `relation.observed`, `artifact.created`, `state.snapshot`. O retorno inclui ranking, memórias e escopo.

Limites: 1–6 memórias; conteúdo 3–600 caracteres cada; rótulo até 120; entrada total até 4.500; consulta 3–600. Dimensão 2.000 e extração em 20 rodadas. Vetores expõem as primeiras 32 coordenadas, hash e política de amostragem.

Cada execução começa com novo Acervo. Persistir eventos não consolida conhecimento na memória global do projeto.

### `archive.traco`

Entrada `{task_index}`. Os índices disponíveis vêm do esquema. Lê tarefa existente em `source/traco.json`, sem executar novamente a busca.

`archive.context` apresenta contexto, primitivas e contagens; `hypothesis.evaluated` expõe candidatos gravados; `operator.applied` apresenta estágios; `archive.result` preserva a conclusão do arquivo.

`recorded_count`, `evaluated_count` e `recording_partial` explicam a cobertura. Os eventos históricos têm `timestamp: null`, `elapsed_ms: null` e `time_basis: ordinal`.

### `import.events`

Entrada `{events: [...]}`, entre 1 e 500 eventos. Cada evento informa `protocol_version: "1.0"`, `module_id`, `kind`, `title`.

Importa conteúdo externo sem executar código nem confirmar autenticidade. O evento original inteiro, inclusive campos desconhecidos, fica em `data._imported_event`; seu objeto `data` também é exposto para inspeção. Pais são remapeados quando já apareceram.

Os eventos ficam ordinais, com `source.verification: external_unverified`. Timestamps fornecidos permanecem no evento original; não viram tempos medidos pelo observatório.

## Acrescentar módulos usando `backend.sdk`

`Recorder.event` emite um fato observado. `Recorder.span` delimita uma operação e mantém a hierarquia. Tipos novos e dados estruturados podem ser inspecionados sem um componente de frontend exclusivo para cada módulo. Um renderer especializado novo pode ser útil, mas o conteúdo permanece acessível no inspetor genérico.

O exemplo abaixo chama a função real e registra seu retorno. Leia um arquivo real de entrada; não substitua resultados por valores ilustrativos. Salve o exemplo como `observar_texto.py` na raiz do observatório e mantenha o servidor local aberto.

```python
import hashlib
import sys
from pathlib import Path
from backend.sdk import Recorder

root = Path.cwd()
source_root = root / "source"
sys.path.insert(0, str(source_root))
from novaphys.lingua import segmenta

input_path = Path(sys.argv[1])
original = input_path.read_text(encoding="utf-8-sig")
text = original[:4000]  # Limite explicitamente registrado.
sequence = "".join(text.split())
if len(sequence) < 2:
    raise ValueError("A entrada precisa conter dois símbolos.")

module_path = source_root / "novaphys" / "lingua.py"
source = {
    "path": "novaphys/lingua.py",
    "sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
    "function": "segmenta",
}
with Recorder("Segmentação de arquivo", "meu-processo", source=source) as recorder:
    recorder.event(
        "input", "Texto lido", module_id="novaphys.lingua",
        data={"path": str(input_path), "text": text,
              "original_characters": len(original),
              "truncated": len(original) > len(text)},
    )
    with recorder.span("Executar segmenta", module_id="novaphys.lingua",
                       data={"rodadas": 20}, source=source):
        tokens, lexicon, before, after = segmenta(
            sequence, rodadas=20, max_len=12, cand=8
        )
        recorder.event(
            "memory.snapshot", "Resultado retornado",
            module_id="novaphys.lingua", source=source,
            data={"tokens": tokens, "lexicon": sorted(lexicon)},
            metrics={"cost_initial_bits": before,
                     "cost_final_bits": after,
                     "gain_bits": before - after},
        )
```

Chamada a partir da raiz do observatório:

```powershell
& 'C:\Program Files\Python312\python.exe' .\observar_texto.py .\source\transcricao_introspeccao.txt
```

O exemplo registra fronteiras e resultado, não cada hipótese intermediária. Para registrar intermediários, use o adaptador existente ou emita eventos junto aos pontos reais do algoritmo. `span` gera `span.started`, `span.completed` ou `span.failed`; exceções continuam sendo propagadas.

O SDK envia de forma síncrona. Falhas de rede local podem gerar exceções e afetam temporização. Um evento pendente fica disponível para reenvio na próxima operação; a sequência permite ignorar eventos já aceitos enquanto a captura continua aberta.

## Metadados e ingestão do SDK

Eventos do SDK são **relatados pelo produtor, não verificados independentemente**. O token vincula envios à captura, sem provar que o produtor descreveu corretamente sua execução.

O servidor atribui IDs e sequência próprios. Os metadados de transporte originais ficam em:

```text
data._producer.seq                 sequência do produtor
data._producer.occurred_at         timestamp declarado
data._producer.original_id        ID original
data._producer.original_parent_id ID original do pai
```

O `timestamp` do envelope externo persistido é o tempo de recepção/registro pelo servidor. O tempo declarado de ocorrência fica em `data._producer.occurred_at`. Não trate os dois relógios como uma medida de latência do algoritmo. Pais são remapeados quando já recebidos.

Fluxo HTTP utilizado pelo SDK:

1. `POST /api/external-runs` com `{title, producer, source}` cria a captura e devolve `id`, `ingest_token`.
2. `POST /api/external-runs/{id}/events` recebe `{token, events}`. Eventos informam versão, `seq` crescente desde 1, `kind` e `title`; o SDK acrescenta os demais campos.
3. Para terminar: `{token, events: [], finish: true, status: "completed", result: ...}`. Estados terminais: `completed`, `failed`, `cancelled`.

Até 100 eventos por lote e 12.000 eventos do produtor por captura. Lacunas são rejeitadas; sequências já aceitas são ignoradas como reenvios. Não reutilize uma sequência para corrigir dados: emita outro evento referindo o anterior. Tokens de captura não são persistidos para retomada após reiniciar.

Campos de domínio devem ficar em `data`, medidas em `metrics`, proveniência em `source`. A ingestão preserva os metadados de transporte listados em `_producer`; não preserva automaticamente todos os campos personalizados no topo do envelope. Para preservar integralmente um envelope histórico arbitrário, use `import.events`.

## Representações futuras

Use `module_id` estável e tipos específicos, por exemplo `representation.created`, `hypothesis.evaluated` e `operator.applied`. Para hipóteses, exponha o conteúdo e critério real. Para relações, forneça IDs e evidência. Para vetores, matrizes e outras representações, informe formato, dimensões, unidades e eventual amostragem.

Uma referência a artefato não transforma automaticamente um arquivo em recurso HTTP servido. Declare o que foi omitido. Não chame uma animação de atividade neural, nem um agrupamento de dados de intenção ou analogia sem que o produtor tenha essa estrutura operacional.

A existência do esquema não significa que todos os módulos já emitam seus estados. Para observar um processo novo, use o SDK. Para permitir acioná-lo pela interface, registre um adaptador com esquema, limites e proveniência. Não há execução arbitrária de código fornecido por HTTP.

## APIs e limites operacionais

| Endpoint | Uso |
|---|---|
| `GET /api/health` | Serviço e versão |
| `GET /api/catalog` | Módulos, imports estáticos, símbolos e hashes |
| `GET /api/capabilities` | Capacidades e esquemas |
| `POST /api/runs` | `{capability_id, input, parameters}` |
| `GET /api/runs` | Lista execuções |
| `GET /api/runs/{id}` | Estado e resultado |
| `GET /api/runs/{id}/events?after=0&limit=1000` | Eventos incrementais; `after` exclusivo; resposta `events,cursor,status,total` |
| `GET /api/runs/{id}/export` | JSON com metadados e eventos |
| `POST /api/runs/{id}/cancel` | Cancelamento local cooperativo ou encerramento da captura externa |
| `GET /api/source?module_id=...` | Fonte por ID catalogado, sem caminho arbitrário |
| `GET /api/metrics` | CPU/RAM da máquina e GPU quando disponível |

Uma execução local por vez. Orçamento de 90 segundos checado cooperativamente e máximo de 12.000 eventos do adaptador. Cancelamento não é interrupção preemptiva de toda operação de biblioteca. Encerrar captura externa não encerra o programa produtor.

Leitura incremental: até 5.000 eventos por chamada. Entradas de `/api/runs`: até 100.000 caracteres na serialização JSON, além dos limites específicos. O middleware rejeita `Content-Length` declarado acima de 2 MB; os endpoints mantêm suas validações próprias.

Métricas de hardware são da máquina inteira. GPU indisponível é `null`; sua consulta pode ficar em cache por cinco segundos. Nada nesses valores constitui uma medida direta de inteligência ou atividade neural do projeto.
