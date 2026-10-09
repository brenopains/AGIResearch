# NovaMind Observatory

Interface local para explorar o código e acompanhar execuções reais do projeto: mapa de módulos, eventos em ordem e hierarquia, inspeção dos dados, memória associativa, cena HDC, replay e exportação.

O trabalho está isolado em `D:\NovaMind_Observatory`. O observatório não editou o projeto original em `C:\Users\usuario\Documents\AGI 2.0`. `source/` é um snapshot congelado: alterações externas no original não são sincronizadas automaticamente e podem fazer as duas versões divergirem.

## Abrir e encerrar

No PowerShell:

```powershell
Set-Location D:\NovaMind_Observatory
.\iniciar.ps1
```

Acesse **http://127.0.0.1:8765**. Para encerrar o servidor iniciado pelo observatório:

```powershell
Set-Location D:\NovaMind_Observatory
.\parar.ps1
```

O inicializador utiliza os runtimes disponíveis nesta máquina e o frontend compilado. Consulte o próprio script para caminhos e diagnósticos de inicialização. Os eventos ficam em `data/runs/` e continuam disponíveis após reiniciar o servidor. As dependências de `node_modules` são acessadas por uma junction compartilhada para leitura; o build e o cache do frontend ficam isolados nesta pasta.

## O que funciona

- **Mapa estrutural dinâmico:** o snapshot contém 256 arquivos Python; funções, classes, docstrings e imports são descobertos por AST. O catálogo se atualiza ao consultar novamente após mudanças na cópia.
- **Execução observada:** três capacidades locais utilizam funções reais de quatro módulos instrumentados: `lingua`, `cena`, `hdc_indice` e `acervo`.
- **Segmentação por MDL:** propõe unidades, mede custos e expõe aceites/rejeições do código original.
- **Cena associativa HDC:** associa papéis a objetos e recupera um objeto pelo codebook, com ranking e amostra rotulada do vetor.
- **Memória associativa:** guarda textos e os ordena por similaridade em um Acervo real, novo a cada execução.
- **Histórico e replay:** importa registros reais de `traco.json`, preservando a informação de que a gravação é parcial e sem tempos conhecidos.
- **Extensibilidade:** importa outros eventos estruturados e recebe observações de módulos futuros pelo SDK Python, sem exigir um componente de frontend para cada módulo.

Use o painel de interação para escolher uma capacidade e fornecer entradas. Os formulários e limites vêm do backend. A interface não oferece, nesta versão, conversa livre com um LLM nem resolve automaticamente qualquer solicitação em linguagem natural.

## Como interpretar a interface

Um nó presente no mapa representa código existente; não significa que esse módulo esteja ativo ou integrado a uma mente única. Conexões estáticas são imports declarados. Chamadas observadas pertencem a uma execução registrada.

A cena 3D organiza relações reais para inspeção humana. O algoritmo HDC não gerou uma imagem fotográfica nem um mundo 3D; a posição de cada nó é apresentação. Custos, similaridades e contagens mantêm seus significados: similaridade não é probabilidade ou confiança.

A cobertura atual é parcial. Não há instrumentação automática de todos os estados internos dos 256 arquivos. O SDK permite acrescentar eventos reais de hipóteses, operadores, abstrações, representações ou outros dados. Eventos desconhecidos permanecem inspecionáveis como dados estruturados; uma visualização especializada nova pode exigir um renderer adicional.

Os dados do SDK são declarações do produtor, sem verificação independente do algoritmo. As importações são identificadas. Estados e métricas ausentes não são substituídos por atividade fictícia.

## Limites atuais

- Uma execução local por vez, orçamento de 90 segundos verificado cooperativamente e até 12.000 eventos do adaptador.
- Cancelamento local é cooperativo. Encerrar uma captura externa não encerra o programa produtor.
- Tokens intermediários e vetores podem ser amostrados; os eventos declaram a política e os tamanhos.
- Replay reapresenta registros; não executa novamente o algoritmo. Em registros ordinais, a duração original é desconhecida.
- Cada execução de memória começa vazia. Persistência de eventos não significa aprendizagem acumulativa na memória global do projeto.
- CPU/RAM/GPU são medições da máquina, não exclusivamente da AGI. GPU indisponível é mostrada como dado ausente.

## Arquivos úteis

- `PROTOCOLO.md`: envelope dos eventos, capacidades, limites, endpoints e exemplo de integração.
- `backend/sdk.py`: `Recorder.event` e `Recorder.span` para módulos novos.
- `backend/adapters.py` e `backend/acervo_adapter.py`: operações permitidas e observações específicas.
- `backend/tracing.py`: rastreamento das funções originais copiadas.
- `source/`: snapshot de código usado pelos adaptadores e catálogo.
- `data/runs/<id>/`: metadados e eventos persistidos de cada execução.

O observatório torna evidências do sistema inspecionáveis. A existência da interface, de um grafo complexo ou de um replay não demonstra AGI, consciência ou compreensão de nível humano.
