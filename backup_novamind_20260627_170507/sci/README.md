# NovaMind-Sci — Três cientistas artificiais para hipóteses de cura

Três sistemas de geração e ranqueamento de hipóteses biomédicas, rodando localmente na sua máquina (RTX 3050 4GB), com o **raciocínio pesado via Claude API** e **utilitários via Nemotron local (LM Studio)**.

## Os três sistemas

| Pasta | O que é | Arquitetura inspirada em |
|---|---|---|
| `co_scientist/` | Clone do AI Co-Scientist | Google DeepMind, Nature 2026 |
| `robin/` | Clone do Robin | FutureHouse, Nature 2026 |
| `novamind_sci/` | **A fusão com nossa AGI** | Vetor multi-atributo + personas evolutivas + composição não-comutativa |

## Pré-requisitos

### 1. LM Studio (Nemotron local) — **já está rodando**
- Modelo: `nvidia/nemotron-3-nano-4b` (4-bit GGUF)
- Servidor local em `http://localhost:1234/v1` (OpenAI-compatible)
- Papel: utilitários (sumarização de abstracts, NER), economiza tokens do Claude

### 2. Chave Anthropic — **VOCÊ PRECISA CRIAR**
Os sistemas precisam do raciocínio do Claude (frontier-LLM). A chave do Claude Code (assinatura) **não** funciona pro SDK Python.

**Passos:**
1. Vá em https://console.anthropic.com/ → API Keys → Create Key
2. (PowerShell, persistente):
   ```powershell
   [System.Environment]::SetEnvironmentVariable('ANTHROPIC_API_KEY','sk-ant-...','User')
   ```
   Feche e reabra o terminal pra carregar.
3. Custo estimado: cada execução de Co-Scientist completa (6 hipóteses, 2 ciclos) gasta ~$0.20–0.50 em Sonnet 4.5. Robin/NovaMind-Sci similares.

### 3. Pacotes Python (já instalados)
- `anthropic`, `biopython`, `python-dotenv`, `requests`

## Como rodar

### Sistema A — Co-Scientist
```bash
cd C:\Users\usuario\Desktop\AGI\novamind\sci
python -m co_scientist.run "alopecia areata" --n 6 --rounds 2 --tourn 2
```

### Sistema B — Robin
```bash
python -m robin.run "alopecia areata" --n 4
```

### Sistema C — NovaMind-Sci (a fusão)
```bash
python -m novamind_sci.run "alopecia areata" --cycles 15 --pop 8
```

### Teste em escada (progressão de dificuldade + assertividade automática)
```bash
python test_progression.py --start 0 --end 0   # só alopecia areata
python test_progression.py --start 0 --end 3   # todas: alopecia, vitiligo, FPI, ALS
```

### Smoke test sem chave Anthropic (Nemotron-only)
```bash
python smoke_nemotron_only.py "alopecia areata"
```
*Aviso honesto: o Nemotron 4B sozinho gera hipóteses fracas. Só prova que a tubulação funciona.*

## A escada de doenças (em ordem crescente de dificuldade)

1. **Alopecia areata** — JAK-STAT bem mapeado; JAK-inhibitors aprovados 2022-23. **Lead claro a recuperar como sinal de assertividade.**
2. **Vitiligo** — autoimune, IFN-γ/CXCL10; ruxolitinib tópico aprovado 2022.
3. **Fibrose pulmonar idiopática** — pirfenidona/nintedanib atrasam, não curam. TGF-β/EMT.
4. **ALS** — riluzol/edaravone modestos. Mecanismo ainda muito aberto. Alto desafio.

## Como assertividade é medida (honestamente)

Para uma doença sem cura conhecida não existe ground-truth. O avaliador compõe:
- **PMID-exist**: % dos PMIDs citados existem mesmo
- **PMID-relevance**: % dos PMIDs citados são da doença
- **Already known**: % hipóteses cuja intervenção+alvo já consta em literatura/trials (recuperação de leads — **alto é bom**, sinal de tração com a realidade)
- **Novel-but-plausible**: % com mecanismo plausível e não-trivial
- **Lead recovery**: intervenção aparece em literatura clínica
- **Diversity**: nº de alvos/vias distintos nas top-K

Score composto `assertiveness` ∈ [0,1].

## Diferença real do NovaMind-Sci

Os outros dois clonam papers. O NovaMind-Sci traz arquitetura nossa:

1. **5 personas com drives diferentes** (explorador, conservador, engenheiro, segurança, sintetizador) — cada uma gera hipóteses *coerentes com seu viés*. Diversidade real, não simulada por temperatura.
2. **Hipótese é um vetor de 9 atributos** (novelty, plausibility, tractability, safety, evidence_strength, orthogonality, cost, clinical_impact, synergy), não um score escalar.
3. **Selecção darwiniana das personas**: a persona cuja hipótese teve score alto ganha peso; baixou peso quem propôs lixo. Por consequência. (Equivalente do `value[ação]` da NovaMind.)
4. **Composição ordenada não-comutativa**: A⇒B (A modulando B) é diferente de B⇒A. Mede a assimetria de ordem, igual a colônia.
5. **Score-por-persona**: cada persona ranqueia com pesos próprios, então a "melhor hipótese" depende de quem julga — pluralismo epistêmico.

## Estado da implementação

- ✅ Infra: Claude client (com prompt-cache), Nemotron client, PubMed E-utilities, avaliador automático
- ✅ Co-Scientist: 6 agentes (Generation, Reflection, Ranking-Elo, Evolution, Proximity, Meta-review)
- ✅ Robin: 4 agentes (Crow, Falcon, Owl, DataAnalyst)
- ✅ NovaMind-Sci: 5 personas + composição ordenada + drives darwinianos
- ⏳ **Falta**: chave ANTHROPIC_API_KEY pra rodar de verdade
