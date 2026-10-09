# Continuar a AGI 2.0 na nuvem

Snapshot de 9 de outubro de 2026. O repositório permanece privado.

O projeto original ocupa a raiz. O trabalho recente do D: está em
`disco_d/AGI_Inventor/`, incluindo organismo, encarnar, representação,
diagnósticos, conectomas, voz, `mente_web/` e seu painel. A outra interface
está em `disco_d/NovaMind_Observatory/`; ela utiliza seu próprio snapshot de
código em `source/`, conforme seu README.

As memórias, estados, modelos e arquivos grandes estão no release
`cloud-2026-10-09`. **Clonar apenas o Git não restaura as memórias.**
`manifesto.json` lista a origem e a localização de cada arquivo;
`arquivos.json` lista os anexos e seus hashes SHA-256.

## Obter o código e restaurar os anexos

Exemplo em Linux, Python 3.12, usando GitHub CLI autenticado com acesso ao repo:

```bash
gh repo clone brenopains/AGIResearch
cd AGI2
mkdir -p /tmp/agi2-assets
gh release download cloud-2026-10-09 --repo brenopains/AGIResearch --dir /tmp/agi2-assets
python3 cloud/restaurar.py /tmp/agi2-assets
```

Os arquivos voltam aos mesmos caminhos relativos da cópia consolidada.
Os anexos também podem ser baixados pela página privada do release. Para
somente verificar tamanho, SHA-256 e integridade dos ZIPs, use `--verificar`.
A restauração substitui arquivos existentes no destino: use uma cópia recém-clonada.

## Retomar a mente e seu painel

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r cloud/requirements-mente.txt
python -m playwright install --with-deps chromium
cd disco_d/AGI_Inventor/mente_web
python guardiao.py vidas/viva v1
```

O guardião continua a vida a partir do checkpoint copiado. Execute-o em um
terminal persistente, por exemplo tmux. A cópia não interrompeu a mente que
estava rodando no computador local; a vida na nuvem será uma continuação
independente a partir desse checkpoint, sem sincronização automática.

Em um segundo terminal, com o mesmo ambiente ativado:

```bash
cd disco_d/AGI_Inventor/mente_web
python painel/servidor.py vidas/viva 8765
```

Abra `http://127.0.0.1:8765` pelo encaminhamento de porta do ambiente ou por
um túnel SSH `ssh -L 8765:127.0.0.1:8765 usuario@servidor`. O painel mantém
seu acesso local. Para uma execução curta, use `python viver.py vidas/viva 10`.
Para parar salvando, crie `vidas/viva/PARAR`; remova-o antes de retomar.

A cópia usa Chromium do Playwright no Linux e mantém Edge no Windows.
Os botões de pausar/retomar usam o Python do ambiente atual no Linux.
Para voz opcional, instale `cloud/requirements-voz.txt`. Os modelos de voz
também são restaurados dos anexos; o comando do painel foi tornado relativo.

## Abrir o Observatory

```bash
pip install -r cloud/requirements-observatory.txt
cd disco_d/NovaMind_Observatory/frontend
npm ci
npm run build
cd ..
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8766
```

Encaminhe a porta 8766 e abra-a no navegador. Os painéis usam portas
diferentes para poderem rodar juntos. O iniciador `.ps1` continua disponível
para o computador Windows original; na nuvem, use os comandos acima.

## Escopo e caminhos locais

Incluídos: arquivos versionados e novos do projeto original, AGI_Inventor,
Observatory, resultados e estados persistidos. Excluídos: ambientes Python,
dependências instaladas, node_modules, caches, arquivos temporários,
credenciais e marcadores locais de processos.

As grandes bases externas de outras pastas do D: (`agi_dados`, `novamind_data`,
`dados_farmaco`, `dados_cancer`, etc.) ficaram locais após a indicação de que
o material desejado estava em `D:\AGI_Inventor`. Experimentos que referenciam
essas bases ainda precisarão dos dados e de ajuste dos caminhos. Diversos
scripts históricos mantêm caminhos absolutos Windows; o caminho de execução
da mente e dos painéis acima recebeu adaptações pontuais para Linux.

Os commits locais anteriores tinham binários acima do limite do GitHub.
O envio é um commit consolidado sobre a antiga `main` remota, com o histórico
local original preservado no computador. A branch local de trabalho não foi
reescrita. Os arquivos grandes atuais foram preservados nos anexos do release.
