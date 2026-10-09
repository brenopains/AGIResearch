"""
make_kaggle_nb.py — gera kaggle_trm.ipynb (notebook Kaggle CORRETO pro TRM).
Usa os comandos REAIS do README oficial (SamsungSAILMontreal/TinyRecursiveModels):
  - install: torch nightly cu126 + requirements + adam-atan2
  - dados: python -m dataset.build_arc_dataset (lê {prefix}_{subset}_challenges/solutions.json)
  - treino: torchrun --nproc-per-node 2 pretrain.py arch=trm ...   (2x T4 no Kaggle, era 4x H100)
HONESTO: o run FIEL (45%) precisa de 4xH100x3dias. No Kaggle T4 grátis dá só um SINAL downscaled, não os 45%.
"""
import json, pathlib

def md(*lines):  return {"cell_type": "markdown", "metadata": {}, "source": [l if l.endswith("\n") else l + "\n" for l in lines]}
def code(*lines): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [l if l.endswith("\n") else l + "\n" for l in lines]}

cells = []

cells.append(md(
    "# TRM no Kaggle (2× T4) — Tiny Recursive Model em ARC-AGI-1",
    "",
    "**Antes de rodar — verdade honesta sobre compute:**",
    "- O TRM oficial treina ARC-1 em **4× H100 por ~3 dias** (≈288 H100-horas).",
    "- O Kaggle dá **2× T4** (cada ~10× mais lenta que H100) + corta sessão em **12h** + **30h/semana**.",
    "- Logo: **o run fiel (45%) NÃO termina na cota grátis.** Aqui dá pra ver o *sinal* do TRM (loss caindo, eval subindo) num run **downscaled** — não os 45%.",
    "",
    "**Setup no Kaggle (faça pela UI, 1 vez):**",
    "1. Notebook → **Settings → Accelerator → GPU T4 ×2**.",
    "2. **Settings → Internet → ON** (precisa pra `git clone` e `pip`).",
    "3. **Add Input** → procure **\"ARC Prize 2024\"** (competição) → Add. Vai montar em `/kaggle/input/arc-prize-2024/`.",
    "",
    "Depois rode as células em ordem.",
))

cells.append(md("## 1) GPU — confirma que tem 2× T4"))
cells.append(code(
    "!nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader",
    "import torch; print('torch', torch.__version__, '| GPUs visíveis:', torch.cuda.device_count())",
))

cells.append(md(
    "## 2) Clone do repo + instalação (oficial)",
    "Instala torch nightly cu126 + requirements + `adam-atan2` (compila — precisa `--no-build-isolation`). Leva alguns minutos.",
))
cells.append(code(
    "%cd /kaggle/working",
    "![ -d TinyRecursiveModels] || git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git",
    "%cd /kaggle/working/TinyRecursiveModels",
    "!pip install -q --upgrade pip wheel setuptools",
    "# torch nightly cu126 (comando oficial). Se quebrar deps do Kaggle, ainda roda nesta sessão.",
    "!pip install -q --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126",
    "!pip install -q -r requirements.txt",
    "!pip install -q --no-cache-dir --no-build-isolation adam-atan2",
    "print('OK install')",
))

cells.append(md(
    "## 3) Monta os dados no formato que o builder espera",
    "O builder lê `kaggle/combined/arc-agi_{subset}_challenges.json` (+ `_solutions.json`).",
    "A competição **ARC Prize 2024** já entrega exatamente esses arquivos — só linkamos pra cá.",
    "Usamos `--subsets training evaluation` (garantido). O subset `concept` (ConceptARC) é **opcional** — célula 3b adiciona se quiser.",
))
cells.append(code(
    "import os, glob",
    "os.makedirs('kaggle/combined', exist_ok=True)",
    "SRC = '/kaggle/input/arc-prize-2024'",
    "print('arquivos no input:', sorted(os.path.basename(p) for p in glob.glob(SRC + '/*.json')))",
    "# linka training/evaluation challenges+solutions pro prefixo que o builder quer",
    "for subset in ['training', 'evaluation']:",
    "    for kind in ['challenges', 'solutions']:",
    "        src = f'{SRC}/arc-agi_{subset}_{kind}.json'",
    "        dst = f'kaggle/combined/arc-agi_{subset}_{kind}.json'",
    "        if os.path.exists(src) and not os.path.exists(dst):",
    "            os.symlink(src, dst)",
    "print('combined:', sorted(os.listdir('kaggle/combined')))",
))

cells.append(md(
    "### 3b) (opcional) adicionar o subset `concept` (ConceptARC)",
    "Só rode se quiser replicar o `arc1concept` do paper. Junta o ConceptARC no mesmo formato `arc-agi_concept_*`.",
    "Se pular isto, mantenha `--subsets training evaluation` na célula 4 (sem `concept`).",
))
cells.append(code(
    "# OPCIONAL — descomente o bloco pra montar o concept a partir do repo ConceptARC.",
    "# import os, glob, json, numpy as np",
    "# os.system('git clone -q https://github.com/victorvikram/ConceptARC.git /kaggle/working/ConceptARC')",
    "# ch = {}; so = {}",
    "# for fp in glob.glob('/kaggle/working/ConceptARC/corpus/*/*.json'):",
    "#     t = json.load(open(fp)); tid = os.path.basename(fp)[:-5]",
    "#     ch[tid] = {'train': t['train'], 'test': [{'input': x['input']} for x in t['test']]}",
    "#     so[tid] = [x['output'] for x in t['test']]",
    "# json.dump(ch, open('kaggle/combined/arc-agi_concept_challenges.json','w'))",
    "# json.dump(so, open('kaggle/combined/arc-agi_concept_solutions.json','w'))",
    "# print('concept montado:', len(ch), 'tarefas')",
))

cells.append(md(
    "## 4) Constrói o dataset (com augmentações)",
    "Primeiro veja os flags REAIS (não chutar). Depois rode o build.",
    "**Downscale:** o nome oficial `aug-1000` = 1000 augmentações/tarefa. No T4, comece menor (ex.: `--num-aug 100` ou `--aug 100` — confira o nome no `--help`) pra caber no tempo.",
))
cells.append(code(
    "!python -m dataset.build_arc_dataset --help",
))
cells.append(code(
    "# Comando OFICIAL do README (sem concept). Ajuste o flag de nº de augmentações conforme o --help acima.",
    "!python -m dataset.build_arc_dataset \\",
    "  --input-file-prefix kaggle/combined/arc-agi \\",
    "  --output-dir data/arc1-aug-1000 \\",
    "  --subsets training evaluation \\",
    "  --test-set-name evaluation",
    "import os; print('saída:', os.listdir('data/arc1-aug-1000') if os.path.exists('data/arc1-aug-1000') else 'FALHOU')",
))

cells.append(md(
    "## 5) Treino — 2× T4 (downscaled)",
    "Comando oficial adaptado: `--nproc-per-node 2` (era 4 no paper, com H100).",
    "Primeiro veja os flags reais do `pretrain.py`; depois rode com os botões de downscale.",
    "",
    "**Botões de downscale (pra caber no T4/12h):** reduza `global_batch_size`, `epochs`, e o nº de augmentações na célula 4. ",
    "Salve checkpoints cedo — a sessão Kaggle morre em 12h.",
))
cells.append(code(
    "import os; os.environ['HYDRA_FULL_ERROR'] = '1'; os.environ['OMP_NUM_THREADS'] = '4'",
    "!python pretrain.py --help 2>/dev/null | head -60 || echo '(pretrain usa hydra; veja config/ no repo)'",
    "!ls config/ config/arch 2>/dev/null",
))
cells.append(code(
    "import os; os.environ['HYDRA_FULL_ERROR'] = '1'; os.environ['OMP_NUM_THREADS'] = '4'",
    "# OFICIAL adaptado p/ 2 GPUs. global_batch_size reduzido p/ caber no T4 (16GB). Ajuste conforme OOM.",
    "run_name = 't4_arc1_downscaled'",
    "!torchrun --nproc-per-node 2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \\",
    "  arch=trm \\",
    "  data_paths=\"[data/arc1-aug-1000]\" \\",
    "  arch.L_layers=2 \\",
    "  arch.H_cycles=3 arch.L_cycles=4 \\",
    "  global_batch_size=32 \\",
    "  +run_name=t4_arc1_downscaled ema=True",
))

cells.append(md(
    "## 6) Checkpoint → salvar em /kaggle/working (vira output do notebook)",
    "Tudo em `/kaggle/working` fica disponível pra download depois da sessão. Baixe o checkpoint e rode inferência até na 3050.",
))
cells.append(code(
    "import glob, shutil, os",
    "cks = sorted(glob.glob('/kaggle/working/TinyRecursiveModels/**/*.pt', recursive=True)) + \\",
    "      sorted(glob.glob('/kaggle/working/TinyRecursiveModels/**/*.ckpt', recursive=True))",
    "print('checkpoints encontrados:', cks[-5:])",
    "# eles já estão sob /kaggle/working, então saem como output do notebook automaticamente.",
))

cells.append(md(
    "---",
    "### Realidade honesta (leia)",
    "- Este notebook roda o **código oficial** com os comandos **reais** — não é reimplementação.",
    "- Mas **2× T4 grátis não chega aos 45%**: isso é 4×H100×3dias. Aqui você vê o **sinal** (loss/eval) de um run downscaled.",
    "- Pra os 45% de verdade: alugar GPU (4× H100 / A100) por algumas horas-dias, OU aceitar o teto downscaled como prova-de-conceito.",
    "- **NÃO use `colab_trm.py`** — aquele tinha comandos chutados (`model.hidden_size`, `data_path`, `\\n` literais) que causaram o erro do Hydra. Os comandos certos são os deste notebook.",
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = pathlib.Path(__file__).parent / "kaggle_trm.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("escrito:", out, "| células:", len(cells))
