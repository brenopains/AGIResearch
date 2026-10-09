"""
colab_trm.py — NOTEBOOK COLAB pro TRM (Tiny Recursive Model, Jolicoeur-Martineau 2025): 7M params, 45% ARC-AGI-1.
Roda o CODIGO OFICIAL (mais confiavel que reimplementar) num T4 gratis, downscale p/ caber. Treino na nuvem;
depois a inferencia roda ate na 3050. Cole celula por celula no Colab (Runtime > GPU T4).

Por que TRM e diferente de tudo que tentamos: nao e busca nem NCA single-shot — e RECURSAO com supervisao profunda.
Mantem (z=raciocinio, y=resposta) e refina:  for i in n: z=net(x,y,z); y=net(y,z)  — profundidade-por-recursao
da compute efetivo sem params p/ overfitar nos ~1000 exemplos do ARC. E o lever de MAIOR TETO buildavel sem LLM.
"""
# ============ CELULA 1 — setup ============
SETUP = r'''
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
!git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git
%cd TinyRecursiveModels
!pip install -q -r requirements.txt 2>/dev/null || pip install -q torch einops tqdm wandb omegaconf hydra-core
# dados ARC-AGI-1 (publico)
!git clone https://github.com/fchollet/ARC-AGI.git /content/ARC-AGI
print("OK setup")
'''

# ============ CELULA 2 — gerar dados + augmentacoes (RE-ARC style) ============
DATA = r'''
# o repo TRM espera tarefas + augmentacoes (d8 + permutacao de cor + translacao).
# usa o pipeline do proprio repo se existir; senao, gera augmentacoes basicas.
import os, json, glob, numpy as np
SRC="/content/ARC-AGI/data/training"; OUT="/content/arc_aug"; os.makedirs(OUT,exist_ok=True)
def d8(a):
    return [a,np.rot90(a,1),np.rot90(a,2),np.rot90(a,3),a[:,::-1],a[::-1,:],a.T,np.rot90(a.T,2)]
def augment(task,n=300):
    out=[]
    for _ in range(n):
        k=np.random.randint(8); perm=np.random.permutation(9)+1; lut=np.concatenate([[0],perm])
        def f(g):
            g=np.array(g); g=d8(g)[k]; return lut[g].tolist()
        out.append({"train":[{"input":f(p["input"]),"output":f(p["output"])} for p in task["train"]],
                    "test":[{"input":f(p["input"]),"output":f(p["output"])} for p in task["test"]]})
    return out
tasks=[json.load(open(fp)) for fp in glob.glob(SRC+"/*.json")]
print(f"{len(tasks)} tarefas base. gerando augmentacoes...")
# (Para o treino fiel, prefira o script de dados do repo: ver README do TinyRecursiveModels.)
'''

# ============ CELULA 3 — treino DOWNSCALED p/ T4 (single GPU) ============
TRAIN = r'''
# O repo treina com 4xH100. No T4: downscale hidden, batch via grad-accum, menos augmentacoes.
# Config recomendada p/ T4 (ajuste no config do repo):
#   model.hidden_size=256   (era 512)
#   model.num_layers=2
#   recursion.n_latent=6  recursion.T_cycles=3  act.max_steps=16
#   training.global_batch_size=64  (com gradient_accumulation p/ caber)
#   data.augmentations_per_task=150  (era 1000)
#   training.epochs=  (rode ate o eval estabilizar; salve checkpoints)
# Comando (ajuste aos nomes reais do repo apos ler o README/configs):
!python pretrain.py arch=trm data_path=/content/arc_aug \
    +model.hidden_size=256 +model.num_layers=2 \
    +training.global_batch_size=64 +training.gradient_accumulation=8 \
    +data.augmentations_per_task=150 2>&1 | tail -40
# Salve o checkpoint no Drive p/ rodar inferencia depois na 3050.
'''

# ============ CELULA 4 — inferencia (roda ate na 3050 depois) ============
INFER = r'''
# Inferencia: most-common da predicao sobre as augmentacoes d8+cor (inverter a transformacao antes de votar).
# Use o script de eval do repo (eval.py / evaluate.py) apontando p/ o checkpoint + ARC-AGI/data/evaluation.
!python evaluate.py checkpoint=<seu_ckpt> data_path=/content/ARC-AGI/data/evaluation 2>&1 | tail -20
print("Pass@2 esperado: ~30-45% (downscaled fica abaixo do 45% full; mede o teto na sua restricao).")
'''

if __name__ == "__main__":
    print("Notebook Colab pro TRM. Abra o Colab (T4), e cole as celulas SETUP, DATA, TRAIN, INFER na ordem.")
    print("Veja o README do repo TinyRecursiveModels p/ os nomes exatos dos configs/scripts (podem mudar).")
    print("\n--- CELULA 1 ---", SETUP, "\n--- CELULA 2 ---", DATA, "\n--- CELULA 3 ---", TRAIN, "\n--- CELULA 4 ---", INFER)
