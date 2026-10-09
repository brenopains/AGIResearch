"""
auditoria_arc.py - os 3,6 pontos sobre copiar sobrevivem a controle?
======================================================================

arc_integrated_results.json registra:

    pixel_accuracy 0.802 . copy_baseline 0.7663 . exact_match 0.0048
    verdict: "APRENDIZADO DETECTADO - Acima do acaso, abaixo do copiar"

TRES DEFEITOS NO INSTRUMENTO, achados lendo o codigo:

(1) O VEREDITO NAO DISCRIMINA. A cascata e:
        if   mex > 0.05        -> BREAKTHROUGH
        elif mpx > mcp * 1.15  -> SINAL FORTE
        elif mpx > chance * 2  -> APRENDIZADO DETECTADO
    Com N_COLORS=11, chance = 0.1, entao o terceiro ramo dispara com pixel
    accuracy > 20%. Copiar a entrada da 76,6%. Um sistema que apenas COPIA
    receberia o mesmo carimbo de "aprendizado detectado".

(2) NAO HA SEMENTE. torch.manual_seed e np.random.seed nunca sao chamados,
    e sample_batch amostra aleatoriamente. Logo 0.802 nao e reproduzivel e a
    variancia nunca foi medida.

(3) NAO HA CONTROLE NEGATIVO. Nada compara contra a MESMA arquitetura sem
    treino. Sem isso nao da para separar "o modelo aprendeu" de "a
    arquitetura tem prior de copia".

O QUE ESTA AUDITORIA FAZ
--------------------------
Tres bracos sobre EXATAMENTE os mesmos lotes (pareado):
    treinado    arc_model_trained.pt
    aleatorio   mesma arquitetura, pesos de init   <- o controle que faltava
    copiar      a entrada como predicao

Varias sementes. Diferenca pareada por lote, com desvio.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  A1  treinado > copiar em todas as sementes E media da diferenca pareada
      acima de 2 erros padrao -> os 3,6pp sao reais
  A2  aleatorio ~ treinado -> o numero mede arquitetura, nao aprendizado
  A3  desvio entre sementes >= a diferenca -> 0.802 e ruido amostral
  A4  exact match em contagem absoluta. 0,48% de 400 = ~2 tarefas.

NAO SOBRESCREVE arc_integrated_results.json.
"""
import os
import sys
import json
import random

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.grid_cognition import GridWorldModel
from core.arc_dataset import ARCDataset

ARC_EVAL_DIR = "data/ARC/data/evaluation"
MAX_H = MAX_W = 30
PAD_VAL = 10
N_COLORS = 11
MODEL_DIM, MODEL_HEADS, MODEL_LAYERS = 128, 4, 3
K_SUPPORT, Q_QUERIES = 3, 1
N_TASKS = 400


def avaliar(model, eval_ds, hw, n_tasks, device):
    """Listas por lote de px, ex, cp, mais contagem absoluta de exact match."""
    px_l, ex_l, cp_l = [], [], []
    n_ex = n_tot = done = 0
    with torch.no_grad():
        while done < n_tasks:
            bs = min(32, n_tasks - done)
            (xs, ys), (xq, yq) = eval_ds.sample_batch(bs, K_SUPPORT, Q_QUERIES)
            valid = (yq != PAD_VAL)
            if model is None:                       # braco COPIAR
                pred = xq.reshape(bs, Q_QUERIES, hw)
            else:
                pred = model(xq, (xs, ys)).argmax(2).reshape(bs, Q_QUERIES, hw)
            px = ((pred == yq) & valid).float().sum() / valid.float().sum().clamp(min=1)
            exv = ((pred == yq) | ~valid).all(dim=2)
            cp = ((xq == yq) & valid).float().sum() / valid.float().sum().clamp(min=1)
            px_l.append(px.item())
            ex_l.append(exv.float().mean().item())
            cp_l.append(cp.item())
            n_ex += int(exv.sum().item())
            n_tot += exv.numel()
            done += bs
    return px_l, ex_l, cp_l, n_ex, n_tot


def main(sementes=(0, 1, 2, 3, 4)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hw = MAX_H * MAX_W
    print("=" * 74)
    print("AUDITORIA ARC - os 3,6 pontos sobre copiar sobrevivem a controle?")
    print("=" * 74)
    print("PRE-REGISTRO:")
    print("  A1  treinado>copiar em todas as sementes e >2 erros padrao -> real")
    print("  A2  aleatorio ~ treinado -> mede arquitetura, nao aprendizado")
    print("  A3  desvio entre sementes >= diferenca -> ruido amostral")
    print("")
    print("device %s . %d queries por semente . sementes %s"
          % (device, N_TASKS, list(sementes)))

    eval_ds = ARCDataset(data_dir=ARC_EVAL_DIR, max_h=MAX_H, max_w=MAX_W,
                         pad_val=PAD_VAL, dev=device)
    print("tasks de avaliacao carregadas: %d" % len(eval_ds.tasks))

    treinado = GridWorldModel(d=MODEL_DIM, heads=MODEL_HEADS,
                              layers=MODEL_LAYERS, max_hw=hw,
                              colors=N_COLORS).to(device)
    treinado.load_state_dict(torch.load("arc_model_trained.pt",
                                        map_location=device))
    treinado.eval()

    acc = {'treinado': [], 'aleatorio': [], 'copiar': []}
    exatos = {'treinado': 0, 'aleatorio': 0, 'copiar': 0}
    total = 0
    dif_pareada = []

    for s in sementes:
        # controle negativo: MESMA arquitetura, pesos de init
        torch.manual_seed(10000 + s)
        aleatorio = GridWorldModel(d=MODEL_DIM, heads=MODEL_HEADS,
                                   layers=MODEL_LAYERS, max_hw=hw,
                                   colors=N_COLORS).to(device)
        aleatorio.eval()

        linha = "  semente %d: " % s
        lotes = {}
        for nome, mdl in (('treinado', treinado), ('aleatorio', aleatorio),
                          ('copiar', None)):
            # MESMOS lotes nos tres bracos: re-semeia antes de cada braco
            torch.manual_seed(s)
            np.random.seed(s)
            random.seed(s)
            px, ex, cp, ne, nt = avaliar(mdl, eval_ds, hw, N_TASKS, device)
            acc[nome].append(float(np.mean(px)))
            exatos[nome] += ne
            lotes[nome] = px
            linha += "%s %.1f%%  " % (nome, 100 * np.mean(px))
        total += nt
        d = np.array(lotes['treinado']) - np.array(lotes['copiar'])
        dif_pareada.extend(d.tolist())
        linha += "| dif pareada %+.1f%%" % (100 * d.mean())
        print(linha)

    print("")
    print("=" * 74)
    print("%-12s%14s%10s%16s" % ('braco', 'pixel acc', 'desvio', 'exact match'))
    print("-" * 74)
    for nome in ('copiar', 'aleatorio', 'treinado'):
        print("%-12s%13.1f%%%9.2f%%%10d/%-5d"
              % (nome, 100 * np.mean(acc[nome]), 100 * np.std(acc[nome]),
                 exatos[nome], total))

    d = np.array(dif_pareada)
    erro_padrao = d.std() / np.sqrt(len(d))
    print("")
    print("=" * 74)
    print("DIFERENCA PAREADA treinado - copiar, por lote (n=%d)" % len(d))
    print("  media %+.2f%%  .  desvio %.2f%%  .  erro padrao %.3f%%  .  "
          "lotes positivos %.0f%%"
          % (100 * d.mean(), 100 * d.std(), 100 * erro_padrao,
             100 * (d > 0).mean()))

    todas_pos = all(t > c for t, c in zip(acc['treinado'], acc['copiar']))
    dois_ep = d.mean() > 2 * erro_padrao
    print("")
    if todas_pos and dois_ep:
        print("A1 . A DIFERENCA SOBREVIVE. O modelo treinado supera copiar de forma")
        print("     consistente entre sementes e alem de 2 erros padrao.")
    else:
        print("A1 . NAO SOBREVIVE COMO DECLARADO. todas_positivas=%s  "
              "acima_2ep=%s" % (todas_pos, dois_ep))

    ga, gt = np.mean(acc['aleatorio']), np.mean(acc['treinado'])
    print("")
    if abs(gt - ga) < 0.05:
        print("A2 . O MODELO NAO TREINADO obtem %.1f%% contra %.1f%% do treinado."
              % (100 * ga, 100 * gt))
        print("     O numero mede a arquitetura e o prior dos dados, nao treino.")
    else:
        print("A2 . o treino acrescenta %+.1f%% sobre init aleatorio."
              % (100 * (gt - ga)))

    print("")
    print("A4 . exact match absoluto - treinado %d/%d, copiar %d/%d, "
          "aleatorio %d/%d"
          % (exatos['treinado'], total, exatos['copiar'], total,
             exatos['aleatorio'], total))
    print("")
    print("NOTA sobre o veredito original: 'APRENDIZADO DETECTADO' dispara com")
    print("pixel accuracy > %.0f%%. Copiar a entrada da %.0f%%. O rotulo nao"
          % (100 * 2 / (N_COLORS - 1), 100 * np.mean(acc['copiar'])))
    print("distingue nada - um sistema que so copia receberia o mesmo carimbo.")

    with open("auditoria_arc_results.json", "w") as f:
        json.dump({'pixel': acc, 'exact_abs': exatos, 'total': total,
                   'dif_pareada_media': float(d.mean()),
                   'dif_pareada_desvio': float(d.std())}, f, indent=2)


if __name__ == "__main__":
    main()
