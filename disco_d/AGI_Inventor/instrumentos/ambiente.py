"""
ambiente — o jogo como a maquina o ve: pixels, recompensa, fim, acoes ANONIMAS
================================================================================

Contrato (PRE_REGISTRO_instrumentos.md): o aprendiz so ve a luminancia 42x32 (media
de blocos), a recompensa e o fim de episodio; as acoes sao indices sem nome. Durante
a COLETA pode salvar e restaurar a situacao (repetir um experimento); todo passo,
inclusive os das ramificacoes, conta no orcamento.
"""

from __future__ import annotations

import numpy as np
from ale_py import ALEInterface, LoggerMode, roms

ALEInterface.setLoggerMode(LoggerMode.Error)

ALT, LARG = 42, 32


class Orcamento(Exception):
    pass


class Mundo:
    def __init__(self, jogo, semente=0, pegajoso=0.0, aparencia=None, orcamento=None):
        self.ale = ALEInterface()
        self.ale.setInt("random_seed", int(semente))
        self.ale.setFloat("repeat_action_probability", float(pegajoso))
        self.ale.setInt("frame_skip", 4)
        self.ale.setInt("max_num_frames_per_episode", 108000)
        self.ale.loadROM(roms.get_rom_path(jogo))
        self.acoes = self.ale.getMinimalActionSet()
        self.nA = len(self.acoes)
        self.aparencia = aparencia            # permutacao de canais (troca de aparencia)
        self.passos = 0
        self.orcamento = orcamento

    def obs(self):
        im = self.ale.getScreenRGB().astype(np.float32)
        if self.aparencia is not None:
            im = im[:, :, list(self.aparencia)]
        lum = im @ np.array([0.299, 0.587, 0.114], np.float32) / 255.0
        return lum.reshape(ALT, 5, LARG, 5).mean((1, 3))

    def reinicia(self):
        self.ale.reset_game()
        return self.obs()

    def passo(self, a):
        if self.orcamento is not None and self.passos >= self.orcamento:
            raise Orcamento()
        r = self.ale.act(self.acoes[int(a)])
        self.passos += 1
        return self.obs(), float(r), self.ale.game_over()

    def salva(self):
        return self.ale.cloneState()

    def restaura(self, s):
        self.ale.restoreState(s)


def base(o, o_ant):
    """r0(h): quadro atual 21x16 e a diferenca para o anterior (672 numeros)."""
    a = o.reshape(21, 2, 16, 2).mean((1, 3))
    b = o_ant.reshape(21, 2, 16, 2).mean((1, 3))
    return np.concatenate([a.ravel(), (a - b).ravel()]).astype(np.float32)


HNS_REF = {  # BBF (Schwarzer et al., 2023), Tab. A.1: aleatorio, humano
    "breakout": (1.7, 30.5), "pong": (-20.7, 14.6), "alien": (227.8, 7127.7),
    "frostbite": (65.2, 4334.7), "private_eye": (24.9, 69571.3),
    "road_runner": (11.5, 7845.0)}

PUBLICADOS = {  # mesma tabela: DER, SPR, EfficientZero, BBF
    "breakout": (9.2, 19.6, 414.1, 370.6), "pong": (-9.7, -5.4, 20.1, 16.7),
    "alien": (802.3, 841.9, 808.5, 1173.2), "frostbite": (871.0, 1170.7, 296.3, 2384.8),
    "private_eye": (72.8, 86.0, 96.7, 40.5), "road_runner": (11843.4, 12213.1, 17751.3, 33426.8)}


def hns(jogo, pontos):
    lo, hi = HNS_REF[jogo]
    return (pontos - lo) / (hi - lo)
