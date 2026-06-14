"""
core/cross_domain.py — MEMORIA / WORLD-MODEL CROSS-DOMINIO FATORADO + ROTEADO POR CONTEXTO.

Tese (e o RISCO que o usuario apontou): memoria cross-dominio INGENUA (jogar tudo num modelo so) sofre
INTERFERENCIA catastrofica e TRANSFERENCIA NEGATIVA — o modelo confunde os mundos e vira mingau.

A solucao NAO e misturar, e FATORAR:
  - ESTRUTURA compartilhada (a "gramatica" de como qualquer mundo funciona) -> ISSO transfere.   [grid cells]
  - CONTEUDO especifico do dominio (entidades/dinamica/recompensa de cada mundo) -> fica separado. [place cells]
  - INFERENCIA DE CONTEXTO ("em que mundo estou?") -> distingue os dominios e roteia o conteudo.   [remapping]
(Inspiracao direta no hipocampo + Tolman-Eichenbaum Machine: estrutura<->conteudo factorizados.)

Este modulo da: o ExperienceStore (memoria que todo dominio escreve) + 3 modelos p/ MEDIR a tese:
  FactoredWorldModel (fatorado+contexto, o seguro) vs NaiveWorldModel (blob, sofre interferencia) vs
  o baseline isolado (treinar do zero por dominio, sem transferencia). As 3 metricas que matam o risco:
  TRANSFERENCIA (few-shot em dominio novo) x INTERFERENCIA (aliasing) x DISCRIMINACAO (contexto separa dominios).
"""
import random, torch, torch.nn as nn, torch.nn.functional as F


class ExperienceStore:
    """Memoria de transicoes (dominio -> [(obs, acao, prox_obs)]). E nisso que mine, xadrez, etc. escrevem."""
    def __init__(self): self.data = {}
    def add(self, domain, obs, a, obs2): self.data.setdefault(domain, []).append((obs, a, obs2))
    def domains(self): return list(self.data)
    def __len__(self): return sum(len(v) for v in self.data.values())
    def sample(self, domain, n):
        d = self.data[domain]; return random.choices(d, k=n)


class StructureCore(nn.Module):
    """ESTRUTURA COMPARTILHADA: transicao no espaco latente z dada uma acao (embedding). MESMOS pesos p/ todo dominio.
    E o 'mapa cognitivo' domino-geral — o que deve transferir entre mundos."""
    def __init__(self, zdim, adim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(zdim + adim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, zdim))
    def forward(self, z, a): return z + self.net(torch.cat([z, a], -1))  # transicao residual


class ContextEncoder(nn.Module):
    """INFERE O CONTEXTO do dominio a partir de um punhado de transicoes de suporte (permutation-invariant).
    c ~= 'em que mundo estou' aprendido por interacao — funciona p/ dominios NOVOS (infere c do suporte)."""
    def __init__(self, obsdim, n_actions, cdim, hidden=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(obsdim * 2 + n_actions, hidden), nn.ReLU(), nn.Linear(hidden, cdim))
    def forward(self, sup_obs, sup_a, sup_obs2):     # [B,K,obs],[B,K,A],[B,K,obs]
        h = self.enc(torch.cat([sup_obs, sup_a, sup_obs2], -1))   # [B,K,cdim]
        return h.mean(1)                                          # agregacao invariante -> [B,cdim]


class FactoredWorldModel(nn.Module):
    """O SEGURO: encoder/decoder de obs (compartilhados) + StructureCore (compartilhado) + acao condicionada
    no CONTEXTO inferido. Estrutura transfere; conteudo (semantica das acoes do dominio) entra via c."""
    def __init__(self, obsdim, n_actions, zdim=32, cdim=16, adim=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(obsdim, 128), nn.ReLU(), nn.Linear(128, zdim))
        self.dec = nn.Sequential(nn.Linear(zdim, 128), nn.ReLU(), nn.Linear(128, obsdim))
        self.ctx = ContextEncoder(obsdim, n_actions, cdim)
        self.act = nn.Sequential(nn.Linear(n_actions + cdim, 64), nn.ReLU(), nn.Linear(64, adim))
        self.core = StructureCore(zdim, adim)
    def context(self, sup): return self.ctx(*sup)
    def forward(self, obs, a_onehot, sup=None, c=None):
        if c is None: c = self.ctx(*sup)
        z = self.enc(obs)
        aemb = self.act(torch.cat([a_onehot, c], -1))
        return self.dec(self.core(z, aemb)), c


class NaiveWorldModel(nn.Module):
    """O INGENUO (blob): (obs, acao) -> prox_obs, SEM nocao de dominio. Quando o mesmo (obs,acao) leva a
    proximos diferentes em mundos diferentes (aliasing), ele e FORCADO a mediar -> INTERFERENCIA."""
    def __init__(self, obsdim, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obsdim + n_actions, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, obsdim))
    def forward(self, obs, a_onehot): return self.net(torch.cat([obs, a_onehot], -1))
