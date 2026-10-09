"""Extrai leituras de experiências intermediárias já observadas.

Para a sonda (a,b), mudança(a,b)=mudança(a)+mudança(b no estado após a).
Isto compõe MEDIÇÕES reais; não introduz estado previsto como rótulo.
Primeiras sondas/estado: até 512 amostras; sondas compostas: até 256 por
primeira ação. Seleção estratificada e determinística só de episódios de treino.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from avaliar import KernelRidge
from avaliar_composicao import features
from avaliar_estado_preditivo import encode, load as load_states

HERE = Path(__file__).resolve().parent


def stratified(episodes, allowed, limit):
    candidates = np.flatnonzero(np.isin(episodes, sorted(allowed)))
    if len(candidates) <= limit:
        return candidates
    rng = np.random.default_rng(9706)
    per_group = [rng.permutation(candidates[episodes[candidates] == e]).tolist()
                 for e in sorted(allowed)]
    selected = []
    while len(selected) < limit and any(per_group):
        for pool in per_group:
            if pool and len(selected) < limit:
                selected.append(pool.pop())
    return np.asarray(selected)


class ExperienceExtractor:
    def __init__(self, game):
        _, pairs, signature, valid, episode = load_states(game, 97)
        self.definition_path = HERE / "resultados" / f"{game}_sondas_decisao_definicoes.json"
        self.definition = json.loads(self.definition_path.read_text(encoding="utf-8"))
        self.probes = self.definition["probes"]
        self.n = len(self.probes)
        basis = np.load(HERE / "resultados" / f"{game}_base_estado_preditivo.npz")
        self.mean, self.basis = basis["mean"], basis["basis"]
        self.root_x = features(pairs[:, 0, 0])
        self.episode = episode
        root_s = signature[:, 0, 0]
        later_s = signature[:, :, 1:].reshape(-1, 4, 336)
        later_x = features(pairs[:, :, 1:].reshape(-1, 2, 336))
        later_e = np.broadcast_to(episode[:, None, None], valid[:, :, 1:].shape).reshape(-1)
        later_valid = valid[:, :, 1:].reshape(-1)
        self.single_s = np.concatenate((root_s, later_s[later_valid]))
        self.single_x = np.concatenate((self.root_x, later_x[later_valid]))
        self.single_e = np.concatenate((episode, later_e[later_valid]))
        self.single_z = encode(self.single_s, self.mean, self.basis)
        self.single_ids = [i for i, probe in enumerate(self.probes) if len(probe["actions"]) == 1]
        self.double = {}
        effects_bank = np.load(HERE / "resultados" / f"{game}_sondas_decisao_97.npz")["effects"]
        errors = []
        for action in range(4):
            indices = [i for i, probe in enumerate(self.probes)
                       if len(probe["actions"]) == 2 and probe["actions"][0] == action]
            current = signature[:, action, :-1]
            next_signature = signature[:, action, 1:]
            # A máscara remove estados já terminados; terminação pelo primeiro
            # comando deixa a segunda mudança em zero, como no executor real.
            composed = current[:, :, action, None, :] + next_signature
            for second in range(4):
                program_index = self.definition["programs"].index([action, second])
                errors.append(float(np.max(np.abs(composed[:, 0, second] - effects_bank[program_index]))))
            mask = valid[:, action, :-1].reshape(-1)
            self.double[action] = {
                "ids": indices,
                "x": features(pairs[:, action, :-1].reshape(-1, 2, 336))[mask],
                "effects": composed.reshape(-1, 4, 336)[mask],
                "episode": np.broadcast_to(episode[:, None], valid[:, action, :-1].shape).reshape(-1)[mask],
            }
        self.certified_root_max_error = max(errors)
        if self.certified_root_max_error > 2e-7:
            raise RuntimeError(f"rótulo composto diverge de ramo executado: {self.certified_root_max_error}")

    def response(self, effects, ids, double=False, random=False):
        out = np.zeros((len(effects), len(ids)))
        for column, index in enumerate(ids):
            probe = self.probes[index]
            action = probe["actions"][1] if double else probe["actions"][0]
            vector = probe["random_readout"] if random else probe["readout"]
            out[:, column] = effects[:, action] @ np.asarray(vector)
        return out

    def predict(self, query_x, allowed):
        selection = stratified(self.single_e, allowed, 512)
        if len(selection) < 4:
            raise RuntimeError("experiências insuficientes no treino")
        regular = self.response(self.single_s, self.single_ids)
        random = self.response(self.single_s, self.single_ids, random=True)
        target = np.column_stack((self.single_z, regular, random))
        prediction = KernelRidge().fit(self.single_x[selection], target[selection]).predict(query_x)
        z = prediction[:, :32]
        q, qr = np.zeros((len(query_x), self.n)), np.zeros((len(query_x), self.n))
        count = len(self.single_ids)
        q[:, self.single_ids] = prediction[:, 32:32 + count]
        qr[:, self.single_ids] = prediction[:, 32 + count:]
        counts = {"single_and_state": int(len(selection))}
        for action, data in self.double.items():
            ids = data["ids"]
            if not ids:
                continue
            selection = stratified(data["episode"], allowed, 256)
            if len(selection) < 4:
                raise RuntimeError("experiências compostas insuficientes")
            regular = self.response(data["effects"], ids, double=True)
            random = self.response(data["effects"], ids, double=True, random=True)
            target = np.column_stack((regular, random))
            prediction = KernelRidge().fit(data["x"][selection], target[selection]).predict(query_x)
            q[:, ids], qr[:, ids] = prediction[:, :len(ids)], prediction[:, len(ids):]
            counts[f"first_action_{action}"] = int(len(selection))
        return z, q, qr, counts

    def cross_predict(self, allowed):
        z, q, qr = np.zeros((64, 32)), np.zeros((64, self.n)), np.zeros((64, self.n))
        for episode in sorted(allowed):
            query = self.episode == episode
            train_groups = set(allowed) - {episode}
            predicted_z, predicted_q, predicted_r, _ = self.predict(self.root_x[query], train_groups)
            z[query], q[query], qr[query] = predicted_z, predicted_q, predicted_r
        return z, q, qr
