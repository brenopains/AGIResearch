"""Contratos físicos, partições, portão contextual e interrupção do monitor."""

import tempfile
from collections import deque
from pathlib import Path

import numpy as np

import coletar_experiencia_cruzada as collector
import experiencia_cruzada as experiment
import piloto_experiencia_cruzada as monitor_module
from avaliar_experiencia_cruzada import observation_collisions


class FakeWorld:
    def __init__(self):
        self.steps = 0

    def act(self, action):
        self.steps += 1
        return 1.

    def game_over(self):
        return self.steps >= 3


def main():
    old_gray = collector.gray
    collector.gray = lambda world: np.full((210, 160), world.steps, np.uint8)
    try:
        world, budget = FakeWorld(), collector.StepBudget(10)
        history = deque([np.zeros((210, 160), np.uint8) for _ in range(16)], maxlen=16)
        row = collector.play(world, [0, 1, 2, 3], [0] * 16, history, budget)
        assert budget.used == row[2] == 3 and world.steps == 3
        assert np.array_equal(row[0], [3.] + [0.] * 15)
        assert np.array_equal(row[1], [True] + [False] * 16)
        assert np.array_equal(history[-1], np.zeros((210, 160), np.uint8))
        limited = collector.StepBudget(2)
        try:
            collector.play(FakeWorld(), [0, 1, 2, 3], [0] * 16, history, limited)
        except RuntimeError:
            assert limited.used == 2
        else:
            raise AssertionError("orçamento não interrompeu")
    finally:
        collector.gray = old_gray
    episode = np.repeat(np.arange(8), 4)
    real = np.zeros((32, 2, 2))
    real[::2, :, 0], real[1::2, :, 1] = 1., 1.
    assert experiment.context_identification(real, episode)["sufficient_contextual_contrast"]
    rows = {arm: {"per_root_regret": [1.] * 32} for arm in experiment.ARMS}
    rows["quadratic"] = {"per_root_regret": [0.] * 32}
    assert experiment.gate(rows, real, episode)["positive"]
    assert experiment.gate(rows, np.zeros_like(real), episode)["status"] == "inconclusive_insufficient_contextual_contrast"
    rows["shuffled_targets"] = {"per_root_regret": [0.] * 32}
    assert not experiment.gate(rows, real, episode)["positive"]
    h = np.asarray([[0.], [0.], [1.]])
    collision = observation_collisions(h, np.asarray([[[1., 0.]], [[0., 1.]], [[1., 0.]]]), [0, 1, 2])
    assert collision["insufficiency_witness_found"]
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / "resultados").mkdir()
        old_here, old_base = monitor_module.HERE, experiment.base
        monitor_module.HERE = root
        experiment.base = lambda game, seed: root / f"{game}_{seed}"
        try:
            history = np.zeros((32, 16, 336), np.float32)
            programs = np.repeat(np.arange(4)[:, None], 16, axis=1)
            rewards = np.broadcast_to(np.arange(4)[None, :, None], (32, 4, 16)).astype(float)
            np.savez_compressed(experiment.base("fake", 131).with_suffix(".npz"), histories=history, programs=programs, rewards=rewards)
            data = experiment.training_data("fake")
            assert data[0].shape == (512, 16, 336) and len(data[1]) == 512
            assert np.array_equal(data[5], np.repeat(np.arange(32), 16))
            assert [len(p) for p in data[1][:16]] == list(experiment.PREFIXES) * 2
            assert data[3][7] - data[4][7] == -16.
            memory = monitor_module.PilotMonitor(seconds=10, rss_bytes=1)
            aborted = memory.run(["-c", "import time; time.sleep(10)"], "teste_memoria", root)
            assert aborted["status"] == "inconclusive_memory_limit" and aborted["exit_code"] != 0
            assert memory.maximum_processes == 2 and memory.peak_rss > 1
            deadline = monitor_module.PilotMonitor(seconds=.15)
            aborted = deadline.run(["-c", "import time; time.sleep(10)"], "teste_prazo", root)
            assert aborted["status"] == "inconclusive_wall_limit" and aborted["exit_code"] != 0
        finally:
            monitor_module.HERE, experiment.base = old_here, old_base
    print("experiencia_cruzada_ok: terminal_absorvente, custo, pares, alvos, inversoes, placebo_no_portao, colisao_e_monitor")


if __name__ == "__main__":
    main()
