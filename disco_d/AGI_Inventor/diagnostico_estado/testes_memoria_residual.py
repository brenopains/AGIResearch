"""Isolamento por episódio, história passada, exportação e absorção."""

import tempfile
from pathlib import Path

import numpy as np

from avaliar import KernelRidge
from memoria_residual import HistoryProjection, MemoryDynamics, crossfit_residual, history_input


def main():
    rng = np.random.default_rng(19)
    history = rng.normal(size=(24, 6, 3))
    before = history.copy()
    scrambled = history_input(history, "permuted_history", np.arange(24)).reshape(history.shape)
    assert np.array_equal(history, before)
    assert np.array_equal(scrambled[:, -2:], history[:, -2:])
    assert all(np.array_equal(np.sort(scrambled[i, :-2], axis=0), np.sort(history[i, :-2], axis=0)) for i in range(24))
    z, next_z = rng.normal(size=(24, 2)), rng.normal(size=(24, 2))
    actions = np.arange(24) % 2
    episodes = np.arange(24) // 4
    reward, continuation = rng.normal(size=24), np.ones(24)
    residual = crossfit_residual(z, next_z, reward, continuation, actions, episodes)
    target = np.column_stack((next_z, reward, continuation))
    for a in range(2):
        for half in (0, 1):
            held, train = (actions == a) & (episodes % 2 == half), (actions == a) & (episodes % 2 != half)
            truth = target[held] - KernelRidge().fit(z[train], target[train]).predict(z[held])
            assert np.allclose(residual[held], truth)
    # The last two frames cannot distinguish the worlds; an earlier observation can.
    h = np.zeros((20, 6, 1))
    h[:, 0, 0] = np.tile([-1., 1.], 10)
    hidden = h[:, 0]
    projection = HistoryProjection().fit(h, hidden, np.zeros(20, int), "residual_history")
    pair_projection = HistoryProjection().fit(h, hidden, np.zeros(20, int), "two_frames")
    assert np.std(projection.transform(h)[:, 0]) > .9
    assert np.all(pair_projection.transform(h) == 0)
    model = MemoryDynamics().fit(z, next_z, history, history * .9, reward, continuation,
                                 actions, residual, "residual_history", np.arange(24), np.arange(24) + 1)
    query = model.state(z[:4], history[:4], np.arange(4))
    plans = np.tile([0, 1, 1, 0], (4, 1))
    expected = model.rollout(query, plans)
    assert np.array_equal(history, before)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "memory.npz"
        model.save(path)
        restored = MemoryDynamics.load(path)
        restored_query = restored.state(z[:4], history[:4], np.arange(4))
        assert np.allclose(restored_query, query, atol=1e-12)
        actual = restored.rollout(restored_query, plans)
        assert np.allclose(actual["returns"], expected["returns"], atol=1e-12)
        assert np.allclose(actual["states"], expected["states"], atol=1e-12)
    terminal = MemoryDynamics().fit(z, next_z, history, history, np.full(24, 3.), np.zeros(24),
                                    np.zeros(24, int), residual, "state_only", np.arange(24), np.arange(24))
    state = terminal.state(z[:4], history[:4], np.arange(4))
    assert np.allclose(terminal.rollout(state, np.zeros((4, 6), int))["returns"], 3.)
    print("memoria_residual_ok: passado, isolamento_episodico, variavel_gerada, corpo_completo, composicao, termino")


if __name__ == "__main__":
    main()
