import runtime
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from circuito import Circuito, Genoma, eval_rule, mutar
from contratos import Diario, Feedback
from importar import build_graph, fuse, rewire, rows
from ponte import Ponte


def fixture(name='fixture'):
    # Somente testes de software; nunca rotulado como dados biológicos ou benchmark de AGI.
    return build_graph(((i + 1, (i + 1) % 12 + 1, 2) for i in range(12)), name, 12, {'synthetic_test_fixture': True})


class Tests(unittest.TestCase):
    def test_missing_ids_not_fabricated(self):
        g = build_graph([(0, 1, 1), (1, 2, 3), (2, 1, 4)], 'unit', 2, {})
        self.assertEqual(g['provenance']['invalid_rows'], 1)
        self.assertEqual(g['counts'], [3, 4])

    def test_large_ids_preserved(self):
        a, b = 864691135000000001, 864691135000000002
        g = build_graph([(a, b, 1)], 'unit', 2, {})
        self.assertEqual(g['node_ids'], [f'unit:{a}', f'unit:{b}'])

    def test_microns_headerless_semantics(self):
        line = '106675816,t,142854,244590,18572,84396795031127211,864691136041487043,142890,244657,18573,84467163775312026,864691135589913867,142946,244642,18573,508\n'
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'x.csv'
            path.write_text(line)
            self.assertEqual(list(rows(path, 'mouse_microns', 1)), [(864691136041487043, 864691135589913867, 1)])

    def test_rule_rejects_external_execution(self):
        with self.assertRaises(ValueError):
            eval_rule(('eval', 'print(1)'), {})

    def test_rewire_preserves_degrees(self):
        g = fixture()
        shuffled = rewire(g, seed=2)
        self.assertEqual(Counter(g['pre']), Counter(shuffled['pre']))
        self.assertEqual(Counter(g['post']), Counter(shuffled['post']))
        self.assertGreater(shuffled['provenance']['swaps'], 0)
        self.assertNotEqual(g['post'], shuffled['post'])

    def test_fusion_keeps_identity_and_marks_bridges(self):
        g = fuse([fixture('a'), fixture('b')], 4)
        self.assertEqual(len(set(g['node_ids'])), 24)
        self.assertGreater(g['edge_measured'].count(False), 0)
        c = Circuito(g, 2, 2)
        self.assertEqual(c.original.tolist(), g['edge_measured'])

    def test_freeze_changes_no_parameters(self):
        c = Circuito(fixture(), 2, 2)
        before = [getattr(c, key).copy() for key in ('w', 'input', 'readout', 'bias')]
        c.step([.2, -.4])
        c.feedback([.3, .1], plastic=False, growth=False, learn_readout=False)
        for key, expected in zip(('w', 'input', 'readout', 'bias'), before):
            np.testing.assert_array_equal(getattr(c, key), expected)

    def test_plasticity_growth_and_log_are_real(self):
        with tempfile.TemporaryDirectory() as d:
            journal = Diario(Path(d) / 'events.jsonl')
            c = Circuito(fixture(), 2, 2, replace(Genoma(), growth_every=2, growth_limit=5), diario=journal)
            w = c.w.copy()
            for t in range(20):
                c.step([.8, -.5])
                c.feedback([.5, -.3])
            self.assertGreater(c.grown_total, 0)
            self.assertLessEqual(c.grown_total, 5)
            self.assertFalse(np.array_equal(w, c.w[:len(w)]))
            self.assertTrue(np.isfinite(c.x).all())
            journal.close()
            events = [json.loads(s) for s in (Path(d) / 'events.jsonl').read_text().splitlines()]
            self.assertEqual(sum(e['kind'] == 'connection_created' for e in events), c.grown_total)

    def test_feedback_has_to_follow_prediction(self):
        c = Circuito(fixture(), 2, 2)
        with self.assertRaises(RuntimeError):
            c.feedback([0, 0])
        c.step([0, 0])
        c.feedback([0, 0])
        with self.assertRaises(RuntimeError):
            c.feedback([0, 0])

    def test_checkpoint_exact_resume_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            c = Circuito(fixture(), 2, 2)
            for _ in range(5):
                c.step([.2, -.1])
                c.feedback([.4, -.1])
            path = Path(d) / 'body.npz'
            c.save(path)
            restored = Circuito.load(path)
            np.testing.assert_array_equal(c.step([.5, .2]), restored.step([.5, .2]))
            c.feedback([.4, -.3])
            restored.feedback([.4, -.3])
            np.testing.assert_array_equal(c.w, restored.w)
            with path.open('ab') as f:
                f.write(b'altered')
            with self.assertRaises(ValueError):
                Circuito.load(path)

    def test_no_checkpoint_side_effect_midstep(self):
        with tempfile.TemporaryDirectory() as d:
            c = Circuito(fixture(), 2, 2)
            c.step([0, 0])
            path = Path(d) / 'mid.npz'
            with self.assertRaises(RuntimeError):
                c.save(path)
            self.assertFalse(path.exists())

    def test_module_removal_and_restoration(self):
        with tempfile.TemporaryDirectory() as d:
            c = Circuito(fuse([fixture('a'), fixture('b')]), 2, 2)
            for _ in range(8):
                c.step([.8, -.2])
                c.feedback([.2, .3])
            path = Path(d) / 'x.npz'
            c.save(path)
            removed = Circuito.load(path)
            removed.remove_module(0)
            removed.step([.8, -.2])
            self.assertTrue(np.all(removed.x[removed.modules == 0] == 0))
            restored = Circuito.load(path)
            np.testing.assert_array_equal(restored.step([.7, .1]), c.step([.7, .1]))

    def test_evolved_rules_valid(self):
        g = Genoma()
        rng = np.random.default_rng(4)
        for _ in range(40):
            g = mutar(g, rng)
            g.validate()

    def test_existing_motor_and_hdc_bridge(self):
        bridge = Ponte()
        x = [[np.sin(t / 3), np.cos(t / 7)] for t in range(32)]
        y = [1 if a > 0 else -1 for a, b in x]
        proposal = bridge.propor(x, y)
        pred = bridge.executar(proposal, x)
        self.assertEqual(pred[2:], y[2:])
        bridge.guardar('experience', x, {'proposal': proposal})
        self.assertEqual(bridge.recuperar(x)[0]['name'], 'experience')

    def test_predictive_learning_reduces_fixture_error(self):
        c = Circuito(fixture(), 2, 2)
        loss = []
        for _ in range(100):
            c.step([.4, -.2])
            loss.append(c.feedback([.7, -.5]))
        self.assertLess(np.mean(loss[-10:]), np.mean(loss[:10]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
