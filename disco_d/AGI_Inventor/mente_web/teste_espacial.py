"""Controle geometrico: recebe apenas pixels/K. Verdade 3D so no avaliador."""
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from espacial import MemoriaEspacial, feicoes, pares, localiza, projeta


def video():
    rng = np.random.default_rng(301)
    h, w = 360, 480
    K = np.array([[360, 0, w/2], [0, 360, h/2], [0, 0, 1.]])
    textures = [cv2.GaussianBlur(rng.integers(0, 256, (600, 800, 3), dtype=np.uint8), (3, 3), .5) for _ in range(2)]
    for tex in textures:
        for _ in range(350):
            x, y = rng.integers([10, 10], [790, 590])
            cv2.circle(tex, (int(x), int(y)), int(rng.integers(2, 9)), tuple(int(a) for a in rng.integers(40, 255, 3)), -1)
    yy, xx = np.mgrid[:h, :w]
    rx, ry = (xx-w/2)/360, (yy-h/2)/360
    frames = []
    # Translacao de camera; dois planos com profundidades diferentes, sem metadados 3D.
    for c in [0, .10, .22, .34, .46, .58]:
        bx, by = c+rx*7, ry*7
        bi = np.clip(((bx+5)/10*799).astype(int), 0, 799)
        bj = np.clip(((by+4)/8*599).astype(int), 0, 599)
        out = textures[0][bj, bi].copy()
        fx, fy = c+rx*3.5, ry*3.5
        mask = (fx > -1.3) & (fx < .05) & (abs(fy) < .95)
        fi = np.clip(((fx+1.3)/1.35*799).astype(int), 0, 799)
        fj = np.clip(((fy+.95)/1.9*599).astype(int), 0, 599)
        out[mask] = textures[1][fj, fi][mask]
        frames.append(out)
    return frames, K


class EspacialTeste(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames, cls.K = video()

    def test_01_reconstrucao_persistencia_e_vista_reservada(self):
        with tempfile.TemporaryDirectory() as d:
            m = MemoriaEspacial(d)
            # Vista 3 nunca entra no treino: ela e reservada integralmente.
            r = m.observa([self.frames[i] for i in [0, 1, 2, 4, 5]], [0, 1, 2, 4, 5], 'teste://sala', self.K)
            self.assertEqual(r['status'], 'reconstruida', r)
            s = m._carrega(r['id'])
            self.assertGreater(len(s['superficie']), 500)
            self.assertGreater(len(s['facetas']), 100)
            self.assertLess(int(s['facetas'].max()), len(s['superficie']))
            # Profundidade relativa deve separar o plano proximo e o distante (~1:2).
            X = s['superficie']
            px0, depths = projeta(X, self.K, s['cameras'][0])
            near = (px0[:, 0] > 120) & (px0[:, 0] < 230) & (px0[:, 1] > 100) & (px0[:, 1] < 255)
            far = (px0[:, 0] > 300) & (px0[:, 0] < 420)
            self.assertGreater(near.sum(), 80)
            self.assertGreater(far.sum(), 200)
            ratio = np.median(depths[far])/np.median(depths[near])
            self.assertGreater(ratio, 1.6)
            self.assertLess(ratio, 2.4)
            px, desc = feicoes(self.frames[3])
            matches = pares(s['desc'], desc)
            loc = localiza(s['X'][matches[:, 0]], px[matches[:, 1]], self.K)
            self.assertIsNotNone(loc)
            self.assertLess(loc[2], 1.8)
            before = (Path(d)/'espacial'/ (r['id']+'.npz')).read_bytes()
            reopened = MemoriaEspacial(d)
            self.assertEqual(before, (Path(d)/'espacial'/ (r['id']+'.npz')).read_bytes())
            r2 = reopened.observa(self.frames[2:5], [2, 3, 4], 'teste://outro-video-da-mesma-sala', self.K)
            self.assertEqual(r2['id'], r['id'])
            self.assertEqual(r2['revisao'], 2)
            public = json.loads((Path(d)/'espacial'/ (r['id']+'.json')).read_text(encoding='utf-8'))
            self.assertEqual(public['regioes_ocultas'], 'desconhecidas')
            self.assertFalse(public['dinamica_3d'])
            self.assertEqual(public['geometria'], 'inferida')
            print('Reconstrucao:', r['pontos'], 'pontos; razao profundidades:', round(ratio, 3),
                  '; erro na vista reservada:', round(loc[2], 3), 'px', flush=True)

    def test_02_camera_fixa_nao_ganha_profundidade(self):
        with tempfile.TemporaryDirectory() as d:
            m = MemoriaEspacial(d)
            r = m.observa([self.frames[0]]*4, range(4), 'teste://fixa', self.K)
            self.assertEqual(r['status'], 'sem_geometria')
            self.assertEqual(m.indice['cenas'], [])

    def test_03_rotacao_pura_nao_ganha_volume(self):
        with tempfile.TemporaryDirectory() as d:
            views = []
            for angle in [0, .03, .06, .09]:
                R = cv2.Rodrigues(np.array([0., angle, 0.]))[0]
                H = self.K @ R @ np.linalg.inv(self.K)
                views.append(cv2.warpPerspective(self.frames[0], H, (480, 360)))
            r = MemoriaEspacial(d).observa(views, range(4), 'teste://rotacao', self.K)
            self.assertEqual(r['status'], 'sem_geometria', r)

    def test_04_cortes_entre_cenas_nao_acoplam_mundos(self):
        with tempfile.TemporaryDirectory() as d:
            rng = np.random.default_rng(19)
            frames = [rng.integers(0, 256, (360, 480, 3), dtype=np.uint8) for _ in range(4)]
            r = MemoriaEspacial(d).observa(frames, range(4), 'teste://cortes', self.K)
            self.assertEqual(r['status'], 'sem_geometria')


if __name__ == '__main__':
    unittest.main(verbosity=2)
