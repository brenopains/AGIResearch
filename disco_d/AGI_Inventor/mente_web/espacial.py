"""Memoria espacial multivista, sem pesos pre-treinados nem geometria da arena.

Estima cameras e pontos 3D pelas correspondencias entre imagens. E um estimador
geometrico explicito, nao um decoder 3D do cortex nem compreensao semantica.
Nao completa superficies escondidas; todas as profundidades sao INFERIDAS.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

MAX_PONTOS = 24000
MAX_CENAS = 20
MIN_PARALAXE = 1.0  # graus; evita dar profundidade a rotacao pura/quadros iguais


def atomico(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    for _ in range(6):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(.05)
    os.replace(tmp, path)


def retina(img, lado=480):
    q = np.asarray(img)
    if q.dtype != np.uint8:
        q = np.clip(q * (255 if q.max() <= 1 else 1), 0, 255).astype(np.uint8)
    if q.ndim == 2:
        q = np.repeat(q[..., None], 3, axis=2)
    h, w = q.shape[:2]
    if max(h, w) > lado:
        q = cv2.resize(q, (round(w * lado / max(h, w)), round(h * lado / max(h, w))), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(q[..., :3])


def camera(forma):
    h, w = forma[:2]
    # Convencao inicial declarada; focal desconhecida nao e medida fisica.
    f = .9 * max(h, w)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.]], float)


def feicoes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    pts, desc = cv2.ORB_create(nfeatures=2400, edgeThreshold=12, fastThreshold=9).detectAndCompute(gray, None)
    return np.array([p.pt for p in pts], np.float32).reshape(-1, 2), desc


def pares(d1, d2):
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return np.empty((0, 2), int)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    a = {m.queryIdx: m.trainIdx for pair in bf.knnMatch(d1, d2, k=2) if len(pair) == 2
         for m, n in [pair] if m.distance < .72 * n.distance}
    b = {m.queryIdx: m.trainIdx for pair in bf.knnMatch(d2, d1, k=2) if len(pair) == 2
         for m, n in [pair] if m.distance < .72 * n.distance}
    return np.array([(i, j) for i, j in a.items() if b.get(j) == i], int).reshape(-1, 2)


def projeta(X, K, pose):
    p = X @ pose[:, :3].T + pose[:, 3]
    z = p[:, 2]
    im = p @ K.T
    return im[:, :2] / np.maximum(z[:, None], 1e-9), z


def triangula(a, b, K, A, B):
    if not len(a):
        return np.empty((0, 3)), np.zeros(0, bool), np.zeros(0), np.zeros(0)
    h = cv2.triangulatePoints(K @ A, K @ B, a.T.astype(float), b.T.astype(float)).T
    X = h[:, :3] / np.where(abs(h[:, 3:]) > 1e-10, h[:, 3:], np.nan)
    qa, za = projeta(X, K, A)
    qb, zb = projeta(X, K, B)
    erro = np.maximum(np.linalg.norm(qa-a, axis=1), np.linalg.norm(qb-b, axis=1))
    ca, cb = -A[:, :3].T @ A[:, 3], -B[:, :3].T @ B[:, 3]
    va, vb = X-ca, X-cb
    cos = np.sum(va*vb, axis=1) / np.maximum(np.linalg.norm(va, axis=1)*np.linalg.norm(vb, axis=1), 1e-12)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    ok = np.isfinite(X).all(axis=1) & (za > 0) & (zb > 0) & (erro < 1.8) & (ang >= MIN_PARALAXE)
    return X, ok, erro, ang


def inicia(a, b, K):
    if len(a) < 30 or np.median(np.linalg.norm(a-b, axis=1)) < 2:
        return None
    E, mask = cv2.findEssentialMat(a, b, K, method=cv2.RANSAC, prob=.999, threshold=1.0)
    if E is None or E.shape != (3, 3):
        return None
    _, R, t, mask = cv2.recoverPose(E, a, b, K, mask=mask)
    A, B = np.c_[np.eye(3), np.zeros(3)], np.c_[R, t]
    X, ok, erro, ang = triangula(a, b, K, A, B)
    ok &= mask.ravel() != 0
    if ok.sum() < 30 or ok.mean() < .45:
        return None
    # Uma homografia que explica quase tudo sugere plano/rotacao: nao fingir volume.
    _, hm = cv2.findHomography(a, b, cv2.RANSAC, 1.2)
    if hm is not None and hm.mean() > .90:
        return None
    return A, B, X, ok, erro, ang


def localiza(X, pixels, K):
    """Pose ajustada em 80%; erro em correspondencias reservadas, nao no ajuste."""
    if len(X) < 24:
        return None
    reserva = np.arange(len(X)) % 5 == 0
    treino = ~reserva
    good, rv, tv, inliers = cv2.solvePnPRansac(X[treino].astype(float), pixels[treino].astype(float), K, None,
                                            iterationsCount=150, reprojectionError=2.2, confidence=.999)
    if not good or inliers is None or len(inliers) < 18:
        return None
    R = cv2.Rodrigues(rv)[0]
    pose = np.c_[R, tv]
    q, z = projeta(X, K, pose)
    errs = np.linalg.norm(q-pixels, axis=1)
    hold = errs[reserva]
    ok = (errs < 2.5) & (z > 0)
    if ok.mean() < .65 or np.median(hold) > 2.5:
        return None
    return pose, ok, float(np.median(hold))


def superficie(img1, img2, K, A, B):
    """Amostras de superficie por fluxo bidirecional e triangulacao, sem placas."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    yy, xx = np.mgrid[12:g1.shape[0]-12:4, 12:g1.shape[1]-12:4]
    a = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    grid_shape = xx.shape
    grid_ids = np.arange(len(a))
    if not len(a):
        return None
    # Regioes lisas nao permitem correspondencias confiaveis.
    eig = cv2.cornerMinEigenVal(g1, blockSize=5)
    keep = eig[a[:, 1].astype(int), a[:, 0].astype(int)] > .00015
    a = a[keep]
    grid_ids = grid_ids[keep]
    if not len(a):
        return None
    b, s, _ = cv2.calcOpticalFlowPyrLK(g1, g2, a[:, None], None, winSize=(19, 19), maxLevel=3)
    back, s2, _ = cv2.calcOpticalFlowPyrLK(g2, g1, b, None, winSize=(19, 19), maxLevel=3)
    b, back = b[:, 0], back[:, 0]
    good = s.ravel().astype(bool) & s2.ravel().astype(bool) & (np.linalg.norm(back-a, axis=1) < .6)
    good &= (b[:, 0] >= 0) & (b[:, 0] < g2.shape[1]) & (b[:, 1] >= 0) & (b[:, 1] < g2.shape[0])
    X, ok, err, angle = triangula(a, b, K, A, B)
    ok &= good
    # Exige tambem consistencia fotometrica; pixels com reflexo/movimento ficam fora.
    bi = np.clip(np.rint(b), [0, 0], [g2.shape[1]-1, g2.shape[0]-1]).astype(int)
    ai = a.astype(int)
    color = img1[ai[:, 1], ai[:, 0]]
    other = img2[bi[:, 1], bi[:, 0]]
    ok &= np.mean(abs(color.astype(float)-other), axis=1) < 35
    # Liga somente vizinhos observados com profundidade continua. Nenhuma face
    # cobre buracos sem correspondencia ou saltos de profundidade/oclusao.
    mapping = np.full(grid_shape[0]*grid_shape[1], -1, int)
    mapping[grid_ids[ok]] = np.arange(ok.sum())
    mapping = mapping.reshape(grid_shape)
    tl, tr, bl, br = mapping[:-1, :-1], mapping[:-1, 1:], mapping[1:, :-1], mapping[1:, 1:]
    faces = np.concatenate([np.stack([tl,tr,bl],axis=-1).reshape(-1,3),
                            np.stack([tr,br,bl],axis=-1).reshape(-1,3)])
    faces = faces[(faces >= 0).all(axis=1)]
    points = X[ok]
    if len(faces):
        vertices = points[faces]
        lens = np.stack([np.linalg.norm(vertices[:,i]-vertices[:,j],axis=1) for i,j in [(0,1),(1,2),(2,0)]],axis=1)
        _, depth = projeta(vertices.mean(axis=1), K, A)
        faces = faces[lens.max(axis=1) < .08*depth]
    return points, color[ok], err[ok], angle[ok], faces


def agrega_superficie(s, surf):
    offset = len(s['superficie'])
    for key, val in zip(('superficie', 'cores', 'erros', 'angulos'), surf[:4]):
        s[key] = np.concatenate([s[key], val])
    s['facetas'] = np.concatenate([s['facetas'], surf[4]+offset])
    drop = max(0, len(s['superficie'])-MAX_PONTOS)
    if drop:
        for key in ('superficie', 'cores', 'erros', 'angulos'):
            s[key] = s[key][drop:]
        s['facetas'] -= drop
        s['facetas'] = s['facetas'][(s['facetas'] >= 0).all(axis=1)]


class MemoriaEspacial:
    def __init__(self, pasta):
        self.pasta = Path(pasta) / 'espacial'
        self.pasta.mkdir(parents=True, exist_ok=True)
        path = self.pasta / 'indice.json'
        self.indice = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
            'versao': 1, 'cenas': [], 'tentativas': [], 'limites': [
                'Geometria inferida das imagens, aparencia observada. Sem pesos pre-treinados.',
                'A focal e aproximada quando a camera nao e calibrada; escala relativa, nao metros.',
                'Superficies escondidas ficam vazias. Nenhum objeto semantico ou dinamica 3D aprendida ainda.',
                'Precisa de textura, mudanca de ponto de vista e uma parte rigida comum. Cortes, camera fixa e rotacao pura podem impedir a reconstrucao.']}

    def _salva_indice(self):
        self.indice['quando'] = time.time()
        atomico(self.pasta / 'indice.json', self.indice)

    def _recusa(self, url, motivo):
        self.indice['tentativas'] = (self.indice['tentativas'] + [{'url': url, 'motivo': motivo, 'quando': time.time()}])[-30:]
        self._salva_indice()
        return {'status': 'sem_geometria', 'motivo': motivo}

    def _carrega(self, id):
        with np.load(self.pasta / (id + '.npz'), allow_pickle=False) as z:
            s = {k: z[k] for k in z.files}
        s.setdefault('facetas', np.empty((0, 3), int))
        return s

    def _grava(self, id, s, urls):
        # Pontos de referencia ORB ficam juntos de seus descritores; superficie e separada.
        for k in ('superficie', 'cores', 'erros', 'angulos'):
            s[k] = s[k][-MAX_PONTOS:]
        dst = self.pasta / (id + '.npz')
        with open(str(dst)+'.tmp', 'wb') as f:
            np.savez_compressed(f, **s)
        os.replace(str(dst)+'.tmp', dst)
        xs = s['superficie'] if len(s['superficie']) else s['X']
        centro = np.median(xs, axis=0)
        raio = float(np.quantile(np.linalg.norm(xs-centro, axis=1), .9))
        public = {'id': id, 'urls': urls, 'quando': time.time(), 'status': 'geometria_estimada',
                  'pontos': xs.round(5).tolist(), 'cores': s['cores'].tolist(),
                  'facetas': s['facetas'].tolist(),
                  'erro_px': s['erros'].round(3).tolist(), 'paralaxe_graus': s['angulos'].round(2).tolist(),
                  'suporte': [2]*len(xs), 'centro': centro.tolist(), 'raio': max(raio, .1),
                  'cameras': s['cameras'].round(6).tolist(), 'tempos': s['tempos'].tolist(),
                  'K': s['K'].tolist(), 'forma': s['forma'].tolist(),
                  'erro_reserva_px': float(s['reserva']), 'geometria': 'inferida', 'aparencia': 'observada',
                  'calibracao': 'fornecida' if int(s['calibrada']) else 'focal_aproximada',
                  'escala': 'relativa', 'dinamica_3d': False, 'regioes_ocultas': 'desconhecidas'}
        atomico(self.pasta / (id+'.json'), public)
        antigos = [c for c in self.indice['cenas'] if c['id'] == id]
        rev = antigos[0]['revisao'] + 1 if antigos else 1
        card = {'id': id, 'urls': urls, 'pontos': len(xs), 'vistas': len(s['cameras']), 'revisao': rev,
                'quando': public['quando'], 'erro_reserva_px': public['erro_reserva_px'], 'status': public['status']}
        self.indice['cenas'] = [c for c in self.indice['cenas'] if c['id'] != id] + [card]
        self._salva_indice()
        return {**card, 'status': 'reconstruida'}

    def observa(self, quadros, tempos, url, K=None):
        if len(quadros) < 3:
            return self._recusa(url, 'Poucas vistas para reconstruir e validar.')
        frames = [retina(q) for q in quadros]
        shape = frames[0].shape
        if min(shape[:2]) < 100 or any(q.shape != shape for q in frames):
            return self._recusa(url, 'Imagens pequenas ou dimensoes variaveis.')
        # Cameras podem ser fornecidas calibradas; nenhum extrinseco ou ponto 3D entra.
        calibrated = K is not None
        K = np.asarray(K, float).copy() if calibrated else camera(shape)
        if calibrated:
            original = np.asarray(quadros[0]).shape
            K[0] *= shape[1]/original[1]
            K[1] *= shape[0]/original[0]
        feature = [feicoes(q) for q in frames]
        chosen = None
        # Reconhecimento geometrico mesmo em outra URL, sem juntar por nome/tema.
        candidates = self.indice['cenas'][-MAX_CENAS:]
        candidates += [c for c in self.indice['cenas'][:-MAX_CENAS] if url in c['urls']]
        for card in reversed(candidates):
            s = self._carrega(card['id'])
            if not np.allclose(s['K'], K) or tuple(s['forma']) != shape[:2]:
                continue
            p = pares(s['desc'], feature[0][1])
            loc = localiza(s['X'][p[:, 0]], feature[0][0][p[:, 1]], K) if len(p) else None
            if loc is not None:
                chosen = (card['id'], s, list(card['urls']), loc[0], 0)
                break
        if chosen is None:
            for j in range(1, min(len(frames), 12)):
                p = pares(feature[0][1], feature[j][1])
                r = inicia(feature[0][0][p[:, 0]], feature[j][0][p[:, 1]], K)
                if r is None:
                    continue
                A, B, X, ok, err, ang = r
                ids = p[ok, 0]
                # Reservar correspondencias para verificar a nova vista tambem no bootstrap.
                loc = localiza(X[ok], feature[j][0][p[ok, 1]], K)
                if loc is None:
                    continue
                surf = superficie(frames[0], frames[j], K, A, B)
                if surf is None or len(surf[0]) < 50:
                    continue
                s = {'X': X[ok].astype(np.float32), 'desc': feature[0][1][ids],
                     'superficie': surf[0].astype(np.float32), 'cores': surf[1], 'erros': surf[2], 'angulos': surf[3], 'facetas': surf[4],
                     'cameras': np.stack([A, B]), 'tempos': np.array([tempos[0], tempos[j]], float),
                     'K': K, 'forma': np.array(shape[:2]), 'calibrada': np.array(int(calibrated)),
                     'reserva': np.array(loc[2])}
                id = hashlib.sha256((url+str(time.time_ns())).encode()).hexdigest()[:16]
                chosen = (id, s, [url], B, j)
                break
        if chosen is None:
            return self._recusa(url, 'Sem paralaxe/estrutura rigida suficiente: profundidade nao identificada.')
        id, s, urls, previous_pose, previous_index = chosen
        if url not in urls:
            urls.append(url)
        for j in range(previous_index+1, len(frames)):
            p = pares(s['desc'], feature[j][1])
            loc = localiza(s['X'][p[:, 0]], feature[j][0][p[:, 1]], K) if len(p) else None
            if loc is None:
                continue
            pose, matched, reserve = loc
            surf = superficie(frames[previous_index], frames[j], K, previous_pose, pose)
            if surf is not None and len(surf[0]) >= 50:
                agrega_superficie(s, surf)
                pm = pares(feature[previous_index][1], feature[j][1])
                X, ok, _, _ = triangula(feature[previous_index][0][pm[:, 0]], feature[j][0][pm[:, 1]], K, previous_pose, pose)
                # Referencias novas permitem localizar vistas progressivamente mais distantes.
                known = set(p[:, 1])
                ok &= np.array([int(i) not in known for i in pm[:, 1]])
                s['X'] = np.concatenate([s['X'], X[ok].astype(np.float32)])[-4000:]
                s['desc'] = np.concatenate([s['desc'], feature[j][1][pm[ok, 1]]])[-4000:]
            s['cameras'] = np.concatenate([s['cameras'], pose[None]])[-120:]
            s['tempos'] = np.r_[s['tempos'], tempos[j]][-120:]
            s['reserva'] = np.array(reserve)
            previous_index, previous_pose = j, pose
        return self._grava(id, s, urls)
