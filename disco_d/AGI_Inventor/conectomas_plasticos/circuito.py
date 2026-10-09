"""Rede esparsa de taxas, plástica. Topologia medida; dinâmica artificial explícita.

Não é emulação biologicamente calibrada. Atualização local não é BPTT.
"""
import runtime
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import numpy as np
from contratos import digest


@dataclass(frozen=True)
class Genoma:
    leak: float = 0.4
    input_rate: float = 0.003
    readout_rate: float = 0.01
    eta: float = 0.005
    trace_decay: float = 0.8
    decay: float = 0.001
    hebb: float = 0.0
    recurrent_bound: float = 1.2
    growth_every: int = 16
    growth_limit: int = 64
    prune_threshold: float = 0.00001
    # Programa de plasticidade interpretado: termos locais, sem eval/Python.
    rule: tuple = ('add', ('mul', ('var', 'eligibility'), ('var', 'error')),
                   ('mul', ('var', 'decay'), ('var', 'weight')))

    def validate(self):
        for key in ('input_rate', 'readout_rate', 'eta', 'decay', 'prune_threshold'):
            value = getattr(self, key)
            if not np.isfinite(value) or not 0 <= value <= 0.1:
                raise ValueError(key)
        for key in ('leak', 'trace_decay'):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(key)
        if not 0 < self.recurrent_bound <= 4 or not 0 <= self.growth_limit <= 10000:
            raise ValueError('Limites de estabilidade/crescimento')
        if not np.isfinite(self.hebb) or abs(self.hebb) > 0.1 or self.growth_every < 1:
            raise ValueError('Genoma inválido')
        eval_rule(self.rule, {k: np.zeros(1) for k in ('eligibility', 'error', 'weight', 'decay', 'pre', 'post')})


def eval_rule(tree, variables, depth=0):
    if depth > 8:
        raise ValueError('Programa plástico excedeu profundidade')
    op = tree[0]
    if op == 'var' and len(tree) == 2 and tree[1] in variables:
        return variables[tree[1]]
    if op == 'const' and len(tree) == 2 and np.isfinite(tree[1]) and abs(tree[1]) <= 4:
        return tree[1]
    if op in ('add', 'mul', 'sub') and len(tree) == 3:
        a, b = (eval_rule(x, variables, depth + 1) for x in tree[1:])
        return {'add': lambda: a + b, 'mul': lambda: a * b, 'sub': lambda: a - b}[op]()
    if op == 'clip' and len(tree) == 2:
        return np.clip(eval_rule(tree[1], variables, depth + 1), -1, 1)
    raise ValueError('Operação inválida no programa plástico')


def mutar(genome, rng):
    """Mutar coeficientes e uma construção do programa local; diversidade inicial limitada."""
    key = rng.choice(['eta', 'input_rate', 'readout_rate', 'trace_decay', 'leak', 'hebb', 'rule'])
    if key == 'rule':
        extra = ('mul', ('const', float(rng.uniform(-0.1, 0.1))),
                 ('mul', ('var', 'pre'), ('var', 'post')))
        rule = ('clip', ('add', genome.rule, extra))
        try:
            candidate = replace(genome, rule=rule)
            candidate.validate()
            return candidate
        except ValueError:
            return genome
    value = float(getattr(genome, key))
    upper = 1 if key in ('leak', 'trace_decay') else 0.1
    lower = -0.1 if key == 'hebb' else 0.00001
    candidate = replace(genome, **{key: float(np.clip(value + rng.normal(0, max(abs(value), .001) * .25), lower, upper))})
    candidate.validate()
    return candidate


class Circuito:
    def __init__(self, graph, input_dim, output_dim, genome=None, seed=0, diario=None):
        self.g = genome or Genoma()
        self.g.validate()
        if not 1 <= input_dim <= 4096 or not 1 <= output_dim <= 4096:
            raise ValueError('Dimensão fora do limite')
        self.rng = np.random.default_rng(seed)
        self.diario = diario
        self.ids = list(graph['node_ids'])
        self.n = len(self.ids)
        if not 1 <= self.n <= 4096:
            raise ValueError('Até 4096 nós neste executor inicial')
        self.pre = np.asarray(graph['pre'], dtype=np.int64).copy()
        self.post = np.asarray(graph['post'], dtype=np.int64).copy()
        self.origin = graph['provenance']
        self.modules = np.asarray(graph.get('modules', [0] * self.n), dtype=np.int64)
        self.active_modules = np.ones(int(self.modules.max()) + 1)
        counts = np.asarray(graph['counts'], dtype=np.float64)
        if len(self.pre) != len(self.post) or counts.shape != self.pre.shape or np.any(counts <= 0):
            raise ValueError('Grafo inconsistente')
        if np.any(self.pre < 0) or np.any(self.pre >= self.n) or np.any(self.post < 0) or np.any(self.post >= self.n):
            raise ValueError('Índice fora do grafo')
        # Sinais imputados, não identificados como neurotransmissores medidos.
        self.sign = self.rng.choice([-1.0, 1.0], self.n, p=[.2, .8])
        self.w = np.log1p(counts) * self.sign[self.pre]
        self._bound(.8)
        self.trace = np.zeros(len(self.w))
        self.original = np.asarray(graph.get('edge_measured', [True] * len(self.w)), dtype=bool)
        self.x = np.zeros(self.n)
        self.previous = self.x.copy()
        self.input_dim, self.output_dim = input_dim, output_dim
        self.input = self.rng.normal(0, 1 / np.sqrt(input_dim), (self.n, input_dim))
        self.readout = np.zeros((output_dim, self.n))
        self.bias = np.zeros(output_dim)
        self.steps = self.learn_steps = self.grown_total = 0
        self.prediction = np.zeros(output_dim)
        self.pending = False
        self.cost = {'edge_ops': 0, 'dense_ops': 0, 'rule_edge_updates': 0, 'structural_proposals': 0}

    def emit(self, kind, **data):
        if self.diario:
            self.diario.event(kind, **data)

    def _bound(self, bound=None):
        totals = np.bincount(self.post, weights=np.abs(self.w), minlength=self.n)
        self.w /= np.maximum(1, totals[self.post] / (bound or self.g.recurrent_bound))

    def reset(self):
        self.x.fill(0)
        self.trace.fill(0)
        self.previous.fill(0)
        self.pending = False

    def step(self, observation, communicating=True):
        obs = np.asarray(observation, dtype=np.float64)
        if obs.shape != (self.input_dim,) or not np.isfinite(obs).all():
            raise ValueError('Observação inválida')
        self.last_observation = np.clip(obs, -1, 1)
        self.previous = self.x.copy()
        permitted = np.ones(len(self.w), dtype=bool) if communicating else self.modules[self.pre] == self.modules[self.post]
        drive = np.bincount(self.post, weights=self.w * self.previous[self.pre] * permitted, minlength=self.n)
        self.activation = np.tanh(drive + self.input @ self.last_observation)
        self.x = ((1 - self.g.leak) * self.previous + self.g.leak * self.activation) * self.active_modules[self.modules]
        self.prediction = self.readout @ self.x + self.bias
        self.trace = self.g.trace_decay * self.trace + (1 - self.g.trace_decay) * self.previous[self.pre] * (1 - self.activation[self.post]**2) * permitted
        self.steps += 1
        self.pending = True
        self.cost['edge_ops'] += len(self.w)
        self.cost['dense_ops'] += self.n * (self.input_dim + self.output_dim)
        return self.prediction.copy()

    def feedback(self, outcome, plastic=True, growth=True, learn_readout=True):
        if not self.pending:
            raise RuntimeError('Feedback exige previsão anterior; não pode ser aplicado duas vezes')
        y = np.asarray(outcome, dtype=np.float64)
        if y.shape != (self.output_dim,) or not np.isfinite(y).all():
            raise ValueError('Resultado inválido')
        error = self.prediction - y
        # Projeção de erro pelo leitor anterior; não usa parâmetros atualizados neste passo.
        delta = np.clip((self.readout.T @ error) * self.g.leak, -1, 1) * self.active_modules[self.modules]
        before = self.w.copy()
        if plastic:
            variables = {'eligibility': self.trace, 'error': delta[self.post], 'weight': self.w,
                         'decay': self.g.decay, 'pre': self.previous[self.pre], 'post': self.x[self.post]}
            update = np.clip(eval_rule(self.g.rule, variables), -1, 1)
            self.w -= self.g.eta * update
            self.w += self.g.hebb * self.previous[self.pre] * self.x[self.post]
            self.w = self.sign[self.pre] * np.abs(self.w)
            self._bound()
            self.input -= self.g.input_rate * np.outer(delta * (1 - self.activation**2), self.last_observation)
            np.clip(self.input, -2, 2, out=self.input)
            self.cost['rule_edge_updates'] += len(self.w)
        if learn_readout:
            scale = max(1., float(self.x @ self.x))
            self.readout -= self.g.readout_rate * np.outer(np.clip(error, -2, 2), self.x) / scale
            self.bias -= self.g.readout_rate * np.clip(error, -2, 2)
            np.clip(self.readout, -4, 4, out=self.readout)
            np.clip(self.bias, -2, 2, out=self.bias)
        self.learn_steps += 1
        changed = float(np.linalg.norm(self.w - before))
        self.pending = False
        self.emit('feedback', step=self.steps, observed_mse=float(np.mean(error**2)), recurrent_change_l2=changed,
                  plastic=plastic, readout_learning=learn_readout)
        if plastic and growth and self.learn_steps % self.g.growth_every == 0:
            self.restructure(delta)
        return float(np.mean(error**2))

    def restructure(self, delta):
        present = set(zip(self.pre.tolist(), self.post.tolist()))
        # Orçamento finito para propostas; seleção pelo erro e atividade observados.
        for post in np.argsort(-np.abs(delta))[:4]:
            for pre in np.argsort(-np.abs(self.previous))[:8]:
                self.cost['structural_proposals'] += 1
                if self.grown_total >= self.g.growth_limit:
                    break
                if pre == post or (int(pre), int(post)) in present:
                    continue
                self.pre = np.append(self.pre, pre)
                self.post = np.append(self.post, post)
                self.w = np.append(self.w, self.sign[pre] * .01)
                self.trace = np.append(self.trace, 0.)
                self.original = np.append(self.original, False)
                self.grown_total += 1
                present.add((int(pre), int(post)))
                self.emit('connection_created', step=self.steps, pre=self.ids[pre], post=self.ids[post], weight=float(self.w[-1]),
                          biological_edge=False)
                break
        remove = np.abs(self.w) < self.g.prune_threshold
        for index in np.flatnonzero(remove):
            self.emit('connection_pruned', step=self.steps, pre=self.ids[self.pre[index]], post=self.ids[self.post[index]],
                      biological_edge=bool(self.original[index]))
        for key in ('pre', 'post', 'w', 'trace', 'original'):
            setattr(self, key, getattr(self, key)[~remove])
        self._bound()

    def remove_module(self, index):
        self.active_modules[index] = 0
        self.x[self.modules == index] = 0

    def save(self, path):
        path = Path(path)
        if self.pending:
            raise RuntimeError('Salvar somente entre passos com feedback concluído')
        if path.exists() or path.with_suffix('.json').exists():
            raise FileExistsError(path)
        with path.open('xb') as f:
            np.savez_compressed(f, pre=self.pre, post=self.post, w=self.w, trace=self.trace, original=self.original,
                                x=self.x, previous=self.previous, sign=self.sign, input=self.input, readout=self.readout,
                                bias=self.bias, modules=self.modules, active_modules=self.active_modules,
                                rng=np.array(json.dumps(self.rng.bit_generator.state)))
        meta = {'sha256': digest(path), 'genome': asdict(self.g), 'node_ids': self.ids, 'provenance': self.origin,
                'input_dim': self.input_dim, 'output_dim': self.output_dim, 'steps': self.steps,
                'learn_steps': self.learn_steps, 'grown_total': self.grown_total, 'cost': self.cost,
                'pending': self.pending, 'model': 'sparse_rate_model_imputed_signs_local_error_plasticity',
                'biologically_calibrated': False}
        path.with_suffix('.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        return meta

    @classmethod
    def load(cls, path, diario=None):
        path = Path(path)
        meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if digest(path) != meta['sha256']:
            raise ValueError('Checkpoint alterado')
        obj = cls.__new__(cls)
        obj.g = Genoma(**meta['genome'])
        obj.g.validate()
        obj.ids, obj.n, obj.origin, obj.diario = meta['node_ids'], len(meta['node_ids']), meta['provenance'], diario
        obj.rng = np.random.default_rng()
        with np.load(path, allow_pickle=False) as data:
            for k in ('pre', 'post', 'w', 'trace', 'original', 'x', 'previous', 'sign', 'input', 'readout', 'bias', 'modules', 'active_modules'):
                setattr(obj, k, data[k].copy())
            obj.rng.bit_generator.state = json.loads(str(data['rng']))
        for k in ('input_dim', 'output_dim', 'steps', 'learn_steps', 'grown_total', 'cost', 'pending'):
            setattr(obj, k, meta[k])
        obj.prediction = np.zeros(obj.output_dim)
        return obj
