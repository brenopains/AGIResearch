"""HDC real para recuperação; subprocesso para busca; feedback observado para circuito."""
import runtime
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from motor_worker import verify

HERE = Path(__file__).resolve().parent


class Ponte:
    def __init__(self, diario=None):
        self.snapshot = verify()
        from motor_snapshot.hdc_indice import Espaco, Loja
        self.space = Espaco(dim=512)
        self.store = Loja(self.space)
        self.diario = diario

    def guardar(self, name, observations, payload):
        from motor_snapshot.hdc_indice import de_fluxo
        # Índice limitado a uma janela; executor completo fica na carga fora do vetor.
        vector = de_fluxo(self.space, np.asarray(observations)[-16:])
        self.store.guarda(name, vector, payload)
        if self.diario:
            self.diario.event('memory_stored', name=name, items=len(self.store.itens))

    def recuperar(self, observations):
        from motor_snapshot.hdc_indice import de_fluxo
        query = de_fluxo(self.space, np.asarray(observations)[-16:])
        ranked = self.store.ordena(query)
        if self.diario:
            self.diario.event('memory_retrieved', items=len(ranked))
        return [{'similarity': s, 'name': name, 'payload': item['carga']} for s, name, item in ranked]

    def propor(self, observations, outcomes, timeout=15):
        request = {'observations': np.asarray(observations, dtype=float).tolist(), 'outcomes': list(map(int, outcomes))}
        result = subprocess.run([sys.executable, '-B', str(HERE / 'motor_worker.py')], input=json.dumps(request),
                                text=True, capture_output=True, timeout=timeout, cwd=HERE)
        if result.returncode:
            raise RuntimeError(result.stderr[-2000:])
        proposal = json.loads(result.stdout)
        if self.diario:
            self.diario.event('motor_proposal', **proposal)
        return proposal

    def executar(self, proposal, observations):
        from motor_snapshot.substrato import No
        def decode(body):
            if body['op'] not in ('LER', 'MEDIA', 'NUM', 'SINAL', 'MENOS', 'MAIOR', 'SE'):
                raise ValueError('Operação externa não permitida')
            return No(body['op'], tuple(decode(x) if isinstance(x, dict) else x for x in body['args']))
        program = decode(proposal['program'])
        x = np.asarray(observations).tolist()
        return [program.aval(x, t) for t in range(len(x))]

    def realimentar(self, circuit, feedback, **options):
        mse = circuit.feedback(feedback.resultado_observado, **options)
        if self.diario:
            self.diario.event('bridge_feedback', scope=feedback.alcance, observed_mse=mse,
                              counterexample=feedback.contraexemplo)
        return mse
