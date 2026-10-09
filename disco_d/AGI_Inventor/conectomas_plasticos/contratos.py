"""Contratos versionados e eventos observados, nunca narrativas de pensamento."""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


@dataclass(frozen=True)
class Experiencia:
    observacao: list
    acao: list
    resultado: list | None
    tempo: int
    proveniencia: str


@dataclass(frozen=True)
class Representacao:
    executor_hash: str
    tipos: dict
    estado: dict
    evidencias: list


@dataclass(frozen=True)
class Feedback:
    resultado_observado: list
    contraexemplo: dict | None
    alcance: str


class Diario:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open('x', encoding='utf-8')
        self.seq = 0

    def event(self, kind, **fields):
        row = {'schema': 1, 'seq': self.seq, 'time_unix': time.time(), 'kind': kind, **fields}
        self.f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        self.f.flush()
        self.seq += 1

    def close(self):
        self.f.close()
