"""Log estruturado por passo: o que foi percebido, por que a acao foi
escolhida. E o ativo de observabilidade do projeto — usado como
DIAGNOSTICO (ver inspect_trace.py), nunca como evidencia de capacidade.
"""
from __future__ import annotations

import json
from pathlib import Path


class TraceLogger:
    def __init__(self):
        self.rows: list = []

    def record(self, **kwargs) -> None:
        self.rows.append(kwargs)

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in self.rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
