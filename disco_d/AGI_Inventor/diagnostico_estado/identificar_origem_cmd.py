"""Rastrear a origem da consulta do sistema, sem mudar retorno da chamada."""

import subprocess
import traceback


original = subprocess.check_output


def traced(*args, **kwargs):
    print("system_query", args, flush=True)
    traceback.print_stack(limit=12)
    return original(*args, **kwargs)


subprocess.check_output = traced
try:
    import avaliar_experiencia_cruzada_v2 as experiment
    from leituras_por_programa import ConditionalReading

    h, p, q, yl, yr, roots = experiment.training_data("breakout")
    ConditionalReading().fit(h[:16], p[:16], q[:16], yl[:16], yr[:16], roots[:16])
finally:
    subprocess.check_output = original
