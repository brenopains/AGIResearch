"""ALE deve ser importado antes da NumPy.testing para cobrir ambas as consultas."""

import coleta
import avaliar_experiencia_cruzada_v2 as experiment
from leituras_por_programa import ConditionalReading

h, p, q, yl, yr, roots = experiment.training_data("breakout")
ConditionalReading().fit(h[:16], p[:16], q[:16], yl[:16], yr[:16], roots[:16])
print("ale_and_numpy_imports_passed_tiny_fit_no_models_saved", flush=True)
