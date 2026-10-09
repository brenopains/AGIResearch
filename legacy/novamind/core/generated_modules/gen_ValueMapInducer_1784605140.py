import numpy as np

import numpy as np

class gen_ValueMapInducer_1784605140:
    """
    Indutor de mapa de valor, gerado para preencher gap de invencao.
    Criado automaticamente pelo ModuleGenerator.
    """
    def __init__(self):
        self.name = "gen_ValueMapInducer_1784605140"
        self.gap_type = "missing_primitive"
        self.trial = "T7"
        self._fn = lambda c: c
        self._params = {}
        self._trained = False

    def train(self, examples):
        from core.primitive_invention.value_maps import induce_value_map
        result = induce_value_map(examples)
        if result.perfect() and result.fn is not None:
            self._fn = result.fn
            self._params = result.params or {}
            self._trained = True
        return self._trained

    def apply(self, grid):
        fn_vec = np.vectorize(self._fn)
        return fn_vec(grid)

    def __call__(self, grid):
        return self.apply(grid)

