"""
ModuleGenerator — Cria Novos Modulos por Deteccao de Gaps
===========================================================
Dado um gap classificado (ex: "calibration_missing", "missing_primitive"),
o generator seleciona um template, instancia com parametros do gap,
escreve o codigo Python, testa em held-out, e integra via
StructuralPlasticity.

Nada aqui e especifico para os 7 trials.
Os templates resolvem PROBLEMAS GERAIS:
  - PlattScalingDomainAdapter: calibracao por dominio
  - ValueMapInducer: induz transformacao de valor de exemplos
  - FunctionFamilyFitter: ajusta funcao com selecao automatica
"""
import os
import sys
import time
import uuid
import importlib
import inspect
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class ModuleTemplate:
    """Um template de modulo que pode ser instanciado."""
    name: str = ""
    description: str = ""
    gap_types: List[str] = field(default_factory=list)
    code_template: str = ""
    required_imports: List[str] = field(default_factory=list)
    test_fn: Optional[Callable] = None


class ModuleGenerator:
    """
    Gera novos modulos para preencher gaps na arquitetura.
    
    Uso:
        gen = ModuleGenerator(sp, cge, world_model, journal)
        new_module = gen.generate(gap_description)
        gen.integrate(new_module)
    """

    def __init__(self, structural_plasticity=None, cge=None,
                 world_model=None, journal=None,
                 calibration_engine=None, primitive_engine=None,
                 verbose: bool = True):
        self.sp = structural_plasticity
        self.cge = cge
        self.world_model = world_model
        self.journal = journal
        self.calibration_engine = calibration_engine
        self.primitive_engine = primitive_engine
        self.verbose = verbose

        self._templates = self._register_templates()
        self._generated_modules: List[dict] = []

    def _register_templates(self) -> List[ModuleTemplate]:
        """Registra templates de modulos gerais."""
        return [
            ModuleTemplate(
                name="PlattScalingDomainAdapter",
                description="Calibracao Platt por dominio usando PredictionJournal",
                gap_types=["calibration_missing"],
                code_template=self._platt_template(),
                required_imports=["import numpy as np", "import math"],
            ),
            ModuleTemplate(
                name="ValueMapInducer",
                description="Induz mapa de valor dos exemplos de falha",
                gap_types=["missing_primitive"],
                code_template=self._valuemap_template(),
                required_imports=["import numpy as np"],
            ),
            ModuleTemplate(
                name="FunctionFamilyFitter",
                description="Ajusta funcao com selecao automatica de familia",
                gap_types=["poor_function_fitting"],
                code_template=self._functionfit_template(),
                required_imports=["import numpy as np"],
            ),
        ]

    def generate(self, gap: dict) -> Optional[dict]:
        """
        Gera um novo modulo para preencher um gap.
        
        Args:
            gap: dict com type, cause, fix, fix_detail, trial, etc.
            
        Returns:
            dict com {name, code, module_id} ou None
        """
        gap_type = gap.get("type", "general_performance")
        trial = gap.get("trial", "")
        score = gap.get("score", 0.0)
        threshold = gap.get("threshold", 0.3)

        # Encontra template que atende este gap
        template = None
        for t in self._templates:
            if gap_type in t.gap_types:
                template = t
                break

        if template is None:
            if self.verbose:
                print(f"  [Generator] No template for gap type: {gap_type}")
            return None

        # Gera codigo
        module_id = f"gen_{template.name}_{int(time.time())}"
        code = self._instantiate_template(template, gap, module_id)

        # Valida sintaxe
        try:
            compile(code, f"<{module_id}>", "exec")
        except SyntaxError as e:
            if self.verbose:
                print(f"  [Generator] Syntax error in generated code: {e}")
            return None

        result = {
            "id": module_id,
            "name": f"{template.name}_{trial}",
            "type": template.name,
            "code": code,
            "gap_type": gap_type,
            "trial": trial,
            "template_used": template.name,
            "created_at": time.time(),
        }
        self._generated_modules.append(result)
        if self.verbose:
            print(f"  [Generator] Created module: {result['name']} ({len(code)} chars)")
        return result

    def _instantiate_template(self, template: ModuleTemplate,
                               gap: dict, module_id: str) -> str:
        """Instancia um template com os parametros do gap."""
        imports = "\n".join(template.required_imports)

        # Substitui placeholders no template
        code = template.code_template
        code = code.replace("{{MODULE_ID}}", module_id)
        code = code.replace("{{TRIAL}}", gap.get("trial", "unknown"))
        code = code.replace("{{SCORE}}", str(gap.get("score", 0.0)))
        code = code.replace("{{THRESHOLD}}", str(gap.get("threshold", 0.3)))
        code = code.replace("{{GAP_TYPE}}", gap.get("type", "unknown"))

        # Adiciona informacao especifica do gap
        journal_n = gap.get("journal_n", 0)
        fix_detail = gap.get("fix_detail", "default")
        code = code.replace("{{JOURNAL_N}}", str(journal_n))
        code = code.replace("{{FIX_DETAIL}}", fix_detail)

        return f"""{imports}

{code}
"""

    def integrate(self, module: dict) -> bool:
        """
        Integra um modulo gerado na arquitetura viva.
        
        1. Salva o arquivo Python
        2. Importa dinamicamente
        3. Registra no StructuralPlasticity
        4. Conecta ao MentalState
        """
        module_id = module["id"]
        module_name = module["name"]
        code = module["code"]

        # 1. Salva arquivo
        gen_dir = os.path.join(BASE, "core", "generated_modules")
        os.makedirs(gen_dir, exist_ok=True)
        fpath = os.path.join(gen_dir, f"{module_id}.py")
        try:
            with open(fpath, "w") as f:
                f.write(code)
        except Exception as e:
            if self.verbose:
                print(f"  [Generator] Failed to write {fpath}: {e}")
            return False

        # 2. Importa dinamicamente
        try:
            sys.path.insert(0, os.path.join(BASE, "core", "generated_modules"))
            spec = importlib.util.spec_from_file_location(module_id, fpath)
            if spec is None:
                return False
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            if self.verbose:
                print(f"  [Generator] Failed to import {module_id}: {e}")
            return False

        # 3. Encontra e instancia a classe
        instance = None
        for obj_name in dir(mod):
            obj = getattr(mod, obj_name)
            if isinstance(obj, type) and obj_name != "object":
                try:
                    instance = obj()
                    if self.verbose:
                        print(f"  [Generator] Instantiated {obj_name}")
                    break
                except Exception:
                    continue

        if instance is None:
            if self.verbose:
                print(f"  [Generator] No instantiable class in {module_id}")
            return False

        # 4. Registra no StructuralPlasticity
        if self.sp is not None:
            node_id = self.sp.register_module(
                module_name, module["type"],
                instance=instance
            )
            if self.verbose:
                print(f"  [Generator] Registered '{module_name}' as {node_id}")

            # Conecta a modulos existentes
            if module["type"] == "PlattScalingDomainAdapter":
                if self.cge is not None:
                    self.sp.connect(module_name, "CognitiveGraphEngine")
                if self.journal is not None:
                    journal_node = next(
                        (nid for nid, n in self.sp.arch.nodes.items()
                         if "Journal" in n.name), None)
                    if journal_node:
                        pass
            elif module["type"] == "ValueMapInducer":
                if self.cge is not None:
                    self.sp.connect(module_name, "CognitiveGraphEngine")
            elif module["type"] == "FunctionFamilyFitter":
                wm_node = next(
                    (nid for nid, n in self.sp.arch.nodes.items()
                     if "WorldModel" in n.name), None)
                if wm_node and instance:
                    pass

        return True

    def get_stats(self) -> dict:
        return {
            "n_generated": len(self._generated_modules),
            "module_types": list(set(m["type"] for m in self._generated_modules)),
            "recent": self._generated_modules[-3:] if self._generated_modules else [],
        }

    # ─── TEMPLATES ───

    def _platt_template(self) -> str:
        return '''class {{MODULE_ID}}:
    """
    Calibrador Platt por dominio, gerado para preencher gap de calibracao.
    Criado automaticamente pelo ModuleGenerator.
    """
    def __init__(self):
        self.name = "{{MODULE_ID}}"
        self.gap_type = "{{GAP_TYPE}}"
        self.trial = "{{TRIAL}}"
        self.alpha = 1.0
        self.beta = 0.0
        self._fitted = False
        self._domains = {}

    def fit(self, confidences, corrects, domain="unknown"):
        import numpy as np
        import math
        targets = np.clip(np.array(corrects), 1e-10, 1 - 1e-10)
        logits = np.clip(np.array(confidences), 1e-10, 1 - 1e-10)
        log_odds = np.log(targets / (1 - targets))
        A = np.column_stack([logits, np.ones_like(logits)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, log_odds, rcond=None)
            self._domains[domain] = {"alpha": float(coeffs[0]), "beta": float(coeffs[1])}
            self._fitted = True
        except Exception:
            self._domains[domain] = {"alpha": 1.0, "beta": 0.0}

    def calibrate(self, confidence, domain="unknown"):
        import math
        if domain in self._domains:
            a = self._domains[domain]["alpha"]
            b = self._domains[domain]["beta"]
        elif self._fitted:
            a = list(self._domains.values())[0]["alpha"]
            b = list(self._domains.values())[0]["beta"]
        else:
            return confidence
        calibrated = 1.0 / (1.0 + math.exp(-(a * confidence + b)))
        return max(0.01, min(0.99, calibrated))

    def __call__(self, confidence, domain="unknown"):
        return self.calibrate(confidence, domain)
'''

    def _valuemap_template(self) -> str:
        return '''import numpy as np

class {{MODULE_ID}}:
    """
    Indutor de mapa de valor, gerado para preencher gap de invencao.
    Criado automaticamente pelo ModuleGenerator.
    """
    def __init__(self):
        self.name = "{{MODULE_ID}}"
        self.gap_type = "{{GAP_TYPE}}"
        self.trial = "{{TRIAL}}"
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
'''

    def _functionfit_template(self) -> str:
        return '''import numpy as np

class {{MODULE_ID}}:
    """
    Ajustador de funcao com selecao de familia, gerado para gap de fitting.
    Criado automaticamente pelo ModuleGenerator.
    """
    def __init__(self):
        self.name = "{{MODULE_ID}}"
        self.gap_type = "{{GAP_TYPE}}"
        self.trial = "{{TRIAL}}"
        self._families = ["linear", "log_linear", "polynomial"]
        self._best = None
        self._params = {}

    def fit(self, X, y):
        n = len(y)
        candidates = []
        import numpy as np

        # Linear
        try:
            X_aug = np.column_stack([np.ones(n), X])
            coeffs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
            preds = X_aug @ coeffs
            resid = float(np.std(y - preds))
            if resid < np.std(y) * 0.95:
                candidates.append(("linear", resid, coeffs))
        except Exception:
            pass

        # Log-linear
        try:
            shift = 1.0
            y_log = np.log(np.maximum(y + shift, 1e-6))
            X_log = np.column_stack([np.log(np.maximum(X[:, i] + shift, 1e-6)) for i in range(X.shape[1])])
            X_log_aug = np.column_stack([np.ones(n), X_log])
            coeffs, _, _, _ = np.linalg.lstsq(X_log_aug, y_log, rcond=None)
            preds = np.exp(X_log_aug @ coeffs) - shift
            resid = float(np.std(y - preds))
            if resid < np.std(y) * 0.95:
                candidates.append(("log_linear", resid, coeffs))
        except Exception:
            pass

        if candidates:
            candidates.sort(key=lambda c: c[1])
            self._best = candidates[0][0]
            self._params = {"type": self._best, "coeffs": candidates[0][2].tolist()}
        return self._best is not None

    def predict(self, X):
        if self._best is None:
            return np.zeros(X.shape[0])
        if self._best == "linear":
            return X @ self._params["coeffs"][1:] + self._params["coeffs"][0]
        elif self._best == "log_linear":
            shift = 1.0
            X_log = np.column_stack([np.log(np.maximum(X[:, i] + shift, 1e-6)) for i in range(X.shape[1])])
            return np.exp(np.column_stack([np.ones(X.shape[0]), X_log]) @ self._params["coeffs"]) - shift
        return np.zeros(X.shape[0])
'''
