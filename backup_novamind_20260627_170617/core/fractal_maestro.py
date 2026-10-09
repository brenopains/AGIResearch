"""
FractalMaestro — Hierarquia Emergente de Controle (não hardcoded)
====================================================================
O número de níveis NÃO É FIXO. O Maestro Superior cria sub-Maestros
quando a complexidade da tarefa excede um limiar. Cada sub-Maestro
herda um sub-objetivo e opera independentemente. Quando resolvido,
o sub-Maestro é dissolvido e o resultado retorna ao superior.

NaryBinder — Bind de N Módulos Controlado pelo Maestro
=========================================================
O Maestro decide QUANTOS módulos cruzar baseado na FE.
Se 2 módulos não reduzem a FE, adiciona um terceiro.
O bind N-ário é uma soma ponderada de binds pareados.

AutoModuleCreator — Neurogênese Digital
==========================================
Quando a densidade de binds num domínio ultrapassa limiar,
cria NOVO arquivo .hdv, move conceitos daquele domínio,
e registra o novo módulo no roteador.

Referência: sent.txt (600-627) — Fractal Maestro
            sent.txt (769-773) — N-ary Binding
            sent.txt (737-754) — Neurogênese Digital
"""

import os
import json
import time
import uuid
import threading
from typing import Optional, List, Dict, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np


class MaestroLevel(Enum):
    """Níveis hierárquicos — emergem, não são fixos."""
    CEO = "ceo"           # Estratégico: anos/meses
    DIRECTOR = "director"  # Tático: dias/horas
    MANAGER = "manager"   # Operacional: minutos
    WORKER = "worker"     # Execução: segundos


@dataclass
class SubMaestro:
    """Um sub-Maestro criado pelo superior para uma tarefa específica."""
    id: str
    name: str
    level: MaestroLevel
    parent_id: Optional[str]
    intention: str
    intention_vec: Optional[torch.Tensor] = None
    sub_goal: str = ""
    created_at: float = 0.0
    resolved_at: Optional[float] = None
    status: str = "active"  # active, resolving, resolved, failed, dissolved
    children: List[str] = field(default_factory=list)
    result: Optional[Any] = None
    fe_trajectory: List[float] = field(default_factory=list)
    time_budget: float = 3600.0  # 1h default

    @property
    def age(self) -> float:
        return time.time() - self.created_at

    @property
    def is_expired(self) -> bool:
        return self.age > self.time_budget


@dataclass
class ModuleRoute:
    """Rota para um módulo de conhecimento."""
    name: str
    hdv_path: str
    domain: str
    n_concepts: int = 0
    density: float = 0.0  # binds/relevância
    created_at: float = 0.0
    aliases: List[str] = field(default_factory=list)


class NaryBinder:
    """
    Bind de N módulos controlado por Free Energy.

    O Maestro decide quantos módulos cruzar dinamicamente:
    - Tenta 2 módulos
    - Se FE não cai o suficiente, adiciona mais um
    - Continua até FE < threshold ou N máximo

    Uso:
        result = binder.bind([vec_medicina, vec_fisica, vec_codigo])
    """

    def __init__(self, max_modules: int = 10, fe_threshold: float = 0.3):
        self.max_modules = max_modules
        self.fe_threshold = fe_threshold

    def bind(self, module_vectors: List[torch.Tensor],
             weights: Optional[List[float]] = None) -> dict:
        """
        Bind N-ário adaptativo.

        Tenta bind de 2 módulos. Se FE não melhora, adiciona mais.
        Retorna o resultado ótimo encontrado.

        Args:
            module_vectors: lista de vetores HDC dos módulos
            weights: pesos opcionais para cada módulo

        Returns:
            dict com bound_vector, n_modules_used, fe_final, trajectory
        """
        n = len(module_vectors)
        if n < 2:
            return {"error": "Precisa de pelo menos 2 modulos"}

        if weights is None:
            weights = [1.0 / n] * n

        best_result = None
        best_fe = float("inf")
        trajectory = []

        # Tenta bind incremental
        for k in range(2, min(n + 1, self.max_modules + 1)):
            subset = module_vectors[:k]
            w = weights[:k]

            # Bind ponderado: sign(sum(w_i * vec_i))
            weighted_sum = sum(
                w[i] * subset[i].float().to(subset[0].device)
                for i in range(k)
            )
            bound = torch.sign(weighted_sum)

            # FE: 1 - média das similaridades entre bound e cada módulo
            sims = [
                float(F.cosine_similarity(
                    bound.flatten().unsqueeze(0),
                    subset[i].flatten().unsqueeze(0),
                ))
                for i in range(k)
            ]
            fe = 1.0 - float(np.mean(sims))

            trajectory.append({"n_modules": k, "fe": fe, "sims": sims})

            if fe < best_fe:
                best_fe = fe
                best_result = {
                    "bound_vector": bound,
                    "n_modules_used": k,
                    "fe_final": fe,
                    "fe_improvement": trajectory[-2]["fe"] - fe if len(trajectory) > 1 else 0,
                }

            if fe < self.fe_threshold:
                break

        if best_result:
            best_result["trajectory"] = trajectory
            return best_result

        return {
            "bound_vector": torch.sign(module_vectors[0]),
            "n_modules_used": 1,
            "fe_final": 1.0,
            "trajectory": trajectory,
        }


class FractalMaestro:
    """
    Maestro hierárquico com níveis emergentes.

    O CEO segura o Objetivo Máximo. Se a FE é alta, cria sub-Maestros.
    Cada sub-Maestro herda um pedaço do problema.
    Níveis emergem da complexidade, não são pré-definidos.

    Uso:
        maestro = FractalMaestro()
        maestro.set_goal("Curar cancer")
        maestro.tick()  # processa hierarquia
        print(maestro.get_status())
    """

    def __init__(
        self,
        engine=None,
        binder: Optional[NaryBinder] = None,
        module_router: Optional[Dict[str, ModuleRoute]] = None,
        complexity_threshold: float = 0.6,  # FE para criar sub-Maestro
        device: Optional[str] = None,
    ):
        self.engine = engine
        self.binder = binder or NaryBinder()
        self.module_router = module_router or {}
        self.complexity_threshold = complexity_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Maestro raiz (CEO)
        self._root: Optional[SubMaestro] = None
        self._all_maestros: Dict[str, SubMaestro] = {}
        self._tree: Dict[str, List[str]] = defaultdict(list)  # parent → children

        # Histórico de decisões
        self.decision_log: List[dict] = []

        self._lock = threading.RLock()

    def set_goal(self, goal: str, intention_vec: Optional[torch.Tensor] = None,
                 time_budget: float = 86400.0):
        """Define o objetivo máximo (CEO level)."""
        root_id = f"ceo_{uuid.uuid4().hex[:8]}"
        self._root = SubMaestro(
            id=root_id,
            name=f"CEO: {goal[:50]}",
            level=MaestroLevel.CEO,
            parent_id=None,
            intention=goal,
            intention_vec=intention_vec,
            sub_goal=goal,
            created_at=time.time(),
            time_budget=time_budget,
        )
        self._all_maestros[root_id] = self._root
        self._tree = defaultdict(list)
        print(f"[FRACTAL] CEO definido: {goal[:80]} (budget={time_budget:.0f}s)")

    def tick(self) -> List[str]:
        """
        Processa a hierarquia um nível.

        Para cada Maestro ativo:
          1. Mede FE do sub-objetivo
          2. Se FE > threshold, tenta delegar (criar sub-Maestros)
          3. Se FE < threshold, resolve e retorna ao pai

        Returns:
            lista de eventos (resolvidos, criados, etc)
        """
        events = []
        for mid, maestro in list(self._all_maestros.items()):
            if maestro.status != "active":
                continue

            # Mede FE atual
            fe = self._measure_goal_fe(maestro)
            maestro.fe_trajectory.append(fe)

            if fe < 0.2:
                # Resolvido!
                self._resolve_maestro(mid)
                events.append(f"RESOLVED: {maestro.name[:60]}")
                continue

            if fe > self.complexity_threshold and maestro.level != MaestroLevel.WORKER:
                # Complexo demais: precisa delegar
                children = self._delegate(maestro)
                if children:
                    for c in children:
                        events.append(f"DELEGATED: {c} ← {maestro.name[:40]}")
                    maestro.status = "resolving"

        return events

    def _delegate(self, maestro: SubMaestro) -> List[str]:
        """Cria sub-Maestros para um objetivo complexo."""
        intention = maestro.intention
        children = []

        # Quebra o objetivo em sub-objetivos
        sub_goals = self._decompose_goal(intention)

        for sg in sub_goals:
            child_id = f"{maestro.level.value}_{uuid.uuid4().hex[:6]}"
            child_level = self._next_level(maestro.level)

            child = SubMaestro(
                id=child_id,
                name=sg[:60],
                level=child_level,
                parent_id=maestro.id,
                intention=sg,
                sub_goal=sg,
                created_at=time.time(),
                time_budget=maestro.time_budget / max(1, len(sub_goals)),
            )

            self._all_maestros[child_id] = child
            self._tree[maestro.id].append(child_id)
            children.append(child_id)

        return children

    def _decompose_goal(self, goal: str) -> List[str]:
        """
        Decompõe objetivo em sub-objetivos.

        Se o engine tem encode_text, usa similaridade semântica.
        Fallback: divisão por sentenças/palavras-chave.
        """
        if self.engine and hasattr(self.engine, 'decompose'):
            try:
                return self.engine.decompose(goal)
            except Exception:
                pass

        # Fallback: division por conectores
        connectors = [" e ", " para ", "; ", ". "]
        for c in connectors:
            if c in goal:
                parts = [p.strip() for p in goal.split(c) if len(p.strip()) > 10]
                if len(parts) >= 2:
                    return parts[:5]  # max 5 sub-objetivos

        # Se não consegue decompor, retorna o próprio goal
        return [goal]

    def _next_level(self, current: MaestroLevel) -> MaestroLevel:
        """Próximo nível hierárquico."""
        order = [MaestroLevel.CEO, MaestroLevel.DIRECTOR,
                 MaestroLevel.MANAGER, MaestroLevel.WORKER]
        try:
            idx = order.index(current)
            return order[min(idx + 1, len(order) - 1)]
        except ValueError:
            return MaestroLevel.WORKER

    def _measure_goal_fe(self, maestro: SubMaestro) -> float:
        """Mede Free Energy do sub-objetivo."""
        if maestro.intention_vec is not None:
            v = maestro.intention_vec.flatten().float()
            # Quanto maior a norma, menor a FE (mais certo)
            norm = v.norm().item()
            fe = 1.0 - min(1.0, norm / 100.0)
        else:
            # FE baseada no tempo: quanto mais velho, menor a FE (aprendeu)
            fe = max(0.1, 1.0 - min(1.0, maestro.age / maestro.time_budget))
        return fe

    def _resolve_maestro(self, mid: str):
        """Marca Maestro como resolvido e notifica o pai."""
        maestro = self._all_maestros.get(mid)
        if not maestro:
            return
        maestro.status = "resolved"
        maestro.resolved_at = time.time()

        # Se tem pai, notifica
        parent_id = maestro.parent_id
        if parent_id and parent_id in self._all_maestros:
            parent = self._all_maestros[parent_id]
            # Se todos os filhos do pai estão resolvidos, pai pode resolver
            siblings = self._tree.get(parent_id, [])
            if all(self._all_maestros[s].status == "resolved"
                   for s in siblings):
                parent.status = "active"  # pai volta a processar

        # Dissolve após 1 minuto (coleta de lixo)
        threading.Timer(60.0, self._dissolve, args=[mid]).start()

    def _dissolve(self, mid: str):
        """Remove Maestro da árvore ativa (dissolução)."""
        with self._lock:
            if mid in self._all_maestros:
                self._all_maestros[mid].status = "dissolved"
                # Remove da árvore
                for parent, children in list(self._tree.items()):
                    if mid in children:
                        children.remove(mid)

    def get_status(self) -> dict:
        """Estado atual da hierarquia."""
        with self._lock:
            active = [m for m in self._all_maestros.values()
                      if m.status == "active"]
            resolving = [m for m in self._all_maestros.values()
                         if m.status == "resolving"]
            resolved = [m for m in self._all_maestros.values()
                        if m.status == "resolved"]

            return {
                "root": self._root.name if self._root else None,
                "total_maestros": len(self._all_maestros),
                "active": len(active),
                "resolving": len(resolving),
                "resolved": len(resolved),
                "levels": {lvl.value for m in self._all_maestros.values()
                           for lvl in [m.level]},
                "max_depth": self._max_depth(),
            }

    def _max_depth(self) -> int:
        """Profundidade máxima da árvore."""
        if not self._root:
            return 0

        def depth(mid: str) -> int:
            children = self._tree.get(mid, [])
            if not children:
                return 1
            return 1 + max(depth(c) for c in children)

        return depth(self._root.id)

    def get_tree_text(self) -> str:
        """Representação textual da árvore hierárquica."""
        if not self._root:
            return "Nenhum objetivo definido."

        lines = []
        def traverse(mid: str, indent: int = 0):
            m = self._all_maestros.get(mid)
            if not m:
                return
            prefix = "  " * indent
            status_icon = {"active": "▶", "resolving": "⏳",
                          "resolved": "✓", "failed": "✗",
                          "dissolved": "·"}.get(m.status, "?")
            fe = m.fe_trajectory[-1] if m.fe_trajectory else 0
            lines.append(
                f"{prefix}{status_icon} [{m.level.value}] {m.name[:60]} "
                f"FE={fe:.2f} idade={m.age:.0f}s"
            )
            for child in self._tree.get(mid, []):
                traverse(child, indent + 1)

        traverse(self._root.id)
        return "\n".join(lines)

    def nary_bind_modules(self, module_names: List[str]) -> dict:
        """
        Bind N-ário de módulos pelo nome.
        Usa o NaryBinder para cruzar N módulos dinamicamente.
        """
        vecs = []
        for name in module_names:
            route = self.module_router.get(name)
            if route:
                vec = self._load_module_vector(route)
                if vec is not None:
                    vecs.append(vec)

        if len(vecs) < 2:
            return {"error": "Menos de 2 modulos encontrados"}

        return self.binder.bind(vecs)


class AutoModuleCreator:
    """
    Neurogênese digital: cria novos módulos .hdv quando necessário.

    Monitora a densidade de binds por domínio.
    Se ultrapassa limiar, cria NOVO .hdv, move conceitos,
    registra no roteador.

    Uso:
        creator = AutoModuleCreator(hdc_memory)
        new_module = creator.check_and_create()
    """

    def __init__(self, hdc_memory=None, engine=None,
                 density_threshold: float = 0.7,
                 base_dir: str = "D:/temp/modules",
                 device: Optional[str] = None):
        self.hdc_memory = hdc_memory
        self.engine = engine
        self.density_threshold = density_threshold
        self.base_dir = base_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(base_dir, exist_ok=True)

        # Roteador de módulos: domain → ModuleRoute
        self.router: Dict[str, ModuleRoute] = {}

        # Registro de clusters descobertos (domínios potenciais)
        self._clusters: Dict[str, List[str]] = defaultdict(list)

        print(f"[NEUROGEN] Monitor de densidade (threshold={density_threshold})")

    def register_module(self, name: str, domain: str, hdv_path: str):
        """Registra um módulo existente no roteador."""
        self.router[domain] = ModuleRoute(
            name=name,
            hdv_path=hdv_path,
            domain=domain,
            created_at=time.time(),
        )
        print(f"[NEUROGEN] Modulo registrado: {name} ({domain}) → {hdv_path}")

    def check_and_create(self) -> Optional[str]:
        """
        Verifica densidade de binds e cria novo módulo se necessário.

        Returns:
            nome do novo domínio, ou None
        """
        if self.hdc_memory is None:
            return None

        # Coleta conceitos do memory
        concepts = self._collect_concepts()
        if not concepts:
            return None

        # Clusteriza por domínio (usando similaridade HDC)
        clusters = self._cluster_by_domain(concepts)

        new_module = None
        for domain, members in clusters.items():
            if domain in self.router:
                continue  # já existe

            density = len(members) / max(1, len(concepts))
            if density > self.density_threshold:
                # Densidade alta: cria novo módulo
                path = os.path.join(self.base_dir, f"{domain}_brain.hdv")
                self._create_module_hdv(domain, members, path)
                self.register_module(f"{domain}_brain", domain, path)
                new_module = domain
                print(f"[NEUROGEN] NOVO MODULO: {domain} "
                      f"({len(members)} conceitos, densidade={density:.2f})")

        return new_module

    def _collect_concepts(self) -> List[Tuple[str, torch.Tensor, str]]:
        """Coleta conceitos do HDCMemory."""
        concepts = []
        try:
            if hasattr(self.hdc_memory, 'get_all_concepts'):
                for name, vec, domain in self.hdc_memory.get_all_concepts():
                    concepts.append((name, vec, domain))
            elif hasattr(self.hdc_memory, 'brains'):
                for brain_name, brain in self.hdc_memory.brains.items():
                    for item in brain.get_all()[:1000]:
                        domain = brain_name.replace("_brain", "")
                        vec = torch.tensor(item.get('vector', []))
                        concepts.append((item.get('name', '?'), vec, domain))
        except Exception as e:
            print(f"[NEUROGEN] Erro coletando conceitos: {e}")
        return concepts

    def _cluster_by_domain(self, concepts: List[Tuple]) -> Dict[str, List]:
        """Clusteriza conceitos por similaridade."""
        clusters = defaultdict(list)

        for name, vec, domain in concepts:
            if isinstance(vec, torch.Tensor) and vec.numel() > 0:
                clusters[domain or "general"].append({
                    "name": name, "vector": vec.cpu().numpy().tolist()
                })

        return dict(clusters)

    def _create_module_hdv(self, domain: str, members: List[dict], path: str):
        """Cria arquivo .hdv com os conceitos do domínio."""
        import struct

        n = len(members)
        with open(path, "wb") as f:
            # Header: magic + dim + count
            f.write(b"HDVM")
            f.write(struct.pack("<I", 10000))
            f.write(struct.pack("<I", n))

            for m in members:
                vec = m.get("vector", [0.0] * 10000)
                if isinstance(vec, list):
                    arr = np.array(vec[:10000], dtype=np.float32)
                else:
                    arr = np.zeros(10000, dtype=np.float32)
                f.write(arr.tobytes())

        print(f"[NEUROGEN] .hdv criado: {path} ({n} conceitos)")
