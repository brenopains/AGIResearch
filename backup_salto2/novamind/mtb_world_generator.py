"""
MTB-1 World Generator — Multiworld Transfer Benchmark

Gera mundos procedurais com:
  - Fisica alterada (fluxo de energia, gravidade, friccao)
  - Materiais desconhecidos (VORIN, MEXAL, etc) com propriedades ocultas
  - Tarefa de transporte de energia A -> B

Cada mundo e uma grade 2D onde a AGI precisa descobrir regras
e construir um caminho de energia.
"""
import random
import hashlib
from typing import Dict, List, Tuple, Optional

# Pool de nomes de materiais (sem semântica prévia)
MATERIAL_NAMES = [
    "VORIN", "MEXAL", "TRUON", "KELP", "ZIND", "BREN", "KORV",
    "NEXIUM", "FEND", "LURK", "THAL", "MORV", "SHEEN", "PLUV",
    "DROX", "QUIL", "BROG", "ZURG", "FLAK", "TREM",
]

# Propriedades base que os materiais podem ter
PROPERTIES_POOL = [
    ("conductive", 0.6, 1.0),   # conduz energia (nome, min, max)
    ("blocking", 0.7, 1.0),     # bloqueia energia
    ("amplifying", 1.5, 3.0),   # amplifica energia
    ("absorbing", 0.5, 1.0),    # absorve energia (some com ela)
    ("unstable", 0.3, 0.7),     # instavel: pode explodir
    ("directional", 0.0, 0.0),  # so conduz em uma direcao
    ("conditional", 0.0, 0.0),  # conduz so sob condicao (ex: calor)
]

# Tipos de fisica para energia
ENERGY_FLOW_TYPES = ["liquid", "pressure", "vibration", "hybrid"]
GRAVITY_TYPES = ["normal", "inverted", "local", "none"]
TIME_MODIFIERS = ["static", "cyclic", "decaying", "amplifying"]


class MTBMaterial:
    """Um material desconhecido com propriedades ocultas."""
    def __init__(self, name: str, properties: Dict):
        self.name = name
        self.properties = properties
        # Estado interno (pode mudar com o tempo em alguns mundos)
        self.state = "solid"  # solid, liquid, gas
        self.temperature = 0.0
        self.energy = 0.0

    def get_conductance(self) -> float:
        return self.properties.get("conductive", 0.0)

    def get_absorption(self) -> float:
        return self.properties.get("absorbing", 0.0)

    def get_amplification(self) -> float:
        return self.properties.get("amplifying", 1.0)

    def is_blocking(self) -> bool:
        return self.properties.get("blocking", 0.0) > 0.5

    def is_unstable(self) -> bool:
        return self.properties.get("unstable", 0.0) > 0.3


class MTBWorld:
    """Um mundo procedural completo."""
    def __init__(self, world_id: int, seed: int):
        self.world_id = world_id
        self.seed = seed
        rng = random.Random(seed)

        # Fisica do mundo
        self.physics = {
            "energy_flow": rng.choice(ENERGY_FLOW_TYPES),
            "gravity": rng.choice(GRAVITY_TYPES),
            "time_modifier": rng.choice(TIME_MODIFIERS),
            "friction_inverse": rng.random() < 0.3,
        }

        # Gerar materiais (5-12 por mundo)
        n_materials = rng.randint(5, 9)
        names = rng.sample(MATERIAL_NAMES, n_materials)
        self.materials: Dict[str, MTBMaterial] = {}
        for name in names:
            props = self._generate_properties(rng)
            self.materials[name] = MTBMaterial(name, props)

        # Garantir pelo menos um material condutor NAO bloqueante
        has_good = any(
            m.get_conductance() > 0.3 and not m.is_blocking()
            for m in self.materials.values()
        )
        if not has_good:
            # Forcar o primeiro material a ser condutor nao-bloqueante
            name = names[0]
            self.materials[name] = MTBMaterial(name, {"conductive": round(rng.uniform(0.5, 1.0), 2)})

        # Grid do mundo (12x12)
        self.width, self.height = 12, 8
        self.grid: List[List[Optional[str]]] = [
            [None for _ in range(self.width)] for _ in range(self.height)
        ]

        # Preencher com materiais aleatorios
        for y in range(self.height):
            for x in range(self.width):
                if rng.random() < 0.4:
                    self.grid[y][x] = rng.choice(names)

        # Fonte e destino de energia
        self.source = (rng.randint(1, 3), rng.randint(1, self.height - 2))
        self.target = (rng.randint(self.width - 4, self.width - 2),
                       rng.randint(1, self.height - 2))

        # Estado energetico do grid
        self.energy_grid = [[0.0 for _ in range(self.width)] for _ in range(self.height)]
        self.energy_grid[self.source[1]][self.source[0]] = 100.0  # fonte com mais energia

        # Historico de interacoes da AGI
        self.interaction_log: List[Dict] = []

    def _generate_properties(self, rng: random.Random) -> Dict:
        """Gera propriedades ocultas para um material."""
        props = {}
        n_props = rng.randint(1, 3)
        chosen = rng.sample(PROPERTIES_POOL, n_props)
        for name, min_val, max_val in chosen:
            if max_val > 0:
                props[name] = round(rng.uniform(min_val, max_val), 2)
            else:
                props[name] = rng.choice([True, False])
        return props

    def get_state_text(self) -> str:
        """Descricao textual do estado atual do mundo para a AGI."""
        lines = [f"MTB World {self.world_id} (seed={self.seed})"]
        lines.append(f"Source: {self.source} | Target: {self.target}")
        materials_list = list(self.materials.keys())
        lines.append(f"Materials: {', '.join(materials_list)}")
        return "\n".join(lines)

    def render_grid(self) -> str:
        """Renderiza grid como texto para visualizacao."""
        lines = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                mat = self.grid[y][x]
                energy = self.energy_grid[y][x]
                if (x, y) == self.source:
                    row.append(f"[S]{energy:.0f}" if energy > 0 else "[S] ")
                elif (x, y) == self.target:
                    row.append(f"[T]{energy:.0f}" if energy > 0 else "[T] ")
                elif mat:
                    m = self.materials[mat]
                    tag = "C" if m.get_conductance() > 0 else "B" if m.is_blocking() else "."
                    e_str = f"{energy:.0f}" if energy > 0 else " "
                    row.append(f"{tag}{e_str} ")
                else:
                    row.append("_   ")
            lines.append("".join(row))
        return "\n".join(lines)

    def step(self, dt: float = 0.1):
        """Simula fisica do mundo por um passo de tempo.

        Modelo simplificado: se existe um caminho condutor continuo
        entre source e target (via 4-vizinhanca), a energia flui
        diretamente. A quantidade depende da condutividade media
        do caminho e da distancia.
        """
        MAX_ENERGY = 100.0

        # BFS da source ao target via celulas condutoras (8-vizinhanca)
        sx, sy = self.source
        tx, ty = self.target
        queue = [(sx, sy)]
        visited = {(sx, sy)}
        parent = {(sx, sy): None}

        while queue:
            cx, cy = queue.pop(0)
            if (cx, cy) == (tx, ty):
                break
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in visited:
                    # Target cell sempre pode ser alcancada (nao precisa de material)
                    if (nx, ny) == (tx, ty):
                        visited.add((nx, ny))
                        parent[(nx, ny)] = (cx, cy)
                        queue.append((nx, ny))
                        continue
                    mat = self.grid[ny][nx]
                    if mat:
                        m = self.materials.get(mat)
                        if m and m.get_conductance() > 0.1:
                            # Materiais bloqueantes deixam passar (com resistencia)
                            visited.add((nx, ny))
                            parent[(nx, ny)] = (cx, cy)
                            queue.append((nx, ny))

        source_energy = min(self.energy_grid[sy][sx], MAX_ENERGY)

        if (tx, ty) in parent:
            # Reconstruir caminho
            path = []
            cur = (tx, ty)
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()  # source -> target
            n_cells = len(path) - 1  # numero de celulas intermediarias

            # Calcular condutividade media do caminho (com penalidade para bloqueantes)
            total_cond = 0.0
            for cx, cy in path[1:-1]:  # exclui source e target
                mat = self.grid[cy][cx]
                if mat:
                    m = self.materials.get(mat)
                    if m:
                        cond = m.get_conductance()
                        if m.is_blocking():
                            cond *= 0.5  # bloqueantes reduzem em 50%
                        total_cond += cond
            avg_cond = total_cond / max(1, n_cells)

            # Energia no target: decay por distancia e condutividade
            decay = 0.75 ** n_cells  # perde 25% por celula intermediaria
            target_energy = source_energy * avg_cond * decay
            self.energy_grid[ty][tx] = min(target_energy, MAX_ENERGY)
        else:
            self.energy_grid[ty][tx] = 0.0

        # Propagar energia no grid (diffusao simples)
        new_grid = [[0.0 for _ in range(self.width)] for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                e = min(self.energy_grid[y][x], MAX_ENERGY)
                if (x, y) == (sx, sy):
                    # Source mantem energia
                    new_grid[y][x] += max(e, 80.0)
                    continue
                if e <= 0.01:
                    continue
                mat = self.grid[y][x]
                if mat is None:
                    new_grid[y][x] += e * 0.5  # perde metade no ar
                elif mat and not self.materials[mat].is_blocking():
                    new_grid[y][x] += e * 0.8  # mantem 80%
                else:
                    new_grid[y][x] += e * 0.9  # mantem 90% em blocking

        # Garantir energia na source
        new_grid[sy][sx] = max(new_grid[sy][sx], 80.0)

        self.energy_grid = new_grid

    def get_energy_at_target(self) -> float:
        """Energia que chega ao destino."""
        return self.energy_grid[self.target[1]][self.target[0]]

    def interact(self, action: str, x: int, y: int,
                 material: Optional[str] = None) -> Dict:
        """AGI interage com o mundo. Retorna observacao."""
        result = {"success": False, "observation": "", "energy_change": 0.0}

        if action == "MOVE":
            if 0 <= x < self.width and 0 <= y < self.height:
                result["success"] = True
                mat = self.grid[y][x]
                if mat:
                    m = self.materials[mat]
                    result["observation"] = (f"Material: {mat}, "
                        f"Props: {list(m.properties.keys())}, "
                        f"Energy: {self.energy_grid[y][x]:.1f}")
                else:
                    result["observation"] = "Empty space"

        elif action == "PLACE":
            if material and material in self.materials:
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.grid[y][x] = material
                    result["success"] = True
                    result["observation"] = f"Placed {material} at ({x},{y})"
                    self.interaction_log.append({
                        "action": "PLACE", "x": x, "y": y,
                        "material": material, "time": len(self.interaction_log)
                    })

        elif action == "REMOVE":
            if 0 <= x < self.width and 0 <= y < self.height:
                old = self.grid[y][x]
                self.grid[y][x] = None
                result["success"] = True
                result["observation"] = f"Removed {old} from ({x},{y})"

        elif action == "MEASURE":
            if 0 <= x < self.width and 0 <= y < self.height:
                e = self.energy_grid[y][x]
                mat = self.grid[y][x]
                if mat:
                    m = self.materials[mat]
                    result["observation"] = (f"Energy: {e:.2f}, Material: {mat}, "
                        f"Temp: {m.temperature:.1f}")
                else:
                    result["observation"] = f"Energy: {e:.2f}, Empty space"

        return result

    def get_material_hint(self, mat_name: str) -> str:
        """Hint sutil sobre propriedades (para facilitar exploracao)."""
        m = self.materials.get(mat_name)
        if not m:
            return "Unknown material"
        hints = {
            "conductive": "feels warm",
            "blocking": "feels dense",
            "amplifying": "hums faintly",
            "absorbing": "feels spongy",
            "unstable": "crackles",
            "directional": "pulls in one direction",
            "conditional": "changes with temperature",
        }
        active = [hints[p] for p in m.properties if p in hints]
        return f"{mat_name}: {', '.join(active[:2])}" if active else f"{mat_name}: seems inert"
