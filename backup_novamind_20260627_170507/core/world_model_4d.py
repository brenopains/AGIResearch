import numpy as np
import time
import random
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque


class WorldModel4D:
    def __init__(self, grid_size: int = 5):
        self.grid_size = grid_size
        self.grid: Dict[Tuple[int, int, int], Dict] = {}
        self.temporal_log: Dict[Tuple[int, int, int], deque] = defaultdict(lambda: deque(maxlen=20))
        self.objects: Dict[str, Tuple[int, int, int]] = {}
        self.history: deque = deque(maxlen=100)

    def set_cell(self, x: int, y: int, z: int, properties: Dict):
        key = (x, y, z)
        if key not in self.grid:
            self.grid[key] = {"type": "empty", "solid": False, "color": None, "objects": []}
        self.grid[key].update(properties)
        self.temporal_log[key].append({
            "time": time.time(),
            "state": dict(self.grid[key]),
        })

    def get_cell(self, x: int, y: int, z: int) -> Optional[Dict]:
        return self.grid.get((x, y, z))

    def place_object(self, obj_name: str, x: int, y: int, z: int):
        self.objects[obj_name] = (x, y, z)
        self.set_cell(x, y, z, {"objects": self.grid.get((x, y, z), {}).get("objects", []) + [obj_name]})

    def move_object(self, obj_name: str, dx: int = 0, dy: int = 0, dz: int = 0) -> bool:
        if obj_name not in self.objects:
            return False
        cx, cy, cz = self.objects[obj_name]
        nx, ny, nz = cx + dx, cy + dy, cz + dz
        if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and 0 <= nz < self.grid_size):
            return False
        target = self.grid.get((nx, ny, nz), {})
        if target.get("solid"):
            return False
        old_cell = self.grid.get((cx, cy, cz), {})
        obj_list = old_cell.get("objects", [])
        if obj_name in obj_list:
            obj_list.remove(obj_name)
        self.objects[obj_name] = (nx, ny, nz)
        self.set_cell(nx, ny, nz, {"objects": self.grid.get((nx, ny, nz), {}).get("objects", []) + [obj_name]})
        return True

    def simulate_step(self, action: str, agent_pos: Tuple[int, int, int]) -> Dict:
        x, y, z = agent_pos
        dx, dy, dz = 0, 0, 0
        if action == "avancar": dy = 1
        elif action == "recuar": dy = -1
        elif action == "virar_esquerda": dx = -1
        elif action == "virar_direita": dx = 1
        nx, ny, nz = x + dx, y + dy, z + dz

        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and 0 <= nz < self.grid_size:
            cell = self.grid.get((nx, ny, nz), {})
            if cell.get("solid"):
                return {"success": False, "reason": "blocked", "new_pos": agent_pos}
            new_pos = (nx, ny, nz)
        else:
            return {"success": False, "reason": "out_of_bounds", "new_pos": agent_pos}

        nearby = []
        for ox, oy, oz in self.objects.values():
            if abs(ox - nx) <= 1 and abs(oy - ny) <= 1 and abs(oz - nz) <= 1:
                nearby.append(ox)
        return {"success": True, "new_pos": new_pos, "nearby_objects": nearby}

    def imagine_trajectory(self, actions: List[str],
                           start_pos: Tuple[int, int, int]) -> List[Dict]:
        trajectory = []
        pos = start_pos
        for action in actions:
            step = self.simulate_step(action, pos)
            trajectory.append({"action": action, "pos_before": pos, **step})
            if step["success"]:
                pos = step["new_pos"]
            else:
                break
        return trajectory

    def observe(self, agent_pos: Tuple[int, int, int], radius: int = 1) -> str:
        x, y, z = agent_pos
        parts = [f"[Pos4D: ({x},{y},{z})]"]
        nearby_objs = []
        for obj_name, (ox, oy, oz) in self.objects.items():
            if abs(ox - x) <= radius and abs(oy - y) <= radius and abs(oz - z) <= radius:
                nearby_objs.append(obj_name)
        if nearby_objs:
            parts.append(f"[Objetos: {', '.join(nearby_objs)}]")
        cell = self.grid.get((x, y, z), {})
        if cell.get("type", "empty") != "empty":
            parts.append(f"[Terreno: {cell['type']}]")
        return " ".join(parts)

    def get_state(self) -> Dict:
        return {
            "grid_size": self.grid_size,
            "occupied_cells": len(self.grid),
            "objects": dict(self.objects),
            "temporal_tracks": {str(k): len(v) for k, v in self.temporal_log.items()},
        }
