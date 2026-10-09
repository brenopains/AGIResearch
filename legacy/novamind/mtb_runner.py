"""
MTB-1 Runner — Multiworld Transfer Benchmark

Executa o experimento completo:
  Fase 0: Gerar N mundos procedureis
  Fase 1-2: Explorar + resolver tarefa local em cada mundo
  Fase 3: Consolidacao entre mundos (Forja)
  Fase 4: Mundo final surpresa (transferencia)

Uso: python mtb_runner.py [n_worlds]
"""
import sys
import time
import random
from typing import List

from mtb_world_generator import MTBWorld, MTBMaterial, MATERIAL_NAMES
from core.cognitive_graph_engine import CognitiveGraphEngine
from core.goal_system import GoalManager

# Tenta carregar BGE
try:
    from core.bge_encoder import BGEEncoder
    bge = BGEEncoder()
    print("[MTB] BGE 768D encoder ativo")
except Exception as e:
    bge = None
    print(f"[MTB] BGE indisponivel ({e}), usando hash fallback")

DEVICE = "cuda"

# Criar CGE e GoalManager
cge = CognitiveGraphEngine(device=DEVICE, bge_encoder=bge)
gm = GoalManager(cge=cge)


def generate_worlds(n: int, base_seed: int = 42) -> List[MTBWorld]:
    """Gera N mundos procedureis."""
    worlds = []
    for i in range(n):
        seed = base_seed + i * 137
        w = MTBWorld(world_id=i + 1, seed=seed)
        worlds.append(w)
    return worlds


def generate_transfer_world(previous_worlds: List[MTBWorld],
                            seed: int = 9999) -> MTBWorld:
    """
    Gera mundo final de transferencia:
    - Combina 3 regras fisicas aleatorias dos mundos anteriores
    - MAS com inversoes
    - Nenhum material se repete
    """
    rng = random.Random(seed)

    # Coletar regras dos mundos anteriores
    all_physics = [w.physics for w in previous_worlds]
    sample = rng.sample(all_physics, min(3, len(all_physics)))

    # Combinar com inversoes
    physics = {
        "energy_flow": rng.choice(["liquid", "pressure", "vibration", "hybrid"]),
        "gravity": "inverted" if rng.random() < 0.5 else "local",
        "time_modifier": "decaying",
        "friction_inverse": True,
    }

    # Material pool EXCLUDING all materials used in previous worlds
    used = set()
    for w in previous_worlds:
        used.update(w.materials.keys())
    available = [m for m in MATERIAL_NAMES if m not in used]
    if len(available) < 5:
        available = MATERIAL_NAMES  # fallback

    # Criar mundo
    world = MTBWorld(world_id=999, seed=seed)
    world.physics = physics

    # Substituir materiais E LIMPAR GRID
    n_materials = rng.randint(5, 8)
    new_names = rng.sample(available, n_materials)
    world.materials = {}
    for name in new_names:
        props = world._generate_properties(rng)
        world.materials[name] = MTBMaterial(name, props)

    # Garantir pelo menos um condutor nao-bloqueante
    has_good = any(
        m.get_conductance() > 0.3 and not m.is_blocking()
        for m in world.materials.values()
    )
    if not has_good:
        name = new_names[0]
        world.materials[name] = MTBMaterial(name, {"conductive": round(rng.uniform(0.5, 1.0), 2)})

    # Limpar grid e repovoar com novos materiais
    world.grid = [[None for _ in range(world.width)] for _ in range(world.height)]
    for y in range(world.height):
        for x in range(world.width):
            if rng.random() < 0.4:
                world.grid[y][x] = rng.choice(new_names)

    return world


def print_world_summary(world: MTBWorld, result: dict = None):
    """Print detalhado de um mundo."""
    physics = world.physics
    print(f"\n  Mundo {world.world_id}: "
          f"flow={physics['energy_flow']}, grav={physics['gravity']}, "
          f"time={physics['time_modifier']}")
    print(f"  Source {world.source} -> Target {world.target}")
    print(f"  Materiais: {', '.join(world.materials.keys())[:60]}...")
    for name, mat in world.materials.items():
        props = list(mat.properties.keys())
        if props:
            print(f"    {name}: {props}")

    if result:
        print(f"  Resultado: {'SOLVED' if result.get('solved') else 'FAIL'} "
              f"({result.get('cycles')} ciclos, "
              f"energia={result.get('final_energy', 0):.1f})")


def main():
    n_worlds = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    print("=" * 70)
    print("MTB-1: MULTIWORLD TRANSFER BENCHMARK")
    print(f"Gerando {n_worlds} mundos + 1 mundo de transferencia...")
    print("=" * 70)

    # FASE 0: Gerar mundos
    worlds = generate_worlds(n_worlds)
    print(f"\n[FASE 0] {len(worlds)} mundos gerados.\n")

    # Import sandbox
    from mtb_sandbox_salto2 import MTBSandboxSalto2 as MTBSandbox
    sandbox = MTBSandbox(cge, gm, bge)

    # FASES 1-3: Explorar + resolver + consolidar cada mundo
    print("=" * 70)
    print("FASES 1-3: EXPLORACAO + TAREFA + CONSOLIDACAO")
    print("=" * 70)

    world_results = []
    start_time = time.time()

    for i, w in enumerate(worlds):
        print(f"\n--- Mundo {i + 1}/{n_worlds} ---")
        print_world_summary(w)
        result = sandbox.run_world(w, max_cycles=60)
        world_results.append(result)

    elapsed = time.time() - start_time

    # Sumario intermediario
    solved = sum(1 for r in world_results if r["solved"])
    print(f"\n{'=' * 70}")
    print(f"SUMARIO: {solved}/{n_worlds} mundos resolvidos "
          f"em {elapsed:.0f}s")
    for r in world_results:
        status = "SOLVED" if r["solved"] else "FAIL"
        print(f"  Mundo {r['world_id']}: {status} "
              f"({r['cycles']} ciclos, energia={r['final_energy']:.1f}, "
              f"nos={r['n_cge_nodes']}, edges={r['n_cge_edges']})")

    # FASE 4: Mundo final de transferencia
    print(f"\n{'=' * 70}")
    print("FASE 4: MUNDO DE TRANSFERENCIA (UNSEEN)")
    print("=" * 70)

    transfer_world = generate_transfer_world(worlds, seed=9999)
    print("\n--- Mundo Final (totalmente novo, sem materiais repetidos) ---")
    print_world_summary(transfer_world)

    transfer_result = sandbox.run_world(transfer_world, max_cycles=80)
    transfer_solved = transfer_result["solved"]
    transfer_cycles = transfer_result["cycles"]

    print(f"\n{'=' * 70}")
    print(f"RESULTADO FINAL")
    print(f"{'=' * 70}")
    print(f"Mundos explorados:       {n_worlds}")
    print(f"Mundos resolvidos:       {solved}/{n_worlds} "
          f"({round(solved / n_worlds * 100, 1)}%)")
    print(f"Transferencia (novo):    {'SOLVED' if transfer_solved else 'FAIL'}")
    print(f"Ciclos no mundo novo:    {transfer_cycles}")
    print(f"Tempo total:             {elapsed:.0f}s")
    print(f"Nos no CGE:              {len(cge.nodes)}")
    print(f"Arestas no CGE:          {len(cge.edges)}")
    print(f"Conceitos abstratos:     {sum(1 for n in cge.nodes if 'abstract' in n)}")
    print(f"\nCriterio AGI:           ", end="")
    if transfer_solved:
        print("PASS - Generalizou para mundo novo!")
        if solved == n_worlds:
            print("PASS - Nao e overfitting (resolver todos + transferir)")
    else:
        print("FAIL - Nao generalizou")

    # Salvar resultados
    import json
    output = {
        "n_worlds": n_worlds,
        "solved": solved,
        "solved_pct": round(solved / n_worlds * 100, 1),
        "transfer_solved": transfer_solved,
        "transfer_cycles": transfer_cycles,
        "cge_nodes": len(cge.nodes),
        "cge_edges": len(cge.edges),
        "elapsed_seconds": round(elapsed, 1),
        "world_results": world_results,
        "transfer_result": transfer_result,
    }
    with open("mtb_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResultados salvos em mtb_results.json")


if __name__ == "__main__":
    main()
