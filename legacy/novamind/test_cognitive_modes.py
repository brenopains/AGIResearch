"""
Teste: Cognitive Mode Manager + WorldModel + integracao no CognitiveExecutive.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, ".")

import numpy as np
from core.cognitive_modes import CognitiveModeManager, CognitiveMode, MODE_CONFIGS
from core.world_model import WorldModel
from core.cognitive_executive import CognitiveExecutive
from core.unified_self import UnifiedSelf
from core.intrinsic_motivation import IntrinsicMotivationEngine
from core.goal_driven_engine import GoalDrivenEngine
from core.program_induction import (
    PrimitiveRegistry, ProgramSynthesizer, CompositionGraph, OperatorGroundingBridge,
)
from core.program_induction.primitive import Primitive
from core.program_induction.primitives import (
    create_recolor_primitive, create_remove_primitive,
    create_move_primitive, create_add_primitive,
)
from core.program_induction.operator_proposals import get_all_proposals
from core.hypothesis_engine import HypothesisEngine
from core.conceptual_discovery.concept_agent import ConceptAgent, ConceptAgentRegistry
from core.conceptual_discovery.constraint_network import ConstraintNetwork


PASS = "[PASS]"
FAIL = "[FAIL]"


def test_mode_transitions():
    """Modos transitam baseado em tarefa, sucesso, surpresa, energia."""
    mgr = CognitiveModeManager()

    # Com tarefa → FOCUSED
    mgr.tick(has_task=True, success_rate=0.8, surprise=0.1, energy=0.9)
    assert mgr.current_mode == CognitiveMode.FOCUSED, f"Esperava FOCUSED, got {mgr.current_mode}"

    # Falha repetida → EXPLORATION
    for _ in range(4):
        mgr.tick(has_task=True, success_rate=0.2, surprise=0.3, energy=0.6)
    assert mgr.current_mode == CognitiveMode.EXPLORATION, f"Esperava EXPLORATION, got {mgr.current_mode}"

    # Sem tarefa + energia alta → INCUBATION
    for _ in range(12):
        mgr.tick(has_task=False, success_rate=0.5, surprise=0.2, energy=0.7)
    assert mgr.current_mode == CognitiveMode.INCUBATION, f"Esperava INCUBATION, got {mgr.current_mode}"

    # Sem tarefa por muito tempo → DREAMING
    for _ in range(25):
        mgr.tick(has_task=False, success_rate=0.5, surprise=0.2, energy=0.7)
    assert mgr.current_mode == CognitiveMode.DREAMING, f"Esperava DREAMING, got {mgr.current_mode}"

    # Energia baixa → CONSOLIDATION
    mgr.tick(has_task=False, success_rate=0.5, surprise=0.2, energy=0.2)
    assert mgr.current_mode == CognitiveMode.CONSOLIDATION, f"Esperava CONSOLIDATION, got {mgr.current_mode}"

    print(f"  Transicoes: FOCUSED → EXPLORATION → INCUBATION → DREAMING → CONSOLIDATION")


def test_mode_config_values():
    """Cada modo tem valores de configuracao diferentes."""
    for mode in CognitiveMode:
        cfg = MODE_CONFIGS[mode]
        assert 0.0 <= cfg.temperature <= 1.0
        assert cfg.beam_width >= 1
        assert cfg.max_search_depth >= 1
        assert 0.0 <= cfg.constraint_strictness <= 1.0
        assert -5 <= cfg.energy_cost <= 5
    print(f"  Todos os {len(CognitiveMode)} modos tem configs validas")


def test_world_model_prediction_error():
    """WorldModel computa surpresa corretamente."""
    wm = WorldModel()
    inp = np.zeros((5, 5), dtype=np.int32)
    out = np.zeros((5, 5), dtype=np.int32)

    # Predicao e erro
    pred = wm.predict(inp)
    error = wm.update(inp, out, pred)

    assert 0.0 <= error.error_magnitude <= 1.0
    assert 0.0 <= error.surprise <= 1.0
    assert wm.get_surprise() >= 0
    assert not wm.is_surprised(threshold=0.9)  # nao deveria estar tao surpreso
    print(f"  Surpresa: {wm.get_surprise():.3f}, Acuracia: {wm.get_accuracy():.3f}")


def test_executive_mode_integration():
    """CognitiveExecutive integra mode manager e world model."""
    # Setup minimo
    reg = PrimitiveRegistry()
    p = Primitive(name="id", structure_type="identity",
                   solver_fn=lambda i, o: {} if i.shape == o.shape else None)
    p._execute = lambda g, params: g.copy()
    p.confidence = 0.5
    reg.register(p)
    cg = CompositionGraph()
    og = OperatorGroundingBridge(reg, cg)
    pie = ProgramSynthesizer(reg, composition_graph=cg, grounding_bridge=og)
    engine = GoalDrivenEngine(program_induction_engine=pie, use_planner=False)
    self_model = UnifiedSelf()
    motivation = IntrinsicMotivationEngine(self_model=self_model)
    mgr = CognitiveModeManager()
    wm = WorldModel()
    he = HypothesisEngine()
    cn = ConstraintNetwork()

    exec_ = CognitiveExecutive(
        goal_engine=engine, self_model=self_model, motivation=motivation,
        mode_manager=mgr, world_model=wm, hypothesis_engine=he,
        synthesizer=pie, constraint_network=cn,
    )

    # Ciclo com tarefa
    inp = np.zeros((5, 5), dtype=np.int32)
    inp[1, 1:4] = 1
    out = inp.copy()

    result = exec_.cycle(inp, out)
    assert "mode" in result
    assert result["mode"] in ("focused", "exploration")
    assert "accuracy" in result
    print(f"  Cycle com tarefa: mode={result['mode']}, acc={result['accuracy']:.2%}")

    # Ciclo idle
    idle = exec_.idle_cycle()
    assert "mode" in idle
    assert "idle_cycle" in idle
    print(f"  Idle cycle: mode={idle['mode']}, temperature={idle.get('temperature', '?')}")


def test_temperature_affects_search():
    """Temperatura alta aumenta beam_width do synthesizer."""
    reg = PrimitiveRegistry()
    p = Primitive(name="id", structure_type="identity",
                   solver_fn=lambda i, o: {} if i.shape == o.shape else None)
    p._execute = lambda g, params: g.copy()
    p.confidence = 0.5
    reg.register(p)
    cg = CompositionGraph()
    og = OperatorGroundingBridge(reg, cg)
    pie = ProgramSynthesizer(reg, composition_graph=cg, grounding_bridge=og)
    engine = GoalDrivenEngine(program_induction_engine=pie, use_planner=False)
    self_model = UnifiedSelf()
    mgr = CognitiveModeManager()
    exec_ = CognitiveExecutive(engine, self_model=self_model, mode_manager=mgr, synthesizer=pie)

    # Modo FOCUSED: beam pequeno
    mgr._transition_to(CognitiveMode.FOCUSED)
    exec_._apply_mode_config()
    focused_beam = pie._beam_width
    focused_depth = pie._max_depth

    # Modo EXPLORATION: beam grande
    mgr._transition_to(CognitiveMode.EXPLORATION)
    exec_._apply_mode_config()
    explored_beam = pie._beam_width
    explored_depth = pie._max_depth

    assert explored_beam >= focused_beam, \
        f"Exploration beam ({explored_beam}) deveria >= Focused beam ({focused_beam})"
    assert explored_depth >= focused_depth, \
        f"Exploration depth ({explored_depth}) deveria >= Focused depth ({focused_depth})"

    print(f"  FOCUSED: beam={focused_beam}, depth={focused_depth}")
    print(f"  EXPLORATION: beam={explored_beam}, depth={explored_depth}")


def test_constraint_strictness_changes():
    """ConstraintNetwork muda strictness com o modo."""
    cn = ConstraintNetwork()
    mgr = CognitiveModeManager()
    self_model = UnifiedSelf()

    # Nenhum checker registrado - validacao depende apenas da strictness
    cn.set_strictness(0.9)
    strict = cn.strictness
    assert strict == 0.9, f"Strictness deveria ser 0.9, got {strict}"

    cn.set_strictness(0.1)
    relaxed = cn.strictness
    assert relaxed == 0.1, f"Strictness deveria ser 0.1, got {relaxed}"
    print(f"  Strictness mudou de {strict} para {relaxed}")


def test_energy_cost_by_mode():
    """Cada modo tem custo/ganho de energia diferente."""
    self_model = UnifiedSelf()
    initial_energy = self_model.energy

    # DREAMING: custa energia
    config = MODE_CONFIGS[CognitiveMode.DREAMING]
    self_model.apply_mode_energy_cost(config.energy_cost)
    assert self_model.energy < initial_energy, "DREAMING deveria custar energia"
    dream_energy = self_model.energy
    print(f"  DREAMING: {initial_energy:.2f} → {dream_energy:.2f}")

    # CONSOLIDATION: recupera
    config = MODE_CONFIGS[CognitiveMode.CONSOLIDATION]
    self_model.apply_mode_energy_cost(config.energy_cost)
    assert self_model.energy >= dream_energy, "CONSOLIDATION deveria recuperar energia"
    print(f"  CONSOLIDATION: {dream_energy:.2f} → {self_model.energy:.2f}")


if __name__ == "__main__":
    print("=== Cognitive Modes + WorldModel ===\n")
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_")):
        print(f"{name}...")
        try:
            fn()
            print(f"  {PASS}\n")
        except Exception as e:
            print(f"  {FAIL}: {e}\n")
            import traceback
            traceback.print_exc()
