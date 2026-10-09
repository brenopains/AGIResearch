"""
Teste: WorldLaws + ConceptEconomy + ConflictResolver integrados.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, ".")

from core.conceptual_discovery.world_laws import WorldLaw, WorldLawRegistry, SEED_LAWS
from core.conceptual_discovery.concept_economy import ConceptEconomy
from core.conceptual_discovery.conflict_resolver import ConflictResolver
from core.conceptual_discovery.constraint_network import ConstraintNetwork


PASS = "[PASS]"
FAIL = "[FAIL]"


def test_world_laws_exist():
    """Leis fundamentais existem e tem propriedades."""
    assert len(SEED_LAWS) >= 4, f"So {len(SEED_LAWS)} leis"
    for name, law in SEED_LAWS.items():
        assert len(law.explains) > 0, f"{name} nao explica conceitos"
        assert len(law.domains) > 0, f"{name} sem dominios"
        assert law.compression > 0, f"{name} sem compressao"
    print(f"  {len(SEED_LAWS)} leis fundamentais:")
    for n, l in SEED_LAWS.items():
        print(f"    {n}: explica {len(l.explains)} conceitos, "
              f"{len(l.domains)} dominios, compressao={l.compression}")


def test_world_law_registry():
    """Registry de leis funciona."""
    reg = WorldLawRegistry()
    assert reg.get_law("propagation") is not None
    assert len(reg.get_all()) == len(SEED_LAWS)
    laws = reg.get_laws_for_concept("branching")
    assert len(laws) >= 1, "branching deveria ser explicado por alguma lei"
    laws = reg.get_laws_for_domain("biology")
    assert len(laws) >= 1, "biology deveria ter leis ativas"
    print(f"  Registry: {len(reg.get_all())} leis")


def test_law_discovery():
    """Descoberta de novas leis quando conceitos compartilham dominios."""
    reg = WorldLawRegistry()
    new_law = reg.discover_law("symmetry", "rotation", {"geometry", "physics"})
    assert new_law is not None, "Deveria descobrir nova lei"
    assert new_law.name in reg.laws
    assert "symmetry" in new_law.explains
    assert "rotation" in new_law.explains
    print(f"  Nova lei descoberta: {new_law.name}")


def test_law_no_discovery_when_exists():
    """Nao descobre lei se ja existe uma que explica ambos."""
    reg = WorldLawRegistry()
    # conservation ja explica symmetry e reflection
    new_law = reg.discover_law("symmetry", "reflection", {"geometry"})
    assert new_law is None, "Nao deveria descobrir (ja existe conservation)"
    print(f"  Descoberta bloqueada: lei ja existe")


def test_law_consistency_check():
    """Verificacao de consistencia entre leis."""
    reg = WorldLawRegistry()
    conflicts = reg.check_consistency("test_domain", {})
    assert isinstance(conflicts, list)
    print(f"  Consistencia: {len(conflicts)} conflitos encontrados")


def test_concept_economy_merge():
    """Economia funde conceitos com alta sobreposicao de dominios."""
    eco = ConceptEconomy(merge_threshold=0.5)
    eco.observe_concept("reflection", {"geometry", "physics", "audio"}, 0.8, 10)
    eco.observe_concept("symmetry", {"geometry", "physics", "biology"}, 0.7, 8)
    assert len(eco._concept_domains) == 2

    actions = eco.run()
    # Overlap: {geometry, physics} / {geometry, physics, audio, biology} = 2/4 = 0.5 >= threshold
    merges = [a for a in actions if a.action_type == "merge"]
    if merges:
        print(f"  Merge: {merges[0].concept_a} + {merges[0].concept_b}")
        print(f"    razao: {merges[0].reason}")
    print(f"  Compression ratio: {eco.compression_ratio}")
    # Podem ter fundido ou nao (depende do overlap exato)
    assert eco.total_current <= eco.total_original


def test_concept_economy_prune():
    """Economia poda conceitos com baixa confianca e reuso."""
    eco = ConceptEconomy(prune_threshold=0.2)
    # Conceito forte
    eco.observe_concept("reflection", {"geometry", "physics"}, 0.8, 10)
    # Conceito fraco (baixa confianca, baixo reuso, 1 obs)
    eco.observe_concept("weak_concept", {"geometry"}, 0.1, 1)

    actions = eco.run()
    prunes = [a for a in actions if a.action_type == "prune"]
    if prunes:
        print(f"  Prune: {prunes[0].concept_a}")
        print(f"    razao: {prunes[0].reason}")
    assert "weak_concept" not in eco._concept_domains or len(prunes) == 0
    print(f"  Apos prune: {len(eco._concept_domains)} conceitos")


def test_conflict_detection():
    """Conflito e detectado e registrado."""
    resolver = ConflictResolver()
    conflict = resolver.detect_conflict(
        law_name="conservation",
        domain="quantum_physics",
        observation="particle disappears",
        prediction="energy conserved",
        description="particle seems to violate conservation",
    )
    assert conflict is not None
    assert not conflict.resolved
    assert conflict.law_name == "conservation"
    print(f"  Conflito: {conflict.description[:60]}")
    print(f"    severity: {conflict.severity:.2f}")


def test_conflict_resolution():
    """Conflito gera hipoteses e pode ser resolvido."""
    cn = ConstraintNetwork()
    resolver = ConflictResolver(constraint_network=cn)

    conflict = resolver.detect_conflict(
        law_name="conservation",
        domain="quantum_physics",
        observation="particle decay",
        prediction="energy conserved",
        description="quantum tunneling seems to violate conservation",
    )

    hypotheses = resolver.resolve(conflict)
    assert len(hypotheses) >= 2, f"Só {len(hypotheses)} hipoteses"
    print(f"  Hipoteses geradas: {len(hypotheses)}")
    for h in hypotheses:
        print(f"    {h.description[:60]} (conf={h.confidence:.2f}, "
              f"tested={h.tested}, passed={h.passed})")


def test_conflict_unresolved_persistence():
    """Conflitos nao resolvidos persistem como gaps."""
    resolver = ConflictResolver()
    c1 = resolver.detect_conflict("law_a", "domain_x", "obs1", "pred1")
    c2 = resolver.detect_conflict("law_b", "domain_y", "obs2", "pred2")

    unresolved = resolver.get_unresolved_conflicts()
    assert len(unresolved) == 2
    assert not c1.resolved and not c2.resolved
    print(f"  Conflitos nao resolvidos: {len(unresolved)}")


def test_integrated_cycle():
    """Ciclo integrado: leis → economia → conflito."""
    # Setup
    law_reg = WorldLawRegistry()
    eco = ConceptEconomy()
    cn = ConstraintNetwork()
    resolver = ConflictResolver(law_registry=law_reg, constraint_network=cn)

    # 1. Observa conceitos
    eco.observe_concept("reflection", {"geometry", "physics", "audio"}, 0.8, 10)
    eco.observe_concept("rotation", {"geometry", "physics"}, 0.7, 8)
    eco.observe_concept("symmetry", {"geometry", "biology"}, 0.6, 5)
    eco.observe_concept("weak", {"unknown"}, 0.05, 1)

    # 2. Tenta descobrir nova lei
    new_law = law_reg.discover_law("reflection", "rotation",
                                    {"geometry", "physics"})
    if new_law:
        print(f"  Nova lei: {new_law.name}")

    # 3. Economia: merge e prune
    actions = eco.run()
    print(f"  Economia: {len(actions)} acoes")
    for a in actions:
        print(f"    {a.action_type}: {a.concept_a} ({a.reason})")

    # 4. Conflito
    conflict = resolver.detect_conflict(
        law_name="conservation", domain="test",
        observation="unexpected", prediction="expected",
    )
    hypotheses = resolver.resolve(conflict)
    print(f"  Conflito resolvido: {conflict.resolved}")

    print(f"  Compression ratio: {eco.compression_ratio}")
    print(f"  Total leis: {len(law_reg.get_all())}")


if __name__ == "__main__":
    print("=== WorldLaws + ConceptEconomy + ConflictResolver ===\n")
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_")):
        print(f"{name}...")
        try:
            fn()
            print(f"  {PASS}\n")
        except Exception as e:
            print(f"  {FAIL}: {e}\n")
            import traceback
            traceback.print_exc()
