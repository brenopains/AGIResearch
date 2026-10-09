"""
Mutual Constraint Network — Integra CGE + Simulacao + Memoria + Programas.

Cada nodo da rede e um verificador de restricoes que avalia se um
conceito proposto e valido do seu ponto de vista.

Se o CGE aprova, a Simulacao aprova, a Memoria aprova, e os Programas
aprovam — o conceito e valido. Se algum rejeita, o conceito e refinado.
"""
from typing import List, Dict, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ConstraintVerdict:
    """Veredito de um verificador de restricoes."""
    source: str                    # "cge", "simulation", "memory", "programs"
    approved: bool = True
    reason: str = ""
    suggestions: List[str] = field(default_factory=list)


class ConstraintNetwork:
    """Rede de restricoes mutuas.
    
    Cada modulo existente (CGE, Simulacao, Memoria, Programas) verifica
    se um conceito proposto e consistente com seu conhecimento.
    
    Uso:
        network = ConstraintNetwork(cge=cge, memory=memory)
        verdicts = network.validate(concept)
        if all(v.approved for v in verdicts):
            print("Conceito VALIDO em toda a rede")
        else:
            for v in verdicts:
                if not v.approved:
                    print(f"Rejeitado por {v.source}: {v.reason}")
    """
    
    def __init__(self, cge=None, memory=None, simulator=None, programs=None,
                 strictness: float = 0.7,
                 law_registry=None, concept_registry=None):
        self.cge = cge
        self.memory = memory
        self.simulator = simulator
        self.programs = programs
        self.law_registry = law_registry
        self.concept_registry = concept_registry
        self.strictness = max(0.0, min(1.0, strictness))

        # Verificadores registrados
        self._checkers: List[Callable] = [
            self._check_domain_coherence,
            self._check_redundancy,
            self._check_cge,
            self._check_simulation,
            self._check_memory,
            self._check_programs,
        ]

        # Checkers de VETO: se um rejeitar, o conceito e automaticamente
        # rejeitado independente da strictness.
        # redundancia e dominio incoerente sao veto.
        self._veto_checkers: Set[str] = {"redundancy", "domain_coherence"}

    def _check_domain_coherence(self, name: str, ctype: str,
                                  domains: set) -> ConstraintVerdict:
        """Verifica se os dominios do conceito/lei sao coerentes entre si.

        Regras:
          - Um conceito com dominios muito diferentes (ex: physics + poetry)
            tem baixa coerencia → provavelmente e uma abstracao forcada
          - Um conceito com 1 dominio so e trivial (coerencia maxima, mas utilidade baixa)
          - 2+ dominios do mesmo campo (ex: physics+chemistry) sao coerentes
        """
        if not domains:
            return ConstraintVerdict(
                source="domain_coherence", approved=False,
                reason="conceito sem dominios — nao e viavel"
            )

        if len(domains) == 1:
            return ConstraintVerdict(
                source="domain_coherence", approved=True,
                reason=f"dominio unico: {list(domains)[0]}"
            )

        # Agrupa dominios por campo semantico
        semantic_fields = {
            "physics": {"physics", "astronomy", "chemistry", "quantum_physics"},
            "life": {"biology", "neuroscience", "medicine", "paleontology"},
            "geo": {"geography", "geology", "climate_science"},
            "math": {"geometry", "mathematics"},
            "cs": {"programming", "computer_science", "deep_learning"},
            "social": {"economics", "sociology", "psychology", "linguistics"},
            "arts": {"art", "music", "linguistics"},
            "engineering": {"robotics", "engineering"},
            "applied": {"epidemiology", "audio"},
        }

        # Verifica se os dominios pertencem a <= 2 campos semanticos
        domain_list = list(domains)
        fields_used = set()
        for d in domain_list:
            for field, members in semantic_fields.items():
                if d in members:
                    fields_used.add(field)
                    break
            else:
                fields_used.add("other")

        if len(fields_used) <= 2:
            return ConstraintVerdict(
                source="domain_coherence", approved=True,
                reason=f"dominios coerentes: {len(domains)} dominios em {len(fields_used)} campos",
                suggestions=list(fields_used),
            )

        return ConstraintVerdict(
            source="domain_coherence", approved=False,
            reason=f"dominios muito dispersos: {len(domains)} dominios em {len(fields_used)} campos",
            suggestions=[f"talvez dividir em {len(fields_used)} conceitos menores"],
        )

    def _check_redundancy(self, name: str, ctype: str,
                           domains: set) -> ConstraintVerdict:
        """Verifica se o conceito/lei e redundante com algo existente.

        Checa tanto sobreposicao de dominios quanto de conceitos explicados.
        """
        if self.law_registry is None:
            return ConstraintVerdict(
                source="redundancy", approved=True,
                reason="redundancy_check_offline"
            )

        if ctype != "world_law":
            return ConstraintVerdict(
                source="redundancy", approved=True,
                reason="nao e uma lei — redundancia nao se aplica"
            )

        existing_laws = self.law_registry.get_all()
        for law in existing_laws:
            if law.name == name:
                continue

            # 1. Se uma lei existente explica os mesmos conceitos → redundante
            if self.concept_registry is not None:
                shared_concepts = law.explains & set(
                    a.name for a in self.concept_registry.get_all()
                )
                if len(shared_concepts) >= 2:
                    return ConstraintVerdict(
                        source="redundancy", approved=False,
                        reason=f"conceitos {shared_concepts} ja explicados por '{law.name}'",
                        suggestions=[f"reutilizar lei existente: {law.name}"],
                    )

            # 2. Se uma lei existente ja cobre os mesmos dominios
            shared_domains = law.domains & domains
            if len(shared_domains) >= 2 and len(shared_domains) >= len(domains) * 0.5:
                return ConstraintVerdict(
                    source="redundancy", approved=False,
                    reason=f"redundante com lei '{law.name}' ({len(shared_domains)} dominios compartilhados)",
                    suggestions=[f"reutilizar lei existente: {law.name}"],
                )

        return ConstraintVerdict(
            source="redundancy", approved=True,
            reason="lei nao redundante com as existentes",
        )

    def set_strictness(self, strictness: float):
        """Ajusta rigidez das restricoes.

        strictness alta (0.8+): so aprova se TUDO confirma
        strictness baixa (0.0-0.3): aprova com evidencia minima
        """
        self.strictness = max(0.0, min(1.0, strictness))
    
    def validate(self, concept_name: str, concept_type: str = "",
                 domains: set = None) -> List[ConstraintVerdict]:
        """Valida um conceito contra todos os verificadores.

        Checkers de veto (domain_coherence, redundancy) podem rejeitar
        automaticamente independente da strictness.
        """
        domains = domains or set()
        verdicts = []
        for checker in self._checkers:
            verdict = checker(concept_name, concept_type, domains)
            verdicts.append(verdict)
            # Veto: se checker de veto rejeitar, para imediatamente
            if not verdict.approved and verdict.source in self._veto_checkers:
                break
        return verdicts

    def is_valid(self, concept_name: str, concept_type: str = "",
                 domains: set = None) -> bool:
        """Um conceito e valido se:
        1. Nenhum veto checker rejeitou
        2. strictness dos demais aprovam
        """
        verdicts = self.validate(concept_name, concept_type, domains)
        if not verdicts:
            return False

        # Se algum veto rejeitou → invalido
        for v in verdicts:
            if not v.approved and v.source in self._veto_checkers:
                return False

        # Demais checkers seguem a strictness
        non_veto = [v for v in verdicts if v.source not in self._veto_checkers]
        if not non_veto:
            return True  # so tem vetos, todos aprovaram
        approved = sum(1 for v in non_veto if v.approved)
        required = max(1, int(len(non_veto) * self.strictness))
        return approved >= required
    
    def _check_cge(self, name: str, ctype: str, domains: set) -> ConstraintVerdict:
        """Verifica se o conceito e consistente com o CGE."""
        if self.cge is None:
            return ConstraintVerdict(source="cge", approved=True, reason="cge_offline")
        
        if name in self.cge.nodes:
            return ConstraintVerdict(
                source="cge", approved=True,
                reason=f"conceito {name} ja existe no CGE",
            )
        
        # Verifica se e consistente com conceitos existentes
        try:
            results = self.cge.query_attention(name, top_k=3)
            if results and results[0][2] > 0.5:
                return ConstraintVerdict(
                    source="cge", approved=True,
                    reason=f"similar a conceito existente: {results[0][0]}",
                )
        except Exception:
            pass
        
        return ConstraintVerdict(
            source="cge", approved=True,
            reason=f"conceito {name} e novo para o CGE",
        )
    
    def _check_simulation(self, name: str, ctype: str, domains: set) -> ConstraintVerdict:
        """Verifica se o conceito e simulaavel."""
        if self.simulator is None:
            return ConstraintVerdict(source="simulation", approved=True, reason="sim_offline")
        
        instances = self.simulator.imagine(name)
        if instances:
            domains_covered = len(set(i.domain for i in instances))
            return ConstraintVerdict(
                source="simulation", approved=True,
                reason=f"instanciavel em {domains_covered} dominios",
                suggestions=[f"tentar dominio: {i.domain}" for i in instances[:3]],
            )
        
        return ConstraintVerdict(
            source="simulation", approved=False,
            reason=f"{name} nao instanciavel em nenhum dominio conhecido",
        )
    
    def _check_memory(self, name: str, ctype: str, domains: set) -> ConstraintVerdict:
        """Verifica se o conceito existe na memoria."""
        if self.memory is None:
            return ConstraintVerdict(source="memory", approved=True, reason="mem_offline")
        
        # Tenta buscar na memoria
        try:
            if hasattr(self.memory, 'query'):
                results = self.memory.query(name, top_k=3)
                if results:
                    return ConstraintVerdict(
                        source="memory", approved=True,
                        reason=f"encontrado na memoria: {len(results)} resultados",
                    )
        except Exception:
            pass
        
        return ConstraintVerdict(
            source="memory", approved=True,
            reason=f"conceito {name} e novo na memoria",
        )
    
    def _check_programs(self, name: str, ctype: str, domains: set) -> ConstraintVerdict:
        """Verifica se o conceito pode ser expresso como programa."""
        if self.programs is None:
            return ConstraintVerdict(source="programs", approved=True, reason="prog_offline")
        
        # Verifica se existe alguma primitiva ou macro com este nome
        if hasattr(self.programs, 'get'):
            prim = self.programs.get(name)
            if prim is not None:
                return ConstraintVerdict(
                    source="programs", approved=True,
                    reason=f"primitiva {name} ja existe no registry",
                )
        
        return ConstraintVerdict(
            source="programs", approved=True,
            reason=f"{name} nao e uma primitiva (pode ser conceito puro)",
        )
    
    def get_stats(self) -> dict:
        return {
            "checkers": ["cge", "simulation", "memory", "programs"],
            "cge_online": self.cge is not None,
            "simulation_online": self.simulator is not None,
            "memory_online": self.memory is not None,
            "programs_online": self.programs is not None,
        }
