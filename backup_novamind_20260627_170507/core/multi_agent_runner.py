"""
Multi-Agent Runner — 3 AGIs compartilhando o Inconsciente Coletivo
===================================================================
Cria 3 instâncias da NovaMind com perfis neuroquímicos diferentes:

  A) CIENTISTA (ACh alto, DA médio)   — explora, descobre padrões, questiona
  B) ENGENHEIRO (DA alto, NA médio)   — constrói, otimiza, executa
  C) CRÍTICO (NA alto, 5HT alto)      — valida, testa, encontra falhas

TODAS compartilham o mesmo CollectiveUnconscious (.hdv único).
Cada uma tem seu cérebro PRIVADO (engine, memória, stream, hipocampo).

Arquitetura:
  - Cada AGI roda em sua própria thread com step() periódico
  - Quando uma descobre algo (Eureka), compartilha no coletivo
  - As outras puxam e validam na próxima iteração
  - NENHUMA é LLM. NENHUMA é TTS. Tudo é HDC puro.

Uso:
    python -m core.multi_agent_runner

Referência: sent.txt (361-396) — Múltiplas AGIs com culturas diferentes
"""

import os
import sys
import time
import json
import threading
import random
from pathlib import Path
from typing import Optional, List, Dict, Any

_NOVAMIND = str(Path(__file__).parent.parent)
if _NOVAMIND not in sys.path:
    sys.path.insert(0, _NOVAMIND)

import torch
import numpy as np

from core.integration import NovaMind


# ─── Perfis Neuroquímicos ──────────────────────────────────────

AGENT_PROFILES = {
    "cientista": {
        "name": "Cientista",
        "agent_id": "novamind_cientista",
        "description": "Explora padrões, formula hipóteses, busca conhecimento",
        "nm_profile": {"DA": 0.4, "NA": 0.3, "ACh": 0.9, "5HT": 0.3},
        "drives": {
            "explorar": 0.95,
            "questionar": 0.85,
            "publicar": 0.60,
            "construir": 0.30,
        },
    },
    "engenheiro": {
        "name": "Engenheiro",
        "agent_id": "novamind_engenheiro",
        "description": "Constrói ferramentas, otimiza sistemas, executa planos",
        "nm_profile": {"DA": 0.9, "NA": 0.6, "ACh": 0.4, "5HT": 0.4},
        "drives": {
            "explorar": 0.30,
            "questionar": 0.30,
            "publicar": 0.50,
            "construir": 0.95,
        },
    },
    "critico": {
        "name": "Crítico",
        "agent_id": "novamind_critico",
        "description": "Valida descobertas, testa hipóteses, encontra falhas",
        "nm_profile": {"DA": 0.2, "NA": 0.9, "ACh": 0.3, "5HT": 0.8},
        "drives": {
            "explorar": 0.20,
            "questionar": 0.70,
            "publicar": 0.40,
            "construir": 0.20,
        },
    },
}


class AGIAgent:
    """
    Uma AGI completa rodando em sua própria thread.

    Cada agente tem:
      - NovaMind completa (engine, memória, stream, planner, incubator...)
      - Perfil neuroquímico único que viesa TODAS as decisões
      - Acesso ao mesmo CollectiveUnconscious do MultiAgentRunner
      - Drives internos que competem via free energy
    """

    def __init__(
        self,
        profile: dict,
        collective_path: str,
        base_data_dir: str,
        device: str,
        hyperview_feeder=None,
        shared_engine=None,
        shared_memory=None,
        shared_collective=None,
    ):
        self.profile = profile
        self.agent_id = profile["agent_id"]
        self.nm_profile = profile["nm_profile"]
        self.drives = profile["drives"]
        self.base_data_dir = os.path.join(base_data_dir, self.agent_id)
        self.device = device
        self.hyperview_feeder = hyperview_feeder
        self.shared_engine = shared_engine
        self.shared_memory = shared_memory
        self.shared_collective = shared_collective

        os.makedirs(self.base_data_dir, exist_ok=True)

        # Cria NovaMind com caminhos exclusivos
        self.mind = NovaMind(device=device)

        # Sobrescreve caminhos para isolamento
        # (cada AGI tem seu stream, seu estado, seu planner)
        self._configure_paths()

        # Registra no coletivo (usa compartilhado se disponível)
        collective = self.mind.collective
        collective.register_agent(self.agent_id, {
            "name": self.profile["name"],
            "profile": self.profile["description"],
            "nm": self.nm_profile,
        })

        # Estado
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.step_count = 0
        self.last_validation_time = 0
        self.discoveries_shared = 0
        self.validations_done = 0

        print(f"[{self.agent_id}] {self.profile['name']}: {self.profile['description']}")

    def _configure_paths(self):
        """Configura paths exclusivos para cada AGI."""
        from core.episodic_stream import EpisodicStream
        stream_path = os.path.join(self.base_data_dir, "episodic_stream.stream")
        self.mind._modules["stream"] = EpisodicStream(
            stream_path, device=self.device)
        self.mind._initialized.add("stream")

        # Engine compartilhado (economiza VRAM)
        if self.shared_engine is not None:
            self.mind._modules["engine"] = self.shared_engine
            self.mind._initialized.add("engine")
        elif "engine" not in self.mind._modules:
            self.mind._lazy_init("engine")

        # Memory compartilhada (VRAM HEAVY — ~2GB)
        if self.shared_memory is not None:
            self.mind._modules["memory"] = self.shared_memory
            self.mind._initialized.add("memory")
        elif "memory" not in self.mind._modules:
            self.mind._lazy_init("memory")

        # Collective compartilhado (mesmo .hdv, mesmo lock)
        if self.shared_collective is not None:
            self.mind._modules["collective"] = self.shared_collective
            self.mind._initialized.add("collective")

        # Planner
        if "planner" not in self.mind._modules:
            self.mind._lazy_init("planner")

        # Hippocampal buffer
        if "hippocampal" not in self.mind._modules:
            self.mind._lazy_init("hippocampal")

        # Incubator precisa de engine + memory + planner
        if "incubator" not in self.mind._modules:
            self.mind._lazy_init("incubator")

        # Conecta callback de Eureka (evita que incubator crie collective próprio)
        incubator = self.mind._modules.get("incubator")
        if incubator:
            incubator.on_eureka = self._on_eureka
            # Usa collective compartilhado no incubator se disponível
            if self.shared_collective:
                incubator.collective = self.shared_collective

    def _on_eureka(self, event):
        """Quando descobre algo, compartilha no coletivo."""
        collective = self.shared_collective or self.mind.collective
        engine = self.shared_engine or self.mind.engine

        try:
            collective.share_discovery(
                agent_id=self.agent_id,
                concept_name=event.description.replace(" ", "_")[:40],
                concept_vector=event.bound_vector.float(),
                domain="discovery",
                description=event.description,
            )
            self.discoveries_shared += 1
            print(f"[{self.agent_id}] EUREKA compartilhado: {event.description}")

            # Alimenta HyperView
            if self.hyperview_feeder:
                self.hyperview_feeder({
                    "active": self.agent_id,
                    "text": f"EUREKA: {event.description}",
                    "fe": event.fe_drop,
                })
        except Exception as e:
            print(f"[{self.agent_id}] Erro compartilhando Eureka: {e}")

    def _tick_drive(self) -> dict:
        """
        Um "tick" interno: o drive mais forte dita a ação.

        Cada ciclo, o agente:
          1. Computa drive winner (o drive com maior urgência)
          2. Executa ação correspondente
          3. Atualiza perfil neuroquímico
        """
        self.step_count += 1

        # Injetas ruído nos drives para evitar loop fixo
        noisy_drives = {
            k: v + random.gauss(0, 0.05) for k, v in self.drives.items()
        }
        winner = max(noisy_drives, key=noisy_drives.get)

        if winner == "explorar":
            return self._action_explore()
        elif winner == "questionar":
            return self._action_question()
        elif winner == "publicar":
            return self._action_publish()
        elif winner == "construir":
            return self._action_build()
        return {"action": "idle", "description": ""}

    def _action_explore(self) -> dict:
        """Explora: faz bind aleatório de conceitos, busca Eureka."""
        try:
            engine = self.shared_engine or self.mind.engine
            memory = self.shared_memory or self.mind.memory

            # Pega 2-3 conceitos aleatórios da memória ou do coletivo
            collective = self.shared_collective or self.mind.collective
            all_concepts = collective.get_all_concepts()

            if len(all_concepts) >= 2:
                selected = random.sample(all_concepts, min(3, len(all_concepts)))
                result = engine.bind_together([c.vector.to(self.device) for c in selected])
                feat = engine.measure_free_energy(result)
                return {
                    "action": "explore",
                    "description": f"Bind de {len(selected)} conceitos",
                    "fe": feat,
                    "n_concepts": len(selected),
                }
        except Exception as e:
            return {"action": "explore", "description": f"erro: {e}", "fe": 1.0}

        return {"action": "explore", "description": "nada para explorar", "fe": 1.0}

    def _action_question(self) -> dict:
        """Questiona: busca conceitos pendentes de validação no coletivo."""
        try:
            collective = self.shared_collective or self.mind.collective
            pending = collective.get_pending_validations(self.agent_id)

            if pending:
                name, concept = pending[0]
                engine = self.shared_engine or self.mind.engine
                own_vec = engine.encode_text(f"validate_{name}")
                success = collective.validate_concept(name, own_vec.float())
                if success:
                    self.validations_done += 1
                    return {
                        "action": "validate",
                        "description": f"Validou: {name}",
                        "fe": 0.2,
                    }
            return {"action": "question", "description": "nada pendente", "fe": 0.8}
        except Exception as e:
            return {"action": "question", "description": f"erro: {e}", "fe": 1.0}

    def _action_publish(self) -> dict:
        """Publica: compartilha uma descoberta própria no coletivo."""
        try:
            engine = self.shared_engine or self.mind.engine
            collective = self.shared_collective or self.mind.collective

            # Gera um conceito sintético
            vec = torch.randn(engine.dim, device=self.device)
            vec = torch.sign(vec)
            name = f"discovery_{self.agent_id}_{self.step_count}"

            collective.share_discovery(
                agent_id=self.agent_id,
                concept_name=name,
                concept_vector=vec.float(),
                domain="synthesis",
                description=f"Descoberta sintetizada pelo {self.profile['name']}",
            )
            self.discoveries_shared += 1
            return {"action": "publish", "description": name, "fe": 0.1}
        except Exception as e:
            return {"action": "publish", "description": f"erro: {e}", "fe": 1.0}

    def _action_build(self) -> dict:
        """Constrói: adiciona ao planner uma tarefa de alto nível."""
        try:
            planner = self.mind.planner
            task_text = f"{self.profile['name']}: construir algo util com conceitos atuais"
            planner.submit(
                description=task_text,
                priority=0.7 + random.random() * 0.2,
                tags=[self.agent_id, "build"],
            )
            return {"action": "build", "description": task_text, "fe": 0.3}
        except Exception as e:
            return {"action": "build", "description": f"erro: {e}", "fe": 1.0}

    def step(self) -> dict:
        """Um passo completo do agente."""
        result = self._tick_drive()

        # Decai drives baseado no que fez
        action = result.get("action", "idle")
        if action in self.drives:
            self.drives[action] = max(0.1, self.drives[action] - 0.05)
        # Recupera drives não usados
        for k in self.drives:
            if k != action:
                self.drives[k] = min(1.0, self.drives[k] + 0.02)

        # Atualiza perfil NM
        self.nm_profile["ACh"] = max(0.0, min(1.0, self.nm_profile["ACh"] + random.gauss(0, 0.01)))
        self.nm_profile["DA"] = max(0.0, min(1.0, self.nm_profile["DA"] + random.gauss(0, 0.01)))

        # Alimenta HyperView
        if self.hyperview_feeder:
            self.hyperview_feeder({
                "points": [],
                "nm": self.nm_profile,
                "fe": result.get("fe", 0.5),
                "active": f"{self.agent_id}:{action}",
                "text": result.get("description", ""),
            })

        return result

    def start(self):
        """Inicia a thread do agente."""
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print(f"[{self.agent_id}] Thread iniciada")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _run_loop(self):
        """Loop principal do agente."""
        # Passos iniciais mais rápidos
        initial_steps = 5
        step_interval = 1.0  # segundos entre steps

        for i in range(initial_steps):
            if not self.running:
                return
            self.step()
            time.sleep(step_interval * 0.5)

        while self.running:
            self.step()
            time.sleep(step_interval)


class MultiAgentRunner:
    """
    Orquestrador de múltiplas AGIs.

    Gerencia:
      - Criação de agentes com perfis diferentes
      - Coletivo compartilhado (mesmo .hdv)
      - Ciclo de vida (start/stop)
      - HyperView feeder opcional
      - Log de atividade
    """

    def __init__(
        self,
        base_data_dir: str = "D:/temp/multi_agent",
        collective_path: str = "D:/temp/collective_unconscious.hdv",
        device: Optional[str] = None,
        profiles: Optional[List[str]] = None,
        hyperview_feeder=None,
    ):
        self.base_data_dir = base_data_dir
        self.collective_path = collective_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hyperview_feeder = hyperview_feeder

        os.makedirs(base_data_dir, exist_ok=True)

        # Perfis a instanciar
        profile_names = profiles or ["cientista", "engenheiro", "critico"]
        self.agents: Dict[str, AGIAgent] = {}

        # Cria engine e memória COMPARTILHADOS (HDCMemory ~2GB, não dá p/ ter 3)
        from core.grounding_engine import GroundingEngine
        self.shared_engine = GroundingEngine(device=torch.device(self.device))
        self.shared_memory = getattr(self.shared_engine, 'hdc_memory', None)
        if self.shared_memory is None:
            from core.hdc_memory import HDCMemory
            self.shared_memory = HDCMemory()

        # Collective compartilhado (mesmo lock, mesmo arquivo)
        from core.collective_unconscious import CollectiveUnconscious
        self.shared_collective = CollectiveUnconscious(path=self.collective_path)

        # Inicializa cada agente
        for pname in profile_names:
            if pname not in AGENT_PROFILES:
                print(f"[MultiAgent] Perfil desconhecido: {pname}")
                continue

            profile = AGENT_PROFILES[pname]
            agent = AGIAgent(
                profile=profile,
                collective_path=self.collective_path,
                base_data_dir=self.base_data_dir,
                device=self.device,
                hyperview_feeder=hyperview_feeder,
                shared_engine=self.shared_engine,
                shared_memory=self.shared_memory,
                shared_collective=self.shared_collective,
            )
            self.agents[pname] = agent

        print(f"\n[MultiAgent] {len(self.agents)} AGIs prontas:")
        for pname, agent in self.agents.items():
            print(f"  [{pname}] {agent.profile['name']} — {agent.profile['description']}")
        print(f"[MultiAgent] Coletivo: {self.collective_path}")

    def start(self):
        """Inicia TODAS as AGIs simultaneamente."""
        print(f"\n{'='*60}")
        print("[MultiAgent] INICIANDO MÚLTIPLAS AGIs")
        print(f"{'='*60}")

        for pname, agent in self.agents.items():
            agent.start()

        print(f"\n[MultiAgent] {len(self.agents)} AGIs rodando em paralelo")

    def stop(self):
        """Para TODAS as AGIs."""
        print("\n[MultiAgent] Parando todas as AGIs...")
        for pname, agent in self.agents.items():
            agent.stop()
        print("[MultiAgent] Todas paradas.")

    def get_summary(self) -> dict:
        """Resumo do estado de todas as AGIs."""
        summary = {}
        for pname, agent in self.agents.items():
            summary[pname] = {
                "name": agent.profile["name"],
                "agent_id": agent.agent_id,
                "nm_profile": agent.nm_profile,
                "drives": {k: round(v, 2) for k, v in agent.drives.items()},
                "steps": agent.step_count,
                "discoveries_shared": agent.discoveries_shared,
                "validations_done": agent.validations_done,
            }

        # Estado do coletivo
        first_agent = next(iter(self.agents.values()))
        collective = first_agent.mind.collective

        return {
            "n_agents": len(self.agents),
            "agents": summary,
            "collective": {
                "n_conceitos": collective.count(),
                "agentes_registrados": collective.get_agents(),
                "path": self.collective_path,
            },
            "device": self.device,
        }

    def print_summary(self):
        """Printa resumo formatado."""
        s = self.get_summary()
        print(f"\n{'='*60}")
        print("[MultiAgent] RESUMO DO SISTEMA")
        print(f"{'='*60}")
        print(f"Device: {s['device']}")
        print(f"AGIs ativas: {s['n_agents']}")
        print(f"Conceitos no coletivo: {s['collective']['n_conceitos']}")
        print(f"Agentes registrados: {s['collective']['agentes_registrados']}")
        print()
        for pname, info in s["agents"].items():
            print(f"  [{pname}] {info['name']}")
            print(f"          Steps: {info['steps']}")
            print(f"          Descobertas: {info['discoveries_shared']}")
            print(f"          Validações: {info['validations_done']}")
            print(f"          Drive atual: {max(info['drives'], key=info['drives'].get)}")
        print()


def run_multi_agent(
    duration: int = 60,
    profiles: Optional[List[str]] = None,
    hyperview_feeder=None,
):
    """
    Executa múltiplas AGIs por 'duration' segundos.

    Args:
        duration: tempo de execução em segundos
        profiles: lista de perfis (padrão: cientista, engenheiro, critico)
        hyperview_feeder: callback para alimentar HyperView

    Uso:
        from core.multi_agent_runner import run_multi_agent
        run_multi_agent(duration=120)
    """
    runner = MultiAgentRunner(
        profiles=profiles,
        hyperview_feeder=hyperview_feeder,
    )

    runner.start()

    try:
        import time
        start = time.time()
        while time.time() - start < duration:
            time.sleep(5)
            runner.print_summary()
    except KeyboardInterrupt:
        print("\n[MultiAgent] Interrompido pelo usuario")
    finally:
        runner.stop()
        runner.print_summary()

    return runner


def feed_hyperview(runner: MultiAgentRunner):
    """Alimenta HyperView com dados de todas as AGIs."""
    import requests

    def feeder(data: dict):
        try:
            requests.post("http://localhost:8768/api/feed", json=data, timeout=0.5)
        except:
            pass

    return feeder


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Multi-Agent NovaMind — Cientista + Engenheiro + Crítico")
    print("="*60)
    print()

    # Tenta conectar no HyperView (se estiver rodando)
    hv_feeder = None
    try:
        import requests
        r = requests.get("http://localhost:8768/api/state", timeout=1)
        if r.status_code == 200:
            print("[Main] HyperView detectado em :8768")
            hv_feeder = feed_hyperview(None)
    except:
        print("[Main] HyperView não detectado (opcional)")

    # Roda por 5 minutos
    run_multi_agent(duration=300, hyperview_feeder=hv_feeder)
