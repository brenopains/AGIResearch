"""
Integração — Plugs todos os módulos no sistema AGI completo
==============================================================
Conecta:
  - SelfImproveLoop → ciclo principal de melhoria
  - PerceptualTargetMatcher → visão por HDC contra referência
  - ExecutivePlanner → fila de tarefas com prioridade
  - BackgroundIncubator → processamento inconsciente + Eureka
  - CollectiveUnconscious → cultura compartilhada entre AGIs
  - BrocaV2 → geração de texto fluente
  - EpisodicStream → memória episódica infinita
  - ContinuousFluxEngine → percepção contínua em tempo real
  - HippocampalBuffer → memória de trabalho circular

Uso:
    from core.integration import NovaMind
    mind = NovaMind()
    mind.start()
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import torch
import numpy as np

# Garante path do novamind
_NOVAMIND = str(Path(__file__).parent.parent)
if _NOVAMIND not in sys.path:
    sys.path.insert(0, _NOVAMIND)


class NovaMind:
    """
    Singleton que integra todos os módulos da AGI.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[NovaMind] Inicializando no device: {self.device}")
        self._modules = {}
        self._initialized = set()
        self._running = False

    # ─── Inicialização Preguiçosa ────────────────────────────────

    def _lazy_init(self, name: str):
        """Inicializa módulo sob demanda."""
        if name in self._initialized:
            return self._modules.get(name)

        init_map = {
            "engine": self._init_engine,
            "memory": self._init_memory,
            "stream": self._init_stream,
            "hippocampal": self._init_hippocampal,
            "planner": self._init_planner,
            "incubator": self._init_incubator,
            "collective": self._init_collective,
            "loop": self._init_loop,
            "ptm": self._init_ptm,
            "flux": self._init_flux,
            "broca": self._init_broca,
            "composer": self._init_broca,
            "context_binder": self._init_context_binder,
            "sandbox": self._init_sandbox,
            "visual_renderer": self._init_visual_renderer,
            "audio_vocoder": self._init_audio_vocoder,
            "maestro": self._init_maestro,
            "neurogen": self._init_neurogen,
            "offload": self._init_offload,
        }

        if name in init_map:
            self._modules[name] = init_map[name]()
            self._initialized.add(name)
            return self._modules.get(name)

        raise KeyError(f"Módulo desconhecido: {name}")

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._lazy_init(name)
        except KeyError:
            raise AttributeError(f"NovaMind não tem módulo: {name}")

    # ─── Init de cada módulo ─────────────────────────────────────

    def _init_engine(self):
        from core.grounding_engine import GroundingEngine
        engine = GroundingEngine(device=torch.device(self.device))
        print("[NovaMind] GroundingEngine pronto")
        return engine

    def _init_memory(self):
        from core.hdc_memory import HDCMemory
        mem = HDCMemory()
        print("[NovaMind] HDCMemory pronto")
        return mem

    def _init_stream(self):
        from core.episodic_stream import EpisodicStream
        stream = EpisodicStream("D:/temp/episodic_stream.stream", device=self.device)
        print(f"[NovaMind] EpisodicStream pronto ({len(stream)} entradas)")
        return stream

    def _init_hippocampal(self):
        from core.hippocampal_buffer import HippocampalBuffer
        buf = HippocampalBuffer(capacity_seconds=13, fps=15)
        print("[NovaMind] HippocampalBuffer pronto")
        return buf

    def _init_planner(self):
        engine = self._lazy_init("engine")
        from core.executive_planner import ExecutivePlanner
        planner = ExecutivePlanner(engine=engine)
        print("[NovaMind] ExecutivePlanner pronto")
        return planner

    def _init_incubator(self):
        engine = self._lazy_init("engine")
        memory = self._lazy_init("memory")
        planner = self._lazy_init("planner")
        from core.background_incubator import BackgroundIncubator
        incubator = BackgroundIncubator(
            hdc_memory=memory,
            engine=engine,
            planner=planner,
            fe_threshold=0.25,
            tick_interval=3.0,
        )
        print("[NovaMind] BackgroundIncubator pronto")
        return incubator

    def _init_collective(self):
        from core.collective_unconscious import CollectiveUnconscious
        collective = CollectiveUnconscious()
        print(f"[NovaMind] CollectiveUnconscious pronto ({collective.count()} conceitos)")
        return collective

    def _init_loop(self):
        from hyperview.backend.self_improve_loop import MonitoredLoop
        loop = MonitoredLoop()
        print("[NovaMind] MonitoredLoop pronto")
        return loop

    def _init_ptm(self):
        engine = self._lazy_init("engine")
        planner = self._lazy_init("planner")
        from core.perceptual_target_matcher import PerceptualTargetMatcher
        ptm = PerceptualTargetMatcher(engine=engine, fe_threshold=0.25)
        print("[NovaMind] PerceptualTargetMatcher pronto")
        return ptm

    def _init_flux(self):
        from core.continuous_flux import ContinuousFluxEngine
        flux = ContinuousFluxEngine(target_fps=15, buffer_seconds=5)
        print("[NovaMind] ContinuousFluxEngine pronto")
        return flux

    def _init_broca(self):
        """Compositor Semântico Puramente HDC — substitui Broca V2 (LSTM)."""
        engine = self._lazy_init("engine")
        memory = getattr(engine, 'hdc_memory', None) or self._lazy_init("memory")
        from core.semantic_composition import SemanticCompositionEngine
        comp = SemanticCompositionEngine(hdc_memory=memory, engine=engine,
                                         device=self.device)
        print("[NovaMind] SemanticCompositionEngine pronto (HDC puro, sem LLM)")
        return comp

    def _init_sandbox(self):
        engine = self._lazy_init("engine")
        from core.physical_sandbox import PhysicalSandbox
        sb = PhysicalSandbox(engine=engine)
        print("[NovaMind] PhysicalSandbox pronto")
        return sb

    def _init_visual_renderer(self):
        from core.visual_renderer import VisualRenderer
        vr = VisualRenderer()
        if os.path.exists("D:/temp/visual_renderer/vqvae.pt"):
            vr.load_pretrained("D:/temp/visual_renderer/vqvae.pt")
        print("[NovaMind] VisualRenderer pronto")
        return vr

    def _init_audio_vocoder(self):
        from core.audio_vocoder import AudioVocoder
        av = AudioVocoder()
        if os.path.exists("D:/temp/audio_vocoder/vocoder.pt"):
            av.load_pretrained("D:/temp/audio_vocoder/vocoder.pt")
        print("[NovaMind] AudioVocoder pronto")
        return av

    def _init_maestro(self):
        engine = self._lazy_init("engine")
        from core.fractal_maestro import FractalMaestro, NaryBinder, AutoModuleCreator
        binder = NaryBinder()
        maestro = FractalMaestro(engine=engine, binder=binder)
        print("[NovaMind] FractalMaestro pronto")
        self._modules["binder"] = binder
        return maestro

    def _init_neurogen(self):
        memory = self._lazy_init("memory")
        engine = self._lazy_init("engine")
        from core.fractal_maestro import AutoModuleCreator
        ng = AutoModuleCreator(hdc_memory=memory, engine=engine)
        # Registra módulos existentes
        for brain_name in ["code_brain", "mega_brain", "msvd_brain", "python_brain", "text_library"]:
            domain = brain_name.replace("_brain", "")
            ng.register_module(brain_name, domain,
                               f"D:/novamind_data/brains/{brain_name}.hdv")
        print("[NovaMind] AutoModuleCreator pronto")
        return ng

    def _init_context_binder(self):
        engine = self._lazy_init("engine")
        from core.emergent_context import EmergentContextBinder
        ctx = EmergentContextBinder(engine=engine, device=self.device)
        print("[NovaMind] EmergentContextBinder pronto")
        return ctx

    def _init_offload(self):
        from core.cognitive_offloading import CognitiveOffloading
        off = CognitiveOffloading()
        print("[NovaMind] CognitiveOffloading pronto")
        return off

    # ─── Controle de Ciclo de Vida ───────────────────────────────

    def start(self):
        """Inicia todos os módulos de background."""
        if self._running:
            print("[NovaMind] Ja rodando")
            return

        print("\n" + "="*60)
        print("[NovaMind] INICIANDO SISTEMA COMPLETO")
        print("="*60)

        self._running = True

        # Inicia planner
        planner = self._lazy_init("planner")
        planner.start_background(on_complete=self._on_task_complete)
        print("[NovaMind] Planner background ativo")

        # Inicia incubator
        incubator = self._lazy_init("incubator")
        incubator.on_eureka = self._on_eureka
        incubator.start()
        print("[NovaMind] Incubator background ativo")

        # Registra esta AGI no coletivo
        collective = self._lazy_init("collective")
        collective.register_agent("novamind_0", {"name": "NovaMind 0"})
        print("[NovaMind] Registrado no CollectiveUnconscious")

        print("="*60)
        print("[NovaMind] SISTEMA OPERACIONAL")
        print("="*60)

    def stop(self):
        """Para todos os módulos de background."""
        if not self._running:
            return

        print("\n[NovaMind] Parando sistema...")

        if "incubator" in self._modules:
            self._modules["incubator"].stop()

        if "planner" in self._modules:
            self._modules["planner"].stop_background()

        if "flux" in self._modules:
            self._modules["flux"].stop()

        self._running = False
        print("[NovaMind] Sistema parado.")

    # ─── Callbacks ───────────────────────────────────────────────

    def _on_task_complete(self, task):
        """Callback quando uma tarefa do planner é completada."""
        collective = self._lazy_init("collective")
        engine = self._lazy_init("engine")

        # Se a tarefa foi bem sucedida, compartilha no coletivo
        if hasattr(task, 'status') and task.status.value == "completed":
            try:
                mf = engine.encode_text(task.description)
                collective.share_discovery(
                    agent_id="novamind_0",
                    concept_name=f"task_result_{task.id}",
                    concept_vector=mf.tensor.float(),
                    domain="task",
                    description=task.description,
                )
            except Exception as e:
                print(f"[NovaMind] Erro compartilhando descoberta: {e}")

    def _on_eureka(self, event):
        """Callback quando o incubator detecta um Eureka."""
        print(f"\n[NovaMind] *** EUREKA DETECTADO ***")
        print(f"[NovaMind] {event.description}")

        # Salva no stream
        try:
            stream = self._lazy_init("stream")
            hdv = event.bound_vector.float()
            stream.append(hdv.cpu(), {
                "type": "eureka",
                "source": event.source_concept,
                "bound_with": event.bound_concept,
                "fe_drop": event.fe_drop,
                "description": event.description,
            }, event.timestamp)
        except Exception as e:
            print(f"[NovaMind] Erro logando Eureka: {e}")

        # Cria tarefa de alta prioridade para validar
        try:
            planner = self._lazy_init("planner")
            planner.submit(
                description=f"Validar Eureka: {event.description}",
                priority=0.95,
                tags=["eureka", "urgent"],
            )
        except Exception as e:
            print(f"[NovaMind] Erro criando tarefa de validacao: {e}")

    # ─── API de Consulta ─────────────────────────────────────────

    def get_status(self) -> dict:
        status = {
            "running": self._running,
            "device": self.device,
            "modules": list(self._initialized),
        }

        if "planner" in self._modules:
            status["planner"] = self._modules["planner"].get_summary()

        if "incubator" in self._modules:
            status["incubator"] = self._modules["incubator"].get_stats()

        if "collective" in self._modules:
            status["collective"] = self._modules["collective"].get_summary()

        if "stream" in self._modules:
            status["stream_entries"] = len(self._modules["stream"])

        if "flux" in self._modules:
            status["flux"] = self._modules["flux"].get_state()

        return status

    def submit_task(self, description: str, **kwargs) -> str:
        """Submete tarefa para execução assíncrona."""
        planner = self._lazy_init("planner")
        return planner.submit(description, **kwargs)

    def get_task_status(self, task_id: str) -> str:
        planner = self._lazy_init("planner")
        return planner.get_status_text(task_id)

    def summarize(self) -> str:
        """Gera resumo textual completo do estado do sistema."""
        parts = [
            "=== NovaMind System Status ===",
            f"Device: {self.device}",
            f"Running: {self._running}",
            f"Modulos carregados: {len(self._initialized)}",
        ]

        if "collective" in self._modules:
            c = self._modules["collective"]
            parts.append(f"Inconsciente Coletivo: {c.count()} conceitos")

        if "planner" in self._modules:
            p = self._modules["planner"]
            parts.append(f"Planner: {p.total_tasks} tarefas, "
                        f"{p.completed_tasks} completadas, "
                        f"{p.failed_tasks} falhas, "
                        f"{p.queue_length()} na fila")

        if "incubator" in self._modules:
            i = self._modules["incubator"]
            parts.append(f"Incubator: {len(i.get_recent_eurekas())} Eurekas detectados")

        if "stream" in self._modules:
            parts.append(f"EpisodicStream: {len(self._modules['stream'])} entradas")

        if "broca" in self._modules:
            parts.append("SemanticCompositionEngine: ativo (HDC puro)")
        if "context_binder" in self._modules:
            ctx = self._modules["context_binder"]
            parts.append(f"ContextBinder: {len(ctx.contexts)} dominios")

        return "\n".join(parts)
