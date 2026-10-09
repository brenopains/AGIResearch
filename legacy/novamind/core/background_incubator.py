"""
BackgroundIncubator — Processamento Inconsciente de Fundo
===========================================================
Thread paralela que faz bind aleatório de conceitos HDC com tarefas
residuais. Monitora Free Energy. Se detectar queda brusca (Eureka!),
injeta solução na thread principal.

Fluxo:
  1. Incubator pega tarefa residual do ExecutivePlanner
  2. Faz bind aleatório com conceitos do HDV
  3. Mede FE da combinação
  4. Se FE cai abaixo do limiar → EUREKA → callback na thread principal
  5. Se não → continua bind aleatório em loop

Referência: sent.txt (417-427) — BackgroundIncubator
"""

import os
import json
import time
import math
import random
import threading
from typing import Optional, List, Dict, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import torch
import numpy as np


@dataclass
class EurekaEvent:
    timestamp: float
    source_task_id: str
    source_concept: str        # texto do conceito original
    bound_concept: str         # texto do conceito que gerou o insight
    fe_before: float           # FE antes do bind
    fe_after: float            # FE depois do bind (deve ser muito menor)
    fe_drop: float             # magnitude da queda
    bound_vector: torch.Tensor # vetor HDC resultante do bind
    description: str           # descrição textual do insight

    def __post_init__(self):
        if isinstance(self.bound_vector, torch.Tensor):
            self.bound_vector = self.bound_vector.cpu()


class BackgroundIncubator:
    """
    Thread de fundo que processa tarefas residuais.

    Args:
        hdc_memory: HDCMemory com conceitos para bind aleatório
        engine: GroundingEngine para codificar texto como HDC
        planner: ExecutivePlanner para pegar tarefas residuais
        fe_threshold: limiar para Eureka (FE < threshold)
        fe_drop_ratio: queda de FE mínima para considerar Eureka
        tick_interval: segundos entre cada bind tentativa
    """

    def __init__(
        self,
        hdc_memory=None,
        engine=None,
        planner=None,
        fe_threshold: float = 0.25,
        fe_drop_ratio: float = 0.5,
        tick_interval: float = 2.0,
        max_concurrent_binds: int = 100,
        device: Optional[str] = None,
    ):
        self.hdc_memory = hdc_memory
        self.engine = engine
        self.planner = planner
        self.fe_threshold = fe_threshold
        self.fe_drop_ratio = fe_drop_ratio
        self.tick_interval = tick_interval
        self.max_concurrent_binds = max_concurrent_binds
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._stop_flag = threading.Event()

        # Histórico de Eurekas
        self.eurekas: List[EurekaEvent] = []

        # Callback chamado quando Eureka acontece
        self.on_eureka: Optional[Callable[[EurekaEvent], None]] = None

        # Cache de FE de binds recentes (para detectar quedas)
        self._fe_cache: Dict[str, float] = {}

        # Pool de conceitos para bind
        self._concept_pool: List[Tuple[str, torch.Tensor]] = []
        self._pool_lock = threading.RLock()
        self._pool_last_refresh = 0.0
        self._pool_refresh_interval = 60.0  # recarrega pool a cada 60s

        print(f"[INCUBATOR] Inicializado (FE threshold={fe_threshold}, "
              f"drop_ratio={fe_drop_ratio}, tick={tick_interval}s)")

    # ─── Controle de Ciclo de Vida ───────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._incubation_loop, daemon=True)
        self._thread.start()
        print("[INCUBATOR] Background thread started")

    def stop(self):
        self._running = False
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        print(f"[INCUBATOR] Parado. {len(self.eurekas)} Eurekas detectados.")

    @property
    def is_running(self) -> bool:
        return self._running

    # ─── Pool de Conceitos ───────────────────────────────────────

    def refresh_concept_pool(self):
        """Recarrega o pool de conceitos do HDCMemory."""
        if self.hdc_memory is None:
            return

        now = time.time()
        if now - self._pool_last_refresh < self._pool_refresh_interval:
            return

        with self._pool_lock:
            concepts = []
            try:
                # Tenta buscar conceitos do HDCMemory
                if hasattr(self.hdc_memory, 'get_all_concepts'):
                    concepts_raw = self.hdc_memory.get_all_concepts(limit=1000)
                    for name, vec in concepts_raw:
                        if isinstance(vec, torch.Tensor):
                            concepts.append((name, vec.cpu()))
                elif hasattr(self.hdc_memory, 'brains'):
                    # Fallback: itera sobre brains
                    for brain_name, brain in self.hdc_memory.brains.items():
                        for item in brain.get_all()[:500]:
                            concepts.append((f"{brain_name}:{item.get('name','?')}",
                                             torch.tensor(item.get('vector', []))))
            except Exception as e:
                print(f"[INCUBATOR] Erro refresh pool: {e}")

            if concepts:
                self._concept_pool = concepts
                self._pool_last_refresh = now
                print(f"[INCUBATOR] Pool refreshed: {len(concepts)} conceitos")

    def _get_random_concepts(self, n: int = 5) -> List[Tuple[str, torch.Tensor]]:
        """Pega N conceitos aleatórios do pool."""
        self.refresh_concept_pool()
        with self._pool_lock:
            if not self._concept_pool:
                return []
            return random.sample(self._concept_pool, min(n, len(self._concept_pool)))

    # ─── Loop de Incubação ───────────────────────────────────────

    def _incubation_loop(self):
        """Loop principal: pega tarefas residuais, faz bind aleatório, detecta Eureka."""
        print("[INCUBATOR] Loop de incubação iniciado")

        while not self._stop_flag.is_set():
            try:
                self._incubation_tick()
            except Exception as e:
                print(f"[INCUBATOR] Erro no tick: {e}")

            time.sleep(self.tick_interval)

    def _incubation_tick(self):
        """Um tick de incubação: pega tarefa residual e tenta bind."""
        task = self._get_residual_task()
        if task is None:
            return  # sem tarefas residuais, dorme

        # Pega conceitos aleatórios do pool
        random_concepts = self._get_random_concepts(3)
        if not random_concepts:
            return

        # Codifica a tarefa como vetor HDC
        task_vec = self._encode_task(task)

        if task_vec is None:
            return

        task_fe_before = task.get("fe_before", 0.5)

        # Tenta binds aleatórios
        for concept_name, concept_vec in random_concepts:
            if self._stop_flag.is_set():
                return

            if not isinstance(concept_vec, torch.Tensor):
                continue

            concept_vec = concept_vec.to(self.device)
            tvec = task_vec.to(self.device)

            # Faz bind: task * concept (HDC binding = element-wise multiplication)
            # Para HDC bipolar: bind = sign(task * concept)
            bound = torch.sign(tvec * concept_vec)

            # Mede FE do bind
            fe_after = self._measure_fe(bound, tvec, concept_vec)
            fe_before = task_fe_before
            fe_drop = fe_before - fe_after

            # Cache para detecção de tendência
            cache_key = f"{task.get('id','?')}_{concept_name}"
            self._fe_cache[cache_key] = fe_after

            # Limpa cache velho
            if len(self._fe_cache) > self.max_concurrent_binds * 10:
                self._fe_cache.clear()

            # Detecta Eureka!
            if fe_after < self.fe_threshold and fe_drop > self.fe_threshold * self.fe_drop_ratio:
                self._fire_eureka(
                    source_task_id=task.get("id", "?"),
                    source_concept=task.get("description", ""),
                    bound_concept=concept_name,
                    fe_before=fe_before,
                    fe_after=fe_after,
                    fe_drop=fe_drop,
                    bound_vector=bound,
                )
                # Depois de um Eureka, pausa para não floodar
                time.sleep(5.0)
                return

            # Se FE está aumentando, tenta outro conceito
            if fe_after > fe_before:
                continue

    def _get_residual_task(self) -> Optional[dict]:
        """Pega tarefa residual do planner (falhou, pendente, ou em pausa)."""
        if self.planner is None:
            # Fallback: gera uma tarefa dummy para incubação
            return {
                "id": "dummy",
                "description": "explorar conceitos aleatorios",
                "fe_before": 0.5,
                "status": "pending",
            }

        try:
            # Busca tarefas que falharam (alta FE residual)
            failed = self.planner.list_tasks(status=None)
            # Filtra tasks com FE alta e não completadas
            candidates = [
                t for t in failed
                if t.status.value in ("failed", "pending", "paused")
                and t.fe_before > 0.3
            ]
            if candidates:
                chosen = max(candidates, key=lambda t: t.fe_before)
                return {
                    "id": chosen.id,
                    "description": chosen.description,
                    "fe_before": chosen.fe_before,
                    "status": chosen.status.value,
                }
        except Exception as e:
            print(f"[INCUBATOR] Erro buscando tarefa residual: {e}")

        return None

    def _encode_task(self, task: dict) -> Optional[torch.Tensor]:
        """Codifica descrição da tarefa como vetor HDC."""
        text = task.get("description", "")
        if not text:
            return None

        if self.engine and hasattr(self.engine, 'encode_text'):
            try:
                mf = self.engine.encode_text(text)
                return mf.tensor.float()
            except Exception:
                pass

        # Fallback: embedding aleatório
        return torch.randn(10000, device=self.device)

    def _measure_fe(
        self, bound_vec: torch.Tensor,
        task_vec: torch.Tensor,
        concept_vec: torch.Tensor,
    ) -> float:
        """
        Mede Free Energy do bind.
        FE = 1 - cosine_similarity(bound, task) * cosine_similarity(bound, concept)
        Um bind bom mantém similaridade com ambos.
        """
        b = bound_vec.flatten().float()
        t = task_vec.flatten().float()
        c = concept_vec.flatten().float()

        cos_bt = torch.nn.functional.cosine_similarity(b.unsqueeze(0), t.unsqueeze(0))
        cos_bc = torch.nn.functional.cosine_similarity(b.unsqueeze(0), c.unsqueeze(0))

        # FE baixa = bound mantém info de ambos
        coherence = float((cos_bt.item() + cos_bc.item()) / 2.0)
        fe = 1.0 - coherence
        return max(0.0, min(1.0, fe))

    # ─── Eureka! ─────────────────────────────────────────────────

    def _fire_eureka(
        self,
        source_task_id: str,
        source_concept: str,
        bound_concept: str,
        fe_before: float,
        fe_after: float,
        fe_drop: float,
        bound_vector: torch.Tensor,
    ):
        """Dispara evento Eureka."""
        desc = (
            f"EUREKA! Enquanto processava \"{source_concept[:40]}\", "
            f"o bind com \"{bound_concept[:40]}\" "
            f"reduziu a FE de {fe_before:.2f} para {fe_after:.2f} "
            f"(queda de {fe_drop:.2f})."
        )

        event = EurekaEvent(
            timestamp=time.time(),
            source_task_id=source_task_id,
            source_concept=source_concept,
            bound_concept=bound_concept,
            fe_before=fe_before,
            fe_after=fe_after,
            fe_drop=fe_drop,
            bound_vector=bound_vector,
            description=desc,
        )

        with self._lock:
            self.eurekas.append(event)

        print(f"\n{'='*60}")
        print(f"[INCUBATOR] {desc}")
        print(f"{'='*60}")

        # Callback para thread principal
        if self.on_eureka:
            try:
                self.on_eureka(event)
            except Exception as e:
                print(f"[INCUBATOR] Erro no callback Eureka: {e}")

    # ─── API de Consulta ─────────────────────────────────────────

    def get_recent_eurekas(self, n: int = 5) -> List[EurekaEvent]:
        with self._lock:
            return self.eurekas[-n:]

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "running": self._running,
                "eurekas_detected": len(self.eurekas),
                "pool_size": len(self._concept_pool),
                "fe_cache_size": len(self._fe_cache),
                "tick_interval": self.tick_interval,
                "fe_threshold": self.fe_threshold,
            }

    def clear_eurekas(self):
        with self._lock:
            self.eurekas.clear()
