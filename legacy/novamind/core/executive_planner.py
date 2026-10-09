"""
ExecutivePlanner — Córtex Pré-Frontal: Decomposição Recursiva Autônoma
========================================================================
O humano dá MISSÕES, não prompts. O Planner assume o controle total.

Fluxo Autônomo:
  1. define_mission(texto) → recebe o problema
  2. process_active_goal() → tenta resolver direto (FE baixa?)
  3. Se FE alta → decompose_goal() → busca sub-conceitos no HDV
  4. Para cada sub-conceito → CRIA SUB-TAREFA → recursão
  5. Quando acha solução na folha → execute_action() → sobe (backtracking)
  6. O Maestro funde a solução da sub-tarefa com o objetivo pai
  7. Processo repete até missão completa ou irredutível

NENHUM HUMANO aponta o caminho. A AGI desce a árvore do conhecimento
sozinha até achar a folha que sabe deduzir, e sobe costurando a solução.

Referência: "A SUA INTUIÇÃO ACABOU DE CORRIGIR A.txt" + sent.txt
"""

import os
import json
import time
import math
import threading
import traceback
from typing import Optional, List, Dict, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import deque

import torch
import numpy as np

DIM = 10000


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    id: str
    description: str
    intention: str                # vetor HDC da intenção (texto ou embed)
    priority: float = 0.5         # 0-1, calculado de FE + DA esperada
    estimated_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    progress: float = 0.0         # 0-1
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    parent_id: Optional[str] = None  # sub-tarefa de
    sub_tasks: List[str] = field(default_factory=list)
    result: Optional[dict] = None
    error: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    fe_before: float = 0.0        # Free Energy antes da tarefa
    fe_after: float = 0.0         # Free Energy depois

    @property
    def da_expected(self) -> float:
        """Dopamina esperada: inverso da FE + progresso."""
        return (1.0 - self.fe_before) * 0.5 + self.progress * 0.5

    @property
    def priority_score(self) -> float:
        """Score combinado: FE alta + DA esperada alta = urgente."""
        return self.fe_before * 0.6 + self.da_expected * 0.4


class ExecutivePlanner:
    """
    Gerencia fila de tarefas com prioridade dinâmica.

    Estima tempo baseado em complexidade do prompt.
    Executa tarefas em background.
    Persiste estado no disco.
    """

    def __init__(
        self,
        engine=None,           # GroundingEngine para codificar intenções
        executor: Optional[Callable] = None,  # função que executa tarefas
        persist_path: str = "D:/temp/executive_planner/tasks.json",
        device: Optional[str] = None,
    ):
        self.engine = engine
        self.executor = executor
        self.persist_path = persist_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(os.path.dirname(persist_path), exist_ok=True)

        self._tasks: Dict[str, Task] = {}   # id → Task
        self._queue: List[str] = []          # ordered list of ids
        self._running_id: Optional[str] = None
        self._lock = threading.RLock()
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._on_complete: Optional[Callable] = None
        self._on_eureka: Optional[Callable] = None

        # Estatísticas
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_estimated_time = 0.0
        self.total_elapsed_time = 0.0

        # Carregar estado persistido
        self._load()

    # ─── API Pública ─────────────────────────────────────────────

    def submit(
        self,
        description: str,
        intention: Optional[str] = None,
        intention_vec: Optional[torch.Tensor] = None,
        priority: Optional[float] = None,
        parent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Submete uma nova tarefa.

        Args:
            description: descrição textual da tarefa
            intention: texto da intenção (opcional, codificado em HDC)
            intention_vec: vetor HDC direto (opcional)
            priority: prioridade 0-1 (auto-estimado se None)
            parent_id: ID da tarefa pai (para sub-tarefas)
            tags: tags para categorização

        Returns:
            task_id: ID da tarefa criada
        """
        task_id = f"task_{int(time.time())}_{self.total_tasks:04d}"
        self.total_tasks += 1

        # Codifica intenção se for texto
        if intention and self.engine and intention_vec is None:
            try:
                mf = self.engine.encode_text(intention)
                intention_vec = mf.tensor.float()
            except Exception:
                pass

        # Estima tempo
        estimated = self._estimate_time(description, tags)

        # FE antes (auto-estimada pelo tamanho da intenção)
        fe_before = self._estimate_fe(description, intention_vec)

        task = Task(
            id=task_id,
            description=description,
            intention=intention or description,
            priority=priority or fe_before * 0.6 + 0.2,
            estimated_seconds=estimated,
            created_at=time.time(),
            parent_id=parent_id,
            tags=tags or [],
            fe_before=fe_before,
        )

        with self._lock:
            self._tasks[task_id] = task
            self._queue.append(task_id)
            self._sort_queue()
            self._save()

        print(f"[PLANNER] Tarefa criada: {task_id}")
        print(f"[PLANNER]   Descrição: {description[:80]}")
        print(f"[PLANNER]   Tempo estimado: {self._format_time(estimated)}")
        print(f"[PLANNER]   Prioridade: {task.priority:.2f}")
        print(f"[PLANNER]   FE: {fe_before:.2f}")

        return task_id

    # ─── Decomposição Recursiva Autônoma ──────────────────────

    def define_mission(self, goal_text: str) -> str:
        """
        Recebe uma MISSÃO (não um prompt) e inicia a decomposição autônoma.

        A AGI desce a árvore do conhecimento sozinha até achar a solução.
        Nenhum humano aponta o caminho.
        """
        print(f"\n[PLANNER] === MISSÃO RECEBIDA ===")
        print(f"[PLANNER] Objetivo: {goal_text}")

        task_id = self.submit(
            description=goal_text,
            intention=goal_text,
            priority=1.0,
            tags=["mission", "autonomous"],
        )
        return task_id

    def process_active_goal(self, task_id: Optional[str] = None) -> Optional[torch.Tensor]:
        """
        Processa recursivamente: tenta resolver, mede FE, decompõe se falhar.

        Algoritmo:
          1. Pega o objetivo ativo (topo da pilha)
          2. Maestro tenta resolver direto (System 1)
          3. Se FE < 0.25 → SOLUÇÃO ENCONTRADA → executa e sobe
          4. Se FE >= 0.25 → DECOMPOSIÇÃO → busca sub-conceitos no HDV
          5. Para cada sub-conceito → CRIA SUB-TAREFA → recursão
          6. Quando sub-tarefa resolvida → funde com objetivo pai → tenta de novo

        Returns:
            Tensor HDC da solução, ou None se irredutível
        """
        if task_id is None:
            if not self._queue:
                return None
            task_id = self._queue[-1]

        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None

        print(f"\n[PLANNER] Processando: {task.description[:80]} "
              f"(FE atual: {task.fe_before:.2f})")

        # 1. TENTA RESOLVER DIRETO (System 1)
        engine = self.engine
        if engine is None:
            return None

        intention_vec = self._get_intention_vec(task)
        if intention_vec is None:
            return None

        candidate = self._synthesize_solution(intention_vec)
        fe = self._measure_fe(intention_vec, candidate)

        task.fe_before = fe

        # 2. SE FE BAIXA → SOLUÇÃO ENCONTRADA
        if fe < 0.25:
            print(f"[PLANNER] Solução deduzida com alta confiança (FE={fe:.2f})")
            task.fe_after = fe
            with self._lock:
                task.status = TaskStatus.COMPLETED
                self.completed_tasks += 1
            return candidate

        # 3. SE FE ALTA → DECOMPOSIÇÃO
        print(f"[PLANNER] FE alta ({fe:.2f}). Decompondo o problema...")
        return self._decompose_goal(task, intention_vec)

    def _decompose_goal(self, task: Any,
                        intention_vec: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Busca no HDV os sub-conceitos que compõem o problema.
        Para cada um, cria sub-tarefa e desce recursivamente.
        """
        engine = self.engine
        if engine is None:
            return None

        # Codifica intenção como texto para query na memória
        memory = getattr(engine, 'hdc_memory', None)
        if memory is None:
            return None

        try:
            sub_concepts = memory.query_vec(intention_vec, top_k=3, min_sim=0.15)
        except Exception:
            sub_concepts = memory.query(task.description, top_k=3, min_sim=0.15)

        if not sub_concepts or sub_concepts[0][0] < 0.10:
            print(f"[PLANNER] Sem conhecimento para decompor. Fim da arvore.")
            return None

        for sim, meta, source in sub_concepts:
            sub_text = meta.get("text", meta.get("id", "unknown"))
            print(f"[PLANNER] Sub-conceito: {sub_text[:60]} (sim={sim:.3f})")

            sub_id = self.submit(
                description=f"Resolver: {sub_text[:100]}",
                intention=sub_text,
                parent_id=task.id,
                tags=task.tags + ["subgoal"],
            )

            with self._lock:
                if task.id in self._tasks:
                    self._tasks[task.id].sub_tasks.append(sub_id)

            result = self.process_active_goal(sub_id)

            if result is not None:
                print(f"[PLANNER] Sub-tarefa resolvida. Fundindo com objetivo pai...")
                merged = torch.sign(intention_vec + result)
                return self.process_active_goal(task.id)

        return None

    def _get_intention_vec(self, task: Any) -> Optional[torch.Tensor]:
        """Obtém o vetor de intenção de uma tarefa."""
        if hasattr(task, 'intention_vec') and task.intention_vec is not None:
            return task.intention_vec.flatten().to(self.device)
        engine = self.engine
        if engine is None:
            return None
        try:
            mf = engine.encode_text(task.intention or task.description)
            return mf.tensor.flatten().to(self.device)
        except Exception:
            return None

    def _synthesize_solution(self, intention_vec: torch.Tensor) -> torch.Tensor:
        """
        Tenta sintetizar uma solução:
          - Se tem engine, faz bind de conceitos similares
          - Senão, retorna o próprio vetor de intenção ruidoso
        """
        engine = self.engine
        if engine is None:
            return intention_vec

        try:
            memory = getattr(engine, 'hdc_memory', None)
            if memory is None:
                return torch.sign(intention_vec + torch.randn(DIM, device=self.device) * 0.1)

            results = memory.query_vec(intention_vec, top_k=3, min_sim=0.1)
            if results and len(results) >= 2:
                vecs = []
                for _, meta, _ in results[:2]:
                    v = meta.get("_vec")
                    if v is not None:
                        vecs.append(v.flatten().to(self.device))
                if len(vecs) >= 2:
                    return torch.sign(vecs[0] * vecs[1] * intention_vec)

            return torch.sign(intention_vec + torch.randn(DIM, device=self.device) * 0.05)
        except Exception:
            return torch.sign(intention_vec + torch.randn(DIM, device=self.device) * 0.05)

    def _measure_fe(self, intention: torch.Tensor,
                    candidate: torch.Tensor) -> float:
        """FE = 1 − cosine(intention, candidate)."""
        i = intention.flatten().float()
        c = candidate.flatten().float()
        cos = float(torch.cosine_similarity(i.unsqueeze(0), c.unsqueeze(0)))
        return 1.0 - cos

    def _estimate_time(self, description: str, tags: Optional[List[str]] = None) -> float:
        """
        Estima tempo baseado em heurísticas:
        - Palavras-chave no prompt
        - Tags
        - Tamanho da descrição
        """
        desc_lower = description.lower()

        # Heurísticas de complexidade
        if any(w in desc_lower for w in ["video", "vídeo", "reference", "comparar"]):
            base = 3600  # 1h
        elif any(w in desc_lower for w in ["train", "treinar", "compile"]):
            base = 7200  # 2h
        elif any(w in desc_lower for w in ["cancer", "câncer", "genetic", "proteína"]):
            base = 14400  # 4h
        elif any(w in desc_lower for w in ["game", "jogo", "pygame", "render"]):
            base = 900  # 15min
        elif any(w in desc_lower for w in ["search", "buscar", "query"]):
            base = 120  # 2min
        elif any(w in desc_lower for w in ["code", "código", "script"]):
            base = 300  # 5min
        else:
            base = 600  # 10min default

        # Ajusta por tamanho da descrição
        length_factor = min(3.0, len(description) / 500)
        estimated = base * (0.5 + length_factor * 0.5)

        # Ajusta por tags
        if tags:
            if "heavy" in tags or "complex" in tags:
                estimated *= 2.0
            if "quick" in tags or "simple" in tags:
                estimated *= 0.3

        # Ajusta por histórico
        if self.completed_tasks > 0:
            avg_time = self.total_elapsed_time / max(1, self.completed_tasks)
            estimated = estimated * 0.7 + avg_time * 0.3

        return max(10.0, estimated)

    def _estimate_fe(self, description: str, intention_vec: Optional[torch.Tensor] = None) -> float:
        """
        Estima Free Energy inicial baseado na incerteza da tarefa.
        """
        desc_lower = description.lower()
        uncertainty = 0.3  # baseline

        # Palavras que indicam incerteza alta
        high_uncertainty = ["unknown", "desconhecido", "new", "novo", "never",
                            "nunca", "complex", "complexo", "difícil", "hard"]
        for w in high_uncertainty:
            if w in desc_lower:
                uncertainty += 0.1
                break

        medium_uncertainty = ["maybe", "talvez", "try", "tentar", "search",
                              "buscar", "find", "encontrar"]
        for w in medium_uncertainty:
            if w in desc_lower:
                uncertainty += 0.05
                break

        # Se tem intenção vetorial, usa similaridade com zero como FE
        if intention_vec is not None:
            try:
                v = intention_vec.flatten().float()
                norm = v.norm()
                if norm > 0:
                    # Quanto menor a norma, maior a incerteza
                    uncertainty = max(uncertainty, 1.0 - min(1.0, norm.item() / 100.0))
            except Exception:
                pass

        return min(0.95, uncertainty)

    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}min"
        elif seconds < 86400:
            return f"{seconds/3600:.1f}h"
        else:
            return f"{seconds/86400:.1f}d"

    def get_status_text(self, task_id: str) -> str:
        """Gera texto de status legível."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return "Tarefa não encontrada."

            if task.status == TaskStatus.PENDING:
                return (
                    f"Tarefa recebida: \"{task.description[:60]}\". "
                    f"Tempo estimado: {self._format_time(task.estimated_seconds)}. "
                    f"Aguardando na fila (posição {self._queue.index(task_id) + 1})."
                )
            elif task.status == TaskStatus.RUNNING:
                remaining = max(0, task.estimated_seconds - task.elapsed_seconds)
                return (
                    f"Executando: \"{task.description[:60]}\". "
                    f"Progresso: {task.progress*100:.0f}%. "
                    f"Tempo restante estimado: {self._format_time(remaining)}."
                )
            elif task.status == TaskStatus.COMPLETED:
                return (
                    f"Concluído: \"{task.description[:60]}\". "
                    f"Tempo: {self._format_time(task.elapsed_seconds)}. "
                    f"FE: {task.fe_before:.2f} → {task.fe_after:.2f}."
                )
            elif task.status == TaskStatus.FAILED:
                return (
                    f"Falhou: \"{task.description[:60]}\". "
                    f"Erro: {task.error[:100] if task.error else 'desconhecido'}."
                )
            return f"Status: {task.status.value}."

    # ─── Execução ────────────────────────────────────────────────

    def start_background(self, on_complete: Optional[Callable] = None):
        """Inicia thread de processamento em background."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._on_complete = on_complete
        self._thread = threading.Thread(target=self._background_loop, daemon=True)
        self._thread.start()
        print("[PLANNER] Background processor started")

    def stop_background(self):
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        print("[PLANNER] Background processor stopped")

    def _background_loop(self):
        """Loop principal que processa tarefas em background."""
        while not self._stop_flag.is_set():
            task = self._dequeue_next()
            if task is None:
                time.sleep(1.0)  # espera ativa
                continue
            self._execute_task(task)
            time.sleep(0.1)

    def _dequeue_next(self) -> Optional[Task]:
        """Pega próxima tarefa da fila (maior prioridade)."""
        with self._lock:
            while self._queue:
                tid = self._queue.pop(0)
                task = self._tasks.get(tid)
                if task and task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.RUNNING
                    task.started_at = time.time()
                    self._running_id = tid
                    self._save()
                    return task
                elif task and task.status == TaskStatus.PAUSED:
                    continue
                # Remove tarefas canceladas/completadas/falhas da fila
                if task and task.status in (TaskStatus.CANCELLED,
                                            TaskStatus.COMPLETED,
                                            TaskStatus.FAILED):
                    continue
            return None

    def _execute_task(self, task: Task):
        """Executa uma tarefa."""
        print(f"\n[PLANNER] Executando: {task.description[:80]}")
        start_time = time.time()

        try:
            if self.executor:
                result = self.executor(task)
            else:
                # Fallback: apenas simula progresso
                result = self._simulate_execution(task)

            task.elapsed_seconds = time.time() - start_time
            task.progress = 1.0
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.result = result
            task.fe_after = task.fe_before * 0.3  # FE reduzida após sucesso

            self.completed_tasks += 1
            self.total_elapsed_time += task.elapsed_seconds

            print(f"[PLANNER] Completado em {self._format_time(task.elapsed_seconds)}")
            print(f"[PLANNER] FE: {task.fe_before:.2f} → {task.fe_after:.2f}")

        except Exception as e:
            task.elapsed_seconds = time.time() - start_time
            task.status = TaskStatus.FAILED
            task.error = traceback.format_exc()
            task.fe_after = min(0.95, task.fe_before * 1.3)  # FE aumenta com falha
            self.failed_tasks += 1
            print(f"[PLANNER] Falhou: {e}")

        finally:
            with self._lock:
                self._running_id = None
                self._sort_queue()
                self._save()

            if self._on_complete:
                try:
                    self._on_complete(task)
                except Exception:
                    pass

    def _simulate_execution(self, task: Task) -> dict:
        """Simula execução (fallback quando não há executor)."""
        total_steps = max(5, int(task.estimated_seconds // 2))
        for step in range(total_steps):
            if self._stop_flag.is_set():
                raise InterruptedError("Tarefa cancelada")
            task.progress = (step + 1) / total_steps
            time.sleep(min(1.0, task.estimated_seconds / total_steps))
        return {"status": "simulated", "steps": total_steps}

    # ─── Gerenciamento da Fila ───────────────────────────────────

    def _sort_queue(self):
        """Ordena fila por prioridade (maior primeiro)."""
        def sort_key(tid):
            task = self._tasks.get(tid)
            if not task:
                return 0
            return task.priority_score
        self._queue.sort(key=sort_key, reverse=True)

    def pause(self, task_id: str):
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.PAUSED
                self._save()
                return True
            return False

    def resume(self, task_id: str):
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.PAUSED:
                task.status = TaskStatus.PENDING
                self._queue.append(task_id)
                self._sort_queue()
                self._save()
                return True
            return False

    def cancel(self, task_id: str):
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status in (TaskStatus.PENDING, TaskStatus.RUNNING,
                                         TaskStatus.PAUSED):
                task.status = TaskStatus.CANCELLED
                if task_id in self._queue:
                    self._queue.remove(task_id)
                self._save()
                return True
            return False

    def get_task(self, task_id: str) -> Optional[Task]:
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Task]:
        with self._lock:
            if status:
                return [t for t in self._tasks.values() if t.status == status]
            return list(self._tasks.values())

    def queue_length(self) -> int:
        with self._lock:
            return sum(1 for t in self._tasks.values()
                       if t.status == TaskStatus.PENDING)

    # ─── Persistência ────────────────────────────────────────────

    def _save(self):
        """Salva estado no disco."""
        try:
            data = {
                "tasks": [self._task_to_dict(t) for t in self._tasks.values()],
                "queue": self._queue,
                "running_id": self._running_id,
                "total_tasks": self.total_tasks,
                "completed_tasks": self.completed_tasks,
                "failed_tasks": self.failed_tasks,
                "total_estimated_time": self.total_estimated_time,
                "total_elapsed_time": self.total_elapsed_time,
                "updated_at": time.time(),
            }
            with open(self.persist_path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[PLANNER] Erro salvando estado: {e}")

    def _load(self):
        """Carrega estado do disco."""
        try:
            if os.path.exists(self.persist_path):
                with open(self.persist_path, "r") as f:
                    data = json.load(f)
                self._tasks = {t["id"]: self._dict_to_task(t)
                               for t in data.get("tasks", [])}
                self._queue = data.get("queue", [])
                self._running_id = data.get("running_id")
                self.total_tasks = data.get("total_tasks", 0)
                self.completed_tasks = data.get("completed_tasks", 0)
                self.failed_tasks = data.get("failed_tasks", 0)
                self.total_estimated_time = data.get("total_estimated_time", 0.0)
                self.total_elapsed_time = data.get("total_elapsed_time", 0.0)
        except Exception as e:
            print(f"[PLANNER] Erro carregando estado: {e}")

    def _task_to_dict(self, task: Task) -> dict:
        return {
            "id": task.id,
            "description": task.description,
            "intention": task.intention,
            "priority": task.priority,
            "estimated_seconds": task.estimated_seconds,
            "elapsed_seconds": task.elapsed_seconds,
            "progress": task.progress,
            "status": task.status.value,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "parent_id": task.parent_id,
            "sub_tasks": task.sub_tasks,
            "error": task.error,
            "tags": task.tags,
            "fe_before": task.fe_before,
            "fe_after": task.fe_after,
        }

    def _dict_to_task(self, d: dict) -> Task:
        return Task(
            id=d["id"],
            description=d.get("description", ""),
            intention=d.get("intention", ""),
            priority=d.get("priority", 0.5),
            estimated_seconds=d.get("estimated_seconds", 0),
            elapsed_seconds=d.get("elapsed_seconds", 0),
            progress=d.get("progress", 0),
            status=TaskStatus(d.get("status", "pending")),
            created_at=d.get("created_at", 0),
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
            parent_id=d.get("parent_id"),
            sub_tasks=d.get("sub_tasks", []),
            error=d.get("error"),
            tags=d.get("tags", []),
            fe_before=d.get("fe_before", 0.5),
            fe_after=d.get("fe_after", 0.5),
        )

    def get_summary(self) -> dict:
        """Resumo geral do planejador."""
        with self._lock:
            active = sum(1 for t in self._tasks.values()
                         if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING))
            return {
                "total_tasks": self.total_tasks,
                "completed": self.completed_tasks,
                "failed": self.failed_tasks,
                "active": active,
                "queue_size": len(self._queue),
                "running_id": self._running_id,
                "total_elapsed_time": self.total_elapsed_time,
                "total_estimated_time": self.total_estimated_time,
            }
