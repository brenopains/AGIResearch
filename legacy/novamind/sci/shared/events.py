"""EventBus leve: orquestradores emitem; FastAPI streama via SSE; CLI imprime."""
import time, json
from queue import Queue, Empty
from threading import Lock
from typing import Callable, Dict, Optional, Any

class EventBus:
    """Fila thread-safe por run_id. Consumidor (SSE) faz .stream(run_id) e itera."""
    def __init__(self):
        self.queues: Dict[str, Queue] = {}
        self.results: Dict[str, dict] = {}
        self.lock = Lock()

    def new_run(self, run_id: str):
        with self.lock:
            self.queues[run_id] = Queue()

    def emit(self, run_id: str, event_type: str, payload: Any = None):
        with self.lock:
            q = self.queues.get(run_id)
        if q is None: return
        q.put({"t": time.time(), "type": event_type, "payload": payload})

    def finish(self, run_id: str, result: dict):
        with self.lock:
            self.results[run_id] = result
        self.emit(run_id, "done", {"summary": result.get("summary","")})
        self.emit(run_id, "__END__", None)

    def stream(self, run_id: str, timeout: float = 0.5):
        """Gera eventos ate receber __END__ ou expirar idle global."""
        with self.lock:
            q = self.queues.get(run_id)
        if q is None:
            yield {"t":time.time(), "type":"error", "payload":{"msg":"run_id desconhecido"}}; return
        last_seen = time.time()
        while True:
            try:
                ev = q.get(timeout=timeout)
                last_seen = time.time()
                if ev.get("type") == "__END__":
                    yield ev; return
                yield ev
            except Empty:
                # keep-alive
                yield {"t":time.time(), "type":"ping"}
                if time.time() - last_seen > 600:   # 10 min sem eventos -> fecha
                    yield {"t":time.time(), "type":"error", "payload":{"msg":"timeout"}}
                    return

    def get_result(self, run_id: str) -> Optional[dict]:
        with self.lock:
            return self.results.get(run_id)

# singleton global
BUS = EventBus()

def cli_printer(run_id: str) -> Callable:
    """Callback pro modo CLI: imprime."""
    def emit(event_type, payload=None):
        line = f"[{event_type}] " + (json.dumps(payload, ensure_ascii=False)[:200] if payload else "")
        print(line)
    return emit

def bus_emitter(run_id: str) -> Callable:
    """Callback pro modo servidor: empurra no BUS."""
    def emit(event_type, payload=None):
        BUS.emit(run_id, event_type, payload)
    return emit
