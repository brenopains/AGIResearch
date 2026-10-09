import time
import random
import threading
import queue
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class ColonyMessage:
    """A message between minds in the colony."""
    sender_id: str
    content: str
    msg_type: str  # "observation", "question", "answer", "correction", "discovery"
    topic: str = ""
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)


class MindProxy:
    """
    A proxy to another mind in the colony.
    Encapsulates communication and knowledge sharing.
    """

    def __init__(self, mind_id: str, send_callback: Callable):
        self.mind_id = mind_id
        self._send = send_callback
        self.knowledge_shared: List[str] = []

    def send(self, content: str, msg_type: str = "observation", topic: str = "",
             confidence: float = 0.5):
        self._send(ColonyMessage(
            sender_id=self.mind_id,
            content=content,
            msg_type=msg_type,
            topic=topic,
            confidence=confidence,
        ))


class Colony:
    """
    Multi-mind colony: a "classroom" where multiple NovaMind instances
    interact, share knowledge, debate, and learn from each other.

    ────────────
    HOW IT WORKS
    ────────────
    Each MindProxy represents a student in the classroom.
    They can:
    • Broadcast observations ("Eu descobri que agua ferve a 100C")
    • Ask questions ("Alguem sabe o que e gravidade?")
    • Answer questions ("Gravidade e a forca que atrai objetos")
    • Correct peers ("Nao, agua ferve a 100C ao nivel do mar")
    • Share discoveries ("Acabei de aprender sobre fotossintese!")

    The classroom dynamics:
    1. No fixed curriculum — knowledge emerges from peer interaction
    2. Each mind has different sources (YouTube, Wikipedia, etc.)
    3. When a mind learns something new, it shares with the group
    4. Other minds can challenge or build upon shared knowledge
    5. Shared knowledge goes to a communal blackboard

    This creates a Vygotskian "Zone of Proximal Development":
    minds learn from peers who are slightly ahead.
    """

    def __init__(self, max_minds: int = 5):
        self.max_minds = max_minds
        self.minds: Dict[str, MindProxy] = {}
        self._message_queue: queue.Queue = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Communal blackboard: knowledge shared across all minds
        self.blackboard: Dict[str, List[Dict]] = defaultdict(list)
        self.blackboard_lock = threading.Lock()

        # Classroom discussion log
        self.discussion_log: List[ColonyMessage] = []
        self.discussion_lock = threading.Lock()

        # Interaction stats
        self.total_exchanges = 0
        self.corrections_made = 0
        self.questions_asked = 0
        self.discoveries_shared = 0

        # Peer ratings: how reliable each mind is
        self.credibility: Dict[str, float] = defaultdict(lambda: 0.5)

    def register_mind(self, mind_id: str) -> MindProxy:
        """Register a mind in the colony. Returns a proxy for communication."""
        proxy = MindProxy(mind_id, self._receive_message)
        self.minds[mind_id] = proxy
        print(f"[COLONY] {mind_id} entrou na sala de aula")
        # Welcome message from colony
        self._broadcast(ColonyMessage(
            sender_id="colony",
            content=f"{mind_id} entrou na sala de aula!",
            msg_type="system",
        ))
        return proxy

    def unregister_mind(self, mind_id: str):
        if mind_id in self.minds:
            del self.minds[mind_id]
            print(f"[COLONY] {mind_id} saiu da sala de aula")

    def _receive_message(self, msg: ColonyMessage):
        """Internal: receive a message from a mind and queue it for processing."""
        self._message_queue.put(msg)

    def _broadcast(self, msg: ColonyMessage):
        """Broadcast a message to all minds (for logging)."""
        with self.discussion_lock:
            self.discussion_log.append(msg)
            if len(self.discussion_log) > 500:
                self.discussion_log = self.discussion_log[-500:]

    def start(self):
        """Start the colony message processing loop in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        print(f"[COLONY] Sala de aula iniciada com {self.max_minds} vagas")

    def stop(self):
        self._running = False

    def _process_loop(self):
        """
        Background loop that processes messages between minds.

        Each message is:
        1. Added to the discussion log
        2. Categorized (question, discovery, correction, etc.)
        3. Posted to the blackboard if it contains new knowledge
        4. Credibility of the sender is updated based on peer feedback
        """
        while self._running:
            try:
                msg = self._message_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            self.total_exchanges += 1

            with self.discussion_lock:
                self.discussion_log.append(msg)

            # Categorize and process
            content_lower = msg.content.lower()

            if "?" in msg.content or "alguem sabe" in content_lower or "o que e " in content_lower:
                self.questions_asked += 1
                self._handle_question(msg)

            elif "nao" in content_lower and ("correto" in content_lower or "errado" in content_lower):
                self.corrections_made += 1
                self._handle_correction(msg)

            elif "descobri" in content_lower or "aprendi" in content_lower or "achei" in content_lower:
                self.discoveries_shared += 1
                self._handle_discovery(msg)

            else:
                self._handle_observation(msg)

    def _handle_question(self, msg: ColonyMessage):
        """A mind asked a question — see if anyone has answered it."""
        # Note the question in the discussion
        with self.blackboard_lock:
            self.blackboard["questions"].append({
                "asked_by": msg.sender_id,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "answered": False,
            })

    def _handle_correction(self, msg: ColonyMessage):
        """A mind corrected someone — update credibility."""
        self.credibility[msg.sender_id] = min(1.0, self.credibility[msg.sender_id] + 0.1)
        with self.blackboard_lock:
            self.blackboard["corrections"].append({
                "by": msg.sender_id,
                "content": msg.content,
                "timestamp": msg.timestamp,
            })
        self._broadcast(ColonyMessage(
            sender_id="colony",
            content=f"[Sala de Aula] {msg.sender_id} fez uma correcao. Credibilidade: {self.credibility[msg.sender_id]:.2f}",
            msg_type="system",
        ))

    def _handle_discovery(self, msg: ColonyMessage):
        """A mind made a discovery — add to blackboard."""
        self.credibility[msg.sender_id] = min(1.0, self.credibility[msg.sender_id] + 0.05)
        with self.blackboard_lock:
            self.blackboard["discoveries"].append({
                "by": msg.sender_id,
                "content": msg.content,
                "topic": msg.topic,
                "confidence": msg.confidence,
                "timestamp": msg.timestamp,
            })

    def _handle_observation(self, msg: ColonyMessage):
        """A mind shared an observation — log it."""
        with self.blackboard_lock:
            self.blackboard["observations"].append({
                "by": msg.sender_id,
                "content": msg.content,
                "topic": msg.topic,
                "timestamp": msg.timestamp,
            })
            if len(self.blackboard["observations"]) > 200:
                self.blackboard["observations"] = self.blackboard["observations"][-200:]

    def get_recent_discussion(self, n: int = 10) -> List[Dict]:
        """Get the most recent discussion messages."""
        with self.discussion_lock:
            return [{
                "sender": m.sender_id,
                "type": m.msg_type,
                "content": m.content[:100],
                "topic": m.topic,
                "time": m.timestamp,
            } for m in self.discussion_log[-n:]]

    def get_state(self) -> dict:
        if not self._running:
            return {"status": "stopped"}
        with self.blackboard_lock:
            return {
                "status": "running",
                "minds_in_class": len(self.minds),
                "total_exchanges": self.total_exchanges,
                "questions_asked": self.questions_asked,
                "corrections_made": self.corrections_made,
                "discoveries_shared": self.discoveries_shared,
                "blackboard": {
                    "observations": len(self.blackboard.get("observations", [])),
                    "questions": len(self.blackboard.get("questions", [])),
                    "discoveries": len(self.blackboard.get("discoveries", [])),
                    "corrections": len(self.blackboard.get("corrections", [])),
                },
                "credibility": dict(self.credibility),
                "recent_exchange": self.discussion_log[-1].content if self.discussion_log else "",
            }


from collections import defaultdict
