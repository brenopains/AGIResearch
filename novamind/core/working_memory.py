import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict


class WorkingMemorySlot:
    def __init__(self, content: str, concept_ids: List[str], attention: float = 0.5,
                 modality: str = "text"):
        self.content = content
        self.concept_ids = concept_ids
        self.attention = attention
        self.modality = modality
        self.created = time.time()
        self.last_accessed = time.time()
        self.access_count = 1

    def touch(self):
        self.last_accessed = time.time()
        self.access_count += 1

    def to_dict(self) -> Dict:
        return {
            "content": self.content[:80],
            "concepts": self.concept_ids[:5],
            "attention": self.attention,
            "modality": self.modality,
            "age_s": time.time() - self.created,
        }


class WorkingMemory:
    MAX_SLOTS = 7

    def __init__(self):
        self.slots: List[Optional[WorkingMemorySlot]] = [None] * self.MAX_SLOTS
        self._focus_idx: int = 0
        self._decay_rate = 0.05

    def focus(self, content: str, concept_ids: List[str], modality: str = "text",
              attention: float = 0.8) -> int:
        best_idx, best_score = -1, -1
        empty_idx = -1
        for i, slot in enumerate(self.slots):
            if slot is None:
                empty_idx = i
                continue
            overlap = len(set(slot.concept_ids) & set(concept_ids))
            score = overlap + slot.attention * 0.5 - (time.time() - slot.last_accessed) * 0.01
            if score > best_score:
                best_score, best_idx = score, i

        if best_score > 1.0 and best_idx >= 0:
            self.slots[best_idx].content = content
            self.slots[best_idx].concept_ids = concept_ids
            self.slots[best_idx].touch()
            self._focus_idx = best_idx
            return best_idx

        idx = empty_idx if empty_idx >= 0 else self._evict_lru()
        self.slots[idx] = WorkingMemorySlot(content, concept_ids, attention, modality)
        self._focus_idx = idx
        return idx

    def _evict_lru(self) -> int:
        oldest_idx, oldest_time = 0, float("inf")
        for i, slot in enumerate(self.slots):
            if slot is None:
                return i
            if slot.last_accessed < oldest_time:
                oldest_time = slot.last_accessed
                oldest_idx = i
        return oldest_idx

    def get_focused(self) -> Optional[WorkingMemorySlot]:
        return self.slots[self._focus_idx]

    def get_all(self) -> List[Dict]:
        return [s.to_dict() if s else None for s in self.slots]

    def decay(self):
        for slot in self.slots:
            if slot is not None:
                slot.attention = max(0.0, slot.attention - self._decay_rate)
                if slot.attention < 0.05:
                    self.slots[self.slots.index(slot)] = None

    def bind(self, concept_a: str, concept_b: str, relation: str) -> Optional[str]:
        combined = f"{concept_a} {relation} {concept_b}"
        idx = self.focus(combined, [concept_a, concept_b], "bound")
        return combined

    def clear(self):
        self.slots = [None] * self.MAX_SLOTS

    def get_state(self) -> Dict:
        return {
            "slots": self.get_all(),
            "focus_idx": self._focus_idx,
            "focused": self.get_focused().to_dict() if self.get_focused() else None,
            "used_slots": sum(1 for s in self.slots if s is not None),
        }
