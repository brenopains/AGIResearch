import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class IntrinsicRL:
    def __init__(self, learning_rate: float = 0.1, gamma: float = 0.9,
                 curiosity_weight: float = 0.3):
        self.lr = learning_rate
        self.gamma = gamma
        self.curiosity_weight = curiosity_weight

        self.q_table: Dict[str, float] = defaultdict(float)
        self.state_values: Dict[str, float] = defaultdict(float)
        self.state_counts: Dict[str, int] = defaultdict(int)
        self.last_state: Optional[str] = None
        self.last_action: Optional[str] = None
        self.total_reward: float = 0.0
        self.episode_count: int = 0
        self.reward_history: List[float] = []

    def compute_reward(self, prediction_error: float, compression_progress: float,
                       novelty: bool, attention: float) -> float:
        r = 0.0
        r += (1.0 - prediction_error) * 0.4
        r += compression_progress * 0.3
        if novelty:
            r += 0.2
        r += max(0, attention - 0.5) * 0.1
        r = max(-1.0, min(1.0, r))
        return r

    def observe(self, state: str, action: str, reward: float, next_state: str):
        self.state_counts[state] += 1
        curiosity_bonus = self.curiosity_weight / np.sqrt(self.state_counts[state] + 1)
        reward += curiosity_bonus

        q_key = f"{state}|{action}"
        old_q = self.q_table[q_key]
        max_next = max(
            [self.q_table[f"{next_state}|{a}"] for a in self._get_possible_actions()],
            default=0.0
        )
        td_target = reward + self.gamma * max_next
        self.q_table[q_key] = old_q + self.lr * (td_target - old_q)

        old_v = self.state_values[state]
        self.state_values[state] = old_v + self.lr * (reward + self.gamma * self.state_values[next_state] - old_v)

        self.total_reward += reward
        self.reward_history.append(reward)
        if len(self.reward_history) > 100:
            self.reward_history = self.reward_history[-100:]

        self.last_state = next_state
        self.last_action = action

    def select_action(self, state: str, available_actions: List[str],
                      exploration_rate: float = 0.2) -> str:
        if not available_actions:
            return "observar"
        if np.random.random() < exploration_rate:
            return np.random.choice(available_actions)

        best_action, best_q = available_actions[0], float("-inf")
        for a in available_actions:
            q = self.q_table[f"{state}|{a}"]
            if q > best_q:
                best_q, best_action = q, a
        return best_action

    def _get_possible_actions(self) -> List[str]:
        return [k.split("|")[1] for k in self.q_table.keys()]

    def get_state(self) -> Dict:
        return {
            "total_reward": round(self.total_reward, 3),
            "episodes": self.episode_count,
            "avg_reward_100": round(np.mean(self.reward_history[-100:]), 3) if self.reward_history else 0,
            "known_states": len(self.state_values),
            "known_actions": len(self.q_table),
            "curiosity_weight": self.curiosity_weight,
        }
