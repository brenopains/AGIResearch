"""
NovaMind — Layer 6: Goal System (Agency)
=========================================

Implements a utility-theoretic goal system with intrinsic motivation,
hierarchical planning, and active inference.

Why goals make it an AGENT instead of a tool?
----------------------------------------------
Without goals, a system only responds. With goals, it INITIATES:
    - It asks questions to fill knowledge gaps (curiosity)
    - It compresses knowledge proactively (understanding drive)
    - It seeks consistency in its beliefs (coherence drive)
    - It predicts and prepares for future queries (anticipation)

This transforms NovaMind from a reactive system (like an LLM) to a
proactive agent (like a mind).

The goal system is inspired by:
    - Utility Theory: goals have expected utility based on value + probability
    - Active Inference: goals arise from expected free energy minimization
    - Intrinsic Motivation: curiosity = expected information gain
    - Hierarchical Planning: complex goals decompose into sub-goals

References:
    - Schmidhuber (2010): "Formal Theory of Creativity, Fun, and Intrinsic Motivation"
    - Friston et al. (2015): "Active Inference and Epistemic Value"
    - Sutton & Barto (2018): "Reinforcement Learning" (Ch. on intrinsic motivation)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import time
import heapq


class Goal:
    """
    A goal with expected utility, priority, and decomposition.

    Goals can be:
        - Instrumental: achieve a specific outcome
        - Epistemic: gain specific knowledge (curiosity-driven)
        - Homeostatic: maintain internal consistency
        - Compressive: compress knowledge (understanding-driven)
    """

    TYPES = ['instrumental', 'epistemic', 'homeostatic', 'compressive']

    def __init__(self, description: str, goal_type: str = 'instrumental',
                 priority: float = 0.5, parent_id: Optional[str] = None):
        self.id = f"goal_{int(time.time() * 1000)}_{np.random.randint(10000)}"
        self.description = description
        self.goal_type = goal_type
        self.priority = priority
        self.parent_id = parent_id

        # State
        self.status = 'active'  # active, completed, failed, deferred
        self.progress = 0.0     # [0, 1]
        self.creation_time = time.time()
        self.deadline = None

        # Utility estimation
        self.expected_utility = priority
        self.estimated_effort = 0.5
        self.information_gain = 0.0  # Expected entropy reduction

        # Sub-goals
        self.sub_goals: List[str] = []

        # Execution trace
        self.actions_taken: List[str] = []
        self.result: Optional[str] = None

    def update_progress(self, increment: float, action: str = ""):
        """Update goal progress."""
        self.progress = min(1.0, self.progress + increment)
        if action:
            self.actions_taken.append(action)
        if self.progress >= 1.0:
            self.status = 'completed'

    @property
    def urgency(self) -> float:
        """Compute urgency based on priority, time, and deadline."""
        age = time.time() - self.creation_time
        age_factor = min(1.0, age / 3600)  # Increases over first hour
        return self.priority * 0.6 + age_factor * 0.2 + (1 - self.progress) * 0.2

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'description': self.description,
            'type': self.goal_type,
            'status': self.status,
            'priority': self.priority,
            'progress': self.progress,
            'urgency': self.urgency,
            'expected_utility': self.expected_utility,
            'information_gain': self.information_gain,
            'sub_goals': self.sub_goals,
            'actions_taken': self.actions_taken[-5:],
        }


class GoalSystem:
    """
    Layer 6: Agency and Goal Management.

    Manages a priority queue of goals, handles decomposition into sub-goals,
    and drives proactive behavior through intrinsic motivation.
    """

    def __init__(self):
        self.goals: Dict[str, Goal] = {}
        self.goal_history: List[Dict] = []
        self.reward_signal: float = 0.5  # Internal satisfaction metric

        # Intrinsic motivation parameters
        self.curiosity_weight = 0.4
        self.compression_weight = 0.3
        self.coherence_weight = 0.3

        # Initialize default homeostatic goals
        self._init_homeostatic_goals()

    def _init_homeostatic_goals(self):
        """
        Create baseline homeostatic goals that are always active.

        These are the system's fundamental drives:
        1. Maintain knowledge consistency
        2. Compress redundant information
        3. Fill critical knowledge gaps
        """
        consistency_goal = Goal(
            "Maintain internal knowledge consistency — resolve contradictions",
            goal_type='homeostatic',
            priority=0.8
        )
        self.goals[consistency_goal.id] = consistency_goal

        compression_goal = Goal(
            "Compress episodic memories into semantic knowledge when patterns emerge",
            goal_type='compressive',
            priority=0.6
        )
        self.goals[compression_goal.id] = compression_goal

        coherence_goal = Goal(
            "Ensure the causal model is internally coherent and well-connected",
            goal_type='homeostatic',
            priority=0.7
        )
        self.goals[coherence_goal.id] = coherence_goal

    def add_goal(self, description: str, goal_type: str = 'instrumental',
                 priority: float = 0.5, parent_id: Optional[str] = None) -> Goal:
        """Add a new goal to the system."""
        goal = Goal(description, goal_type, priority, parent_id)
        self.goals[goal.id] = goal

        # If parent exists, add as sub-goal
        if parent_id and parent_id in self.goals:
            self.goals[parent_id].sub_goals.append(goal.id)

        return goal

    def update_from_metacognition(self, metacognition_report: Dict):
        """
        Update goals based on metacognitive assessment.

        This is where PROACTIVE behavior emerges:
        - Low confidence → add goal to seek more information
        - Knowledge gaps → add epistemic goals to fill gaps
        - High surprise → add goal to update world model
        - Poor coherence → add goal to resolve contradictions
        """
        recommendations = metacognition_report.get('recommendations', [])
        emotional_state = metacognition_report.get('emotional_state', {})

        for rec in recommendations:
            if rec['action'] == 'seek_information':
                goal = self.add_goal(
                    f"Seek information: {rec['description']}",
                    goal_type='epistemic',
                    priority=0.7
                )
                goal.information_gain = 0.5

            elif rec['action'] == 'resolve_contradictions':
                goal = self.add_goal(
                    f"Resolve contradictions: {rec['description']}",
                    goal_type='homeostatic',
                    priority=0.8
                )

            elif rec['action'] == 'increase_deliberation':
                goal = self.add_goal(
                    f"Deepen reasoning: {rec['description']}",
                    goal_type='instrumental',
                    priority=0.6
                )

        # Curiosity-driven goals
        if emotional_state.get('curiosity', 0) > 0.7:
            surprise = metacognition_report.get('surprise', {})
            if surprise.get('current_level', 0) > 0.5:
                goal = self.add_goal(
                    "Explore novel concepts detected in recent input",
                    goal_type='epistemic',
                    priority=0.5
                )
                goal.information_gain = 0.7

        # Compression-driven goals
        if emotional_state.get('satisfaction', 0) < 0.3:
            self.add_goal(
                "Attempt knowledge compression — look for compactable patterns",
                goal_type='compressive',
                priority=0.5
            )

    def get_next_goal(self) -> Optional[Goal]:
        """
        Get the highest-urgency active goal.

        Priority is computed as:
            score = priority × (1 - progress) + information_gain × curiosity_weight
        """
        active_goals = [g for g in self.goals.values() if g.status == 'active']

        if not active_goals:
            return None

        # Score each goal
        scored = []
        for goal in active_goals:
            score = (
                goal.urgency * 0.4 +
                goal.expected_utility * 0.3 +
                goal.information_gain * self.curiosity_weight * 0.3
            )
            scored.append((score, goal))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else None

    def complete_goal(self, goal_id: str, result: str = "", success: bool = True):
        """Mark a goal as completed."""
        if goal_id in self.goals:
            goal = self.goals[goal_id]
            goal.status = 'completed' if success else 'failed'
            goal.progress = 1.0 if success else goal.progress
            goal.result = result

            self.goal_history.append({
                'goal_id': goal_id,
                'description': goal.description,
                'type': goal.goal_type,
                'success': success,
                'timestamp': time.time(),
            })

            # Update reward signal
            if success:
                self.reward_signal = min(1.0, self.reward_signal + 0.1)
            else:
                self.reward_signal = max(0.0, self.reward_signal - 0.05)

    def compute_reward(self, compression_progress: float, coherence: float,
                       information_gain: float) -> float:
        """
        Compute intrinsic reward signal.

        Reward = weighted combination of:
        - Compression progress (Schmidhuber's compression progress theory)
        - Coherence maintenance
        - Information gain (curiosity satisfaction)
        """
        reward = (
            compression_progress * self.compression_weight +
            coherence * self.coherence_weight +
            information_gain * self.curiosity_weight
        )

        self.reward_signal = 0.9 * self.reward_signal + 0.1 * reward
        return self.reward_signal

    def prune_completed_goals(self, max_age: float = 3600):
        """Remove old completed goals to prevent memory bloat."""
        now = time.time()
        pruned = {}
        for gid, goal in self.goals.items():
            if goal.status in ('completed', 'failed') and (now - goal.creation_time) > max_age:
                continue  # Skip (prune)
            pruned[gid] = goal
        self.goals = pruned

    def get_state(self) -> Dict:
        """Return goal system state for visualization."""
        active = [g for g in self.goals.values() if g.status == 'active']
        completed = [g for g in self.goals.values() if g.status == 'completed']
        failed = [g for g in self.goals.values() if g.status == 'failed']

        return {
            'active_goals': [g.to_dict() for g in sorted(active, key=lambda x: x.urgency, reverse=True)],
            'completed_goals': len(completed),
            'failed_goals': len(failed),
            'reward_signal': self.reward_signal,
            'total_goals_ever': len(self.goal_history) + len(self.goals),
            'goal_type_distribution': {
                t: len([g for g in self.goals.values() if g.goal_type == t])
                for t in Goal.TYPES
            },
        }
