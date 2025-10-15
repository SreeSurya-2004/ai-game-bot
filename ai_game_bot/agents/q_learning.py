from __future__ import annotations

import json
import random
from typing import Any, Callable, Dict, List, Tuple


class QLearningAgent:
    """
    Simple tabular Q-learning agent for discrete state/action spaces.
    States are serialized to strings using the provided serializer so we can save to JSON.
    """

    def __init__(
        self,
        actions: List[str],
        alpha: float = 0.3,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        state_serializer: Callable[[Any], str] | None = None,
    ) -> None:
        self.actions: List[str] = actions
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self._q: Dict[str, Dict[str, float]] = {}
        self._serialize: Callable[[Any], str] = state_serializer or (lambda s: json.dumps(s, sort_keys=True))

    # --- Core Q-learning operations ---
    def _ensure_state(self, state_key: str) -> Dict[str, float]:
        if state_key not in self._q:
            self._q[state_key] = {a: 0.0 for a in self.actions}
        return self._q[state_key]

    def get_q_values(self, state: Any) -> Dict[str, float]:
        key = self._serialize(state)
        return self._ensure_state(key)

    def best_action_value(self, state: Any) -> Tuple[str, float]:
        q_values = self.get_q_values(state)
        # break ties randomly for stability
        best_a = max(q_values, key=lambda a: (q_values[a], random.random()))
        return best_a, q_values[best_a]

    def choose_action(self, state: Any, training: bool = True) -> str:
        if training and random.random() < self.epsilon:
            return random.choice(self.actions)
        best_a, _ = self.best_action_value(state)
        return best_a

    def update(self, state: Any, action: str, reward: float, next_state: Any, done: bool) -> None:
        state_key = self._serialize(state)
        next_key = self._serialize(next_state)
        q_values = self._ensure_state(state_key)
        next_values = self._ensure_state(next_key)
        max_next = 0.0 if done else max(next_values.values())
        target = reward + self.gamma * max_next
        q_values[action] = (1 - self.alpha) * q_values[action] + self.alpha * target

    # --- Persistence ---
    def to_json(self) -> str:
        return json.dumps({"actions": self.actions, "q": self._q}, sort_keys=True)

    @classmethod
    def from_json(cls, data: str) -> "QLearningAgent":
        payload = json.loads(data)
        agent = cls(actions=payload["actions"])  # use defaults for hyperparams
        agent._q = {k: {a: float(v) for a, v in row.items()} for k, row in payload["q"].items()}
        return agent

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str, actions: List[str]) -> "QLearningAgent":
        try:
            with open(path, "r", encoding="utf-8") as f:
                return cls.from_json(f.read())
        except FileNotFoundError:
            return cls(actions=actions)
