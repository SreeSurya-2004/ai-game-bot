from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Tuple, List, Optional
import random
import math
import json
import os


@dataclass
class QConfig:
    learning_rate: float = 0.1
    discount: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000
    seed: Optional[int] = 42


class EpsilonGreedyScheduler:
    def __init__(self, cfg: QConfig):
        self.cfg = cfg
        self.step = 0
        self.random = random.Random(cfg.seed)

    def epsilon(self) -> float:
        if self.step >= self.cfg.epsilon_decay_steps:
            return self.cfg.epsilon_end
        frac = self.step / max(1, self.cfg.epsilon_decay_steps)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def next_action(self, actions: List[Hashable], q_values: List[float]) -> Hashable:
        eps = self.epsilon()
        self.step += 1
        if self.random.random() < eps:
            return self.random.choice(actions)
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return self.random.choice(best_actions)


class QTable:
    def __init__(self, cfg: QConfig):
        self.cfg = cfg
        self.table: Dict[Tuple[Hashable, Hashable], float] = {}
        self.random = random.Random(cfg.seed)

    def get(self, state: Hashable, action: Hashable) -> float:
        return self.table.get((state, action), 0.0)

    def set(self, state: Hashable, action: Hashable, value: float) -> None:
        self.table[(state, action)] = value

    def best_action(self, state: Hashable, actions: List[Hashable]) -> Tuple[Hashable, float]:
        q_vals = [self.get(state, a) for a in actions]
        max_q = max(q_vals)
        best_actions = [a for a, q in zip(actions, q_vals) if q == max_q]
        return self.random.choice(best_actions), max_q

    def update(self, state: Hashable, action: Hashable, reward: float, next_state: Optional[Hashable], next_actions: List[Hashable]) -> float:
        old_q = self.get(state, action)
        if next_state is None or not next_actions:
            target = reward
        else:
            _, next_max_q = self.best_action(next_state, next_actions)
            target = reward + self.cfg.discount * next_max_q
        new_q = old_q + self.cfg.learning_rate * (target - old_q)
        self.set(state, action, new_q)
        return new_q

    def to_json(self) -> str:
        serializable = {f"{s}|{a}": v for (s, a), v in self.table.items()}
        payload = {"config": self.cfg.__dict__, "q": serializable}
        return json.dumps(payload)

    @staticmethod
    def from_json(text: str) -> "QTable":
        payload = json.loads(text)
        cfg = QConfig(**payload.get("config", {}))
        qt = QTable(cfg)
        for k, v in payload.get("q", {}).items():
            s, a = k.split("|")
            qt.table[(s, a)] = float(v)
        return qt

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @staticmethod
    def load(path: str) -> "QTable":
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return QTable.from_json(text)
