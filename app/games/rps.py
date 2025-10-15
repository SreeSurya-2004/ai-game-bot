from __future__ import annotations

from enum import Enum
from typing import Dict, Tuple
import random


class Move(str, Enum):
    ROCK = "rock"
    PAPER = "paper"
    SCISSORS = "scissors"


ALL_MOVES = [Move.ROCK, Move.PAPER, Move.SCISSORS]


def outcome(bot: Move, human: Move) -> int:
    if bot == human:
        return 0
    if (
        (bot == Move.ROCK and human == Move.SCISSORS)
        or (bot == Move.PAPER and human == Move.ROCK)
        or (bot == Move.SCISSORS and human == Move.PAPER)
    ):
        return 1
    return -1


class FrequencyModel:
    def __init__(self, seed: int = 42):
        self.counts: Dict[Move, int] = {m: 1 for m in ALL_MOVES}
        self.random = random.Random(seed)

    def update(self, human_move: Move):
        self.counts[human_move] += 1

    def predict_human(self) -> Move:
        total = sum(self.counts.values())
        r = self.random.uniform(0, total)
        upto = 0
        for m in ALL_MOVES:
            upto += self.counts[m]
            if r <= upto:
                return m
        return self.random.choice(ALL_MOVES)

    def best_response(self) -> Move:
        pred = self.predict_human()
        if pred == Move.ROCK:
            return Move.PAPER
        if pred == Move.PAPER:
            return Move.SCISSORS
        return Move.ROCK
