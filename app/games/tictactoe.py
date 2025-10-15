from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import itertools


PLAYERS = {"human": "X", "bot": "O"}


def check_winner(board: List[str]) -> Optional[str]:
    wins = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)
    ]
    for a, b, c in wins:
        if board[a] != " " and board[a] == board[b] == board[c]:
            return board[a]
    if all(cell != " " for cell in board):
        return "draw"
    return None


def available_actions(board: List[str]) -> List[int]:
    return [i for i, v in enumerate(board) if v == " "]


def board_to_state(board: List[str]) -> str:
    return "".join(board)


def state_to_board(state: str) -> List[str]:
    return list(state)


def make_move(board: List[str], index: int, symbol: str) -> List[str]:
    new_board = board.copy()
    new_board[index] = symbol
    return new_board


def reward_for_outcome(outcome: Optional[str], for_symbol: str) -> float:
    if outcome is None:
        return 0.0
    if outcome == "draw":
        return 0.2
    if outcome == for_symbol:
        return 1.0
    return -1.0
