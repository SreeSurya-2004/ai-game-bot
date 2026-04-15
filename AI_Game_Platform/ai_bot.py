import os
import time
import json
import random
from typing import Any, Dict, Optional

import requests

class AIBot:
    """
    AI Bot with Heuristic + Reinforcement Learning (Q-learning) for:
    - Tic-Tac-Toe
    - Memory Game
    - Snake
    Works with webpage hints interface: { game, suggestion?, text }
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.memory = []
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self._last_call = 0
        self._min_interval = 0.25
        self.rng = random.Random(42)

        # Q-tables for RL
        self.q_tt = {}     # Tic-Tac-Toe
        self.q_mem = {}    # Memory
        self.q_snake = {}  # Snake

        # RL hyperparameters
        self.alpha = 0.3
        self.gamma = 0.95
        self.epsilon = 0.15

        self._load_qtables()

    # -----------------------------
    # Public interface for webpage
    # -----------------------------
    def get_hint(self, player_action: Any) -> Dict[str, Any]:
        payload = player_action if isinstance(player_action, dict) else {"raw": str(player_action)}
        game = payload.get("game") or payload.get("game_name") or self._guess_game_from_payload(payload)
        entry = {"time": time.time(), "input": payload, "game": game}
        self.memory.append(entry)

        # Try OpenAI API if available
        if self.api_key:
            hint = self._call_openai_for_hint(payload, game)
            if hint:
                return hint

        # fallback: RL heuristic
        return self._heuristic_hint(payload, game)

    # -----------------------------
    # Game detection
    # -----------------------------
    def _guess_game_from_payload(self, payload: Dict[str, Any]) -> str:
        if "board" in payload or payload.get("mark") or payload.get("move") is not None:
            return "tictactoe"
        if "visible" in payload or payload.get("flip") is not None:
            return "memory"
        if payload.get("dir") or payload.get("head"):
            return "snake"
        return "general"

    # -----------------------------
    # Tic-Tac-Toe RL helpers
    # -----------------------------
    @staticmethod
    def tt_encode(board: list) -> str:
        return "".join([c if c in ("X","O") else "_" for c in board])

    def tt_qget(self, state: str) -> list:
        if state not in self.q_tt:
            self.q_tt[state] = [0.0]*9
        return self.q_tt[state]

    def tt_choose_action(self, board: list, explore=True) -> int:
        state = self.tt_encode(board)
        q = self.tt_qget(state)
        allowed = [i for i, v in enumerate(board) if v == ""]
        if not allowed:
            return 0
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(allowed)
        best_val = None; best_idxs=[]
        for i in allowed:
            val = q[i]
            if best_val is None or val>best_val:
                best_val=val; best_idxs=[i]
            elif val==best_val:
                best_idxs.append(i)
        return self.rng.choice(best_idxs)

    def tt_update(self, s_board, action:int, reward:float, s2_board, done:bool):
        s_key = self.tt_encode(s_board)
        s2_key = self.tt_encode(s2_board)
        q = self.tt_qget(s_key)
        q2 = self.tt_qget(s2_key) if not done else [0.0]*9
        max_q2 = 0.0 if done else max(q2)
        q[action] = q[action] + self.alpha * (reward + self.gamma*max_q2 - q[action])
        self.q_tt[s_key] = q

    def tt_get_reward(self, board_before, action, board_after, done):
        # Win for suggested move
        if done:
            if self._is_winner(board_after, "O"):  # bot suggested 'O'
                return 1.0
            elif self._is_winner(board_after, "X"):
                return -1.0
            else:
                return 0.0
        # small reward if action blocked opponent
        trial = board_before.copy()
        trial[action] = "X"
        if self._is_winner(trial, "X"):
            return 0.5
        return 0.0

    # -----------------------------
    # Memory RL helpers
    # -----------------------------
    def mem_key(self, visible: list) -> str:
        return "|".join(['#' if v is None else str(v) for v in visible])

    def mem_qget(self, key:str, n:int) -> list:
        if key not in self.q_mem:
            self.q_mem[key] = [0.0]*n
        return self.q_mem[key]

    def mem_choose(self, visible:list, explore=True) -> int:
        key = self.mem_key(visible)
        n = len(visible)
        q = self.mem_qget(key, n)
        hidden = [i for i, v in enumerate(visible) if v is None]
        if not hidden:
            return 0
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(hidden)
        best_val = None; best_idxs=[]
        for i in hidden:
            val = q[i]
            if best_val is None or val>best_val:
                best_val=val; best_idxs=[i]
            elif val==best_val:
                best_idxs.append(i)
        return self.rng.choice(best_idxs)

    def mem_update(self, s_vis, action:int, reward:float, s2_vis, done:bool):
        k1 = self.mem_key(s_vis)
        k2 = self.mem_key(s2_vis)
        q = self.mem_qget(k1, len(s_vis))
        q2 = self.mem_qget(k2, len(s_vis)) if not done else [0.0]*len(s_vis)
        max_q2 = 0.0 if done else max(q2)
        q[action] = q[action] + self.alpha*(reward+self.gamma*max_q2 - q[action])
        self.q_mem[k1] = q

    def mem_get_reward(self, revealed_before, action, revealed_after, done):
        if revealed_after[action] is not None:
            return 1.0
        return -0.1

    # -----------------------------
    # Snake RL helpers
    # -----------------------------
    @staticmethod
    def snake_state_key(state_tuple) -> str:
        return json.dumps(state_tuple)

    def snake_qget(self, key:str) -> list:
        if key not in self.q_snake:
            self.q_snake[key] = [0.0,0.0,0.0]  # straight, left, right
        return self.q_snake[key]

    def snake_choose(self, key:str, explore=True) -> int:
        q = self.snake_qget(key)
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice([0,1,2])
        best_val = max(q)
        best_idxs = [i for i,v in enumerate(q) if v==best_val]
        return self.rng.choice(best_idxs)

    def snake_update(self, s_key, action:int, reward:float, s2_key, done:bool):
        q = self.snake_qget(s_key)
        q2 = self.snake_qget(s2_key) if not done else [0.0,0.0,0.0]
        max_q2 = 0.0 if done else max(q2)
        q[action] = q[action] + self.alpha*(reward+self.gamma*max_q2 - q[action])
        self.q_snake[s_key] = q

    def snake_get_reward(self, alive:bool, ate_food:bool):
        if not alive:
            return -1.0
        if ate_food:
            return 1.0
        return 0.1

    # -----------------------------
    # Heuristic RL fallback for webpage
    # -----------------------------
    def _heuristic_hint(self, payload: Dict[str, Any], game: str) -> Dict[str, Any]:
        if game == "tictactoe":
            board = payload.get("board", [""]*9)
            action = self.tt_choose_action(board)
            return {"game": "tictactoe", "suggestion": {"index": action, "reason":"qlearning"}, "text": f"Try move at {action}"}

        if game == "memory":
            visible = payload.get("visible", [])
            action = self.mem_choose(visible)
            return {"game":"memory","suggestion":{"index":action,"reason":"qlearning"},"text":"Try this card"}

        if game == "snake":
            dirp = payload.get("dir") or {}
            state_key = self.snake_state_key((str(dirp),))
            action = self.snake_choose(state_key)
            return {"game":"snake","suggestion":{"dir":action,"reason":"qlearning"},"text":"Try RL move"}

        return {"game":"general","text":"Heuristic suggestion (fallback). Provide game state for more detail."}

    # -----------------------------
    # OpenAI API (if available)
    # -----------------------------
    def _call_openai_for_hint(self, payload: Dict[str, Any], game: str) -> Optional[Dict[str, Any]]:
        return None  # simplified; keep your previous implementation if needed

    # -----------------------------
    # Q-table persistence
    # -----------------------------
    def save_qtables(self):
        os.makedirs("qtables_saved", exist_ok=True)
        json.dump(self.q_tt, open("qtables_saved/tt_q.json","w"))
        json.dump(self.q_mem, open("qtables_saved/mem_q.json","w"))
        json.dump(self.q_snake, open("qtables_saved/snake_q.json","w"))

    def _load_qtables(self):
        try:
            self.q_tt = json.load(open("qtables_saved/tt_q.json"))
        except: pass
        try:
            self.q_mem = json.load(open("qtables_saved/mem_q.json"))
        except: pass
        try:
            self.q_snake = json.load(open("qtables_saved/snake_q.json"))
        except: pass

    # -----------------------------
    # Tic-Tac-Toe win check helper
    # -----------------------------
    def _is_winner(self, board, mark):
        combos = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        return any(board[a]==board[b]==board[c]==mark for a,b,c in combos)
