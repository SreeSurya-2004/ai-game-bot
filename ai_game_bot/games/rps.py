from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st

from agents.q_learning import QLearningAgent
from utils.storage import model_path

GAME_NAME = "rps"
ACTIONS = ["rock", "paper", "scissors"]

WIN_MAP = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
LOSE_MAP = {v: k for k, v in WIN_MAP.items()}


def Beats(move: str) -> str:
    return LOSE_MAP[move]


def play_result(player: str, opponent: str) -> int:
    if player == opponent:
        return 0
    return 1 if WIN_MAP[player] == opponent else -1


# --- Human-then-train paradigm ---
# State is previous human move (or "START"). Agent is trained to pick the action
# that beats the NEXT human move, learning from human sequences.

def run() -> None:
    st.subheader("Rock–Paper–Scissors (learns your patterns)")
    st.caption("Step 1: Play a few rounds. Step 2: Train bot on your rounds. Step 3: Play vs bot.")

    if "rps_history" not in st.session_state:
        st.session_state.rps_history: List[str] = []  # sequence of human moves
    if "rps_trained" not in st.session_state:
        st.session_state.rps_trained = False

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**1) Human play session**")
        move = st.radio("Choose your move", ACTIONS, horizontal=True, key="rps_move")
        if st.button("Add round"):
            st.session_state.rps_history.append(move)
        st.write("Your sequence:", st.session_state.rps_history)
        if st.button("Clear session"):
            st.session_state.rps_history = []
            st.session_state.rps_trained = False

    with col2:
        st.markdown("**2) Train bot on your session**")
        alpha = st.slider("Learning rate (alpha)", 0.05, 1.0, 0.3, 0.05)
        gamma = st.slider("Discount (gamma)", 0.5, 0.99, 0.95, 0.01)
        epsilon = st.slider("Exploration (epsilon)", 0.0, 1.0, 0.1, 0.05)
        path = model_path(GAME_NAME)
        agent = QLearningAgent.load(path, actions=ACTIONS)
        agent.alpha, agent.gamma, agent.epsilon = alpha, gamma, epsilon

        if st.button("Train on my rounds"):
            history = st.session_state.rps_history
            if len(history) < 2:
                st.warning("Add at least 2 rounds so the bot can learn transitions.")
            else:
                prev = "START"
                for current in history:
                    # Reward the action that would beat the next human move
                    target_action = Beats(current)
                    agent.update(prev, target_action, 1.0, current, done=False)
                    prev = current
                agent.save(path)
                st.session_state.rps_trained = True
                st.success("Bot trained on your sequence and saved.")

    st.divider()
    st.markdown("**3) Play vs Bot**")
    path = model_path(GAME_NAME)
    agent = QLearningAgent.load(path, actions=ACTIONS)

    if "rps_vs_state" not in st.session_state:
        st.session_state.rps_vs_state = "START"
        st.session_state.rps_vs_log: List[Tuple[str, str, int]] = []

    play_col, log_col = st.columns([1, 1])
    with play_col:
        your_move = st.selectbox("Your move", ACTIONS, index=0)
        if st.button("Play round vs Bot"):
            bot_move = agent.choose_action(st.session_state.rps_vs_state, training=False)
            outcome = play_result(your_move, bot_move)
            st.session_state.rps_vs_log.append((your_move, bot_move, outcome))
            st.session_state.rps_vs_state = your_move  # next state is your last move
        if st.button("Reset vs Bot"):
            st.session_state.rps_vs_state = "START"
            st.session_state.rps_vs_log = []

    with log_col:
        if st.session_state.rps_vs_log:
            wins = sum(1 for _, _, r in st.session_state.rps_vs_log if r == 1)
            losses = sum(1 for _, _, r in st.session_state.rps_vs_log if r == -1)
            draws = sum(1 for _, _, r in st.session_state.rps_vs_log if r == 0)
            st.write(f"Results — Wins: {wins}, Losses: {losses}, Draws: {draws}")
            st.table(
                {
                    "Your move": [m[0] for m in st.session_state.rps_vs_log],
                    "Bot move": [m[1] for m in st.session_state.rps_vs_log],
                    "Outcome": ["Win" if m[2] == 1 else ("Loss" if m[2] == -1 else "Draw") for m in st.session_state.rps_vs_log],
                }
            )

    # Suggestions: analyze transition frequencies and bot counters
    if st.session_state.rps_history:
        st.divider()
        st.markdown("**Suggestions based on your session**")
        history = st.session_state.rps_history
        transitions: Dict[str, Dict[str, int]] = {}
        prev = "START"
        for move in history:
            transitions.setdefault(prev, {}).setdefault(move, 0)
            transitions[prev][move] += 1
            prev = move
        rows = []
        total = sum(sum(d.values()) for d in transitions.values())
        for s, dist in transitions.items():
            total_s = sum(dist.values())
            likely = max(dist, key=dist.get)
            bot_counter = Beats(likely)
            rows.append((s, likely, dist[likely] / total_s, bot_counter))
        if rows:
            st.write("The bot will exploit your most likely next move in each situation.")
            st.table({
                "After you played": [r[0] for r in rows],
                "You often play": [r[1] for r in rows],
                "Frequency": [f"{r[2]*100:.0f}%" for r in rows],
                "Bot will play": [r[3] for r in rows],
            })
            st.info("Tip: Be less predictable, especially after the scenarios above.")
