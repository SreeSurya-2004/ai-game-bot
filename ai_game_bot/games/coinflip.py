from __future__ import annotations

from typing import Dict, List, Tuple

import streamlit as st

from agents.q_learning import QLearningAgent
from utils.storage import model_path

GAME_NAME = "coinflip"
ACTIONS = ["heads", "tails"]


def run() -> None:
    st.subheader("Coin Flip Predictor (learns your tendencies)")
    st.caption("Add your coin outcomes as a sequence. Train the bot to predict your next outcome. Then see if it guesses right.")

    if "cf_history" not in st.session_state:
        st.session_state.cf_history: List[str] = []
    if "cf_trained" not in st.session_state:
        st.session_state.cf_trained = False

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**1) Add your coin outcomes**")
        move = st.radio("Outcome", ACTIONS, horizontal=True, key="cf_move")
        if st.button("Add outcome"):
            st.session_state.cf_history.append(move)
        st.write("Your sequence:", st.session_state.cf_history)
        if st.button("Clear sequence"):
            st.session_state.cf_history = []
            st.session_state.cf_trained = False

    with col2:
        st.markdown("**2) Train bot on your sequence**")
        alpha = st.slider("Learning rate (alpha)", 0.05, 1.0, 0.3, 0.05)
        gamma = st.slider("Discount (gamma)", 0.5, 0.99, 0.95, 0.01)
        epsilon = st.slider("Exploration (epsilon)", 0.0, 1.0, 0.1, 0.05)
        path = model_path(GAME_NAME)
        agent = QLearningAgent.load(path, actions=ACTIONS)
        agent.alpha, agent.gamma, agent.epsilon = alpha, gamma, epsilon

        if st.button("Train on my sequence"):
            history = st.session_state.cf_history
            if len(history) < 2:
                st.warning("Add at least 2 outcomes so the bot can learn transitions.")
            else:
                prev_state = "START"
                for current in history:
                    # Reward the action equal to your next move
                    agent.update(prev_state, current, 1.0, current, done=False)
                    prev_state = current
                agent.save(path)
                st.session_state.cf_trained = True
                st.success("Bot trained and saved.")

    st.divider()
    st.markdown("**3) Bot guesses your next outcome**")
    path = model_path(GAME_NAME)
    agent = QLearningAgent.load(path, actions=ACTIONS)

    if "cf_vs_state" not in st.session_state:
        st.session_state.cf_vs_state = "START"
        st.session_state.cf_vs_log: List[Tuple[str, str, int]] = []

    play_col, log_col = st.columns([1, 1])
    with play_col:
        your_next = st.selectbox("Your next outcome", ACTIONS, index=0)
        if st.button("Reveal and score"):
            bot_guess = agent.choose_action(st.session_state.cf_vs_state, training=False)
            outcome = 1 if your_next == bot_guess else -1
            st.session_state.cf_vs_log.append((your_next, bot_guess, outcome))
            st.session_state.cf_vs_state = your_next
        if st.button("Reset bot session"):
            st.session_state.cf_vs_state = "START"
            st.session_state.cf_vs_log = []

    with log_col:
        if st.session_state.cf_vs_log:
            correct = sum(1 for _, _, r in st.session_state.cf_vs_log if r == 1)
            wrong = sum(1 for _, _, r in st.session_state.cf_vs_log if r == -1)
            st.write(f"Results — Correct: {correct}, Wrong: {wrong}")
            st.table(
                {
                    "Your outcome": [m[0] for m in st.session_state.cf_vs_log],
                    "Bot guess": [m[1] for m in st.session_state.cf_vs_log],
                    "Score": ["Correct" if m[2] == 1 else "Wrong" for m in st.session_state.cf_vs_log],
                }
            )

    # Suggestions from your sequence
    if st.session_state.cf_history:
        st.divider()
        st.markdown("**Suggestions based on your sequence**")
        history = st.session_state.cf_history
        transitions: Dict[str, Dict[str, int]] = {}
        prev = "START"
        for move in history:
            transitions.setdefault(prev, {}).setdefault(move, 0)
            transitions[prev][move] += 1
            prev = move
        rows = []
        for s, dist in transitions.items():
            total_s = sum(dist.values())
            likely = max(dist, key=dist.get)
            rows.append((s, likely, dist[likely] / total_s))
        if rows:
            st.write("The bot will guess your most likely next outcome in each situation.")
            st.table(
                {
                    "After state": [r[0] for r in rows],
                    "You often do": [r[1] for r in rows],
                    "Frequency": [f"{r[2]*100:.0f}%" for r in rows],
                }
            )
            st.info("Tip: Mix up your sequence to reduce predictability.")
