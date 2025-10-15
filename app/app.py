from __future__ import annotations

import streamlit as st
import os
from typing import List

from app.rl.qlearning import QTable, QConfig, EpsilonGreedyScheduler
from app.games.tictactoe import (
    PLAYERS,
    check_winner,
    available_actions,
    board_to_state,
    state_to_board,
    make_move,
    reward_for_outcome,
)
from app.games.rps import Move, ALL_MOVES, FrequencyModel, outcome

SAVE_DIR = "data"
TTT_Q_PATH = os.path.join(SAVE_DIR, "tictactoe_qtable.json")
RPS_SAVE = os.path.join(SAVE_DIR, "rps_freq.json")


def load_qtable() -> QTable:
    if os.path.exists(TTT_Q_PATH):
        try:
            return QTable.load(TTT_Q_PATH)
        except Exception:
            pass
    return QTable(QConfig())


def save_qtable(q: QTable):
    q.save(TTT_Q_PATH)


def ttt_ui():
    st.subheader("Tic-Tac-Toe: Human vs RL Bot")
    st.caption("You are X, the bot is O. Train the bot by playing multiple rounds.")

    if "ttt_board" not in st.session_state:
        st.session_state.ttt_board = [" "] * 9
        st.session_state.ttt_history = []
        st.session_state.ttt_player = "human"  # human goes first
        st.session_state.ttt_q = load_qtable()
        st.session_state.ttt_sched = EpsilonGreedyScheduler(st.session_state.ttt_q.cfg)

    board = st.session_state.ttt_board
    q = st.session_state.ttt_q
    sched = st.session_state.ttt_sched

    cols = st.columns(3)
    for r in range(3):
        for c in range(3):
            i = r * 3 + c
            with cols[c]:
                if st.button(board[i] if board[i] != " " else " ", key=f"cell_{i}"):
                    if board[i] == " " and st.session_state.ttt_player == "human":
                        board[i] = PLAYERS["human"]
                        st.session_state.ttt_history.append((board_to_state(board), None))
                        st.session_state.ttt_player = "bot"

    outcome_now = check_winner(board)
    if outcome_now is None and st.session_state.ttt_player == "bot":
        actions = available_actions(board)
        state = board_to_state(board)
        q_vals = [q.get(state, a) for a in actions]
        action = sched.next_action(actions, q_vals)
        board[action] = PLAYERS["bot"]
        st.session_state.ttt_history.append((state, action))
        st.session_state.ttt_player = "human"
        outcome_now = check_winner(board)

    st.write("")
    if st.button("Reset board"):
        st.session_state.ttt_board = [" "] * 9
        st.session_state.ttt_history = []
        st.session_state.ttt_player = "human"
        st.rerun()

    if outcome_now is not None:
        st.success(f"Game finished: {outcome_now}")
        # Backpropagate rewards
        history = st.session_state.ttt_history
        for idx in range(len(history) - 1, -1, -1):
            state, action = history[idx]
            if action is None:
                # Human move: we can optionally update using negative of bot's reward
                continue
            next_state = None
            next_actions: List[int] = []
            if idx + 1 < len(history):
                next_state, _ = history[idx + 1]
                if next_state is not None:
                    next_actions = available_actions(state_to_board(next_state))
            r = reward_for_outcome(outcome_now, PLAYERS["bot"])
            q.update(state, action, r, next_state, next_actions)
        save_qtable(q)

        # Suggestion: explain bot's learning focus
        st.info("Suggestion: Try to set up forks. The bot learns to block and win lines based on your play history.")

    # Hint system: recommend move for human based on current Q-values as if we were the bot
    if outcome_now is None and st.session_state.ttt_player == "human":
        state = board_to_state(board)
        actions = available_actions(board)
        if actions:
            best, _ = q.best_action(state, actions)
            st.caption(f"Hint: Avoid letting O take cell {best}. Consider playing strategically to create two threats.")


import json

def rps_ui():
    st.subheader("Rock-Paper-Scissors: Bot learns your pattern")
    if "rps_model" not in st.session_state:
        model = FrequencyModel()
        if os.path.exists(RPS_SAVE):
            try:
                with open(RPS_SAVE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.get("counts", {}).items():
                    model.counts[Move(k)] = int(v)
            except Exception:
                pass
        st.session_state.rps_model = model
        st.session_state.rps_history = []

    model: FrequencyModel = st.session_state.rps_model

    col1, col2, col3 = st.columns(3)
    choice = None
    with col1:
        if st.button("🪨 Rock"):
            choice = Move.ROCK
    with col2:
        if st.button("📄 Paper"):
            choice = Move.PAPER
    with col3:
        if st.button("✂️ Scissors"):
            choice = Move.SCISSORS

    if choice is not None:
        bot_move = model.best_response()
        res = outcome(bot_move, choice)
        st.session_state.rps_history.append((choice.value, bot_move.value, res))
        model.update(choice)
        os.makedirs(SAVE_DIR, exist_ok=True)
        with open(RPS_SAVE, "w", encoding="utf-8") as f:
            json.dump({"counts": {m.value: c for m, c in model.counts.items()}}, f)
        st.write(f"You chose {choice.value}, bot chose {bot_move.value}. Result: {'Win' if res==-1 else 'Lose' if res==1 else 'Draw'}")
        st.caption("Suggestion: The bot models your move frequencies and plays the counter. Mix your choices!")

    if st.button("Reset RPS model"):
        st.session_state.rps_model = FrequencyModel()
        if os.path.exists(RPS_SAVE):
            os.remove(RPS_SAVE)
        st.rerun()

    if st.session_state.rps_history:
        st.write("Recent rounds:")
        st.dataframe(st.session_state.rps_history, columns=["You", "Bot", "Score(+1 bot)"])


def main():
    st.title("AI Game Bot with Reinforcement Learning")
    game = st.sidebar.selectbox("Choose a game", ["Tic-Tac-Toe", "Rock-Paper-Scissors"]) 

    if game == "Tic-Tac-Toe":
        ttt_ui()
    else:
        rps_ui()

    st.sidebar.markdown("---")
    st.sidebar.write("Training Data & Models saved under `data/`.")
    # Summary suggestions based on current data
    if os.path.exists(TTT_Q_PATH):
        st.sidebar.caption("Tic-Tac-Toe has learned from your previous games. If it is still easy to beat, try opening in the corner to create fork opportunities. The bot will adapt by blocking and prioritizing winning lines.")
    if os.path.exists(RPS_SAVE):
        st.sidebar.caption("RPS bot tracks your move frequencies. If you repeat moves, it will counter more often. Randomize to reduce predictability.")
    st.sidebar.write("Author: Student Project Demo")


if __name__ == "__main__":
    main()
