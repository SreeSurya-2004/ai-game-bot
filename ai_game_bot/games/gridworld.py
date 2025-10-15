from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import streamlit as st

from agents.q_learning import QLearningAgent
from utils.storage import model_path

GAME_NAME = "gridworld"
ACTIONS = ["up", "down", "left", "right"]
ACTION_TO_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


@dataclass
class Env:
    size: int = 4
    start: Tuple[int, int] = (0, 0)
    goal: Tuple[int, int] = (3, 3)
    walls: Tuple[Tuple[int, int], ...] = ()

    def reset(self) -> Tuple[int, int]:
        return self.start

    def step(self, state: Tuple[int, int], action: str) -> Tuple[Tuple[int, int], float, bool]:
        dr, dc = ACTION_TO_DELTA[action]
        r, c = state
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            nr, nc = r, c  # bump into wall
        if (nr, nc) in self.walls:
            nr, nc = r, c
        next_state = (nr, nc)
        reward = 10.0 if next_state == self.goal else -1.0
        done = next_state == self.goal
        return next_state, reward, done


def run() -> None:
    st.subheader("GridWorld 4x4 (learns from your path)")
    st.caption("Navigate to the goal. After your run, the bot learns your trajectory and improves.")

    env = Env()
    path = model_path(GAME_NAME)
    agent = QLearningAgent.load(path, actions=ACTIONS)

    if "gw_state" not in st.session_state:
        st.session_state.gw_state = env.reset()
        st.session_state.gw_traj: List[Tuple[Tuple[int, int], str, float]] = []
        st.session_state.gw_done = False

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Human play**")
        grid = np.zeros((env.size, env.size), dtype=int)
        r, c = st.session_state.gw_state
        grid[r, c] = 1
        gr, gc = env.goal
        grid[gr, gc] = 2
        st.write("Legend: You=1, Goal=2")
        st.dataframe(grid, use_container_width=True)

        move = st.radio("Move", ACTIONS, horizontal=True)
        if st.button("Step") and not st.session_state.gw_done:
            ns, reward, done = env.step(st.session_state.gw_state, move)
            st.session_state.gw_traj.append((st.session_state.gw_state, move, reward))
            st.session_state.gw_state = ns
            st.session_state.gw_done = done
        if st.button("Reset run"):
            st.session_state.gw_state = env.reset()
            st.session_state.gw_traj = []
            st.session_state.gw_done = False

    with col2:
        st.markdown("**Train bot from your run**")
        alpha = st.slider("Learning rate (alpha)", 0.05, 1.0, 0.3, 0.05)
        gamma = st.slider("Discount (gamma)", 0.5, 0.99, 0.95, 0.01)
        epsilon = st.slider("Exploration (epsilon)", 0.0, 1.0, 0.1, 0.05)
        agent.alpha, agent.gamma, agent.epsilon = alpha, gamma, epsilon

        if st.button("Train from trajectory"):
            traj = st.session_state.gw_traj
            if not traj:
                st.warning("Do some steps first.")
            else:
                for i, (s, a, r) in enumerate(traj):
                    next_state = traj[i + 1][0] if i + 1 < len(traj) else st.session_state.gw_state
                    done = i + 1 == len(traj) and st.session_state.gw_done
                    agent.update(s, a, r, next_state, done)
                agent.save(path)
                st.success("Bot updated and model saved.")

    st.divider()
    st.markdown("**Bot play (visualize learned policy)**")
    start_pos = st.selectbox("Start position", [(r, c) for r in range(env.size) for c in range(env.size)], index=0)
    if st.button("Run bot from start"):
        pos = start_pos
        steps: List[Tuple[Tuple[int, int], str]] = []
        for _ in range(50):
            action = agent.choose_action(pos, training=False)
            next_state, _, done = env.step(pos, action)
            steps.append((pos, action))
            pos = next_state
            if done:
                steps.append((pos, "GOAL"))
                break
        st.write({"state": [s for s, _ in steps], "action": [a for _, a in steps]})

    if st.session_state.gw_traj:
        st.divider()
        st.markdown("**Suggestions**")
        # Compare your action vs agent argmax at the same states
        rows = []
        for (s, a, _r) in st.session_state.gw_traj:
            best, _ = agent.best_action_value(s)
            if best != a:
                rows.append((s, a, best))
        if rows:
            st.table({"At state": [r[0] for r in rows], "You played": [r[1] for r in rows], "Agent suggests": [r[2] for r in rows]})
        else:
            st.info("Nice! Your choices already match the agent's best actions.")
