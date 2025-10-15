from __future__ import annotations

import streamlit as st

from games import rps
from utils.storage import clear_model


def sidebar():
    st.sidebar.title("AI Game Bot (RL)")
    page = st.sidebar.radio("Choose a page", ["Home", "Rock-Paper-Scissors", "Models"])
    return page


def home_page():
    st.title("AI Game Bot using Reinforcement Learning")
    st.write(
        "Play a game, then train a bot on your gameplay. The bot learns a policy and suggests improvements."
    )
    st.markdown(
        "- Rock–Paper–Scissors: bot learns your transition patterns."
    )


def models_page():
    st.header("Model management")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reset RPS model"):
            clear_model("rps")
            st.success("RPS model cleared")
    with col2:
        st.caption("More model controls will appear as you add games.")


PAGE_MAP = {
    "Home": home_page,
    "Rock-Paper-Scissors": rps.run,
    "Models": models_page,
}


def main():
    page = sidebar()
    PAGE_MAP[page]()


if __name__ == "__main__":
    main()
