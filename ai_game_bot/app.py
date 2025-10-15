from __future__ import annotations

import streamlit as st

from games import rps, coinflip, dice
from utils.storage import clear_model


def sidebar():
    st.sidebar.title("AI Game Bot (RL)")
    page = st.sidebar.radio(
        "Choose a page",
        [
            "Home",
            "Rock-Paper-Scissors",
            "Coin Flip",
            "Dice Predictor",
            "Models",
        ],
    )
    return page


def home_page():
    st.title("AI Game Bot using Reinforcement Learning")
    st.write(
        "Play a game, then train a bot on your gameplay. The bot learns a policy and suggests improvements."
    )
    st.markdown(
        "- Rock–Paper–Scissors: bot learns your transition patterns.\n"
        "- Coin Flip Predictor: bot predicts your next heads/tails based on your sequence.\n"
        "- Dice Predictor: bot predicts your next dice face from your roll history."
    )


def models_page():
    st.header("Model management")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Reset RPS model"):
            clear_model("rps")
            st.success("RPS model cleared")
    with col2:
        if st.button("Reset Coin Flip model"):
            clear_model("coinflip")
            st.success("Coin Flip model cleared")
    with col3:
        if st.button("Reset Dice model"):
            clear_model("dice")
            st.success("Dice model cleared")


PAGE_MAP = {
    "Home": home_page,
    "Rock-Paper-Scissors": rps.run,
    "Coin Flip": coinflip.run,
    "Dice Predictor": dice.run,
    "Models": models_page,
}


def main():
    page = sidebar()
    PAGE_MAP[page]()


if __name__ == "__main__":
    main()
