# AI Game Bot using Reinforcement Learning (Student Mini Project)

This mini-project demonstrates a friendly UI where a human plays simple games, and a bot learns from human gameplay using reinforcement learning and simple models. Built with Streamlit for quick demos.

## Features
- Tic-Tac-Toe with Q-learning (tabular)
- Rock-Paper-Scissors with a frequency model (predict and counter human choices)
- Hints/suggestions after play
- Persistent learning across sessions (saved in `data/`)

## Quick Start

```bash
pip install -r requirements.txt
./scripts/run.sh
```
Then open your browser to http://localhost:8501

If `./scripts/run.sh` is not executable, run `chmod +x scripts/run.sh`.

## Project Structure
```
app/
  app.py                # Streamlit UI
  games/
    tictactoe.py        # Game logic for Tic-Tac-Toe
    rps.py              # Game logic for Rock-Paper-Scissors
  rl/
    qlearning.py        # Q-learning utilities
scripts/
  run.sh                # Start the Streamlit app
requirements.txt        # Dependencies
data/                   # Models & gameplay are stored here
```

## How it learns
- Tic-Tac-Toe: After each game, the bot back-propagates a reward to states it visited and actions it took. Rewards: win=+1, loss=-1, draw=+0.2.
- Rock-Paper-Scissors: Tracks frequencies of your choices and plays the best response.

## Extend with more games
You can add new games by creating a module in `app/games/` with:
- A state representation
- Legal actions
- Transition and reward functions
- A small UI in `app/app.py`

## Notes
- Designed for teaching. Algorithms are intentionally simple (no deep RL) to make it easy to explain.
- For more advanced RL (DQN, policy gradients), add PyTorch or TensorFlow and create an environment wrapper.
