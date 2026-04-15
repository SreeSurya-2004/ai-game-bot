# AI Game Platform using Reinforcement Learning

A mini-project for students: a friendly web-based game platform where an AI bot learns from gameplay using tabular Q-learning and gives hints in real time. This project is built with Flask and Flask-SocketIO, and currently supports multiple simple games through a browser interface. [1][2]

## Features

- **Tic-Tac-Toe**: the bot learns from board states and suggests the best next move using a Q-table. [2]
- **Memory Game**: the bot tracks visible card patterns and learns which hidden card to suggest next. [2]
- **Snake**: the bot learns simple directional decisions such as going straight, left, or right based on the game state. [2]
- **Real-time hint system** using Flask routes and SocketIO events. [1]
- **Persistent Q-table storage** so learned behavior can be saved and reused across sessions in `qtables_saved/`. [2]
- **Beginner-friendly project structure** with backend, templates, static files, and reinforcement learning logic separated clearly. [3]

## Quick start

1. **Create a virtual environment (optional)**

```bash
python3 -m venv .venv && source .venv/bin/activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the app**

```bash
python app.py
```

4. **Open in browser**

```text
http://127.0.0.1:5000/
```

The project uses Flask, Flask-SocketIO, and eventlet as its core dependencies. [4]

## How it works

The AI bot uses tabular Q-learning with separate Q-tables for each game. It receives the current game state, predicts a useful next action, and then updates the Q-table after the player's move based on the reward. [2][1]

- **For Tic-Tac-Toe**: the state is the current board layout, and the action is the suggested board index for the next move. Rewards are updated based on wins, losses, draws, or blocking useful moves. [2][1]
- **For Memory Game**: the state is the visible card layout, and the action is the suggested hidden card index. Rewards are based on whether the chosen reveal is useful. [2][1]
- **For Snake**: the state is encoded from the current movement/game situation, and the action is one of three directional choices: straight, left, or right. Rewards are based on survival and food collection. [2][1]

The backend can return hints through a normal Flask API endpoint (`/send_action`) and also through a SocketIO event (`player_action`) for more interactive gameplay. [1]

## Project structure

```text
AI_Game_Platform/
├── app.py                  # Flask backend and SocketIO integration
├── ai_bot.py               # AI bot logic with Q-learning and hint generation
├── requirements.txt        # Python dependencies
├── static/
│   ├── style.css           # CSS styling
│   └── script.js           # Frontend JavaScript
├── templates/
│   ├── index.html          # Home page
│   └── game.html           # Game interface
└── qtables_saved/          # Created at runtime for saved Q-tables
```

This structure is based on the project notes you attached and the Python files currently used by the app. [3][1][2]

## API flow

- The frontend sends player actions to the backend. [1]
- The backend asks the AI bot for a hint using `get_hint()`. [2][1]
- After the move, the backend calculates a reward and updates the game-specific Q-table. [2][1]
- The learned Q-tables are saved so the bot can improve over time. [2]

## Packaging

To create a zip file for submission:

```bash
cd ..
zip -r AI_Game_Platform.zip AI_Game_Platform
```

## Notes

- This is an educational demo project for learning reinforcement learning concepts through games. [2]
- The bot combines heuristic suggestions with tabular Q-learning logic. [2]
- Q-table learning improves with more gameplay and repeated updates. [2]
- The project currently depends on local browser templates and static frontend files in addition to the Python backend. [3]
- Your `app.py` imports `AIBot` from `ai_bot`, so the filename should remain `ai_bot.py` unless the import is changed. [1][2]

## Learning goals

This project helps students understand:

- Flask-based web application development, [1]
- real-time communication using Flask-SocketIO, [1][4]
- tabular Q-learning in simple games, [2]
- reward-based policy updates for different game environments, [2]
- and how AI-based hints can be integrated into interactive browser games. [2][1]
