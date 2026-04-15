# AI Game Platform using Reinforcement Learning

An interactive browser-based game platform that demonstrates how reinforcement learning can enhance gameplay through intelligent, real-time hints. Built with Flask and Flask-SocketIO, this project features a Q-learning powered AI bot that learns from player interactions, improves over time, and provides move suggestions across classic games like Tic-Tac-Toe, Memory Game, and Snake.

## Features

- **Tic-Tac-Toe**: the bot learns from board states and suggests the best next move using a Q-table.
- **Memory Game**: the bot tracks visible card patterns and learns which hidden card to suggest next.
- **Snake**: the bot learns simple directional decisions such as going straight, left, or right based on the game state.
- **Real-time hint system** using Flask routes and SocketIO events.
- **Persistent Q-table storage** so learned behavior can be saved and reused across sessions in `qtables_saved/`.
- **Beginner-friendly project structure** with backend, templates, static files, and reinforcement learning logic separated clearly.

## Quick start

### Local execution

1. **Open the project folder**

```bash
cd AI_Game_Platform
```

2. **Create a virtual environment (optional but recommended)**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
pip install requests
```

4. **Run the app**

```bash
python app.py
```

5. **Open in browser**

```text
http://127.0.0.1:5000/
```

### Execution in GitHub Codespaces

1. Open the repository in **GitHub Codespaces**.
2. Open the terminal.
3. Move into the project folder if needed:

```bash
cd /workspaces/ai-game-bot/AI_Game_Platform
```

4. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

5. Install dependencies:

```bash
pip install -r requirements.txt
pip install requests
```

6. Run the project:

```bash
python app.py
```

7. Open the forwarded port or browser preview for port **5000**.

## How it works

The AI bot uses tabular Q-learning with separate Q-tables for each game. It receives the current game state, predicts a useful next action, and then updates the Q-table after the player's move based on the reward.

- **For Tic-Tac-Toe**: the state is the current board layout, and the action is the suggested board index for the next move. Rewards are updated based on wins, losses, draws, or blocking useful moves.
- **For Memory Game**: the state is the visible card layout, and the action is the suggested hidden card index. Rewards are based on whether the chosen reveal is useful.
- **For Snake**: the state is encoded from the current movement or game situation, and the action is one of three directional choices: straight, left, or right. Rewards are based on survival and food collection.

The backend can return hints through a normal Flask API endpoint (`/send_action`) and also through a SocketIO event (`player_action`) for more interactive gameplay.

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

## API flow

- The frontend sends player actions to the backend.
- The backend asks the AI bot for a hint using `get_hint()`.
- After the move, the backend calculates a reward and updates the game-specific Q-table.
- The learned Q-tables are saved so the bot can improve over time.

## Packaging

To create a zip file for submission:

```bash
cd ..
zip -r AI_Game_Platform.zip AI_Game_Platform
```

## Notes

- This is an educational demo project for learning reinforcement learning concepts through games.
- The bot combines heuristic suggestions with tabular Q-learning logic.
- Q-table learning improves with more gameplay and repeated updates.
- The project currently depends on local browser templates and static frontend files in addition to the Python backend.
- Since `ai_bot.py` imports `requests`, install it before running the app.

## Learning goals

This project helps students understand:

- Flask-based web application development,
- real-time communication using Flask-SocketIO,
- tabular Q-learning in simple games,
- reward-based policy updates for different game environments,
- and how AI-based hints can be integrated into interactive browser games.
