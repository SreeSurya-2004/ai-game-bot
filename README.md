AI Game Bot using Reinforcement Learning
A mini-project for students:
A friendly Streamlit UI with multiple simple games where a bot learns from your gameplay using tabular Q-learning, then plays against you and offers suggestions.

Features
Rock–Paper–Scissors: learns your transition patterns (what you play after a previous move) and counters them
Coin Flip Predictor: learns your heads/tails transition tendencies and predicts next
Dice Predictor: learns your roll transitions (1–6) and predicts next
Per-game model persistence to models/
Clean, beginner-friendly UI, configurable learning hyperparameters
Quick start
# 1) Create venv (optional)
python3 -m venv .venv && source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
How it works (brief)
Q-learning agent stores a Q-table mapping state-action pairs to values.
For RPS, the state is your previous move (or START). Training rewards the counter-move for your next move.
For Coin Flip and Dice, the state is your previous outcome (or START). Training rewards the actual next outcome you produced.
The bot then picks greedy actions (with low exploration) and suggests better moves by comparing your actions to its argmax policy.
Project structure
ai_game_bot/
  app.py
  requirements.txt
  agents/
    q_learning.py
  games/
    rps.py
    coinflip.py
    dice.py
  utils/
    storage.py
  models/                # created at runtime
  .streamlit/
    config.toml
Packaging
To create a zip for submission:

cd ..
zip -r ai_game_bot.zip ai_game_bot
Notes
This is an educational demo. Policies improve with more data and training.
You can reset models from the Models page in the app.
