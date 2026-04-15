from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from ai_bot import AIBot

app = Flask(__name__)
socketio = SocketIO(app)
bot = AIBot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/game/<game_name>')
def game_page(game_name):
    return render_template('game.html', game_name=game_name)

@app.route('/send_action', methods=['POST'])
def send_action():
    """
    Expects JSON from webpage with keys:
    - game: "tictactoe", "memory", "snake"
    - action: current player action payload
    - prev_state: previous game state
    - suggested_action: bot's previous suggestion
    - new_state: updated game state after player move
    - done: game over flag
    - extra: optional (for snake: alive, ate_food)
    """
    data = request.json
    game = data.get('game')
    prev_state = data.get('prev_state')
    suggested_action = data.get('suggested_action')
    new_state = data.get('new_state')
    done = data.get('done', False)
    extra = data.get('extra', {})

    # Get bot hint
    hint = bot.get_hint(data.get('action'))

    # Update RL Q-table
    if game == "tictactoe" and prev_state is not None:
        reward = bot.tt_get_reward(prev_state, suggested_action, new_state, done)
        bot.tt_update(prev_state, suggested_action, reward, new_state, done)

    elif game == "memory" and prev_state is not None:
        reward = bot.mem_get_reward(prev_state, suggested_action, new_state, done)
        bot.mem_update(prev_state, suggested_action, reward, new_state, done)

    elif game == "snake":
        alive = extra.get('alive', True)
        ate_food = extra.get('ate_food', False)
        reward = bot.snake_get_reward(alive, ate_food)
        prev_key = bot.snake_state_key(prev_state)
        new_key = bot.snake_state_key(new_state)
        bot.snake_update(prev_key, suggested_action, reward, new_key, done)

    # Save Q-tables to persist learning
    bot.save_qtables()

    return jsonify({'hint': hint})

@socketio.on('player_action')
def handle_player_action(data):
    """
    SocketIO handler for real-time gameplay.
    Data payload same as /send_action endpoint.
    """
    game = data.get('game')
    prev_state = data.get('prev_state')
    suggested_action = data.get('suggested_action')
    new_state = data.get('new_state')
    done = data.get('done', False)
    extra = data.get('extra', {})

    hint = bot.get_hint(data['action'])

    # RL update
    if game == "tictactoe" and prev_state is not None:
        reward = bot.tt_get_reward(prev_state, suggested_action, new_state, done)
        bot.tt_update(prev_state, suggested_action, reward, new_state, done)

    elif game == "memory" and prev_state is not None:
        reward = bot.mem_get_reward(prev_state, suggested_action, new_state, done)
        bot.mem_update(prev_state, suggested_action, reward, new_state, done)

    elif game == "snake":
        alive = extra.get('alive', True)
        ate_food = extra.get('ate_food', False)
        reward = bot.snake_get_reward(alive, ate_food)
        prev_key = bot.snake_state_key(prev_state)
        new_key = bot.snake_state_key(new_state)
        bot.snake_update(prev_key, suggested_action, reward, new_key, done)

    bot.save_qtables()

    emit('bot_hint', {'hint': hint})

if __name__ == '__main__':
    socketio.run(app, debug=True)
