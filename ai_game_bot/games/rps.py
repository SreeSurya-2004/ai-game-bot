# Improved state: use last 2 moves
def get_state(history):
    return tuple(history[-2:]) if len(history) >= 2 else ("START",)

# Epsilon decay (within training session)
initial_epsilon = 0.2
decay_rate = 0.99

for round_num, current in enumerate(history):
    agent.epsilon = max(0.01, initial_epsilon * (decay_rate ** round_num))
    prev_state = get_state(history[:round_num])
    target_action = Beats(current)
    agent.update(prev_state, target_action, 1.0, current, done=False)

# Visualize Q-table in Streamlit
if st.button("Show Q-table"):
    st.write(agent.q_table)  # assuming QLearningAgent.q_table attribute exists

# Show most exploited patterns to user
if len(history) > 5:
    pattern_counts = {}
    for i in range(2, len(history)):
        pattern = tuple(history[i-2:i])
        next_move = history[i]
        pattern_counts.setdefault(pattern, {}).setdefault(next_move, 0)
        pattern_counts[pattern][next_move] += 1
    st.write("Most frequent move after each pattern:")
    for pattern, moves in pattern_counts.items():
        likely = max(moves, key=moves.get)
        freq = moves[likely] / sum(moves.values())
        st.write(f"After {pattern}, you play {likely} ({freq*100:.1f}%), bot will play {Beats(likely)}")
