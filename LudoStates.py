import numpy as np

# Define state space dimensions
player_turn = 4  # 4 players in total
num_pieces = 4  # each player has 4 pieces
board_size = 56  # board size is 56 for Ludo

# Initialize state space
# each piece has 2 features: position and state
state_shape = (player_turn, num_pieces, 2)
# initialize with -1 (not on board)
state_space = np.zeros((board_size,) + state_shape, dtype=int) - 1

# Set initial positions of pieces
for player in range(player_turn):
    # set piece position to start and state to not in play
    state_space[0, player, :] = [0, 0]
