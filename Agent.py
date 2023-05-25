import numpy as np
import random
# from ludo_engine import LudoGame
# assuming you have a ludo_engine module
import ludopy

# Define the possible states
NUM_STATES = 27
states = range(NUM_STATES)

# Define the possible actions
NUM_ACTIONS = 4
actions = range(NUM_ACTIONS)

# Set the learning parameters
learning_rate = 0.8
discount_factor = 0.95
exploration_rate = 1.0
exploration_decay_rate = 0.99

# Set the number of episodes and maximum number of steps per episode
num_episodes = 5000
max_steps_per_episode = 100

# Initialize the Q-table
Q = np.zeros((NUM_STATES, NUM_ACTIONS))

# Initialize the Ludo board and set the initial state of the game
game = ludopy.Game()
game.reset()


def choose_action(state, exploration_rate):
    # Choose an action using an epsilon-greedy policy based on the Q-values
    if random.uniform(0, 1) < exploration_rate:
        # Explore: choose a random action
        action = random.choice(actions)
    else:
        # Exploit: choose the action with the highest Q-value for this state
        action = np.argmax(Q[state, :])
    return action


def play_game():
    # Play a game of Ludo against a random opponent
    game.reset()
    while not game.is_over():
        state = game.get_state()
        action = choose_action(state, exploration_rate)
        reward, done = game.move(action)
        new_state = game.get_state()
        # Update the Q-value for the current state-action pair using the Bellman equation
        Q[state, action] = Q[state, action] + learning_rate * \
            (reward + discount_factor *
             np.max(Q[new_state, :]) - Q[state, action])
        state = new_state



def train():
    # Train the agent for the specified number of episodes
    for episode in range(num_episodes):
        # Set the initial state of the game
        state = game.get_state()
        for step in range(max_steps_per_episode):
            # Choose an action and take it
            action = choose_action(state, exploration_rate)
            reward, done = game.move(action)
            new_state = game.get_state()
            # Update the Q-value for the current state-action pair using the Bellman equation
            Q[state, action] = Q[state, action] + learning_rate * \
                (reward + discount_factor *
                 np.max(Q[new_state, :]) - Q[state, action])
            state = new_state
            # Check if the game is over
            if done:
                break
        # Update the exploration rate
        exploration_rate *= exploration_decay_rate
        # Evaluate the performance of the agent
        if episode % 100 == 0:
            total_rewards = 0
            for i in range(100):
                # Play a game and calculate the total reward
                game.reset()
                while not game.is_over():
                    state = game.get_state()
                    action = np.argmax(Q[state, :])
                    reward, done = game.move(action)
                    total_rewards += reward
            avg_reward = total_rewards / 100
            print("Episode: {}, Average Reward: {}".format(episode, avg_reward))


# Train the agent
train()
