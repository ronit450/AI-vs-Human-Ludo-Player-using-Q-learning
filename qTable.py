'''
Code References: 
https://github.com/niconielsen32/LudoRL
https://www.youtube.com/watch?v=piTnn3dJ9QE&t=1874s
'''

import random
import numpy as np
from stateSpace import Action
import pickle

class Rewards():

    rewardsTable = np.zeros(len(Action))
    q_table = None
    epoch = 0
    iteration = 0

    def __init__(self, states: int, actions: int, gamma: float, lr: float, epsilon: float, train: bool= True) -> None:
        '''
        Description:
            - constructor for Rewards class.
        Parameters:
            - states: 
            - actions: 
            - gamma: 
            - lr: 
            - epsilon: 
            - train: 
        Returns:
        None
        '''
        super().__init__()
        if train:
            self.q_table = np.zeros([states, actions])
        else:
            self.q_table = self.loadPolicy("policy_AI")

        self.epsilonGreedy = epsilon
        self.gamma = gamma
        self.lr = lr

        self.max_expected_reward = 0

        VERY_BAD = -0.8
        BAD = -0.4
        GOOD = 0.4
        VERY_GOOD = 1.2

        self.rewardsTable[Action.SAFE_MoveOut.value] = 0.4
        self.rewardsTable[Action.SAFE_MoveDice.value] = 0.01
        self.rewardsTable[Action.SAFE_Goal.value] = 0.8
        self.rewardsTable[Action.SAFE_Star.value] = 0.8
        self.rewardsTable[Action.SAFE_Globe.value] = 0.4
        self.rewardsTable[Action.SAFE_Protect.value] = 0.2
        self.rewardsTable[Action.SAFE_Kill.value] = 1.5
        self.rewardsTable[Action.SAFE_Die.value] = -0.5
        self.rewardsTable[Action.SAFE_GoalZone.value] = 0.2

        self.rewardsTable[Action.UNSAFE_MoveOut.value] = self.rewardsTable[Action.SAFE_MoveOut.value] + BAD
        self.rewardsTable[Action.UNSAFE_MoveDice.value] = self.rewardsTable[Action.SAFE_MoveDice.value] + BAD
        self.rewardsTable[Action.UNSAFE_Star.value] = self.rewardsTable[Action.SAFE_Star.value] + BAD
        self.rewardsTable[Action.UNSAFE_Globe.value] = self.rewardsTable[Action.SAFE_Globe.value] + GOOD
        self.rewardsTable[Action.UNSAFE_Protect.value] = self.rewardsTable[Action.SAFE_Protect.value] + GOOD
        self.rewardsTable[Action.UNSAFE_Kill.value] = self.rewardsTable[Action.SAFE_Kill.value] + GOOD
        self.rewardsTable[Action.UNSAFE_Die.value] = self.rewardsTable[Action.SAFE_Die.value] + VERY_BAD
        self.rewardsTable[Action.UNSAFE_GoalZone.value] = self.rewardsTable[Action.SAFE_GoalZone.value] + GOOD
        self.rewardsTable[Action.UNSAFE_Goal.value] = self.rewardsTable[Action.SAFE_Goal.value] + GOOD

        self.rewardsTable[Action.HOME_MoveOut.value] = self.rewardsTable[Action.SAFE_MoveOut.value] + VERY_GOOD
        self.rewardsTable[Action.HOME_MoveDice.value] = self.rewardsTable[Action.SAFE_MoveDice.value] + VERY_BAD
        self.rewardsTable[Action.HOME_Star.value] = self.rewardsTable[Action.SAFE_Star.value] + VERY_BAD
        self.rewardsTable[Action.HOME_Globe.value] = self.rewardsTable[Action.SAFE_Globe.value] + VERY_BAD
        self.rewardsTable[Action.HOME_Protect.value] = self.rewardsTable[Action.SAFE_Protect.value] + VERY_BAD
        self.rewardsTable[Action.HOME_Kill.value] = self.rewardsTable[Action.SAFE_Kill.value] + VERY_BAD
        self.rewardsTable[Action.HOME_Die.value] = self.rewardsTable[Action.SAFE_Die.value] + VERY_BAD
        self.rewardsTable[Action.HOME_GoalZone.value] = self.rewardsTable[Action.SAFE_GoalZone.value] + VERY_BAD
        self.rewardsTable[Action.HOME_Goal.value] = self.rewardsTable[Action.SAFE_Goal.value] + VERY_BAD

    def update_epsilon(self, newEpsilon: float):
        '''
        Description:
            - updates the epsilon value
        Parameters:
            - newEpsilon: new epsilon value
        Returns:
        None
        '''
        self.epsilonGreedy = newEpsilon

    def getStateAction(self, value, array)->tuple:
        '''
        Description:
            - for the given value and array, it returns a state and action tuple if it exists.
            In case of NaN value, it returns (-1,-1)
        Parameters:
            - value: value to find in the array
            - array: an array
        Returns:
            - A tuple (state, action)
        '''
        if np.isnan(value):
            return (-1, -1)
        idx = np.where(array == value)
        random_idx = random.randint(0, len(idx[0]) - 1)
        state = idx[0][random_idx]
        action = idx[1][random_idx]
        return (state, action)

    def choose_next_action(self, player, action_table: 'np.array'):
        '''
        Description:
            - This function is used to choose the next action that the agent will take.
        Parameters:
            - player: This is an integer representing the index of the player for whom the function is choosing the next action.
            - action_table: a table specifying available actions
        Returns:
            - A tuple containing state and action
        '''
        q_table_options = np.multiply(self.q_table, action_table)

        if random.uniform(0, 1) < self.epsilonGreedy:
            self.iteration = self.iteration + 1
            nz = action_table[np.logical_not(np.isnan(action_table))]
            randomValue = nz[random.randint(0, len(nz) - 1)]
            state, action = self.getStateAction(
                randomValue, action_table)
        else:
            maxVal = np.nanmax(q_table_options)
            if not np.isnan(maxVal):
                state, action = self.getStateAction(
                    maxVal, q_table_options)
            else:
                nz = action_table[np.logical_not(np.isnan(action_table))]
                random_value = nz[random.randint(0, len(nz) - 1)]
                state, action = self.getStateAction(
                    random_value, action_table)
        return (state, action)

    def reward(self, state: str, new_action_table: 'np.array', action: str)->None:
        '''
        Description:
            - This function updates the Q-table based on the reward received from a previous action taken in a certain state. 
        Parameters:
            - state: A string representing the current state of the game.  
            - new_action_table: A 2-dimensional NumPy array representing the action table for the current state 
                after the agent has taken an action
            - action:  A string representing the action taken by the agent in the current state
        Returns:
        None
        '''
        state = int(state)
        action = int(action)

        # Q-learning equation
        reward = self.rewardsTable[action]
        # Q-learning
        estimate_of_optimal_future_value = np.max(
            self.q_table * new_action_table)
        old_q_value = self.q_table[state, action]
        delta_q = (self.lr * (reward + self.gamma *
                             estimate_of_optimal_future_value - old_q_value))

        self.max_expected_reward += reward

        # Update the Q table from the new action taken in the current state
        self.q_table[state, action] = old_q_value + delta_q
        # print("update q table, state: {0}, action:{1}".format(state,action))

    def save_policy(self) -> None:
        '''
        Description:
            - saves the policy of the agent
        Parameters:
            None
        Returns:
        None
        '''
        fw = open('policy_' + str("AI"), 'wb')
        pickle.dump(self.q_table, fw)
        fw.close()

    def loadPolicy(self, file: str) -> 'np.array':
        '''
        Description:
            - loads the save policy of the agent.
        Parameters:
            - file: a file path of the policy
        Returns:
        None
        '''
        fr = open(file, 'rb')
        states_value = pickle.load(fr)
        fr.close()
        return states_value
