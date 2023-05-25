import numpy as np
from qTable import Rewards
from stateSpace import Action, State, StateSpace


class QLearningAgent(StateSpace):
    playerIndex = -1
    q_learning = None
    state = None
    action = None

    def __init__(self, playerIndex : int, gamma: float, learning_rate: float, 
                 epsilon: float, isTrain: bool= True) -> None:
        '''
        Description:
            - Constructor for Qlearning Agent class.
        Parameters:
            - playerIndex: the index of the current player
            - gamma: it is a discount factor that determines the importance of future rewards 
            - learning_rate: it is a parameter in Q-Learning that controls how much Q-Value is to be updated.
            - epsilon: a paramater that controls exploration and exploitation in Q-Learning
            - isTrain: is agent in training mood.
        Returns:
            None
        '''
        super().__init__()
        self.isTrain = isTrain
        self.q_learning = Rewards(len(State), len(Action), gamma=gamma, lr=learning_rate, epsilon=epsilon, train=isTrain)
        self.playerIndex = playerIndex

    def update(self, players: 'np.array', selectedPiece: 'np.array', dice: int) -> 'np.array':
        '''
        Description:
            This function updates the Q-learning agent by choosing the next action to take given the current game
            state and updating the agent's state and action accordingly.
        Parameters:
            - players: the players currently in the game.
            - selectedPiece: the list specifying places available to move.
            - dice: dice value for the player.
        Returns:
            an array indicating updated pieces to move.
        '''
        super().update(players, self.playerIndex, selectedPiece, dice)
        action_table = self.action_table_player.get_action_table()
        temp = self.q_learning.choose_next_action(self.playerIndex, action_table)
        state, action = temp[0], temp[1]
        selectedPiece = self.action_table_player.get_piece_to_move(state, action)
        self.state = state
        self.action = action
        return selectedPiece

    def reward(self, players: 'np.array', selectedPiece: 'np.array') -> None:
        '''
        Description:
            - This function updates the Q-table with the reward for the current state-action pair.
        Parameters:
            - players:  numbers of players in the game.
            - selectedPiece: a list specifying places available to move
        Returns:
        None
        '''
        super().get_possible_actions(players, self.playerIndex, selectedPiece)
        new_action_table = np.nan_to_num(self.action_table_player.get_action_table(), nan=0.0)
        self.q_learning.reward(self.state, new_action_table, self.action)

    def save_policy(self) -> None:
        '''
        Description:
            - saves the agent policy
        Parameters:
            - None
        Returns:
            None
        '''
        self.q_learning.save_policy()
