import matplotlib.pyplot as plt
import QAgent
import ludopy
import numpy as np
import cv2
import random
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 720

class ReinforcementLearning:
    def __init__(self, episodes : int, epsilon : float,  decayRate: float, learnRate: float, 
                 gamma : float, agent : 'QAgent', numPlayers=1) -> None:
        '''
        Description:
            Constructor for ReinforcementLearning class.    

        Parameters:
         - episodes: an integer that specifies the number of episodes to be run for agent to train.
         - epsilon: a paramater that controls exploration and exploitation in Q-Learning
         - decayRate: it is the rate at which epsilon is reduced 
         - learnRate: it is a parameter in Q-Learning that controls how much Q-Value is to be updated.
         - gamma: it is a discount factor that determines the importance of future rewards 
         - agent: it is an object of Qlearning Agent class 

        Returns:
        None
        '''
        self.episodes = episodes
        self.epsilon = epsilon
        self.no_of_players = numPlayers
        self.decayRate = decayRate
        self.learnRate = learnRate
        self.gamma = gamma
        self.agent = agent
        self.width = WINDOW_WIDTH
        self.height = WINDOW_HEIGHT
        self.game = ludopy.Game(ghost_players=[1, 3])

    def epsilon_decay(self, epsilon: float, decay_rate: float, episode: int) -> float:
        '''
        Description:
            - calculate the new value of epsilon based on the decay rate and current episode number.
        Parameters:
            - epsilon: A float representing the initial value of epsilon
            - decay_rate: A float representing the rate at which epsilon decays over time.
            - episode: An integer representing the current episode number.
        Returns:
            - new epsilon value.
        '''
        return epsilon * np.exp(-decay_rate*episode)
    
    def playAI_Human(self, isWinner: bool) -> None:
        while not isWinner:
            gameState = self.game.get_observation()
            (dice, piecesAvailable, player_pieces, enemy_pieces, player_is_a_winner,
            isWinner), player_i = gameState
            if len(piecesAvailable):
                if self.agent.playerIndex == player_i:
                    selectedPiece = self.agent.update(self.game.players, piecesAvailable, dice)
                    if not selectedPiece in piecesAvailable:
                        self.game.render_environment()
                else:
                    if not self.agent.isTrain:
                        print(f'Dice Roll = {dice}')
                        print(f'Available Pieces to Move = {piecesAvailable}')
                        userInput = int(input("Select which piece you want to move:"))
                        while userInput not in piecesAvailable:
                            print(f'Available Pieces to Move = {piecesAvailable}')
                            userInput = int(input("You've made an invalid choice, select from the avaialable values:"))
                        idx = np.where(piecesAvailable == userInput)
                        selectedPiece = piecesAvailable[int(idx[0])]
                    else:
                        selectedPiece = piecesAvailable[np.random.randint(
                            0, len(piecesAvailable))]
            else:
                selectedPiece = -1
            _, _, _, _, playerIsAWinner, isWinner = self.game.answer_observation(
                selectedPiece)
            
            if self.agent.playerIndex == player_i and selectedPiece != -1:
                self.agent.reward(self.game.players, [selectedPiece])

                '''
            uncomment this code if you want to visualize the agent's training.
            '''
            if not self.agent.isTrain:
                board = self.game.render_environment()
                board = cv2.resize(board, (WINDOW_WIDTH, WINDOW_HEIGHT))
                cv2.imshow("Ludo Board", board)
                cv2.waitKey(10)


    def is_exploration_episode(self):
        return random.random() < self.epsilon

    def teachAgent(self) -> None:
        '''
        Description:
            - Ludo agent is trained in this function with episodes given. Win rate of the agent is calculated, and the agent's policy
            is saved.
        Parameters:
            None
        Returns:
            None
        '''
        agentWinAvg = []
        epsilonLst = []
        idx = []
        winRateLst = []
        agentWonNum = 0
        exploration_count = 0
        exploitation_count = 0
        for episode in range(self.episodes):
            if self.is_exploration_episode():
                exploration_count += 1
        # Perform exploration action
            else:
                exploitation_count += 1
            isWinner = False
            self.game.reset()
            self.playAI_Human(isWinner)
            self.epsilon = self.epsilon_decay(
                epsilon=self.epsilon, decay_rate=self.decayRate, episode=episode)
            epsilonLst.append(self.epsilon)
            self.agent.q_learning.update_epsilon(self.epsilon)
            if self.game.first_winner_was == self.agent.playerIndex:
                agentWinAvg.append(1)
                agentWonNum = agentWonNum + 1
            else:
                agentWinAvg.append(0)
            idx.append(episode)

            # Print some results
            winRate = agentWonNum / len(agentWinAvg)
            winRatePercentage = winRate * 100
            winRateLst.append(winRatePercentage)
        print(np.mean(winRateLst))
        self.agent.save_policy()
        self.PlotGraph(winRateLst, epsilonLst )
        print(exploitation_count)
        print(exploration_count)

    def play_against_human(self)-> None:
        '''
        Description:
            A game is run that allows human player to play with the agent. 
        Parameters:
            None
        Returns:
            None
        '''
        isWinner = False
        self.game.reset()
        self.playAI_Human(isWinner)

    def PlotGraph(self, winRateLst: 'list[float]', epsilonLst: 'list[float]') -> None:
        '''
        Description:
            - From the data collected from the agent's training, it is plotted using matplotlib library
        Parameters:
            - winRateLst: a list that stores wins of agent against random player.
            - epsilonLst: a list contains epsilon rate of agent against random player.
        Returns:
            None
        '''
        # Plot win rates against opponents
        fig, axs = plt.subplots(1)
        axs.set_title("Win Rate with 1000 Episodes")

        axs.set_xlabel('Episodes')
        axs.set_ylabel('Win Rate %')
        axs.plot(winRateLst, color='tab:red')
        axs.legend(['1 Opponent', '2 Opponents', '3 Opponents'])

        # Plot epsilon decay
        fig, axs = plt.subplots(1)
        axs.set_title("Epilson Decay")
        axs.set_xlabel('Episodes')
        axs.set_ylabel('Epsilon')
        axs.plot(epsilonLst, color='tab:red')
        temp = "Epsilon Decay = " + str(self.decayRate)
        axs.legend([temp])
        plt.show()
    
    def alpha_beta_graph(self):

        # The source of this code is from https://stackoverflow.com/questions/22239691/code-for-line-coefficients-given-two-points


        x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        y = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        z = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        w = [1348.099397489995, 1009.2277286535249, 882.199860269852, 853.3586743652411,
            843.0602471138308, 812.0323477103043, 827.2248686759375, 824.7540700673145, 810.3706665228863]

        # Create a 4D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(x, y, z, c=w, cmap='viridis')

        # Add a colorbar
        fig.colorbar(scatter, shrink=0.5, aspect=5)

        # Set the plot labels
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Beta')
        ax.set_zlabel('Evaporation Rate')

        # Show the plot
        plt.savefig("Controlled.png")
        plt.show()
