from RL import ReinforcementLearning
from QAgent import QLearningAgent


def trainingModel(gamma : float, lr : float, epsilon : float, decay : float, episodes : int) -> None:
    '''
    Description:
    Training Model for QLearning Agent. This function begins with random policy
    Uses QLearning to train our agent over number of episodes to come up with
    the best possible policy.

    Parameters:
    - gamma: float rate of gamma hyperparameter
    - lr: float learning rate, hyperparameter
    - epsilon: float Initial rate of epsilon for epsilon greedy
    - decay: float decay rate for epsilon
    - episodes: integer number of episodes for training

    Returns:
    None
    '''
    agent = QLearningAgent(0, gamma, lr, epsilon)
    Qlearn = ReinforcementLearning(episodes, epsilon, decay,lr, gamma, agent)
    Qlearn.teachAgent()


def testingModel(gamma : float, lr : float, epsilon : float, decay : float, episodes : int) -> None:
    '''
    Description:
    Testing Model to test the effectiveness of the policy learned by our agent by making it play
    against Human.

    Parameters:
    - gamma: float rate of gamma hyperparameter
    - lr: float learning rate, hyperparameter
    - epsilon: float Initial rate of epsilon for epsilon greedy
    - decay: float decay rate for epsilon
    - episodes: integer number of episodes for training

    Returns:
    None
    '''
    agent = QLearningAgent(0, gamma, lr, epsilon, False)
    Qlearn = ReinforcementLearning(episodes, epsilon, decay,lr, gamma, agent)
    Qlearn.play_against_human()

def main() -> None:
    '''
    Description:
    Main function to call in order to run the entire project.

    Parameters:

    Returns:
    None
    '''

    learning_rate = 0.3
    gamma = 0.7
    epsilon = 0.1
    epsilon_decay_rate = 0.1
    episodes = 10
    # for i in range(len(learning_rate)):
    trainingModel(gamma, learning_rate, epsilon, epsilon_decay_rate, episodes)    
    testingModel(gamma, learning_rate, epsilon, epsilon_decay_rate, episodes)


if __name__ == '__main__':
    main()