import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

def tempg():


# Define the hyperparameters and win rates
    learning_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.7, 0.8, 0.9]
    gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    epsilon = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1]
    win_rates = [68.60278783020127,
        72.2573312527102,
        69.39149114017977,
        68.231850473421,
        70.52165547071914,
        70.43176741134323,
        59.412266731274904,
        50.20583163965508,
        48.91605899951959]


    # Convert the lists to numpy arrays
    learning_rates = np.array(learning_rate)
    gammas = np.array(gamma)
    epsilons = np.array(epsilon)
    win_rates = np.array(win_rates)

    # Reshape the numpy arrays to create a 3D matrix
    lr_matrix = learning_rates.reshape(3, 3, 1)
    gamma_matrix = gammas.reshape(3, 3, 1)
    epsilon_matrix = epsilons.reshape(3, 3, 1)
    win_rate_matrix = win_rates.reshape(3, 3, 1)

    # Stack the matrices along the third dimension
    data = np.concatenate(
        (lr_matrix, gamma_matrix, epsilon_matrix, win_rate_matrix), axis=2)

    # Create the heatmap
    plt.imshow(data[:, :, 3], cmap='coolwarm')

    # Set the x and y axis labels
    plt.xticks(np.arange(3), [0.1, 0.2, 0.3])
    plt.yticks(np.arange(3), [0.1, 0.2, 0.3])

    # Add a color bar
    plt.colorbar()

    # Set the plot title and axis labels
    plt.title('Win Rates by Hyperparameters')
    plt.xlabel('Learning Rates')
    plt.ylabel('Gammas')

    # Show the plot
    plt.show()



def alpha_beta_graph():

    # The source of this code is from https://stackoverflow.com/questions/22239691/code-for-line-coefficients-given-two-points

    learning_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.7, 0.8, 0.9]
    gamma = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    epsilon = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9,1]
    w = [68.60278783020127,
         72.2573312527102,
         69.39149114017977,
         68.231850473421,
         70.52165547071914,
         70.43176741134323,
         59.412266731274904,
         50.20583163965508,
         48.91605899951959]

    # Create a 4D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(learning_rate, gamma, epsilon, c=w, cmap='viridis')

    # Add a colorbar
    fig.colorbar(scatter, shrink=0.5, aspect=5)

    # Set the plot labels
    ax.set_xlabel('learning_rate')
    ax.set_ylabel('gamma')
    ax.set_zlabel('epsilon')


    plt.title("Change in WinRate with respect to Alpha, Beta and Epsilon" )
    # Show the plot
    plt.savefig("Controlled.png")
    plt.show()





def sec():
    q_values = np.array([[1.92482316,  0.2944787,  0, 0, 0, 0, 0, 0,  0, 0, 0, 0],
                         [0,  0.,  0.8,  0.7998552, 0.38286184, 0., 0., 0.75968246, 0.65876347, 0.56113421,
                          3.32423763, -1.24747078, ],
                         [0., 0.,  0.8,  0.76263429,
                          0.60354909, 0., -0.16047999, 0.76000001, 0.69529766,
                          0.95944443,  3.05552635, -1.23417999]])
# replace this with your own Q value array


    for i in q_values:
        print(len(i))


    # Define the labels for the rows and columns
    rows = ["Home", "Safe", "UnSafe"]
    cols = ['HOME_MoveOut',   'HOME_Kill', 'SAFE_Goal',  'SAFE_Globe',
            'SAFE_Protect', 'SAFE_Kill', 'SAFE_Die', 'SAFE_GoalZone', 'UNSAFE_MoveOut',  'UNSAFE_Star', 'UNSAFE_Kill', 'UNSAFE_Die',]

    # Define the colors for the heatmap
    cmap = "YlGnBu"

    # Create the heatmap
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(q_values, cmap=cmap)

    # Set ticks for the middle of the cells
    ax.set_xticks(np.arange(q_values.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(q_values.shape[0]) + 0.5, minor=False)

    # Rotate the x-axis labels
    plt.xticks(rotation=45)

    # Add row and column labels
    ax.set_xticklabels(cols, minor=False)
    ax.set_yticklabels(rows, minor=False)

    # Add a colorbar
    cbar = plt.colorbar(heatmap)

    # Add text to each cell
    for i in range(q_values.shape[0]):
        for j in range(q_values.shape[1]):
            plt.text(j + 0.5, i + 0.5,
                    "{:.2f}".format(q_values[i, j]), ha="center", va="center", color="black")

    # Add a title
    plt.title("State-Action Values")

    # Show the plot
    plt.show()

# alpha_beta_graph()
sec()
