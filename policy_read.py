import matplotlib.pyplot as plt
import numpy as np

# Generate some sample data (replace with your actual data)
num_episodes = 10000
episodes = np.arange(1, num_episodes + 1)
exploration_proportions = np.random.uniform(0, 1, size=num_episodes)
print(exploration_proportions)
exploitation_proportions = 1 - exploration_proportions

# Create the stacked area graph
plt.stackplot(episodes, exploration_proportions,
              exploitation_proportions, labels=['Exploration', 'Exploitation'])
plt.xlabel('Episodes')
plt.ylabel('Proportion')
plt.title('Exploration vs. Exploitation')
plt.legend(loc='upper right')

# Show the graph
plt.show()
