import numpy as np

class GridWorld:
    def __init__(self):
        self.grid_size = 4
        self.start = (0, 0)
        self.goal = (3, 3)
        self.walls = [(1, 1), (1, 2), (2, 1)]
        self.agent_pos = self.start

    def reset(self):
        self.agent_pos = self.start
        return np.array(self.agent_pos).reshape(2, 1)

    def step(self, action):
        row, col = self.agent_pos
        if action == 0:
            new_pos = (row - 1, col)
        elif action == 1:
            new_pos = (row + 1, col)
        elif action == 2:
            new_pos = (row, col - 1)
        elif action == 3:
            new_pos = (row, col + 1)

        if new_pos in self.walls or not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
            reward = -1
            done = False
        elif new_pos == self.goal:
            self.agent_pos = new_pos
            reward = 1
            done = True
        else:
            self.agent_pos = new_pos
            reward = 0
            done = False

        return self.agent_pos, reward, done

    def get_state(self):
        return np.array(self.agent_pos).reshape(2, 1)
        # turning it into a 2x1 matrix so that it is initialised for the neural network that has 2 inputs
