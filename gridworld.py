import numpy as np

class GridWorld:
    def __init__(self):
        self.grid_size = 6
        self.start = (0, 0)
        self.goal = (5, 5)
        self.walls = [(2, 2), (2, 3), (5, 3)]
        self.agent_pos = self.start
        self.key_pos = None
        self.has_key = False


    def reset(self):
        self.agent_pos = self.start
        self.has_key = False

        # place key randomly. Not on start, goal, or a wall
        while True:
            candidate = (np.random.randint(0, self.grid_size),
                         np.random.randint(0, self.grid_size))
            if candidate != self.start and candidate != self.goal and candidate not in self.walls:
                self.key_pos = candidate
                break

        return self.get_ground_state()


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
            reward = -0.1
            done = False
        else:
            self.agent_pos = new_pos

            if not self.has_key and self.agent_pos == self.key_pos:
                self.has_key = True
                reward = 0.5
                done = False
            elif self.has_key and self.agent_pos == self.goal:
                reward = 1.0
                done = True
            else:
                reward = -0.01
                done = False

        return self.agent_pos, reward, done


    def get_ground_state(self):
        # ground agent sees: its position + whether it has the key
        return np.array([self.agent_pos[0], self.agent_pos[1], int(self.has_key)],
                        dtype=float).reshape(3, 1) # initialise ground vector for NN

    def get_sky_obs(self):
        # sky agent sees: agent position + key position
        # -1,-1 signals key already collected so sky NN learns its job is done
        if self.has_key:
            kx, ky = -1, -1
        else:
            kx, ky = self.key_pos
        return np.array([self.agent_pos[0], self.agent_pos[1], kx, ky],
                        dtype=float).reshape(4, 1) # intiialise sky vector for NN
