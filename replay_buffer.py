import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        # deque automatically discards the oldest experience when full
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        # store one experience — works for both baseline and MARL
        # (they store different numbers of fields, but push handles both)
        self.buffer.append(args)

    def sample(self, batch_size):
        # return a random subset of experiences — this is what breaks
        # temporal correlation and makes training stable
        return random.sample(self.buffer, batch_size)

    def is_ready(self, batch_size):
        # don't start training until the buffer has enough experiences to sample from
        return len(self.buffer) >= batch_size

    def __len__(self):
        return len(self.buffer)
