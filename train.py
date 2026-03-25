import numpy as np
from neural_network import NeuralNetwork
from gridworld import GridWorld

# initialising grid world and neural network
env = GridWorld()
nn = NeuralNetwork(input_size=2, hidden_size=64, output_size=4)

epsilon = 1.0
learning_rate = 0.01
gamma = 0.99
episodes = 1000

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    max_steps = 200
    steps = 0
    while not done and steps < max_steps:
        # steps < max_steps done so that agent doesn't wander around aimlessly
        steps += 1
        # epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.randint(4)
        else:
            q_values = nn.forward(state)
            action = np.argmax(q_values)

        # taking action
        next_state, reward, done = env.step(action)
        next_state = env.get_state()
        total_reward += reward 

        # bellman to generate target
        next_q = nn.forward(next_state)
        target = nn.forward(state).copy()
        target[action][0] = reward + gamma * np.max(next_q) * (not done)

        # training
        nn.forward(state)
        nn.backward(target)
        nn.update(learning_rate)

        state = next_state

    # decaying epsilon / increasing greed
    epsilon = max(0.1, epsilon * 0.995)

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")