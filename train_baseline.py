import numpy as np
from neural_network import NeuralNetwork
from gridworld import GridWorld
from replay_buffer import ReplayBuffer

# environment and networks
env = GridWorld()
ground_nn = NeuralNetwork(input_size=3, hidden_size=64, output_size=4)
target_nn = NeuralNetwork(input_size=3, hidden_size=64, output_size=4)
target_nn.copy_weights_from(ground_nn)  # target starts as a copy of ground

buffer = ReplayBuffer(capacity=10000)

BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 100   # copy ground → target every 100 steps

epsilon = 1.0
learning_rate = 0.01
gamma = 0.99
episodes = 3000
max_steps = 300

reward_log = []
key_log = []
total_steps = 0

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0

    while not done and steps < max_steps:
        steps += 1
        total_steps += 1

        # select action
        q_values = ground_nn.forward(state)
        if np.random.rand() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(q_values)

        # step environment
        _, reward, done = env.step(action)
        next_state = env.get_ground_state()
        total_reward += reward

        # store experience in replay buffer
        buffer.push(state, action, reward, next_state, done)

        # only train once buffer has enough experiences
        if buffer.is_ready(BATCH_SIZE):
            batch = buffer.sample(BATCH_SIZE)
            states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*batch)

            # stack individual (3,1) vectors into (3, batch_size) matrices
            states_mat      = np.hstack(states_b)
            next_states_mat = np.hstack(next_states_b)

            # target network computes stable next-Q values (frozen weights)
            next_q = target_nn.forward(next_states_mat)       # (4, batch_size)

            # ground network computes current Q and stores activations for backprop
            current_q = ground_nn.forward(states_mat)         # (4, batch_size)

            # build bellman targets — only update the Q-value for the action taken
            targets = current_q.copy()
            for i in range(BATCH_SIZE):
                targets[actions_b[i], i] = (rewards_b[i]
                    + gamma * np.max(next_q[:, i]) * (not dones_b[i]))

            # backprop on the batch
            ground_nn.backward(targets)
            ground_nn.update(learning_rate / BATCH_SIZE)

        # periodically copy ground network weights into the target network
        if total_steps % TARGET_UPDATE_FREQ == 0:
            target_nn.copy_weights_from(ground_nn)

        state = next_state

    key_log.append(int(env.has_key))
    epsilon = max(0.1, epsilon * 0.995)
    reward_log.append(total_reward)

    if episode % 100 == 0:
        avg = np.mean(reward_log[-100:]) if len(reward_log) >= 100 else np.mean(reward_log)
        key_rate = np.mean(key_log[-100:]) * 100 if len(key_log) >= 100 else np.mean(key_log) * 100
        print(f"Episode {episode} | Reward: {total_reward:.2f} | "
              f"Avg100: {avg:.2f} | Key%: {key_rate:.1f}% | Epsilon: {epsilon:.3f}")

np.save("baseline_rewards.npy", np.array(reward_log))
np.save("baseline_keys.npy", np.array(key_log))
print("Training complete. Rewards saved to baseline_rewards.npy, and keys success rate saved to baseline_keys.npy")