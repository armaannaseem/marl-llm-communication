import numpy as np
from neural_network import NeuralNetwork
from gridworld import GridWorld
from sky_agent import SkyAgent
from replay_buffer import ReplayBuffer

# environment and agents
env = GridWorld()
sky = SkyAgent(hidden_size=64, learning_rate=0.01)
ground_nn = NeuralNetwork(input_size=7, hidden_size=64, output_size=4)

# target networks — frozen copies updated every TARGET_UPDATE_FREQ steps
target_ground_nn = NeuralNetwork(input_size=7, hidden_size=64, output_size=4)
target_sky_nn    = NeuralNetwork(input_size=4, hidden_size=64, output_size=4)
target_ground_nn.copy_weights_from(ground_nn)
target_sky_nn.copy_weights_from(sky.nn)

buffer = ReplayBuffer(capacity=10000)

BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 100

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
    sky_obs = env.get_sky_obs()
    total_reward = 0
    done = False
    steps = 0

    while not done and steps < max_steps:
        steps += 1
        total_steps += 1

        # action selection (not stored in buffer — we store raw obs)
        comm = sky.get_comm(sky_obs)
        ground_input = np.vstack([state, comm])
        q_values = ground_nn.forward(ground_input)

        if np.random.rand() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(q_values)

        # step environment
        _, reward, done = env.step(action)
        next_state = env.get_ground_state()
        next_sky_obs = env.get_sky_obs()
        total_reward += reward

        # store raw observations — comm is recomputed at training time
        buffer.push(state, sky_obs, action, reward, next_state, next_sky_obs, done)

        if buffer.is_ready(BATCH_SIZE):
            batch = buffer.sample(BATCH_SIZE)
            states_b, sky_obs_b, actions_b, rewards_b, next_states_b, next_sky_obs_b, dones_b = zip(*batch)

            # stack into matrices
            states_mat        = np.hstack(states_b)
            sky_obs_mat       = np.hstack(sky_obs_b)
            next_states_mat   = np.hstack(next_states_b)
            next_sky_obs_mat  = np.hstack(next_sky_obs_b)

            # target networks compute stable next-Q values
            next_comm_target   = target_sky_nn.forward(next_sky_obs_mat)
            next_ground_target = np.vstack([next_states_mat, next_comm_target])
            next_q             = target_ground_nn.forward(next_ground_target)

            # live sky NN computes comm for current states (stores sky activations)
            comm_batch    = sky.nn.forward(sky_obs_mat)
            ground_inputs = np.vstack([states_mat, comm_batch])

            # live ground NN forward (stores ground activations for backprop)
            current_q = ground_nn.forward(ground_inputs)

            # build bellman targets
            targets = current_q.copy()
            for i in range(BATCH_SIZE):
                targets[actions_b[i], i] = (rewards_b[i]
                    + gamma * np.max(next_q[:, i]) * (not dones_b[i]))

            # backprop ground NN
            ground_nn.backward(targets)
            ground_nn.update(learning_rate / BATCH_SIZE)

            # backprop sky NN using gradient that flowed back through ground NN inputs
            comm_grad = ground_nn.dx[3:]   # (4, batch_size)
            sky.nn.backward_from_grad(comm_grad)
            sky.nn.update(sky.learning_rate / BATCH_SIZE)

        # periodically freeze updated weights into both target networks
        if total_steps % TARGET_UPDATE_FREQ == 0:
            target_ground_nn.copy_weights_from(ground_nn)
            target_sky_nn.copy_weights_from(sky.nn)

        state = next_state
        sky_obs = next_sky_obs

    key_log.append(int(env.has_key))
    epsilon = max(0.1, epsilon * 0.995)
    reward_log.append(total_reward)

    if episode % 100 == 0:
        avg = np.mean(reward_log[-100:]) if len(reward_log) >= 100 else np.mean(reward_log)
        key_rate = np.mean(key_log[-100:]) * 100 if len(key_log) >= 100 else np.mean(key_log) * 100
        print(f"Episode {episode} | Reward: {total_reward:.2f} | "
              f"Avg100: {avg:.2f} | Key%: {key_rate:.1f}% | Epsilon: {epsilon:.3f}")

np.save("marl_rewards.npy", np.array(reward_log))
np.save("marl_keys.npy", np.array(key_log))
print("Training complete. Rewards saved to marl_rewards.npy and key success rate saved to marl_keys.npy")
