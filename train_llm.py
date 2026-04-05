import numpy as np
import json
import os
from dotenv import load_dotenv
from google import genai
from neural_network import NeuralNetwork
from gridworld import GridWorld
from replay_buffer import ReplayBuffer

# load API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None


# load cache if it exists — allows running without API key once warmed up
CACHE_FILE = "llm_cache.json"
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        cache = json.load(f)
    print(f"Loaded {len(cache)} cached LLM responses.")
else:
    cache = {}

def get_llm_comm(sky_obs):
    coords = tuple(int(x) for x in sky_obs.flatten())
    cache_key = str(coords)

    if cache_key in cache:
        return np.array(cache[cache_key]).reshape(4, 1)

    agent_x, agent_y, key_x, key_y = coords

    if key_x == -1:
        task = "The key has already been collected. Guide the agent toward the exit at row=5, col=5."
        key_info = "Key: already collected"
    else:
        task = f"Guide the agent toward the key at row={key_x}, col={key_y}."
        key_info = f"Key: row={key_x}, col={key_y}"

    prompt = f"""You are a sky agent in a 6x6 grid navigation task (rows 0-5, cols 0-5).

Grid layout:
- Walls at positions: (row=2,col=2), (row=2,col=3), (row=5,col=3)
- Exit at: row=5, col=5
- Agent currently at: row={agent_x}, col={agent_y}
- {key_info}

Actions: up=row-1, down=row+1, left=col-1, right=col+1

Task: {task}

Return ONLY a JSON object with Q-value estimates in range [-1, 1].
Use higher values for directions toward the target, negative values for directions into walls or boundaries.
{{"up": <float>, "down": <float>, "left": <float>, "right": <float>}}"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        text = response.text.strip()
        start = text.find("{")
        # raw_decode parses only the FIRST JSON object, ignores anything after
        decoder = json.JSONDecoder()
        q_dict, _ = decoder.raw_decode(text[start:])
        q_values = [float(q_dict["up"]), float(q_dict["down"]),
                    float(q_dict["left"]), float(q_dict["right"])]
        q_values = list(np.clip(q_values, -1, 1))
        # only write to cache if parsing succeeded — never cache zeros
        cache[cache_key] = q_values
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"LLM parse error: {e} — using zeros (not cached)")
        q_values = [0.0, 0.0, 0.0, 0.0]

    return np.array(q_values).reshape(4, 1)

# environment and DQN setup (ground agent only — LLM is not trained)
env = GridWorld()
ground_nn = NeuralNetwork(input_size=7, hidden_size=64, output_size=4)
target_nn  = NeuralNetwork(input_size=7, hidden_size=64, output_size=4)
target_nn.copy_weights_from(ground_nn)

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

        comm = get_llm_comm(sky_obs)
        ground_input = np.vstack([state, comm])
        q_values = ground_nn.forward(ground_input)

        if np.random.rand() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(q_values)

        _, reward, done = env.step(action)
        next_state = env.get_ground_state()
        next_sky_obs = env.get_sky_obs()
        total_reward += reward

        buffer.push(state, sky_obs, action, reward, next_state, next_sky_obs, done)

        if buffer.is_ready(BATCH_SIZE):
            batch = buffer.sample(BATCH_SIZE)
            states_b, sky_obs_b, actions_b, rewards_b, next_states_b, next_sky_obs_b, dones_b = zip(*batch)

            states_mat      = np.hstack(states_b)
            next_states_mat = np.hstack(next_states_b)

            comms_mat      = np.hstack([get_llm_comm(s) for s in sky_obs_b])
            next_comms_mat = np.hstack([get_llm_comm(s) for s in next_sky_obs_b])

            ground_inputs      = np.vstack([states_mat, comms_mat])
            next_ground_inputs = np.vstack([next_states_mat, next_comms_mat])

            next_q    = target_nn.forward(next_ground_inputs)
            current_q = ground_nn.forward(ground_inputs)

            targets = current_q.copy()
            for i in range(BATCH_SIZE):
                targets[actions_b[i], i] = (rewards_b[i]
                    + gamma * np.max(next_q[:, i]) * (not dones_b[i]))

            ground_nn.backward(targets)
            ground_nn.update(learning_rate / BATCH_SIZE)

        if total_steps % TARGET_UPDATE_FREQ == 0:
            target_nn.copy_weights_from(ground_nn)

        state = next_state
        sky_obs = next_sky_obs

    key_log.append(int(env.has_key))
    epsilon = max(0.1, epsilon * 0.995)
    reward_log.append(total_reward)

    if episode % 100 == 0:
        avg = np.mean(reward_log[-100:]) if len(reward_log) >= 100 else np.mean(reward_log)
        key_rate = np.mean(key_log[-100:]) * 100 if len(key_log) >= 100 else np.mean(key_log) * 100
        print(f"Episode {episode} | Reward: {total_reward:.2f} | "
              f"Avg100: {avg:.2f} | Key%: {key_rate:.1f}% | "
              f"Cache size: {len(cache)} | Epsilon: {epsilon:.3f}")

np.save("llm_rewards.npy", np.array(reward_log))
np.save("llm_keys.npy", np.array(key_log))
print(f"Training complete. Rewards saved to llm_rewards.npy and key success rate to llm_keys.npy")
print(f"LLM cache: {len(cache)} unique observations cached in {CACHE_FILE}")
