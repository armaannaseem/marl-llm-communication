# LLM-Based vs. Emergent Communication in Multi-Agent Reinforcement Learning

This project investigates whether Large Language Models (LLMs) can provide superior, zero-shot guidance in a Multi-Agent Reinforcement Learning (MARL) environment compared to traditional emergent communication channels.

This repository is an evolution of my previous project [[ANN](https://github.com/armaannaseem/ANN)], which established the foundational NumPy DQN engine (**No ML libraries, only NumPy**) for a single agent. This extension upgrades the environment to a 6x6 partial-observability grid and introduces the Sky/Ground multi-agent problem solving environment.

## Project Overview
The environment is a 6x6 GridWorld where a "Ground Agent" must navigate around obstacles to collect a key and then proceed to the exit. However, the ground agent has **partial observability** with it only able to see its current location, location of the goal state, and understanding of walls on the grid if and when it collides with them.

A second "Sky Agent" also has **partial observability** of the grid, with it able to see the Ground Agent's location, the Key's location, and nothing else. It cannot move and its sole purpose is to communicate to the Ground Agent to help it find the key which it cannot see. 

We compare three paradigms for how the Sky Agent assists the Ground Agent:
1. **Baseline**: No communication whatsoever.
2. **Emergent MARL**: The Sky Agent is a Deep Q-Network that passes continuous vector messages to the ground agent, trained jointly via backpropagation.
3. **LLM Communication**: The Sky Agent is replaced by Gemini 2.5 Flash Lite, which parses the grid state and zero-shot generates directional Q-value priors for the ground agent.

_Note: All neural networks and deep Q-learning logic (experience replay, target networks, bellman equations) are implemented entirely from scratch in pure NumPy (**no ML libraries at all**)._

## Results
The LLM communication protocol dramatically outperforms the emergent MARL baseline, achieving up to 90%+ successful key collection rates compared to the emergent network's peak of ~40%. Most notably, the zero-shot nature of the LLM provides high-quality guidance from the very first episode.

![Results comparison showing reward and key success rate](results.png)

## Installation & Setup
This project requires no deep learning frameworks — just standard Python libraries.

```bash
# 1. Clone the repository
git clone https://github.com/armaannaseem/marl-llm-communication.git
cd marl-llm-communication

# 2. Install requirements
python3 -m pip install numpy python-dotenv google-genai matplotlib
```

### API Key & Caching Strategy
To ensure the LLM experiments are easily reproducible without requiring graders or reviewers to supply their own API keys, **I have committed `llm_cache.json` to this repository.**

This cache stores the LLM's pre-computed responses to every possible spatial configuration in the GridWorld. When you run the LLM script, it will fetch responses from this local cache (taking microseconds) rather than making live API calls. 

*(If you wish to make live API calls or modify the prompt, copy `.env.example` to `.env`, add your own Gemini API key, and delete llm_cache.json).*

## Running the Experiments

You can train the agents in all three paradigms using the provided scripts. Each script trains for 3,000 episodes and will save its results as `.npy` files. You can then finally run plot_results.py to generate graphs similar to the one above.

```bash
# 1. Run Baseline (No Comm)
python3 train_baseline.py

# 2. Run Emergent MARL Protocol
python3 train_marl.py

# 3. Run LLM Protocol (Uses local cache by default)
python3 train_llm.py

# 4. Generate the comparative graph
python3 plot_results.py
```

## Technical Architecture & Implementation
A major feature of this project is that **no high-level deep learning frameworks (like PyTorch or TensorFlow) were used.** The entire Reinforcement Learning engine is built from scratch in pure NumPy.

### Deep Q-Network (DQN)
The Ground Agent is powered by an explicitly coded Neural Network featuring a hidden layer with 64 neurons, ReLU activations, and a custom backpropagation engine. The network learns to map the 7-dimensional input vector (3 local sensors + 4 communication signals) directly to Q-values for the 4 possible actions.

### Reinforcement Learning Mechanics
To ensure convergence and prevent catastrophic forgetting, the agent utilizes several advanced RL stabilization techniques:
* **The Bellman Equation**: The network's loss is calculated fundamentally upon Bellman targets, bootstrapping the value of the next state: `Q*(s,a) = r + γ * max(Q(s', a'))`. Backpropagation manually computes gradients based on this temporal difference error.
* **Epsilon-Greedy Exploration**: Action selection balances exploitation (trusting the NN's Q-values) with exploration (taking random actions). Epsilon decays multiplicatively from `1.0` down to `0.1`, forcing the agent to discover the environment early and refine its policy later.
* **Experience Replay**: Transitions `(s, a, r, s', done)` are stored in a rolling `deque` buffer of capacity 10,000. Gradients are calculated over random mini-batches of size 32, successfully breaking the temporal correlation of sequential grid steps. 
* **Target Network Stabilization**: A separate, frozen copy of the Ground Agent's neural network is used to calculate the `max(Q(s', a'))` Bellman targets. The weights of the active network are hard-copied to the target network every 100 steps, preventing the "moving target" destabilization problem famous in standard Q-learning.

### Communication Integration
For the **Emergent MARL** condition, joint backpropagation was achieved by passing the temporal difference gradients backward from the Ground Network, through the 4-dimensional communication channel, and directly into the Sky Network. For the **LLM** condition, the sky network is bypassed, and the language model's zero-shot dimensional rankings seamlessly replace the continuous vector.
