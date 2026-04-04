from neural_network import NeuralNetwork

class SkyAgent:
    def __init__(self, hidden_size=64, learning_rate=0.01):
        # sky NN: takes [agent_x, agent_y, key_x, key_y] -> outputs 4-dim comm vector
        # same shape as Q-values so it feeds naturally into the ground NN's input
        self.nn = NeuralNetwork(input_size=4, hidden_size=hidden_size, output_size=4)
        self.learning_rate = learning_rate

    def get_comm(self, sky_obs):
        # forward pass: produce the 4-dim communication vector
        return self.nn.forward(sky_obs)

    def learn(self, comm_grad):
        # comm_grad: gradient of the loss w.r.t. the comm vector
        # extracted from ground_nn.dx[3:] in the training loop
        self.nn.backward_from_grad(comm_grad)
        self.nn.update(self.learning_rate)
