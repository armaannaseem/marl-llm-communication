import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # initialising weights and biases, using 0.01 to ensure small initial values
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))

    def forward(self, x):
        self.x = x
        self.z1 = np.dot(self.W1, x) + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        return self.z2

    def loss(self, predicted, target):
        # Mean Squared Error
        return np.mean((predicted - target) ** 2)

    def backward(self, target):
        dz2 = self.z2 - target
        self.dW2 = np.dot(dz2, self.a1.T)
        self.db2 = np.sum(dz2, axis=1, keepdims=True)
        da1 = np.dot(self.W2.T, dz2)
        dz1 = da1 * (self.z1 > 0)
        self.dW1 = np.dot(dz1, self.x.T)
        self.db1 = np.sum(dz1, axis=1, keepdims=True)

    def update(self, learning_rate):
        self.W1 = self.W1 - learning_rate * self.dW1
        self.b1 = self.b1 - learning_rate * self.db1
        self.W2 = self.W2 - learning_rate * self.dW2
        self.b2 = self.b2 - learning_rate * self.db2
