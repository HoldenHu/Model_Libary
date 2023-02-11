'''
This implementation uses a single layer GRU cell with input size input_size and hidden size hidden_size. The GRU cell has three weight matrices Wz, Wr, and Wh for the update gate, reset gate, and hidden state transformations, respectively, and three bias vectors bz, br, and bh. The GRU cell also has two recurrent weight matrices Uz and Ur for the update and reset gates, respectively, and Uh for the hidden state transformation.
In the forward method, the update gate z and the reset gate r are calculated using the sigmoid activation function, and the candidate hidden state h_tilde is calculated using the tanh activation function. The final hidden state h_next is a combination of the previous hidden state h_prev and the candidate hidden state, weighted by the update gate.
'''

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class GRU:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wz = np.random.randn(input_size, hidden_size)
        self.Wr = np.random.randn(input_size, hidden_size)
        self.Wh = np.random.randn(input_size, hidden_size)
        self.Uz = np.random.randn(hidden_size, hidden_size)
        self.Ur = np.random.randn(hidden_size, hidden_size)
        self.Uh = np.random.randn(hidden_size, hidden_size)
        self.bz = np.zeros((1, hidden_size))
        self.br = np.zeros((1, hidden_size))
        self.bh = np.zeros((1, hidden_size))
    
    def forward(self, x, h_prev):
        z = sigmoid(np.dot(x, self.Wz) + np.dot(h_prev, self.Uz) + self.bz)
        r = sigmoid(np.dot(x, self.Wr) + np.dot(h_prev, self.Ur) + self.br)
        h_tilde = tanh(np.dot(x, self.Wh) + np.dot(r * h_prev, self.Uh) + self.bh)
        h_next = (1 - z) * h_prev + z * h_tilde
        return h_next
