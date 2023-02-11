import numpy as np

'''
This implementation uses a single layer LSTM cell with input size input_size and hidden size hidden_size. The LSTM cell has four linear weight matrices Wf, Wi, Wc, and Wo for the forget gate, input gate, candidate state, and output gate transformations, respectively, and four bias vectors bf, bi, bc, and bo for the forget gate, input gate, candidate state, and output gate transformations, respectively.
In the forward method, the forget gate f, the input gate i, the candidate state c_candidate, and the output gate o are calculated using the sigmoid and tanh activation functions, respectively. The final cell state c_next and hidden state h_next are calculated using the forget gate, input gate, and output gate.
'''
class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, x, h_prev, c_prev):
        concat = np.concatenate((x, h_prev), axis=0)
        f = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
        i = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
        c_candidate = self.tanh(np.dot(self.Wc, concat) + self.bc)
        c_next = f * c_prev + i * c_candidate
        o = self.sigmoid(np.dot(self.Wo, concat) + self.bo)
        h_next = o * self.tanh(c_next)
        return h_next, c_next
