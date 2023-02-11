'''
In this example, SAGPool implements a single SAGPool layer. The forward method performs the forward pass of the layer, which consists of four steps: (1) transforming the node representations to an intermediate representation using a fully-connected layer, (2) applying a tanh activation function, (3) computing the attention scores using another fully-connected layer, (4) computing the weighted average of the node representations using the attention scores. Note that this is just a basic implementation, and you may need to add additional components, such as regularization, etc. to make it more robust in practice.
'''

import numpy as np

class SAGPool(object):
    def __init__(self, in_features, out_features, num_nodes, dropout=0.5):
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.weights1 = np.random.randn(in_features, out_features)
        self.weights2 = np.random.randn(out_features, 1)
        self.bias1 = np.zeros(out_features)
        self.bias2 = np.zeros(1)

    def forward(self, input):
        h = np.dot(input, self.weights1) + self.bias1
        h = np.tanh(h)
        h = np.dot(h, self.weights2) + self.bias2
        h = np.reshape(h, (self.num_nodes,))
        attention = np.exp(h)
        attention = attention / np.sum(attention)
        output = np.average(input, axis=0, weights=attention)
        output = np.tanh(output)
        if self.dropout > 0:
            output = np.random.binomial(1, 1 - self.dropout, size=output.shape) * output
        return output
