'''
In this example, DiffPool implements a single DiffPool layer. The forward method performs the forward pass of the layer, which consists of multiple steps: (1) computing the hidden representation h with a ReLU activation function, (2) computing the affinity scores a based on the hidden representation, (3) performing pooling to compute the output. Note that this is just a basic implementation, and you may need to add additional components, such as dropout, regularization, etc. to make it more robust in practice.
'''

import numpy as np

class DiffPool(object):
    def __init__(self, in_features, out_features, adjacency_matrix, num_nodes):
        self.in_features = in_features
        self.out_features = out_features
        self.adjacency_matrix = adjacency_matrix
        self.num_nodes = num_nodes
        self.weights_1 = np.random.randn(in_features, out_features)
        self.weights_2 = np.random.randn(in_features, out_features)
        self.weights_3 = np.random.randn(out_features, out_features)
        self.bias_1 = np.zeros(out_features)
        self.bias_2 = np.zeros(out_features)
        self.bias_3 = np.zeros(out_features)

    def forward(self, input):
        h = np.dot(input, self.weights_1) + self.bias_1
        h = np.maximum(h, 0)
        s = np.dot(input, self.weights_2) + self.bias_2
        s = np.maximum(s, 0)
        a = np.exp(np.dot(h, self.weights_3) + self.bias_3)
        a = a * self.adjacency_matrix
        a = a / np.sum(a, axis=1, keepdims=True)
        output = np.zeros((self.num_nodes, self.out_features))
        for i in range(self.num_nodes):
            output[i, :] = np.dot(a[i, :], s)
        return output
