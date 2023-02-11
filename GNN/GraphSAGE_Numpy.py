'''
In this example, GraphSAGE implements a single GraphSAGE layer. The forward method performs the forward pass of the layer, which consists of two steps: (1) aggregating the representations of the neighbors of each node, (2) transforming the aggregated representation to the output. The aggregation method can be either mean or sum, and you can choose the method by setting the aggregation argument in the constructor. Note that this is just a basic implementation, and you may need to add additional components, such as activation functions, dropout, regularization, etc. to make it more robust in practice.
'''

import numpy as np

class GraphSAGE(object):
    def __init__(self, in_features, out_features, adjacency_matrix, num_nodes, aggregation="mean"):
        self.in_features = in_features
        self.out_features = out_features
        self.adjacency_matrix = adjacency_matrix
        self.num_nodes = num_nodes
        self.aggregation = aggregation
        self.weights = np.random.randn(in_features, out_features)
        self.bias = np.zeros(out_features)

    def forward(self, input):
        output = np.zeros((self.num_nodes, self.out_features))
        for i in range(self.num_nodes):
            neighbors = np.where(self.adjacency_matrix[i, :] == 1)[0]
            if self.aggregation == "mean":
                output[i, :] = np.mean(input[neighbors, :], axis=0)
            elif self.aggregation == "sum":
                output[i, :] = np.sum(input[neighbors, :], axis=0)
            else:
                raise ValueError("Invalid aggregation method")
        output = np.dot(output, self.weights) + self.bias
        return output
