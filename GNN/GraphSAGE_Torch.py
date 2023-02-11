import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSAGE(nn.Module):
    def __init__(self, in_features, out_features, adjacency_matrix, num_nodes, aggregation="mean"):
        super(GraphSAGE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adjacency_matrix = adjacency_matrix
        self.num_nodes = num_nodes
        self.aggregation = aggregation
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, input):
        output = torch.zeros((self.num_nodes, self.out_features), dtype=torch.float32)
        for i in range(self.num_nodes):
            neighbors = (self.adjacency_matrix[i, :] == 1).nonzero().squeeze(1)
            if self.aggregation == "mean":
                output[i, :] = torch.mean(input[neighbors, :], dim=0)
            elif self.aggregation == "sum":
                output[i, :] = torch.sum(input[neighbors, :], dim=0)
            else:
                raise ValueError("Invalid aggregation method")
        output = self.fc(output)
        return output
