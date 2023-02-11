import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adjacency_matrix):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adjacency_matrix, support)
        return output

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN, self).__init__()
        self.layer1 = GCNLayer(in_features, hidden_features)
        self.layer2 = GCNLayer(hidden_features, out_features)

    def forward(self, input, adjacency_matrix):
        hidden = torch.relu(self.layer1(input, adjacency_matrix))
        output = self.layer2(hidden, adjacency_matrix)
        return output
