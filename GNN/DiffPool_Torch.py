import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffPool(nn.Module):
    def __init__(self, in_features, out_features, adjacency_matrix, num_nodes):
        super(DiffPool, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adjacency_matrix = adjacency_matrix
        self.num_nodes = num_nodes
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(in_features, out_features)
        self.fc3 = nn.Linear(out_features, out_features)

    def forward(self, input):
        h = F.relu(self.fc1(input))
        s = F.relu(self.fc2(input))
        a = torch.exp(self.fc3(h))
        a = a * self.adjacency_matrix
        a = a / torch.sum(a, dim=1, keepdim=True)
        output = torch.zeros((self.num_nodes, self.out_features), dtype=torch.float32)
        for i in range(self.num_nodes):
            output[i, :] = torch.mm(a[i, :].view(1, -1), s)
        return output
