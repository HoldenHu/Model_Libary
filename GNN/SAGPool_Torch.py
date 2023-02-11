import torch
import torch.nn as nn
import torch.nn.functional as F

class SAGPool(nn.Module):
    def __init__(self, in_features, out_features, num_nodes, dropout=0.5):
        super(SAGPool, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        h = self.fc1(input)
        h = torch.tanh(h)
        h = self.fc2(h)
        h = h.squeeze(1)
        attention = torch.exp(h)
        attention = attention / torch.sum(attention)
        output = torch.matmul(attention, input)
        output = torch.tanh(output)
        output = self.dropout(output)
        return output
