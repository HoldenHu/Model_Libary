
import torch
import torch.nn as nn

class RGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, num_rels, bias=True):
        super(RGCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_rels = num_rels
        self.bias = bias
        self.weights = nn.Parameter(torch.FloatTensor(num_rels, in_features, out_features))
        nn.init.xavier_uniform_(self.weights)
        if self.bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
            nn.init.zeros_(self.bias)

    def forward(self, input, adjacency_matrix, rel_matrix):
        support = torch.einsum("ij,ijk->ik", input, self.weights)
        output = torch.einsum("ijk,ik->ij", rel_matrix, support)
        output = torch.matmul(adjacency_matrix, output)
        if self.bias:
            output += self.bias
        return output

class RGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_rels1, num_rels2):
        super(RGCN, self).__init__()
        self.layer1 = RGCNLayer(in_features, hidden_features, num_rels1)
        self.layer2 = RGCNLayer(hidden_features, out_features, num_rels2)

    def forward(self, input, adjacency_matrix, rel_matrix1, rel_matrix2):
        hidden = torch.relu(self.layer1(input, adjacency_matrix, rel_matrix1))
        output = self.layer2(hidden, adjacency_matrix, rel_matrix2)
        return output
