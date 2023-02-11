import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wz = nn.Linear(input_size, hidden_size)
        self.Wr = nn.Linear(input_size, hidden_size)
        self.Wh = nn.Linear(input_size, hidden_size)
        self.Uz = nn.Linear(hidden_size, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size)
        self.Uh = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x, h_prev):
        z = F.sigmoid(self.Wz(x) + self.Uz(h_prev))
        r = F.sigmoid(self.Wr(x) + self.Ur(h_prev))
        h_tilde = F.tanh(self.Wh(x) + r * self.Uh(h_prev))
        h_next = (1 - z) * h_prev + z * h_tilde
        return h_next
