import torch
import torch.nn as nn

'''
This implementation uses a single layer LSTM cell with input size input_size and hidden size hidden_size. PyTorch provides a built-in nn.LSTM module for LSTM cells, which makes it easier to implement LSTMs compared to using Numpy.
In the forward method, the input sequence x is reshaped and passed through the nn.LSTM module, which returns the output and the hidden state h_next and cell state c_next. These states are reshaped to their original shapes and returned.
'''
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
    
    def forward(self, x, h_prev, c_prev):
        output, (h_next, c_next) = self.lstm(x.view(1, 1, -1), (h_prev.view(1, 1, -1), c_prev.view(1, 1, -1)))
        return h_next.view(-1), c_next.view(-1)
