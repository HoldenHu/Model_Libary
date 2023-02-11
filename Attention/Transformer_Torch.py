import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size = x.shape[0]

        x = x.view(batch_size, -1, self.num_heads, self.depth)
        x = x.transpose(1, 2)

        return x

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        query = self.W_Q(query)
        key = self.W_K(key)
        value = self.W_V(value)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        scores = torch.matmul(query, key.transpose(2, 3)) / np.sqrt(self.depth)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)

        attention_output = torch.matmul(attention_weights, value)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        attention_output = self.fc(attention_output)

        return attention_output, attention_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, dff):
        super().__init__()

        self.fc1 = nn.Linear(d_model, dff)
        self.fc2 = nn.Linear(dff, d_model)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, dff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        attention_output, attention_weights = self.multi_head_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))

        feed_forward_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(feed_forward_output))

        return x

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, dff, num_layers, target_vocab_size, max_seq_length, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(target_vocab_size, d_model)
        self.position_encoding = nn.Embedding(max_seq_length, d_model)

        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, target_vocab_size)

    def create_mask(self, input):
        mask = (input == 0)

        return mask

    def forward(self, input):
        batch_size = input.shape[0]
        seq_length = input.shape[1]

        input = self.embedding(input)

        position = torch.arange(0, seq_length).unsqueeze(0).repeat(batch_size, 1).to(input.device)
        position_encoded = self.position_encoding(position)

        input += position_encoded

        mask = self.create_mask(input)

        for i in range(self.num_layers):
            input = self.transformer_blocks[i](input, mask)

        input = self.fc(input)

        return input

