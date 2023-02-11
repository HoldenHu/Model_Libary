import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def gelu(x):
    return x * 0.5 * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def layer_norm(x, epsilon=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + epsilon)

def scaled_dot_product_attention(Q, K, V, mask=None, scale=None):
    dot_product = np.matmul(Q, np.transpose(K, (0, 2, 1)))

    if scale:
        dot_product = dot_product / np.sqrt(scale)

    if mask is not None:
        dot_product = np.where(mask, dot_product, -1e9)

    attention_weights = softmax(dot_product, axis=-1)
    attention_output = np.matmul(attention_weights, V)
    return attention_output, attention_weights

class MultiHeadAttention:
    def __init__(self, d_model, num_heads, scale=True):
        self.d_model = d_model
        self.num_heads = num_heads
        self.scale = scale

        self.W_Q = np.random.randn(d_model, d_model)
        self.W_K = np.random.randn(d_model, d_model)
        self.W_V = np.random.randn(d_model, d_model)
        self.W_O = np.random.randn(d_model, d_model)

    def __call__(self, Q, K, V, mask=None):
        batch_size, seq_len_Q, d_model = Q.shape
        seq_len_K = K.shape[1]

        Q = np.matmul(Q, self.W_Q)
        K = np.matmul(K, self.W_K)
        V = np.matmul(V, self.W_V)

        Q = np.concatenate(np.split(Q, self.num_heads, axis=2), axis=0)
        K = np.concatenate(np.split(K, self.num_heads, axis=2), axis=0)
        V = np.concatenate(np.split(V, self.num_heads, axis=2), axis=0)

        attention_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, scale=self.scale)

        attention_output = np.concatenate(np.split(attention_output, self.num_heads, axis=0), axis=2)
        attention_output = np.matmul(attention_output, self.W_O)

        return attention_output, attention_weights

class FeedForward:
    def __init__(self, d_model, dff):
        self.d_model = d_model
        self.dff = dff

        self.W_1 = np.random.randn(d_model, dff)
        self.W_2 = np.random.randn(dff, d_model)

    def __call__(self, x):
        hidden_layer = gelu(np.matmul(x, self.W_1))
        output_layer = np.matmul(hidden_layer, self.W_2)
        return output_layer

class TransformerBlock:
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, dff)

        self.layer_norm1 = layer_norm(d_model)
        self.layer_norm2 = layer_norm(d_model)

        self.dropout = np.random.rand(*d_model.shape)

    def __call__(self, x, mask=None):
        attention_output, attention_weights = self.multi_head_attention(x, x, x, mask=mask)
        attention_output = self.dropout * attention_output
        x = x + attention_output
        x = self.layer_norm1(x)

        feed_forward_output = self.feed_forward(x)
        feed_forward_output = self.dropout * feed_forward_output
        x = x + feed_forward_output
        x = self.layer_norm2(x)

        return x

class Transformer:
    def __init__(self, d_model, num_heads, dff, num_layers, dropout_rate=0.1):
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.transformer_blocks = [TransformerBlock(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]

    def __call__(self, x, mask=None):
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask=mask)
        return x

