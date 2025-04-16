import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):
    """Layer normalization layer for the transformer model."""
    def __init__(self, features:int, eps:float=1e-6)-> None:
        super().__init__()
        """
        Args:
            features (int): Number of features in the input.
            eps (float): Small value to avoid division by zero.
        """
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Forward pass for the layer normalization layer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, features).
        Returns:
            torch.Tensor: Layer normalized tensor of shape (batch_size, seq_len, features).
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        return self.alpha * x + self.bias
class FeedForward(nn.Module):
    """Feed forward layer for the transformer model."""
    def __init__(self, d_model:int, d_ff:int, dropout:float)-> None:
        super().__init__()
        """
        Args:
            d_model (int): Dimension of the model.
            d_ff (int): Dimension of the feed forward layer.
            dropout (float): Dropout probability.
        """
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Forward pass for the feed forward layer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        x = torch.relu(self.linear_1(x))
        x = self.linear_2(self.dropout(x))
        return x


class InputEmbedding(nn.Module):
    """Input embedding layer for the transformer model."""
    def __init__(self, d_model:int, vocab_size:int)-> None:
        """
        Args:
            d_model (int): Dimension of the model.
            vocab_size (int): Size of the vocabulary.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Forward pass for the input embedding layer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).
        Returns:
            torch.Tensor: Embedded tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        return x
    
class PositionalEncoding(nn.Module):
    """Positional encoding layer for the transformer model."""
    def __init__(self, d_model:int, seq_len:int, dropout:float=0.1)-> None:
        """
        Args:
            d_model (int): Dimension of the model.
            seq_len (int): Length of the input sequence.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (sqe_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # Calculate the positional encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Forward pass for the positional encoding layer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            torch.Tensor: Positional encoded tensor of shape (batch_size, seq_len, d_model).
        """
        x = x + (self.pe[:, :x.size(1), :]).require_grad_(False)
        return self.dropout(x)
    
class MultiHeadAttention(nn.Module):
    """Multi-head attention layer for the transformer model."""
    def __init__(self, d_model:int, h:int, dropout:float)-> None:
        """
        Args:
            d_model (int): Dimension of the model.
            h (int): Number of attention heads.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.h = h
        # make sure d_model is divisible by h
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, mask:torch.Tensor, dropout:nn.Dropout)->torch.Tensor:
        """
        Scaled dot-product attention.
        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len, d_k).
            key (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len, d_k).
            value (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len, d_k).
            mask (torch.Tensor): Optional mask tensor of shape (batch_size, 1, 1, seq_len).
            dropout (nn.Dropout): Dropout layer.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_heads, seq_len, d_k).
        """
        d_k = query.shape[-1]
        attenttion_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attenttion_scores = attenttion_scores.masked_fill(mask == 0, -1e9)
        attenttion_scores = torch.softmax(attenttion_scores, dim=-1)
        if dropout is not None:
            attenttion_scores = dropout(attenttion_scores)
        return (attenttion_scores @ value), attenttion_scores

    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask:torch.Tensor=None)->torch.Tensor:
        """
        Forward pass for the multi-head attention layer.
        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_model).
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_model).
            v (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): Optional mask tensor of shape (batch_size, 1, 1, seq_len).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate the attention scores
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # Concatenate the heads and apply the final linear layer
        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0] ,-1 ,self.h * self.d_k)
        return self.w_o(x)