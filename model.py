import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):
    """Layer normalization layer for the transformer model."""
    def __init__(self, features:int, eps:float=1e-6)-> None:
        super(LayerNormalization,self).__init__()
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
        
class InputEmbedding(nn.Module):
    """Input embedding layer for the transformer model."""
    def __init__(self, d_model:int, vocab_size:int)-> None:
        """
        Args:
            d_model (int): Dimension of the model.
            vocab_size (int): Size of the vocabulary.
        """
        super(InputEmbedding,self).__init__()
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
        super(PositionalEncoding,self).__init__()
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
