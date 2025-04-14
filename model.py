import torch
import torch.nn as nn
import math

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