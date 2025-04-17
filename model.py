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
        var = ((x - mean) ** 2).mean(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.alpha * x + self.bias

class FeedForwardBlock(nn.Module):
    """Feed forward layer for the transformer model."""
    def __init__(self, d_model:int, d_ff:int, dropout:float)-> None:
        """
        Args:
            d_model (int): Dimension of the model.
            d_ff (int): Dimension of the feed forward layer.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        # Use GELU activation for better performance
        self.activation = nn.GELU()
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Forward pass for the feed forward layer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.activation(self.linear_1(x))
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
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    """Residual connection layer for the transformer model."""
    def __init__(self,features, dropout:float)-> None:
        """
        Args:
            features (int): Number of features in the input.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.norm = LayerNormalization(features)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x:torch.Tensor, sublayer:nn.Module)->torch.Tensor:
        """
        Forward pass for the residual connection layer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, features).
            sublayer (nn.Module): Sublayer to apply the residual connection to.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, features).
        """
        return x + self.dropout(sublayer(self.norm(x)))

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
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

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
    
class EncoderBlock(nn.Module):
    """Encoder layer for the transformer model."""
    def __init__(self, features:int, self_attention_block:MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout:float)-> None:
        """
        Args:
            features (int): Number of features in the input.
            self_attention_block (MultiHeadAttention): Multi-head attention block.
            feed_forward_block (FeedForwardBlock): Feed forward block.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x:torch.Tensor, mask:torch.Tensor)->torch.Tensor:
        """
        Forward pass for the encoder layer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, features).
            mask (torch.Tensor): Optional mask tensor of shape (batch_size, 1, 1, seq_len).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, features).
        """
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, features:int, layers:nn.ModuleList)-> None:
        """
        Encoder for the transformer model.
        Args:
            features (int): Number of features in the input.
            layers (nn.ModuleList): List of encoder layers.
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
    def forward(self, x:torch.Tensor, mask:torch.Tensor)-> torch.Tensor:
        """
        Forward pass for the encoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, features).
            mask (torch.Tensor): Optional mask tensor of shape (batch_size, 1, 1, seq_len).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, features).
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    """ Decoder block for the Transformer model. """
    def __init__ (self, features:int, self_attention_block:MultiHeadAttention, cross_attention_block:MultiHeadAttention, feed_forward_block:FeedForwardBlock, dropout:float)-> None:
        """
        Args:
            features (int): Number of features in the input.
            self_attention_block (MultiHeadAttention): Multi-head attention block for self-attention.
            cross_attention_block (MultiHeadAttention): Multi-head attention block for cross-attention.
            feed_forward_block (FeedForwardBlock): Feed forward block.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x:torch.Tensor, encoder_output:torch.Tensor, src_mask:torch.Tensor, tgt_mask:torch.Tensor)-> torch.Tensor:
        """
        Forward pass for the decoder block.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, features).
            encoder_output (torch.Tensor): Encoder output tensor of shape (batch_size, seq_len, features).
            src_mask (torch.Tensor): Source mask tensor of shape (batch_size, 1, 1, seq_len).
            tgt_mask (torch.Tensor): Target mask tensor of shape (batch_size, 1, 1, seq_len).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, features).
        """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
        
class Decoder(nn.Module):
    """ Decoder for the Transformer model. """
    def __init__(self, features:int, layers:nn.ModuleList)-> None:
        """
        Args:
            features (int): Number of features in the input.
            layers (nn.ModuleList): List of decoder layers.
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x:torch.Tensor, encoder_output:torch.Tensor, src_mask:torch.Tensor, tgt_mask:torch.Tensor)-> torch.Tensor:
        """
        Forward pass for the decoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, features).
            encoder_output (torch.Tensor): Encoder output tensor of shape (batch_size, seq_len, features).
            src_mask (torch.Tensor): Source mask tensor of shape (batch_size, 1, 1, seq_len).
            tgt_mask (torch.Tensor): Target mask tensor of shape (batch_size, 1, 1, seq_len).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, features).
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    """Projection layer for the transformer model."""
    def __init__(self, d_model:int, vocab_size:int)-> None:
        """
        Args:
            d_model (int): Dimension of the model.
            vocab_size (int): Size of the vocabulary.
        """
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        """
        Forward pass for the projection layer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, vocab_size).
        """
        return self.proj(x)
    
class Transformer(nn.Module):
    """Transformer model."""
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer)-> None:
        """
        Args:
            encoder (Encoder): Encoder module.
            decoder (Decoder): Decoder module.
            src_embed (InputEmbeddings): Source input embedding module.
            tgt_embed (InputEmbeddings): Target input embedding module.
            src_pos (PositionalEncoding): Source positional encoding module.
            tgt_pos (PositionalEncoding): Target positional encoding module.
            projection_layer (ProjectionLayer): Projection layer for the output.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    def encode(self, src:torch.Tensor, src_mask:torch.Tensor)-> torch.Tensor:
        """
        Encode the source sequence.
        Args:
            src (torch.Tensor): Source input tensor of shape (batch_size, seq_len).
            src_mask (torch.Tensor): Source mask tensor of shape (batch_size, 1, 1, seq_len).
        Returns:
            torch.Tensor: Encoded output tensor of shape (batch_size, seq_len, features).
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output:torch.Tensor, src_mask:torch.Tensor, tgt:torch.Tensor, tgt_mask:torch.Tensor)-> torch.Tensor:
        """
        Decode the target sequence.
        Args:
            encoder_output (torch.Tensor): Encoder output tensor of shape (batch_size, seq_len, features).
            src_mask (torch.Tensor): Source mask tensor of shape (batch_size, 1, 1, seq_len).
            tgt (torch.Tensor): Target input tensor of shape (batch_size, seq_len).
            tgt_mask (torch.Tensor): Target mask tensor of shape (batch_size, 1, 1, seq_len).
        Returns:
            torch.Tensor: Decoded output tensor of shape (batch_size, seq_len, features).
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x:torch.Tensor)-> torch.Tensor:
        """
        Project the output to the vocabulary size.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, vocab_size).
        """
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    """
    Build a transformer model.
    Args:
        src_vocab_size (int): Source vocabulary size.
        tgt_vocab_size (int): Target vocabulary size.
        src_seq_len (int): Source sequence length.
        tgt_seq_len (int): Target sequence length.
        d_model (int): Dimension of the model.
        N (int): Number of encoder/decoder layers.
        h (int): Number of attention heads.
        dropout (float): Dropout probability.
        d_ff (int): Dimension of the feed forward layer.
    Returns:
        Transformer: Transformer model.
    """
    # Create the input and output embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer model
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    return transformer