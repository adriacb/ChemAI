import torch
import torch.nn as nn

from llm.models.mha import MultiHeadAttentionBlock
from llm.tools import PositionalEncoding

class FeedForward(nn.Module):
    act_functions = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'silu': nn.SiLU()
    }
    
    def __init__(self, 
                 embedding_dim: int, 
                 dropout: float,
                 activation: str = 'relu'):
        
        if activation not in self.act_functions:
            raise ValueError(f'Activation function {activation} not supported.')
        
        act_fn = self.act_functions[activation]

        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            act_fn,
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the feed forward block.

        Args:
        - x: torch.Tensor of shape (batch_size, context_size, embedding_dim)

        Returns:
        - x: torch.Tensor of shape (batch_size, context_size, embedding_dim)"""
        return self.ff(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, 
                 dff: int, rate: float = 0.1):
        super(EncoderLayer, self).__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
    
    def __call__(self, x: torch.Tensor, 
                 training: bool, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder layer.

        Args:
        - x: torch.Tensor of shape (batch_size, input_seq_len, d_model)
        - training: bool
        - mask: torch.Tensor of shape (batch_size, 1, 1, input_seq_len)

        Returns:
        - x: torch.Tensor of shape (batch_size, input_seq_len, d_model)
        """
        attn_output, _ = self.mha(x, x, x, attn_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

class Encoder(nn.Module):
    def __init__(self, num_layers: int, 
                 d_model: int, num_heads: int, 
                 dff: int, input_vocab_size: int, 
                 max_pe_input: int, rate: float = 0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        ps_encoding = PositionalEncoding(d_model, max_pe_input)
        self.pos_encoding = ps_encoding.get_positional_encoding(d_model, max_pe_input)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = nn.Dropout(rate)
    
    def __call__(self, input: torch.Tensor, training: bool, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
        - input: torch.Tensor of shape (batch_size, input_seq_len)
        - training: bool
        - mask: torch.Tensor of shape (batch_size, 1, 1, input_seq_len)

        Returns:
        - x: torch.Tensor of shape (batch_size, input_seq_len, d_model)
        """
        seq_len = input.size(1)
        x = self.embedding(input)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x += self.pos_encoding(torch.arange(seq_len).unsqueeze(0))
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        
        return x
    

class DecoderBlock(nn.Module):
    def __init__(self,
                embedding_dim: int,
                num_heads: int,
                context_size: int,
                dropout: float,
                activation: str):
        super(DecoderBlock, self).__init__()

        self.mha = MultiHeadAttentionBlock(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            context_size=context_size
        )
        self.ff = FeedForward(embedding_dim, dropout, activation)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the decoder block.
        
        Args:
        - x: torch.Tensor of shape (batch_size, context_size, embedding_dim)
        
        Returns:
        - x: torch.Tensor of shape (batch_size, context_size, embedding_dim)"""
        x = x + self.mha(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x
    
class Decoder(nn.Module):
    def __init__(self,
                n_layers: int,
                embedding_dim: int,
                context_size: int,
                num_heads: int,
                dropout: float,
                activation: str):
        super(Decoder, self).__init__()
        self.blocks = nn.Sequential(*[
            DecoderBlock(embedding_dim, num_heads, context_size, dropout, activation)
            for _ in range(n_layers)
        ])

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the decoder.
        
        Args:
        - x: torch.Tensor of shape (batch_size, context_size, embedding_dim)
        
        Returns:
        - x: torch.Tensor of shape (batch_size, context_size, embedding_dim)"""
        return self.blocks(x)