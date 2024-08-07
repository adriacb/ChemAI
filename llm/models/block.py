import torch
import torch.nn as nn

from llm.models.mha import AttentionBlock
from llm.tools import PositionalEncoding
from llm.logger import model_logger

class FeedForward(nn.Module):
    def __init__(self, 
                 embedding_dim: int, 
                 ff_dim: int):
        """Initialize the feed-forward block.
        
        Args:
        - embedding_dim: int - The dimension of the input embeddings
        - ff_dim: int - The dimension of the feed-forward network"""
        super(FeedForward, self).__init__()

        self.linear_1 = nn.Linear(embedding_dim, ff_dim)
        self.relu = nn.ReLU()                           # Try with GeLU
        self.linear_2 = nn.Linear(ff_dim, embedding_dim)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the feed-forward block.

        Args:
        - x: torch.Tensor of shape (batch_size, context_size, embedding_dim)

        Returns:
        - x: torch.Tensor of shape (batch_size, context_size, embedding_dim)"""
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x

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
    
class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim: int,
                 head_dim: int,
                 context_size: int,
                 ff_dim: int,
                 dropout: float = 0.2):
        """Initialize the decoder layer.

        1- Multi-head attention block
        2- Feed-forward block
        3- Layer normalization
        4- Layer normalization 

        Args:
        - embedding_dim: int - The dimension of the input embeddings
        - head_dim: int - The dimension of the heads
        - context_size: int - The size of the context
        - ff_dim: int - The dimension of the feed-forward network
        - dropout: float - The dropout rate"""
        super(DecoderLayer, self).__init__()

        self.attention = AttentionBlock(embedding_dim, head_dim, context_size)
        self.feedforward = FeedForward(embedding_dim, ff_dim)
        self.norm_1 = nn.LayerNorm(embedding_dim)
        self.norm_2 = nn.LayerNorm(embedding_dim)
        self.drop_shortcut = nn.Dropout(p=dropout)                     # Necessary?
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the decoder layer.

        Args:
        - x: torch.Tensor of shape (batch_size, context_size, embedding_dim)

        Returns:
        - x: torch.Tensor of shape (batch_size, context_size, embedding_dim)"""
        # Shortcut connection for the attention block
        shortcut = x                      # Shortcut connection
        x = self.attention(x)             # Multi-head attention
        x = self.drop_shortcut(x)         # Dropout
        x = self.norm_1(x + shortcut)     # Layer normalization
        # Shortcut connection for the feed-forward block
        shortcut = x
        x = self.feedforward(x)
        x = self.drop_shortcut(x)
        x = self.norm_2(x + shortcut)
        return x

class Decoder(nn.Module):
    def __init__(self, n_layers: int, decoder: DecoderLayer):
        """Initialize the decoder.

        Args:
        - n_layers: int - The number of decoder layers
        - decoder: DecoderLayer - The decoder layer"""
        super(Decoder, self).__init__()
        #        self.layers = nn.ModuleList([decoder for _ in range(n_layers)])
        self.layers = nn.Sequential(*[decoder for _ in range(n_layers)])   # the difference between ModuleList and Sequential is that ModuleList is a list of modules, while Sequential is a module that contains modules
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the decoder.

        Args:
        - x: torch.Tensor of shape (batch_size, context_size, embedding_dim)

        Returns:
        - x: torch.Tensor of shape (batch_size, context_size, embedding_dim)"""
        model_logger.debug(f"Decoder input shape: {x.size()}")
        return self.layers(x)
