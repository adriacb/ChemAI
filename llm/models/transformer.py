import torch
import torch.nn as nn
from typing import Tuple

from .tools import PositionalEncodings
from .block import FeedForward, \
    EncoderLayer, DecoderLayer, Encoder, Decoder

class DecoderTransformer(nn.Module):
    # GPT-2 like transformer
    def __init__(self,
                 vocab_size: int,          # Add vocab_size as parameter
                 n_layers: int,
                 embedding_dim: int,
                 num_heads: int,
                 context_size: int,
                 dropout: float):
        super(DecoderTransformer, self).__init__()  # Initialize parent class
        self.context_size = context_size
        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # positional encoding
        self.pos_embedding = nn.Embedding(context_size, embedding_dim)
        # decoder
        self.decoder = Decoder(
            n_layers, 
            embedding_dim, 
            num_heads, 
            context_size, 
            dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        # output layer
        self.output = nn.Linear(embedding_dim, vocab_size)

        # initialize the weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, 
                idx: torch.Tensor, 
                targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the decoder transformer.

        Args:
        - idx: torch.Tensor of shape (batch_size, context_size)
        - targets: torch.Tensor of shape (batch_size, context_size)

        Returns:
        - x: torch.Tensor of shape (batch_size, context_size, vocab_size)"""
        B, T = idx.size()

        # embedding
        token_emb = self.embedding(idx) 
        pos_emb = self.pos_embedding(torch.arange(T, device=idx.device))
        x = token_emb + pos_emb
        x = self.decoder(x)
        x = self.norm(x)
        logits = self.output(x) # (B, T, V)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=0)
        
        return logits, loss
    
    def generate(
            self,
            idx: torch.Tensor,           # idx of shape (batch_size, context_size)
            max_len: int = 100,          # max length of the generated sequence
            temperature: float = 1.0     # temperature for sampling
            ) -> torch.Tensor:
        
        for _ in range(max_len):
            # get the last block_size
            idx_cond = idx[:, -self.context_size:]  # (B, T)
            # get the predictions for the next token
            logits, _ = self.forward(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]               # becomes (B, C)
            # apply temperature
            logits = logits / temperature           # used to control the randomness of the sampling
            # apply softmax
            probs = F.softmax(logits, dim=-1)       # (B, C)
            # sample the next token
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
            # concatenate the new token
            idx = torch.cat([idx, next_token], dim=-1)
        return idx