import torch
import torch.nn as nn
import torch.nn.functional as F

#TODO: Use FlexAttention https://pytorch.org/blog/flexattention/


class AttentionBlock(nn.Module):
    def __init__(self, 
                 embedding_dim: int, 
                 head_dim: int,
                 context_size: int,
                 dropout: float = 0.2):
        """Initialize the attention block.
        
        Args:
        - embedding_dim: int - The dimension of the input embeddings
        - context_size: int - The size of the context
        - dropout: float - The dropout rate"""
        super(AttentionBlock, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.query = nn.Linear(in_features=embedding_dim, out_features=head_dim)
        self.key = nn.Linear(in_features=embedding_dim, out_features=head_dim)
        self.value = nn.Linear(in_features=embedding_dim, out_features=head_dim)

        ones = torch.ones(size=(context_size, context_size), dtype=torch.float)
        self.register_buffer("mask", torch.tril(input=ones)) # Lower triangular matrix


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the attention block.
        
        Attention is all you need, equation (1): 
        Attention(Q, K, V) = softmax(Q*K.T / sqrt(d_k)) * V

        Args:
        - x: torch.Tensor of shape (batch_size, context_size, embedding_dim)
        
        Returns:
        - x: torch.Tensor of shape (batch_size, context_size, embedding_dim)"""
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        tril = torch.tril(torch.ones(T, T, device=x.device))
        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        y = wei @ v
        return y

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int, 
                 num_heads: int, 
                 context_size: int):
        """Initialize the multi-head attention block.
        Instead of performing a single attention operation, 
        the input embeddings are split into multiple heads and 
        the attention operation is performed in parallel.

        MultiHeadAttention(Q, K, V) = Concat(head_1, ..., head_h) * W_O
        where head_i = Attention(QWQi, KW Ki, V WV)
        
        Args:
        - embedding_dim: int - The dimension of the input embeddings
        - num_heads: int - The number of heads
        - context_size: int - The size of the context"""
        super(MultiHeadAttentionBlock, self).__init__()

        head_dim = embedding_dim // num_heads
        assert head_dim * num_heads == embedding_dim, "Embedding dimension must be divisible by number of heads"

        self.attention = nn.ModuleList(modules=[
            AttentionBlock(embedding_dim, head_dim, context_size) for _ in range(num_heads)])
        self.linear = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([
            attention(x) for attention in self.attention], # Perform attention for each head
            dim=-1)
        x = self.linear(x) # Linear transformation
        return x