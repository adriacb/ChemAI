import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'
        self.d_model = d_model                          # Dimension of the model
        self.num_heads = num_heads                      # Number of heads
        self.head_dim = d_model // num_heads            # Dimension of each head
        
        self.dropout = nn.Dropout(dropout)              # Dropout layer
        
        self.fc_q = nn.Linear(d_model, d_model)         # Fully connected layer for queries
        self.fc_k = nn.Linear(d_model, d_model)         # Fully connected layer for keys
        self.fc_v = nn.Linear(d_model, d_model)         # Fully connected layer for values
        
        self.fc_o = nn.Linear(d_model, d_model)         # Fully connected layer for output
    
    def split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Split the input tensor into multiple heads.
        
        Args:
        - x: torch.Tensor of shape (batch_size, seq_len, d_model)
        - batch_size: int
        
        Returns:
        - split: torch.Tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)                     # (batch_size, num_heads, seq_len, head_dim)
    
    def __call__(self, v: torch.Tensor, k: torch.Tensor, q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MultiHeadAttention layer.
        
        Args:
        - v: torch.Tensor of shape (batch_size, seq_len, d_model)
        - k: torch.Tensor of shape (batch_size, seq_len, d_model)
        - q: torch.Tensor of shape (batch_size, seq_len, d_model)
        - mask: torch.Tensor of shape (batch_size, 1, seq_len, seq_len)
        
        Returns:
        - x: torch.Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size = q.shape[0]
        
        q = self.fc_q(q)                                # (batch_size, seq_len, d_model)
        k = self.fc_k(k)                                # (batch_size, seq_len, d_model)
        v = self.fc_v(v)                                # (batch_size, seq_len, d_model)
        
        q = self.split_heads(q, batch_size)             # (batch_size, num_heads, seq_len, head_dim)
        k = self.split_heads(k, batch_size)             # (batch_size, num_heads, seq_len, head_dim)
        v = self.split_heads(v, batch_size)             # (batch_size, num_heads, seq_len, head_dim)
        
        # Scaled dot-product attention (multiple heads)
        scalled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scalled_attention = scalled_attention.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len, num_heads, head_dim)

        concat_attention = scalled_attention.view(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)
        x = self.fc_o(concat_attention)                                         # (batch_size, seq_len, d_model)

        return x, attention_weights
    
    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Scaled dot-product attention mechanism.
        
        Args:
        - q: torch.Tensor of shape (batch_size, num_heads, seq_len, head_dim)
        - k: torch.Tensor of shape (batch_size, num_heads, seq_len, head_dim)
        - v: torch.Tensor of shape (batch_size, num_heads, seq_len, head_dim)
        - mask: torch.Tensor of shape (batch_size, 1, seq_len, seq_len)
        
        Returns:
        - output: torch.Tensor of shape (batch_size, num_heads, seq_len, head_dim)
        - attention_weights: torch.Tensor of shape (batch_size, num_heads, seq_len, seq_len)
        """
        matmul_qk = torch.matmul(q, k.permute(0, 1, 3, 2))
        scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)

        return output, attention_weights
