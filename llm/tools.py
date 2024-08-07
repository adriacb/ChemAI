import torch
import torch.nn as nn

from transformers import AutoModel

from .logger import model_logger

# https://github.com/MTxSouza/MediumArticleGenerator/blob/main/model/embedding.py

class CustomSchedule:
    def __init__(self, d_model: int, warmup_steps: int = 4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def get_lr(self, step: int) -> float:
        """
        Compute the learning rate for the given step.

        Args:
        - step: int

        Returns:
        - lr: float
        """
        step_tensor = torch.tensor(step, dtype=torch.float32)
        arg1 = torch.rsqrt(step_tensor)
        arg2 = step_tensor * (self.warmup_steps ** -1.5)
        arg2 = torch.tensor(arg2, dtype=torch.float32)  # Convert arg2 to a tensor
        min_val = torch.min(arg1, arg2)
        return float(torch.rsqrt(torch.tensor(self.d_model, dtype=torch.float32)) * min_val)


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_context_size: int):
        """Initialize the positional encoding.

        Args:
        - embedding_dim: int - The dimension of the input embeddings
        - max_context_size: int - The maximum size of the context"""
        super(PositionalEncoding, self).__init__()
        
        even_i = torch.arange(0, embedding_dim, 2, dtype=torch.float32)
        odd_i = torch.arange(1, embedding_dim, 2, dtype=torch.float32) - 1

        even_denom = torch.pow(10_000, even_i / embedding_dim)
        odd_denom = torch.pow(10_000, odd_i / embedding_dim)

        pos = torch.arange(end=max_context_size, dtype=torch.float32).reshape(max_context_size, 1)

        even = torch.sin(pos / even_denom)
        odd = torch.cos(pos / odd_denom)
        self.register_buffer("pos_encoding", torch.cat([even, odd], dim=1)).expand(1, -1, -1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the positional encoding.

        Args:
        - x: torch.Tensor of shape (batch_size, context_size, embedding_dim)

        Returns:
        - x: torch.Tensor of shape (batch_size, context_size, embedding_dim)
        """
        B, T, D = x.size() # B: batch size, T: context size, D: embedding dimension
        x_pe = x + self.pos_encoding[:, :T, :]        # Add the positional encoding to the input
        model_logger.debug(f"Positional encoding shape: {x_pe.size()}")
        return x_pe
    
    
class Embedding(nn.Module):

    def __init__(self, vocab_size: int, emb_dim: int):
        """
        Embedding layer for the transformer model.

        Args:
            vocab_size (int) : The size of the vocabulary.
            emb_dim (int) : The dimension of the embedding.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)

    @property
    def embedding_dim(self):
        """
        Get the embedding dimension.

        Returns:
            int : The embedding dimension.
        """
        return self.embedding.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the embedding layer.

        Args:
            x (torch.Tensor) : The input tensor.

        Returns:
            torch.Tensor : The output tensor.
        """
        model_logger.debug("Embedding: x shape: %s", x.size())
        return self.embedding(x)
    
class GPTEmbedding(nn.Module):

    def __init__(self):
        """
        Embedding layer for the GPT model.
        """
        super().__init__()
        model = AutoModel.from_pretrained("gpt2")
        self.embedding = model.wte
        self.pe = model.wpe

        # Freeze the parameters
        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.pe.parameters():
            param.requires_grad = False

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the embedding layer.

        Args:
            x (torch.Tensor) : The input tensor.

        Returns:
            torch.Tensor : The output tensor.
        """
        i = x.size(1)
        emb = self.embedding(x)
        pe = self.pe(torch.arange(i).unsqueeze(0).to(x.device))
        return emb + pe