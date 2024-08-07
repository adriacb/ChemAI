import torch
import torch.nn as nn
from typing import Tuple

from .tools import PositionalEncoding
from .mha import MultiHeadAttention
from .block import FeedForward, \
    EncoderLayer, DecoderLayer, Encoder, Decoder

class Transformer(nn.Module):
    def __init__(self, 
                 num_layers: int, 
                 d_model: int, 
                 num_heads: int, 
                 dff: int, 
                 input_vocab_size: int, 
                 target_vocab_size: int, 
                 max_pe_input: int, 
                 max_pe_target: int, 
                 rate: float = 0.1):
        super(Transformer, self).__init__()
