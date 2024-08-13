# tests for llm module
import os
import sys
import pytest
import torch
# add the llm module to the path
LLM_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
print(f"Adding {LLM_PATH} to the path.")
sys.path.append(LLM_PATH)
from llm.models.block import FeedForward, Encoder, Decoder
from llm.models.mha import MultiHeadAttentionBlock, AttentionBlock
from llm.models.transformer import DecoderTransformer
from llm.logger import model_logger

def test_feed_forward():
    model_logger.info("Testing the feed forward block.")
    ff = FeedForward(embedding_dim=512, dropout=0.1)
    x = torch.randn(64, 10, 512)
    y = ff(x)
    assert y.size() == (64, 10, 512)

def test_multi_head_attention_block():
    model_logger.info("Testing the multi-head attention block.")
    mha = MultiHeadAttentionBlock(
        embedding_dim=512,
        num_heads=8,
        context_size=10
    )
    x = torch.randn(64, 10, 512)
    y = mha(x)
    assert y.size() == (64, 10, 512)

def test_attention_block():
    model_logger.info("Testing the attention block.")
    embedding_dim = 512
    head_dim = embedding_dim // 8  # Assuming 8 heads
    context_size = 10
    batch_size = 64

    attention = AttentionBlock(embedding_dim=embedding_dim, head_dim=head_dim, context_size=context_size)
    x = torch.randn(batch_size, context_size, embedding_dim)
    y = attention(x)
    assert y.size() == (batch_size, context_size, head_dim)

# Uncomment and adjust the following test if necessary
# def test_encoder():
#     model_logger.info("Testing the encoder.")
#     encoder = Encoder(
#         num_layers=6,
#         d_model=512,
#         num_heads=8,
#         dff=2048,
#         input_vocab_size=8500,
#         max_pe_input=10000,
#         rate=0.1
#     )
#     x = torch.randn(64, 10, 512)
#     y = encoder(x, training=True)
#     assert y.size() == (64, 10, 512)

def test_decoder():
    model_logger.info("Testing the decoder.")
    decoder = Decoder(
        n_layers=6,
        embedding_dim=512,
        context_size=10,
        num_heads=8,
        dropout=0.1
    )
    x = torch.randn(64, 10, 512)
    y = decoder(x)
    assert y.size() == (64, 10, 512)

def test_decoder_transformer():
    model_logger.info("Testing the decoder transformer.")
    decoder = DecoderTransformer(
        vocab_size=1000,
        n_layers=6,
        embedding_dim=512,
        num_heads=8,
        context_size=10,
        dropout=0.1
    )
    idx = torch.randint(0, 1000, (64, 10))
    logits, loss = decoder(idx, idx)
    
    # Check the shape after flattening
    assert logits.size() == (64 * 10, 1000)