import torch
import torch.nn as nn

from models.model.transformer import Transformer
from models.model.encoder import Encoder
from models.model.decoder import Decoder
from models.block.encoder_block import EncoderBlock
from models.block.decoder_block import DecoderBlock
from models.layer.multi_head_attention_layer import MultiHeadAttentionLayer
from models.layer.position_wise_feed_forward_layer import PositionWiseFeedForwardLayer
from models.embedding.transformer_embedding import TransformerEmbedding
from models.embedding.token_embedding import TokenEmbedding
from models.embedding.positional_encoding import PositionalEncoding

def build_model(src_vocab_size,
                tgt_vocab_size,
                device=torch.device('cuda:0'),
                max_len=256,
                d_embed=512,
                n_layer=6,
                d_model=512,
                h=8,
                d_ff=2048,
                dr_rate=0.1,
                norm_eps=1e-5):
    import copy
    copy = copy.deepcopy

    # Initialize source and target token embeddings
    src_token_embed = TokenEmbedding(
                                     d_embed=d_embed,
                                     vocab_size=src_vocab_size)
    tgt_token_embed = TokenEmbedding(
                                     d_embed=d_embed,
                                     vocab_size=tgt_vocab_size)
    # Initialize positional encoding
    pos_embed = PositionalEncoding(
                                   d_embed=d_embed,
                                   max_len=max_len,
                                   device=device)

    # Combine token embeddings with positional encodings for source and target
    src_embed = TransformerEmbedding(
                                     token_embed=src_token_embed,
                                     pos_embed=copy(pos_embed),
                                     dr_rate=dr_rate)
    tgt_embed = TransformerEmbedding(
                                     token_embed=tgt_token_embed,
                                     pos_embed=copy(pos_embed),
                                     dr_rate=dr_rate)

    # Initialize multi-head attention layer
    attention = MultiHeadAttentionLayer(
                                        d_model=d_model,
                                        h=h,
                                        qkv_fc=nn.Linear(d_embed, d_model),
                                        out_fc=nn.Linear(d_model, d_embed),
                                        dr_rate=dr_rate)
    # Initialize position-wise feed-forward layer
    position_ff = PositionWiseFeedForwardLayer(
                                               fc1=nn.Linear(d_embed, d_ff),
                                               fc2=nn.Linear(d_ff, d_embed),
                                               dr_rate=dr_rate)
    # Initialize layer normalization
    norm = nn.LayerNorm(d_embed, eps=norm_eps)

    # Create encoder block
    encoder_block = EncoderBlock(
                                 self_attention=copy(attention),
                                 position_ff=copy(position_ff),
                                 norm=copy(norm),
                                 dr_rate=dr_rate)
    # Create decoder block
    decoder_block = DecoderBlock(
                                 self_attention=copy(attention),
                                 cross_attention=copy(attention),
                                 position_ff=copy(position_ff),
                                 norm=copy(norm),
                                 dr_rate=dr_rate)

    # Initialize encoder with multiple layers
    encoder = Encoder(
                      encoder_block=encoder_block,
                      n_layer=n_layer,
                      norm=copy(norm))
    # Initialize decoder with multiple layers
    decoder = Decoder(
                      decoder_block=decoder_block,
                      n_layer=n_layer,
                      norm=copy(norm))
    # Initialize the generator layer to produce output vocabulary probabilities
    generator = nn.Linear(d_model, tgt_vocab_size)

    # Build the Transformer model
    model = Transformer(
                        src_embed=src_embed,
                        tgt_embed=tgt_embed,
                        encoder=encoder,
                        decoder=decoder,
                        generator=generator).to(device)
    model.device = device

    return model

