import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):

    def __init__(self, src_embed, tgt_embed, encoder, decoder, generator):
        super(Transformer, self).__init__()
        # Initialize source and target embedding layers, encoder, decoder, and generator
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def encode(self, src, src_mask):
        # Apply source embedding and pass through the encoder
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        # Apply target embedding and pass through the decoder
        return self.decoder(self.tgt_embed(tgt), encoder_out, tgt_mask, src_tgt_mask)

    def forward(self, src, tgt):
        # Create masks and pass through the encoder and decoder
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        src_tgt_mask = self.make_src_tgt_mask(src, tgt)
        encoder_out = self.encode(src, src_mask)
        decoder_out = self.decode(tgt, encoder_out, tgt_mask, src_tgt_mask)
        # Generate output and apply log softmax
        out = self.generator(decoder_out)
        out = F.log_softmax(out, dim=-1)
        return out, decoder_out

    def make_src_mask(self, src):
        # Create padding mask for the source
        pad_mask = self.make_pad_mask(src, src)
        return pad_mask

    def make_tgt_mask(self, tgt):
        # Create padding and subsequent masks for the target
        pad_mask = self.make_pad_mask(tgt, tgt)
        seq_mask = self.make_subsequent_mask(tgt, tgt)
        mask = pad_mask & seq_mask
        return mask

    def make_src_tgt_mask(self, src, tgt):
        # Create padding mask for the source-target
        pad_mask = self.make_pad_mask(tgt, src)
        return pad_mask

    def make_pad_mask(self, query, key, pad_idx=1):
        # Create a padding mask for a given query and key sequence 
        # to ignore padding tokens in both queries and key sequences

        # Get the sequence lengths of the query and key inputs
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        # Create a mask for the key where padding tokens (pad_idx) are marked as False
        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)  # Shape: (n_batch, 1, 1, key_seq_len)
        # Repeat the key mask across the query sequence length to match dimensions
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1)    # Shape: (n_batch, 1, query_seq_len, key_seq_len)

        # Create a mask for the query where padding tokens (pad_idx) are marked as False
        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)  # Shape: (n_batch, 1, query_seq_len, 1)
        # Repeat the query mask across the key sequence length to match dimensions
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len)  # Shape: (n_batch, 1, query_seq_len, key_seq_len)

        # Combine key_mask and query_mask to form the final mask, marking padding tokens as False
        mask = key_mask & query_mask
        # Ensure the mask does not require gradient calculations
        mask.requires_grad = False

        return mask

    def make_subsequent_mask(self, query, key):
        # Create a mask to hide future tokens in the target
        # to prevent decoders from seeing future tokens

        # Get the sequence lengths of the query and key inputs
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        # Create a lower triangular matrix with ones below and on the diagonal, and zeros above
        tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('uint8')

        # Convert the numpy array to a PyTorch tensor, ensuring it has the same device as the query tensor
        mask = torch.tensor(tril, dtype=torch.bool, requires_grad=False, device=query.device)

        return mask
