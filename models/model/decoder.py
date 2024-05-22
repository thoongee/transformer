import copy
import torch.nn as nn

class Decoder(nn.Module):

    def __init__(self, decoder_block, n_layer, norm):
        super(Decoder, self).__init__()
        # Number of decoder layers
        self.n_layer = n_layer
        # Create a list of decoder blocks, each a deep copy of the provided decoder_block
        # to ensure each decoder block works independently
        self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(self.n_layer)])
        # Normalization layer to be applied after the decoder layers
        self.norm = norm

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        # Initialize output with the target input
        out = tgt
        # Pass the input through each decoder block
        for layer in self.layers:
            out = layer(out, encoder_out, tgt_mask, src_tgt_mask)
        # Apply normalization to the final output
        out = self.norm(out)
        return out
