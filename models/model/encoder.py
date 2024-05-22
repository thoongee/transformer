import copy
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, encoder_block, n_layer, norm):
        super(Encoder, self).__init__()
        # Number of encoder layers
        self.n_layer = n_layer
        # Create a list of encoder blocks, each a deep copy of the provided encoder_block
        self.layers = nn.ModuleList([copy.deepcopy(encoder_block) for _ in range(self.n_layer)])
        # Normalization layer to be applied after the encoder layers
        self.norm = norm

    def forward(self, src, src_mask):
        # Initialize output with the source input
        out = src
        # Pass the input through each encoder block
        for layer in self.layers:
            out = layer(out, src_mask)
        # Apply normalization to the final output
        out = self.norm(out)
        return out
