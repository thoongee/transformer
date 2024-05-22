import copy
import torch.nn as nn

from models.layer.residual_connection_layer import ResidualConnectionLayer

class EncoderBlock(nn.Module):

    def __init__(self, self_attention, position_ff, norm, dr_rate=0):
        super(EncoderBlock, self).__init__()
        # Initialize self-attention layer
        self.self_attention = self_attention
        # Initialize first residual connection layer with dropout
        self.residual1 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        # Initialize position-wise feed-forward layer
        self.position_ff = position_ff
        # Initialize second residual connection layer with dropout
        self.residual2 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)

    def forward(self, src, src_mask):
        # Pass source through self-attention and first residual connection
        out = src
        out = self.residual1(out, lambda out: self.self_attention(query=out, key=out, value=out, mask=src_mask))
        # Pass output through position-wise feed-forward and second residual connection
        out = self.residual2(out, self.position_ff)
        return out
