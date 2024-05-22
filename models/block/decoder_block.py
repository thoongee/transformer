import copy
import torch.nn as nn

from models.layer.residual_connection_layer import ResidualConnectionLayer

class DecoderBlock(nn.Module):

    def __init__(self, self_attention, cross_attention, position_ff, norm, dr_rate=0):
        super(DecoderBlock, self).__init__()
        # Initialize self-attention layer
        self.self_attention = self_attention
        # Initialize first residual connection layer with dropout
        self.residual1 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        # Initialize cross-attention layer
        self.cross_attention = cross_attention
        # Initialize second residual connection layer with dropout
        self.residual2 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        # Initialize position-wise feed-forward layer
        self.position_ff = position_ff
        # Initialize third residual connection layer with dropout
        self.residual3 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        # Pass target through self-attention and first residual connection
        out = tgt
        out = self.residual1(out, lambda out: self.self_attention(query=out, key=out, value=out, mask=tgt_mask))
        # Pass output through cross-attention and second residual connection
        out = self.residual2(out, lambda out: self.cross_attention(query=out, key=encoder_out, value=encoder_out, mask=src_tgt_mask))
        # Pass output through position-wise feed-forward and third residual connection
        out = self.residual3(out, self.position_ff)
        return out
