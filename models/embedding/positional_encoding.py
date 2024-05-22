import math
import torch
import torch.nn as nn

"""
Use the sin function for even indices and the cos function for odd indices,
using only values between -1 and 1 as positional information

"""


class PositionalEncoding(nn.Module):

    def __init__(self, d_embed, max_len=256, device=torch.device('cuda:0')):
        super(PositionalEncoding, self).__init__()
        # Create a tensor to hold the positional encodings
        encoding = torch.zeros(max_len, d_embed)
        # Disable gradients for the positional encoding tensor
        encoding.requires_grad = False
        # Create a tensor with positions from 0 to max_len-1
        position = torch.arange(0, max_len).float().unsqueeze(1)
        # Calculate the div_term for the encoding formula (div_term : Scaling factor for each location)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))
        # Apply sine to even indices in the encoding tensor
        encoding[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices in the encoding tensor
        encoding[:, 1::2] = torch.cos(position * div_term)
        # Add a batch dimension and move encoding tensor to the specified device
        self.encoding = encoding.unsqueeze(0).to(device)

    def forward(self, x):
        # Extract sequence length from the input tensor
        _, seq_len, _ = x.size()
        # Slice the positional encoding tensor to match the sequence length
        pos_embed = self.encoding[:, :seq_len, :]
        # Add positional encoding to the input tensor
        out = x + pos_embed
        return out
