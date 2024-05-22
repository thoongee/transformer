import math
import torch.nn as nn

class TokenEmbedding(nn.Module):

    def __init__(self, d_embed, vocab_size):
        super(TokenEmbedding, self).__init__()
        # Initialize an embedding layer with vocabulary size and embedding dimension
        self.embedding = nn.Embedding(vocab_size, d_embed)
        # Store the embedding dimension
        self.d_embed = d_embed

    def forward(self, x):
        # To stabilise the size of the embedding vector and make the model's training more efficient, we ..
        # Perform the embedding lookup and scale the embeddings by the square root of the embedding dimension
        out = self.embedding(x) * math.sqrt(self.d_embed)
        return out
