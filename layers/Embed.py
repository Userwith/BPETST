import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
from Discretization import Discretization, BPEEncode_Channel_Splitting

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class BPEEmbedding_univariate(nn.Module):
    def __init__(self, d_model,is_channel_splitting=True,num_bins=100):
        super(BPEEmbedding_univariate, self).__init__()
        self.discrete = Discretization(num_bins=num_bins)
        if is_channel_splitting:
            self.encoder = BPEEncode_Channel_Splitting()
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.embedding_linear = nn.Linear(1,d_model)

    def forward(self,x):
        x = self.discrete(x)
        x = self.encoder(x)[0]



