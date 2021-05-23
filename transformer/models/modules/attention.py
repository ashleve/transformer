import torch
from torch import nn
from torch.nn.init import xavier_uniform_


class SelfAttention(nn.Module):
    def __init__(self, emb_dim, input_length):
        super().__init__()

        self.Q = nn.parameter.Parameter(torch.empty((emb_dim, input_length)))
        self.K = nn.parameter.Parameter(torch.empty((emb_dim, input_length)))
        self.V = nn.parameter.Parameter(torch.empty((emb_dim, input_length)))

    def _reset_parameters(self):
        xavier_uniform_(self.Q)
        xavier_uniform_(self.K)
        xavier_uniform_(self.V)

    def forward(self, x):

        q = torch.matmul(self.Q, x)
        k = torch.matmul(self.K, x)
        v = torch.matmul(self.V, x)

        probs = torch.softmax(torch.matmul(q.T, k))
        output = torch.cat(torch.matmul(v, probs), x)

        return output


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):

        return None
