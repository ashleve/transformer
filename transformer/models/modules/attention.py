import torch
from torch import nn
from torch.nn.init import xavier_uniform_


class SelfAttention(nn.Module):
    
    def __init__(self, embedding_dim, input_length):
        super().__init__()

        self.Q = nn.parameter.Parameter(torch.empty((embedding_dim, input_length)))
        self.K = nn.parameter.Parameter(torch.empty((embedding_dim, input_length)))
        self.V = nn.parameter.Parameter(torch.empty((embedding_dim, input_length)))
        
    def _reset_parameters(self):
        xavier_uniform_(self.Q)
        xavier_uniform_(self.K)
        xavier_uniform_(self.V)

    def forward(self, x):

        return x
    
    
    

    