import torch


def generate_square_mask(size):
    mask = torch.triu(torch.ones(size, size))
    mask = torch.flip(mask, dims=(-1,))
    return mask
