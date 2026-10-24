import torch
from torch import Tensor
from jaxtyping import Bool, Float, Int

def sigmoid(in_features): 
    return 1 / (1 + torch.exp(- in_features))


def silu(in_features):
    return in_features * sigmoid(in_features)

