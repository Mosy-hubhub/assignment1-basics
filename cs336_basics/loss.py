import torch
from torch import Tensor
from jaxtyping import Bool, Float, Int

def Cross_entropy_loss(inputs: Float[Tensor, " batch_size vocab_size"],
                       targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
        loss across examples.
    
        Args:
            inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
                unnormalized logit of jth class for the ith example.
            targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
                Each value must be between 0 and `num_classes - 1`.
    
        Returns:
            Float[Tensor, ""]: The average cross-entropy loss across examples.
        """
    output = torch.logsumexp(inputs, dim=-1) - inputs[torch.arange(inputs.shape[0]), targets]
    return torch.mean(output, dim = 0)
    
    
