import torch
from torch import Tensor
import numpy as np 
from collections.abc import Iterable
from jaxtyping import Bool, Float, Int
from typing import BinaryIO, IO
import math
import os
import numpy.typing as npt

def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
        Given a tensor of inputs, return the output of softmaxing the given `dim`
        of the input.
    
        Args:
            in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
            dim (int): Dimension of the `in_features` to apply softmax to.
    
        Returns:
            Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
            softmax normalizing the specified `dim`.
    """
    # actually logsumexp have used stablizing trick, we don't need to deal with it again explicitly
    in_features_shifted = in_features - torch.amax(in_features, dim=dim, keepdim=True)
    output = in_features_shifted - torch.logsumexp(in_features_shifted, dim = dim, keepdim = True)
    return torch.exp(output)
    
    
def lr_cosine_schedule(it: int,
                       max_learning_rate: float,
                       min_learning_rate: float,
                       warmup_iters: int,
                       cosine_cycle_iters: int,
                       ):
    """
        Given the parameters of a cosine learning rate decay schedule (with linear
        warmup) and an iteration number, return the learning rate at the given
        iteration under the specified schedule.
    
        Args:
            it (int): Iteration number to get learning rate for.
            max_learning_rate (float): alpha_max, the maximum learning rate for
                cosine learning rate schedule (with warmup).
            min_learning_rate (float): alpha_min, the minimum / final learning rate for
                the cosine learning rate schedule (with warmup).
            warmup_iters (int): T_w, the number of iterations to linearly warm-up
                the learning rate.
            cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.
    
        Returns:
            Learning rate at the given iteration under the specified schedule.
        """
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    elif it >= warmup_iters and it <= cosine_cycle_iters:
        return min_learning_rate + 1/2 * ((1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate))
    else:
        return min_learning_rate
    
    
    
def lr_constant_schedule(it: int, learning_rate: float) -> float:
    """每一步返回相同的学习率，不进行 warmup 或衰减。"""
    if it < 0 or learning_rate < 0:
        raise ValueError("it and learning_rate must be nonnegative")
    return learning_rate


def lr_sgd_schedule(it: int, learning_rate: float) -> float:
    """从第 0 步起返回 learning_rate / sqrt(it + 1)，不叠加 warmup/cosine。"""
    if it < 0 or learning_rate < 0:
        raise ValueError("it and learning_rate must be nonnegative")
    return learning_rate / math.sqrt(it + 1)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    
    if not grads:
        return
    
    else:
        grad_norm = torch.linalg.vector_norm(torch.stack(
            [torch.linalg.vector_norm(g) for g in grads]))
        if grad_norm <= max_l2_norm:
            return
        else:
            scale =  max_l2_norm / (1e-6 + grad_norm)
            with torch.no_grad():
                for g in grads:
                    g.mul_(scale)
    
    
def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(
        src,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]
    
    
    
    
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(checkpoint, out)
    
    
def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    
    if len(dataset) <= context_length:
        raise ValueError("dataset should be more than context_length token")
    
    inputs = torch.empty((batch_size, context_length), dtype=torch.long, device=device)
    targets = torch.empty((batch_size, context_length), dtype=torch.long, device=device)
    
    starts = np.random.randint(0, len(dataset) - context_length, (batch_size,))
    
    ##for i,j in enumerate(starts):
    ##  inputs[i] = torch.tensor(dataset[j:j+context_length-1],
    ##                           dtype=torch.long,
    ##                           device=device)
    ##  targets[i] = torch.tensor(dataset[j+1:j+context_length],
    ##                           dtype=torch.long,
    ##                           device=device)
    
    # starts[:, None].shape 将会从[batch_size, 1]广播为[batch_size, context_length]
    indices = starts[:, None] + np.arange(context_length + 1)
    batch = torch.tensor(dataset[indices], dtype=torch.long, device=device)
    inputs = batch[:,:-1]
    targets = batch[:,1:]
    
    return inputs, targets
