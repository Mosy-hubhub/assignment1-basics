from torch import Tensor
from jaxtyping import Float
from .utils import softmax
import torch

def decode(
    logits: Float[Tensor, "... vocab_size"],
    temperature: float = 1.0,
    top_p: float | None = None,
):
    if temperature <= 0:
        raise ValueError("temperature must bigger than 0")

    probs = softmax(logits / temperature, dim=-1)

    if top_p is not None:
        if not 0 < top_p <= 1:
            raise ValueError("top_p must in (0, 1]")

        # 按概率从大到小排序，同时保留原 token ID
        sorted_probs, sorted_indices = torch.sort(
            probs, dim=-1, descending=True
        )

        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 保留使累计概率首次达到或超过 top_p 的那个 token
        remove_mask = cumulative_probs > top_p
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False

        sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)

        # 对保留下来的概率重新归一化
        sorted_probs = sorted_probs / sorted_probs.sum(
            dim=-1, keepdim=True
        )

        # 从排序后的顺序还原到原 token ID 顺序
        probs = torch.zeros_like(probs).scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_probs,
        )

    return probs