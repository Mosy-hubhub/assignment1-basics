from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr * grad # Learning rate scheduling is handled by the caller.

        return loss
    
    
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay = 0.01, eps=1e-8):
            if lr < 0:
                raise ValueError(f"Invalid learning rate: {lr}")
            defaults = {"lr": lr,
                        "beta1": betas[0],
                        "beta2": betas[1],
                        "weight_decay": weight_decay,
                        "eps": eps}
            super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, or 0.
                if t == 1:
                    # 状态与参数保持相同的 dtype 和 device，避免恢复时发生类型转换。
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                m = state["m"]
                v = state["v"]
                grad = p.grad.data # Get the gradient of loss with respect    
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                # weight decay
                p.data *= 1 - lr * weight_decay
                # Update the first moment estimate
                m = beta1 * m + (1 - beta1) * grad
                # Update the second moment estimate
                v = beta2 * v + (1 - beta2) * grad**2
                p.data -= lr_t * m / (torch.sqrt(v) + eps) 
                state["t"] = t + 1 # Increment iteration number.
                state["m"] = m # Update m in state
                state["v"] = v # Update v in state
                
                
        return loss
    
