import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO, Literal
import torch.nn as nn
from einops import einsum, rearrange
from .utils import softmax
from .activation import silu

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math

class Linear(nn.Module):
    def __init__(self,
                 in_features:int, 
                 out_features:int,
                 device=None,
                 dtype=None):
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        
        std = math.sqrt(2/(in_features+out_features))
        self.W = nn.Parameter(
            torch.empty(
                (out_features, in_features),
                device=device,
                dtype=dtype,
            )
        )
        nn.init.trunc_normal_(self.W, mean = 0, std = std, a = -3 * std, b = 3 * std)
        
    def forward(self, x: torch.Tensor):
        return einsum(x, self.W, "batch sequence d_in, d_out d_in -> batch sequence d_out")
            
        

class Embedding(nn.Module):
    def __init__(self,
                 num_embeddings:int, 
                 embedding_dim:int,
                 device=None,
                 dtype=None):
        
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        
        self.W = nn.Parameter(
            torch.empty(
                (num_embeddings, embedding_dim),
                device=device,
                dtype=dtype,
            )
        )
        nn.init.trunc_normal_(self.W, mean = 0, std = 1, a = -3, b = 3)
        
    def forward(self, token_ids: torch.Tensor):
        return self.W[token_ids]
       

class SwiGLU(nn.Module):
    def __init__(self, 
                 d_model: int,
                 d_ff: int,
                 device = None,
                 dtype = None
                 ):
        """
        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.proj1 = Linear(d_model,d_ff, device=device, dtype = dtype)
        self.proj2 = Linear(d_ff,d_model, device=device, dtype = dtype)
        self.proj3 = Linear(d_model,d_ff, device=device, dtype = dtype)
        

    def forward(self,
                in_features: Float[Tensor, " ... d_model"]
                ):
        output = silu(self.proj1(in_features)) * self.proj3(in_features)
        output = self.proj2(output)
        return output
        




class SiLUFFN(nn.Module):
    """Vanilla SiLU FFN, no gating,  d_ff = 4 * d_model when running abalation"""

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.proj1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.proj2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, in_features: Float[Tensor, " ... d_model"]):
        return self.proj2(silu(self.proj1(in_features)))


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module. This function should accept the following parameters:
            d_model: int  Hidden dimension of the model
            eps: float = 1e-5  Epsilon value for numerical stability
            device: torch.device | None = None  Device to store the parameters on
            dtype: torch.dtype | None = None  Data type of the parameters
        """
        super().__init__()
        
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.d_model = d_model
        self.g = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape.
        """
        
        in_type = x.dtype
        x = x.to(torch.float32)
        RMS_x = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        output = x / RMS_x * self.g
        
        return output.to(in_type)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """ Construct the RoPE module and create buffers if needed.
        theta: float  Θ value for the RoPE
        d_k: int  dimension of query and key vectors
        max_seq_len: int  Maximum sequence length that will be input
        device: torch.device | None = None  Device to store the buffer on
        """
        
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("dimension of query and key vectors should be even")
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        
         # position：(max_seq_len,)
        positions = torch.arange(
            max_seq_len, device=device, dtype=torch.float32
        )

        # 每一对分量对应的频率：(d_k // 2,), 同一个query或者key之间的不同对分量也用的不同频率
        inv_freq = theta ** (
            -torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
            / d_k
        )

        # 广播：(max_seq_len, 1) * (d_k // 2,) -> (max_seq_len, d_k // 2)
        # 不同位置之间的频率与位置呈现线性，保证平移不变性
        angles = positions[:, None] * inv_freq

        self.register_buffer("sin", torch.sin(angles), persistent=False)
        self.register_buffer("cos", torch.cos(angles), persistent=False)

        

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
         Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. 
         Note that you should tolerate x with an arbitrary number of batch dimensions. 
         You should assume that the token positions are a tensor of shape (..., seq_len) 
         specifying the token positions of x along the sequence dimension.
        
        """
        # 查出各个 token 位置对应的旋转系数
        sin = self.sin[token_positions]
        cos = self.cos[token_positions]

        # 分别取出d_k这一维序号为偶和奇的序号
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # shape:(max_seq_len, d_k // 2)
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        # 先把每对分量放在一起，再合并最后两维
        output = torch.stack(
            (rotated_even, rotated_odd), dim=-1
        ).flatten(start_dim=-2)

        return output.to(x.dtype)


class multihead_self_attention(nn.Module):
    """
    Given the key, query, and value projection weights of a naive unbatched
        implementation of multi-head attention, return the output of an optimized batched
        implementation. This implementation should handle the key, query, and value projections
        for all heads in a single matrix multiply.
        RoPE is enabled by default; set use_rope=False to disable it.
        See section 3.2.2 of Vaswani et al., 2017.
    
        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            num_heads (int): Number of heads to use in multi-headed attention.
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
            q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
            k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
            v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
            o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
            in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
    
        Returns:
            Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
            implementation with the given QKV projection weights and input features.
    """
    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 max_seq_len: int,
                 theta: float = 10000.0,
                 use_rope: bool = True,
                 device = None,
                 dtype = None
                 ):
        super().__init__()
        self.d_model = d_model
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("num_heads must be positive and divide d_model")
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        self.device = device
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        if use_rope:
            if theta is None:
                raise ValueError("theta must be provided when use_rope=True")
            self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device)
        
        # 最后两维的顺序是(queries, keys),所以下半部分应该保留
        mask = torch.tril(torch.ones((max_seq_len, max_seq_len),
                                     dtype=torch.bool,
                                     device=device,
                                     )
                          )
        self.register_buffer("causal_mask", mask, persistent=False)
        
        
    def forward(self, 
                in_features: Float[Tensor, " ... sequence_length d_model"],
                token_positions: Int[Tensor, " ... sequence_length"] | None = None
                ):
        """
        Args:
            in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
            token_positions (int[Tensor, " ... sequence_length"]):
            
        Returns:
            Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
            implementation with the given QKV projection weights and input features.
        
        """
        seq_len = in_features.shape[-2]
        if seq_len > self.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")

        Q = self.q_proj(in_features)
        K = self.k_proj(in_features)
        V = self.v_proj(in_features)
        if token_positions is None:
            token_positions = torch.arange(0, seq_len, dtype=int, device=in_features.device)
        # (..., seq_len, d_model) -> (..., num_heads, seq_len, d_per_head)
        Q = rearrange(Q, "... s (h d) -> ... h s d", h=self.num_heads)
        K = rearrange(K, "... s (h d) -> ... h s d", h=self.num_heads)
        V = rearrange(V, "... s (h d) -> ... h s d", h=self.num_heads)
        if self.use_rope:
            # 这里的操作是为了让广播机制正常运行
            positions = token_positions.unsqueeze(-2)
            Q = self.rope(Q, positions)
            K = self.rope(K, positions)
        mask = self.causal_mask[:seq_len, :seq_len]
        output = scaled_dot_product_attention(Q, K, V, mask)
        output = rearrange(output, "... h s d -> ... s (h d)")
        return self.o_proj(output)



class transformer_block(nn.Module):
    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 max_seq_len: int,
                 theta: float = 10000.0,
                 use_rope: bool = True,
                 norm_position: Literal["pre", "post"] = "pre",
                 use_norm: bool = True,
                 use_swiglu: bool = True,
                 device = None,
                 dtype = None
                 ):
        """
        Args:
            d_model (int): The dimensionality of the Transformer block input.
            num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
                evenly divisible by `num_heads`.
            d_ff (int): Dimensionality of the feed-forward inner layer.
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
            theta (float): RoPE parameter.
            use_swiglu (bool): Use SwiGLU if True, otherwise a plain SiLU FFN.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        if use_rope:
            self.theta = theta
        self.use_norm = use_norm
        if use_norm:
            if norm_position not in ("pre", "post"):
                raise ValueError("norm_position must be pre or post!")
            self.norm_position = norm_position
            self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.use_swiglu = use_swiglu
        ffn_class = SwiGLU if use_swiglu else SiLUFFN
        self.FFN = ffn_class(d_model, d_ff, device=device, dtype=dtype)
        self.attention = multihead_self_attention(d_model,
                                                  num_heads,
                                                  max_seq_len,
                                                  theta,
                                                  use_rope=use_rope,
                                                  device=device,
                                                  dtype=dtype
                                                  )


    def forward(self, 
                in_features: Float[Tensor, " ... sequence_length d_model"],
                token_positions: Int[Tensor, " ... sequence_length"] | None = None
                ):
        if self.use_norm and self.norm_position == "pre":
            res1 = self.attention(self.norm1(in_features), token_positions)
        else:
            res1 = self.attention(in_features, token_positions)
        in_features = res1 + in_features
        if self.use_norm and self.norm_position == "post":
            in_features = self.norm1(in_features)
        if self.use_norm and self.norm_position == "pre":
            res2 = self.FFN(self.norm2(in_features))
        else:
            res2 = self.FFN(in_features)
        out_features = res2 + in_features
        if self.use_norm and self.norm_position == "post":
            out_features = self.norm2(out_features)
        return out_features



class transformer_lm(nn.Module):
    """
    Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
    """
    def __init__(self, 
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float = 10000.0,
                 use_rope: bool = True,
                 norm_position: Literal["pre", "post"] = "pre",
                 use_norm: bool = True,
                 use_swiglu: bool = True,
                 device = None,
                 dtype = None
                 ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.context_length = context_length
        self.use_rope = use_rope
        self.use_swiglu = use_swiglu
        if use_rope:
            self.rope_theta = rope_theta
        self.device = device
        self.dtype = dtype
        self.use_norm = use_norm
        self.norm_position = norm_position
        self.Transform_Blocks = nn.ModuleList([
            transformer_block(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                use_rope = use_rope,
                use_norm = use_norm,
                use_swiglu = use_swiglu,
                norm_position = norm_position,
                device=device,
                dtype=dtype,
                )
            for _ in range(num_layers)
            ])
        self.Token_Embedding = Embedding(num_embeddings=vocab_size,
                                         embedding_dim=d_model,
                                         device=device,
                                         dtype=dtype
                                         )
        if use_norm:
            self.Norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.Output_Embedding = Linear(in_features=d_model,
                                       out_features=vocab_size,
                                       device=device,
                                       dtype=dtype
                                       )
        
        
    def forward(self, 
                in_indices: Int[Tensor, " batch_size sequence_length"],
                token_positions: Int[Tensor, " ... sequence_length"] | None = None
                ):
        """
            in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
                    `sequence_length` is at most `context_length`.
        
            Returns:
                Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
                next-word distribution for each token.
        """
        in_features = self.Token_Embedding(in_indices)
        for block in self.Transform_Blocks:
            in_features = block(in_features, token_positions)
        if self.use_norm:
            logits = self.Output_Embedding(self.Norm(in_features))
        else:
            logits = self.Output_Embedding(in_features)
        return logits


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    scores = scores / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
        
    weights = softmax(scores, dim=-1)
    output = einsum(weights, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return output
