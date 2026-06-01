import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 2
    n_head: int = 6
    n_embd: int = 768

    # Qwen-style additions. Old configs still work because all are optional.
    n_kv_head: Optional[int] = None
    intermediate_size: Optional[int] = None
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )

        # Qwen-style default: use GQA rather than full MHA.
        if self.n_kv_head is None:
             self.n_kv_head = self.n_head

        if self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})"
            )

        if self.intermediate_size is None:
            # Qwen-style gated MLP with a larger intermediate size than classic 4d ReLU MLP.
            # We use a stable default close to common Qwen-family practice.
            self.intermediate_size = _round_to_multiple((16 * self.n_embd) / 3, 256)


def _round_to_multiple(x: float, multiple: int) -> int:
    return int(round(x / multiple) * multiple)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(var + self.eps)
        return (x_norm.to(dtype=x.dtype) * self.weight)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 1_000_000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = None
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def init_inv_freq(self):
        self.inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )

    def forward(self, x: torch.Tensor):
        assert self.inv_freq is not None, "inv_freq not initialized"
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached or self.cos_cached.device != x.device:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().to(dtype=x.dtype)
            self.sin_cached = freqs.sin().to(dtype=x.dtype)
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 4
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.kv_dim = self.n_kv_head * self.head_dim
        self.n_rep = self.n_head // self.n_kv_head

        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.k_proj = nn.Linear(self.n_embd, self.kv_dim, bias=True)
        self.v_proj = nn.Linear(self.n_embd, self.kv_dim, bias=True)
        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        self.rotary = Rotary(self.head_dim, base=config.rope_theta)
        self.attention_dropout = config.attention_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.size()

        q = self.q_proj(x).view(bsz, seqlen, self.n_head, self.head_dim)
        k = self.k_proj(x).view(bsz, seqlen, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(bsz, seqlen, self.n_kv_head, self.head_dim)

        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = q.transpose(1, 2)  # (B, n_head, T, D)
        k = k.transpose(1, 2)  # (B, n_kv_head, T, D)
        v = v.transpose(1, 2)  # (B, n_kv_head, T, D)

        if self.n_kv_head != self.n_head:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        dropout_p = self.attention_dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.n_embd)
        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = config.intermediate_size
        self.gate_proj = nn.Linear(config.n_embd, hidden, bias=False)
        self.up_proj = nn.Linear(config.n_embd, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.self_attn = CausalSelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                norm=RMSNorm(config.n_embd, eps=config.rms_norm_eps),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.transformer.wte.weight

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, Rotary):
            module.init_inv_freq()

    def _forward_emb(self, idx):
        return self.transformer.wte(idx)

    def _forward(self, x, targets, return_logits, return_hidden=False, hidden_layer_idx=-1):
        h = None
        last_layer = len(self.transformer.h) - 1
        pick_layer = last_layer if hidden_layer_idx == -1 else hidden_layer_idx

        for li, block in enumerate(self.transformer.h):
            x = block(x)
            if return_hidden and li == pick_layer:
                h = x

        x = self.transformer.norm(x)

        logits = None
        loss = None

        if return_logits or (targets is not None):
            logits = self.lm_head(x)
            logits = logits.float()

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        if return_hidden and return_logits:
            if targets is None:
                return logits, h
            return loss, logits, h

        if return_logits and (targets is not None):
            return loss, logits

        if return_logits and (targets is None):
            return logits

        if return_hidden and (targets is not None):
            return loss, h

        if return_hidden and (targets is None):
            return h

        return loss

    def compile(self):
        self._forward_eager = self._forward
        self._forward_emb_eager = self._forward_emb
        self._forward_compiled = torch.compile(self._forward_eager)
        self._forward_emb_compiled = torch.compile(self._forward_emb_eager)
        self._is_compiled = True

    def forward(self, idx, targets=None, return_logits=False,
                return_hidden=False, hidden_layer_idx=-1):
        use_compiled = getattr(self, "_is_compiled", False) and (not return_hidden)

        if use_compiled:
            x = self._forward_emb_compiled(idx)
            return self._forward_compiled(
                x, targets, return_logits,
                return_hidden=return_hidden,
                hidden_layer_idx=hidden_layer_idx,
            )
        else:
            x = self._forward_emb_eager(idx) if getattr(self, "_is_compiled", False) else self._forward_emb(idx)
            fwd = self._forward_eager if getattr(self, "_is_compiled", False) else self._forward
            return fwd(
                x, targets, return_logits,
                return_hidden=return_hidden,
                hidden_layer_idx=hidden_layer_idx,
            )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.LongTensor:
        idx = input_ids
        for _ in range(max_new_tokens):
            if idx.size(1) > self.config.sequence_len:
                idx = idx[:, -self.config.sequence_len :]

            logits = self(idx, targets=None, return_logits=True)
            next_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


def parallelize_gpt_model(
    model: GPT,
    device_mesh: DeviceMesh,
    dp_name: Optional[str] = "dp",
    fs_name: Optional[str] = "fs",
    tp_name: Optional[str] = "tp",
    fsdp_reshard_after_forward: bool = True,
):
    required_mesh_names = [x for x in (dp_name, fs_name, tp_name) if x]
    target_ndim = len(required_mesh_names)
    if target_ndim == 0:
        raise ValueError(
            "At least one of dp_name, fs_name, or tp_name must be provided"
        )

    if dp_name and not fs_name:
        raise ValueError("Data parallelism with fully_shard() requires 2D FSDP mesh")

    if device_mesh.ndim < target_ndim:
        raise ValueError(
            f"Expected {target_ndim}-D device mesh {required_mesh_names}, but got mesh with {device_mesh.ndim} dimensions"
        )

    actual_names = list(device_mesh.mesh_dim_names)
    if not all(name in actual_names for name in required_mesh_names):
        raise ValueError(
            f"Expected device mesh to have names {required_mesh_names}, but got {actual_names}"
        )

    tp_enabled = False
    if tp_name:
        tp_mesh = device_mesh[tp_name]
        if tp_mesh.size() > 1:
            _apply_tp(model, tp_mesh)
            tp_enabled = True

    if fs_name:
        fsdp_mesh = (
            device_mesh[fs_name] if not dp_name else device_mesh[dp_name, fs_name]
        )
        _apply_fsdp(model, fsdp_mesh, fsdp_reshard_after_forward, tp_enabled=tp_enabled)


def _apply_tp(model: GPT, tp_mesh: DeviceMesh):
    tp_plan = {
        "transformer.wte": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Replicate(),
        ),
        "lm_head": ColwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Replicate(),
        ),
    }
    parallelize_module(model, tp_mesh, parallelize_plan=tp_plan)

    for block in model.transformer.h:
        attn = block.self_attn
        if attn.n_head % tp_mesh.size() != 0:
            raise ValueError(
                f"n_head {attn.n_head} must be divisible by TP size {tp_mesh.size()}"
            )
        if attn.n_kv_head % tp_mesh.size() != 0:
            raise ValueError(
                f"n_kv_head {attn.n_kv_head} must be divisible by TP size {tp_mesh.size()} for this TP layout"
            )

        tp_plan = {
            "self_attn.q_proj": ColwiseParallel(),
            "self_attn.k_proj": ColwiseParallel(),
            "self_attn.v_proj": ColwiseParallel(),
            "self_attn.o_proj": RowwiseParallel(),
            "mlp.gate_proj": ColwiseParallel(),
            "mlp.up_proj": ColwiseParallel(),
            "mlp.down_proj": RowwiseParallel(),
        }
        parallelize_module(block, tp_mesh, parallelize_plan=tp_plan)

        attn.n_head = attn.n_head // tp_mesh.size()
        attn.n_kv_head = attn.n_kv_head // tp_mesh.size()
        attn.n_rep = attn.n_head // attn.n_kv_head
        attn.kv_dim = attn.n_kv_head * attn.head_dim


def _apply_fsdp(
    model: GPT,
    fsdp_mesh: DeviceMesh,
    reshard_after_forward: bool = True,
    tp_enabled: bool = False,
):
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )

    for block in model.transformer.h:
        shard_placement_fn = None
        if tp_enabled:
            shard_map = {
                block.self_attn.q_proj.weight: Shard(1),
                block.self_attn.k_proj.weight: Shard(1),
                block.self_attn.v_proj.weight: Shard(1),
                block.self_attn.o_proj.weight: Shard(0),
                block.mlp.gate_proj.weight: Shard(1),
                block.mlp.up_proj.weight: Shard(1),
                block.mlp.down_proj.weight: Shard(0),
            }
            shard_placement_fn = lambda param: shard_map.get(param)

        fully_shard(
            block,
            mesh=fsdp_mesh,
            shard_placement_fn=shard_placement_fn,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )

    shard_placement_fn = None
    if tp_enabled:
        shard_map = {
            model.transformer.wte.weight: Shard(1),
            model.lm_head.weight: Shard(1),
        }
        shard_placement_fn = lambda param: shard_map.get(param)

    fully_shard(
        model,
        mesh=fsdp_mesh,
        mp_policy=mp_policy,
        shard_placement_fn=shard_placement_fn,
        reshard_after_forward=False,
    )
