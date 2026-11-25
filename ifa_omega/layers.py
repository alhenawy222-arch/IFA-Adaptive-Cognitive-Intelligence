"""Building blocks: RMSNorm, SSM (Mamba-style, pure PyTorch), Attention, SwiGLU, MoE."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class SSMBlock(nn.Module):
    """Selective state-space block (Mamba-style) in plain PyTorch.

    Uses a sequential scan: simple and correct, but slower than the CUDA
    kernel of mamba-ssm. Fine for small models and short sequences on Colab.
    """

    def __init__(self, cfg):
        super().__init__()
        d, e = cfg.d_model, cfg.expand * cfg.d_model
        self.d_inner, self.d_state = e, cfg.d_state
        self.dt_rank = max(1, math.ceil(d / 16))
        self.in_proj = nn.Linear(d, 2 * e, bias=False)
        self.conv = nn.Conv1d(e, e, cfg.d_conv, groups=e, padding=cfg.d_conv - 1)
        self.x_proj = nn.Linear(e, self.dt_rank + 2 * cfg.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, e)
        A = torch.arange(1, cfg.d_state + 1).float().repeat(e, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(e))
        self.out_proj = nn.Linear(e, d, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[..., :L].transpose(1, 2)  # causal
        u = F.silu(u)
        dt, Bm, Cm = self.x_proj(u).split([self.dt_rank, self.d_state, self.d_state], -1)
        dt = F.softplus(self.dt_proj(dt))                # (B,L,E)
        A = -torch.exp(self.A_log)                       # (E,N)
        dA = torch.exp(dt.unsqueeze(-1) * A)             # (B,L,E,N)
        dBu = dt.unsqueeze(-1) * Bm.unsqueeze(2) * u.unsqueeze(-1)
        h = x.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(L):                               # sequential scan
            h = dA[:, t] * h + dBu[:, t]
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, 1) + u * self.D
        return self.out_proj(y * F.silu(z))


class Attention(nn.Module):
    """Causal multi-head self-attention."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.h = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = cfg.dropout

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.drop if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(B, L, D))


class SwiGLU(nn.Module):
    def __init__(self, d, mult):
        super().__init__()
        hdim = int(d * mult)
        self.w1 = nn.Linear(d, hdim, bias=False)
        self.w2 = nn.Linear(hdim, d, bias=False)
        self.w3 = nn.Linear(d, hdim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """Top-k routed mixture of SwiGLU experts with a load-balancing loss."""

    def __init__(self, cfg):
        super().__init__()
        self.k, self.n = cfg.top_k, cfg.n_experts
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(cfg.d_model, cfg.mlp_mult) for _ in range(cfg.n_experts))
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        B, L, D = x.shape
        flat = x.reshape(-1, D)
        probs = F.softmax(self.router(flat), dim=-1)
        topv, topi = probs.topk(self.k, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            tok, slot = (topi == e).nonzero(as_tuple=True)
            if tok.numel():
                out.index_add_(0, tok, expert(flat[tok]) * topv[tok, slot].unsqueeze(-1))
        # Switch-style balance loss: fraction routed * mean prob per expert
        frac = F.one_hot(topi[:, 0], self.n).float().mean(0)
        self.aux_loss = self.n * (frac * probs.mean(0)).sum()
        return out.view(B, L, D)
