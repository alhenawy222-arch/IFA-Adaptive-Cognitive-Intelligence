import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import RMSNorm, SSMBlock, Attention, SwiGLU, MoE


class HybridLayer(nn.Module):
    """[SSM or Attention] mixer, then [MLP or MoE] feed-forward, both pre-norm residual."""

    def __init__(self, cfg, idx):
        super().__init__()
        self.use_attn = (idx + 1) % cfg.attn_every == 0
        self.use_moe = (idx + 1) % cfg.moe_every == 0
        self.n1, self.n2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.mixer = Attention(cfg) if self.use_attn else SSMBlock(cfg)
        self.ff = MoE(cfg) if self.use_moe else SwiGLU(cfg.d_model, cfg.mlp_mult)

    def forward(self, x):
        x = x + self.mixer(self.n1(x))
        return x + self.ff(self.n2(x))


class IFAOmega(nn.Module):
    """Hybrid language model: SSM backbone + sparse attention + MoE."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(HybridLayer(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight  # tie weights
        nn.init.normal_(self.emb.weight, std=0.02)  # small init -> sane starting loss

    def forward(self, ids, targets=None):
        x = self.emb(ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            aux = sum(l.ff.aux_loss for l in self.layers if l.use_moe)
            loss = loss + self.cfg.moe_aux_weight * aux
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=100, temperature=1.0):
        for _ in range(max_new_tokens):
            logits, _ = self(ids[:, -self.cfg.max_seq_len:])
            p = F.softmax(logits[:, -1] / temperature, dim=-1)
            ids = torch.cat([ids, torch.multinomial(p, 1)], dim=1)
        return ids

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
