from dataclasses import dataclass


@dataclass
class IFAConfig:
    vocab_size: int = 50257      # GPT-2 tokenizer; char mode overrides this
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 4
    attn_every: int = 4          # attention layer every Nth layer, rest are SSM
    moe_every: int = 2           # MoE replaces the MLP in every Nth layer
    n_experts: int = 4
    top_k: int = 2
    d_state: int = 16            # SSM state size
    d_conv: int = 4              # SSM causal conv width
    expand: int = 2              # SSM inner width = expand * d_model
    mlp_mult: float = 2.667      # SwiGLU hidden multiplier
    max_seq_len: int = 256
    dropout: float = 0.0
    moe_aux_weight: float = 0.01
