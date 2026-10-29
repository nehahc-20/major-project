"""
A GPT-style decoder-only transformer, implemented from scratch (no
`transformers` model classes, no pretrained weights, no download). This is
what "building an LLM" actually means at the architecture level: token +
positional embeddings, stacked self-attention + MLP blocks with residual
connections, a tied output head. Every weight starts as random noise and
is only ever updated by scratch_llm/train.py.

Because we write the attention module ourselves, we get real, reliable
per-head attention weights back from every forward pass -- no dependency
on a library's output_attentions flag or attn_implementation quirks (the
exact fragility that src/generation.py has to work around for Qwen/Llama).
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8000
    block_size: int = 256      # max sequence length (context window)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask, implemented explicitly
    (not via F.scaled_dot_product_attention) so we can return the real
    attention weight matrix every forward pass -- that's the whole point
    for this project."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(
            1, 1, config.block_size, config.block_size
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B, nh, T, hd]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)     # [B, nh, T, T]
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v                                                  # [B, nh, T, hd]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.proj(out))
        return out, att  # return real attention weights -- consumed by AttentionFeedbackMiner


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        attn_out, att_weights = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, att_weights


class GPT(nn.Module):
    """The model itself: embed tokens+positions, run N transformer blocks,
    project back to vocab logits. Weight-tied output head (standard GPT-2
    trick: saves params, and ties input/output token representations)."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT model initialized: {n_params/1e6:.2f}M parameters "
              f"({config.n_layer} layers, {config.n_head} heads, {config.n_embd} dim)")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None,
                return_attentions: bool = False):
        B, T = input_ids.shape
        assert T <= self.config.block_size, f"sequence length {T} exceeds block_size {self.config.block_size}"

        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))

        all_attentions = [] if return_attentions else None
        for block in self.blocks:
            x, att = block(x)
            if return_attentions:
                all_attentions.append(att)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100,
            )

        return {"logits": logits, "loss": loss, "attentions": all_attentions}

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64,
                 temperature: float = 0.8, top_k: int = 40, eos_token_id: int = None):
        """Greedy-ish autoregressive sampling with temperature + top-k.
        Returns (generated_ids, per_step_logprob, last_step_attentions) so
        callers get everything scratch_llm/soar_adapter.py needs to build
        a GenerationResult without a second forward pass."""
        self.eval()
        logprobs = []
        last_attentions = None
        for step in range(max_new_tokens):
            cond = input_ids[:, -self.config.block_size:]
            out = self.forward(cond, return_attentions=(step == 0))
            if step == 0:
                last_attentions = out["attentions"]  # first step sees the FULL prompt (matches src/generation.py's convention)
            logits = out["logits"][:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            logprobs.append(float(torch.log(probs[0, next_id[0, 0]] + 1e-12)))
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if eos_token_id is not None and next_id.item() == eos_token_id:
                break
        return input_ids, logprobs, last_attentions
