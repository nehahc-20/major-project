"""
Lightweight stand-ins for KnowledgeBase and LLMGenerator that need no
model downloads, so the pipeline's control-flow / math can be unit-tested
offline (e.g. in CI, or this sandbox). Swap in the real classes from
src/knowledge_base.py and src/generation.py for actual runs.
"""
from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass
from typing import List
import numpy as np
import torch

from src.knowledge_base import Document
from src.generation import GenerationResult


def _hash_embed(text: str, dim: int = 32) -> np.ndarray:
    """Deterministic bag-of-words hashing embedding -- not semantically
    meaningful, only used so tests don't need real embedding models."""
    vec = np.zeros(dim, dtype="float32")
    for tok in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class MockKnowledgeBase:
    def __init__(self, texts: List[str]):
        self.documents = [Document(doc_id=i, text=t) for i, t in enumerate(texts)]
        self._embs = np.stack([_hash_embed(t) for t in texts])

    def embed_query(self, query: str) -> np.ndarray:
        return _hash_embed(query).reshape(1, -1)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        sims = self._embs @ query_embedding.reshape(-1)
        order = np.argsort(-sims)[:top_k]
        return [(self.documents[i], float(sims[i])) for i in order]


class MockGenerator:
    """Fabricates plausible attentions/logprobs so AttentionFeedbackMiner
    and ConfidenceEstimator have real tensors to operate on."""

    def __init__(self, answer: str = "paris is the capital", n_layers: int = 2, n_heads: int = 2):
        self.answer = answer
        self.n_layers, self.n_heads = n_layers, n_heads

    def generate(self, query: str, documents: List[str]) -> GenerationResult:
        prompt_len = sum(len(d.split()) for d in documents) + len(query.split()) + 5
        prompt_len = max(prompt_len, 4)
        spans, cursor = [], 0
        for d in documents:
            span_len = max(len(d.split()), 1)
            spans.append((cursor, cursor + span_len))
            cursor += span_len

        gen_ids = torch.randint(0, 100, (5,))
        # attention concentrated on the first document to exercise the
        # "low-attention doc" pruning path deterministically
        attn = torch.zeros(1, self.n_heads, 1, prompt_len)
        if spans:
            s, e = spans[0]
            attn[0, :, 0, s:e] = 1.0 / max(e - s, 1)
        attentions = tuple(attn.clone() for _ in range(self.n_layers))

        return GenerationResult(
            text=self.answer,
            generated_token_ids=gen_ids,
            input_token_ids=torch.arange(prompt_len),
            attentions=attentions,
            token_logprobs=[-0.1, -0.2, -0.15, -0.3, -0.05],
            doc_token_spans=spans,
        )
