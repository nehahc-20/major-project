"""
Attention-Based Feedback Mining (paper III.F, Fig. 3).
Turns raw cross-attention tensors into: token importance, per-document
relevance scores (Eq. 8), and an uncertainty/entropy signal.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import torch


@dataclass
class AttentionFeedback:
    doc_scores: Dict[int, float]      # Score(D_i), Eq. 8 -- keyed by position in the retrieved list
    token_importance: np.ndarray      # a_1..a_n over prompt tokens, Eq. 7
    entropy: float                    # attention entropy -> uncertainty signal
    low_attention_docs: List[int]     # indices (into the retrieved list) flagged as low-contribution


class AttentionFeedbackMiner:
    def __init__(self, low_attention_percentile: float = 30.0):
        self.low_attention_percentile = low_attention_percentile

    def mine(self, attentions, doc_token_spans: List[tuple]) -> AttentionFeedback:
        """`attentions`: tuple(layers) of [1, heads, q_len, k_len] from the
        first decoding step. We average over layers and heads to get a
        single attention-from-query-token-to-context-token distribution,
        matching A = {a_1, ..., a_n} (Eq. 7)."""
        if attentions is None or len(attentions) == 0 or not doc_token_spans:
            n = 1
            return AttentionFeedback(doc_scores={}, token_importance=np.zeros(n), entropy=0.0, low_attention_docs=[])

        # stack layers -> [layers, heads, q_len, k_len], average over layers+heads
        stacked = torch.stack([layer[0] for layer in attentions])  # [L, H, q, k]
        avg = stacked.mean(dim=(0, 1))  # [q, k]
        # attention FROM the last query position (the token about to be
        # generated) TO every key/context position is the standard proxy
        # for "what did the model actually look at"
        token_importance = avg[-1].float().cpu().numpy()  # [k]
        token_importance = np.clip(token_importance, 0, None)
        if token_importance.sum() > 0:
            token_importance = token_importance / token_importance.sum()

        doc_scores: Dict[int, float] = {}
        for i, (start, end) in enumerate(doc_token_spans):
            start = max(0, min(start, len(token_importance)))
            end = max(start, min(end, len(token_importance)))
            doc_scores[i] = float(token_importance[start:end].sum())  # Eq. 8: sum_t a_t(D_i)

        # attention entropy as an uncertainty signal (low entropy = focused/confident,
        # high entropy = attention smeared everywhere = uncertain retrieval)
        p = token_importance[token_importance > 0]
        entropy = float(-(p * np.log(p + 1e-12)).sum()) if p.size else 0.0

        if doc_scores:
            threshold = np.percentile(list(doc_scores.values()), self.low_attention_percentile)
            low_docs = [i for i, s in doc_scores.items() if s <= threshold]
        else:
            low_docs = []

        return AttentionFeedback(
            doc_scores=doc_scores, token_importance=token_importance,
            entropy=entropy, low_attention_docs=low_docs,
        )
