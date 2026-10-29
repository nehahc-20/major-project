"""
Feedback Optimization Engine + Dynamic Reranking (paper III.G, III.H):
context pruning (Eq. 9), query expansion (Eq. 10), dynamic Top-K (Eq. 11),
and hybrid relevance scoring (Eq. 12).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class RerankedDoc:
    doc_id: int
    text: str
    similarity: float
    attention_score: float
    confidence: float
    hybrid_score: float


class FeedbackOptimizationEngine:
    def __init__(self, alpha: float = 0.4, beta: float = 0.4, gamma: float = 0.2,
                 min_k: int = 2, max_k: int = 10):
        """alpha/beta/gamma weight semantic similarity, attention contribution,
        and confidence respectively in Eq. 12. min_k/max_k bound Eq. 11."""
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.min_k, self.max_k = min_k, max_k

    def context_pruning(self, retrieved: list, doc_scores: dict, low_attention_idx: List[int]) -> list:
        """R_p = R0 - D_low (Eq. 9): drop documents flagged as low-attention,
        but never prune below 1 remaining document."""
        if not low_attention_idx or len(retrieved) <= 1:
            return retrieved
        keep = [i for i in range(len(retrieved)) if i not in set(low_attention_idx)]
        if not keep:
            keep = [max(doc_scores, key=doc_scores.get)] if doc_scores else [0]
        return [retrieved[i] for i in keep]

    def dynamic_top_k(self, query_complexity: float, attention_feedback_score: float) -> int:
        """K = f(c_q, a_f) (Eq. 11): more complex queries / more diffuse
        attention (i.e. the model needed to look around more) -> larger K."""
        raw = self.min_k + (self.max_k - self.min_k) * (0.6 * query_complexity + 0.4 * attention_feedback_score)
        return int(round(np.clip(raw, self.min_k, self.max_k)))

    def hybrid_rerank(self, retrieved: List[Tuple], doc_scores: dict, confidence: float) -> List[RerankedDoc]:
        """R_score(D_i) = alpha*Sim + beta*A(D_i) + gamma*Conf(D_i) (Eq. 12).
        `retrieved` is a list of (Document, similarity) tuples in the SAME
        order used when doc_scores keys were assigned (positional index)."""
        results = []
        max_attn = max(doc_scores.values()) if doc_scores else 1.0
        max_attn = max_attn if max_attn > 0 else 1.0
        for i, (doc, sim) in enumerate(retrieved):
            a_score = doc_scores.get(i, 0.0) / max_attn  # normalize to [0,1]
            hybrid = self.alpha * sim + self.beta * a_score + self.gamma * confidence
            results.append(RerankedDoc(
                doc_id=doc.doc_id, text=doc.text, similarity=sim,
                attention_score=a_score, confidence=confidence, hybrid_score=hybrid,
            ))
        results.sort(key=lambda d: d.hybrid_score, reverse=True)
        return results

    def needs_query_expansion(self, confidence: float, low_conf_threshold: float = 0.4) -> bool:
        return confidence < low_conf_threshold
