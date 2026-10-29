"""
Confidence Estimation (paper III.J, Eq. 14):
Conf = mean token probability, blended with attention-entropy consistency
and retrieval-score consistency.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import math
import numpy as np


@dataclass
class ConfidenceReport:
    token_prob_conf: float
    attention_consistency: float
    retrieval_consistency: float
    overall: float


class ConfidenceEstimator:
    def __init__(self, w_token: float = 0.5, w_attention: float = 0.25, w_retrieval: float = 0.25):
        self.w_token, self.w_attention, self.w_retrieval = w_token, w_attention, w_retrieval

    def token_probability_confidence(self, token_logprobs: List[float]) -> float:
        """Eq. 14: Conf = (1/n) * sum P(t_i). We average the *probabilities*
        (exp of logprobs), not the log-probabilities, per the paper formula."""
        if not token_logprobs:
            return 0.0
        probs = [math.exp(lp) for lp in token_logprobs]
        return float(np.mean(probs))

    def attention_consistency(self, entropy: float, max_entropy: float) -> float:
        """Low attention entropy -> focused/confident retrieval use.
        Normalized and inverted so higher == more confident."""
        if max_entropy <= 0:
            return 0.5
        norm = min(entropy / max_entropy, 1.0)
        return float(1.0 - norm)

    def retrieval_consistency(self, similarities: List[float]) -> float:
        """High, tightly-clustered similarity scores among the top-k hits
        indicate the retriever is confident about relevance; a big gap
        between the best and the rest indicates a clean, confident hit too.
        We combine mean similarity with (1 - normalized spread)."""
        if not similarities:
            return 0.0
        sims = np.array(similarities)
        mean_sim = float(np.clip(sims.mean(), 0, 1))
        spread = float(np.std(sims))
        return float(np.clip(mean_sim * (1.0 - min(spread, 1.0)), 0, 1))

    def estimate(self, token_logprobs: List[float], attn_entropy: float,
                 max_possible_entropy: float, retrieval_sims: List[float]) -> ConfidenceReport:
        tconf = self.token_probability_confidence(token_logprobs)
        aconf = self.attention_consistency(attn_entropy, max_possible_entropy)
        rconf = self.retrieval_consistency(retrieval_sims)
        overall = self.w_token * tconf + self.w_attention * aconf + self.w_retrieval * rconf
        return ConfidenceReport(tconf, aconf, rconf, float(overall))
