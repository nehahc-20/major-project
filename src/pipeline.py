"""
SOAR-RAG Pipeline (paper III.A/III.I, Fig. 1 & Fig. 4):
Q -> R0 -> G0 -> Af -> Rf -> Gf, iterated to convergence.

  1. Retrieval        4. Reranking
  2. Generation        5. Refinement
  3. Attention analysis    (loop back to 1 if not converged)

Stopping criteria (III.I.1): confidence threshold reached, attention
entropy minimized, or max iterations reached. Convergence via Eq. 13.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import time
import math
import numpy as np

from .query_processing import QueryProcessor
from .attention_feedback import AttentionFeedbackMiner
from .confidence import ConfidenceEstimator
from .reranking import FeedbackOptimizationEngine


@dataclass
class IterationTrace:
    iteration: int
    query_used: str
    retrieved_doc_ids: List[int]
    response: str
    confidence: float
    attention_entropy: float
    top_k: int
    latency_s: float
    doc_attention_scores: dict = field(default_factory=dict)  # positional-idx -> raw attention mass (Eq. 8)


@dataclass
class SOARResult:
    final_response: str
    trace: List[IterationTrace]
    converged_reason: str
    total_latency_s: float
    context_compression_ratio: float  # len(final context) / len(first-round context)


class SOARRAGPipeline:
    def __init__(self, knowledge_base, generator, *,
                 confidence_threshold: float = 0.75,
                 entropy_threshold: float = 1.0,
                 convergence_epsilon: float = 0.02,
                 max_iterations: int = 3,
                 top_k_default: int = 5):
        self.kb = knowledge_base
        self.generator = generator
        self.qp = QueryProcessor(knowledge_base=knowledge_base)
        self.attn_miner = AttentionFeedbackMiner()
        self.conf_estimator = ConfidenceEstimator()
        self.foe = FeedbackOptimizationEngine()
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.convergence_epsilon = convergence_epsilon
        self.max_iterations = max_iterations
        self.top_k_default = top_k_default

    def _generation_distance(self, prev_emb, cur_emb) -> float:
        """|G_t - G_{t-1}| (Eq. 13), computed as 1 - cosine similarity
        between sentence embeddings of successive generations (both are
        already L2-normalized by the KB encoder)."""
        if prev_emb is None:
            return 1.0
        cos = float(np.dot(prev_emb.flatten(), cur_emb.flatten()))
        return 1.0 - cos

    def run(self, query: str, verbose: bool = False) -> SOARResult:
        t_start = time.time()
        pq = self.qp.process(query)
        current_query = query
        prev_gen_emb = None
        trace: List[IterationTrace] = []
        first_round_ctx_len = None
        reason = "max_iterations_reached"

        for it in range(1, self.max_iterations + 1):
            iter_t0 = time.time()

            # --- 1. Retrieval ---
            q_emb = self.kb.embed_query(current_query)
            k = self.top_k_default if it == 1 else self.foe.dynamic_top_k(
                query_complexity=self.qp.complexity_score(pq),
                attention_feedback_score=min(trace[-1].attention_entropy / max(self.entropy_threshold, 1e-6), 1.0) if trace else 0.5,
            )
            retrieved = self.kb.search(q_emb, top_k=k)
            if first_round_ctx_len is None:
                first_round_ctx_len = sum(len(d.text) for d, _ in retrieved) or 1

            # --- 2. Generation ---
            docs_text = [d.text for d, _ in retrieved]
            gen = self.generator.generate(current_query, docs_text)

            # --- 3. Attention analysis ---
            feedback = self.attn_miner.mine(gen.attentions, gen.doc_token_spans)
            max_entropy = math.log(max(len(gen.input_token_ids), 2))
            sims = [sim for _, sim in retrieved]
            conf = self.conf_estimator.estimate(
                gen.token_logprobs, feedback.entropy, max_entropy, sims,
            )

            trace.append(IterationTrace(
                iteration=it, query_used=current_query,
                retrieved_doc_ids=[d.doc_id for d, _ in retrieved],
                response=gen.text, confidence=conf.overall,
                attention_entropy=feedback.entropy, top_k=k,
                latency_s=time.time() - iter_t0,
                doc_attention_scores={d.doc_id: feedback.doc_scores.get(i, 0.0) for i, (d, _) in enumerate(retrieved)},
            ))
            if verbose:
                print(f"[iter {it}] k={k} conf={conf.overall:.3f} entropy={feedback.entropy:.3f} -> {gen.text[:80]!r}")

            # --- convergence checks ---
            cur_gen_emb = self.kb.embed_query(gen.text) if gen.text else None
            dist = self._generation_distance(prev_gen_emb, cur_gen_emb) if cur_gen_emb is not None else 1.0
            prev_gen_emb = cur_gen_emb

            if conf.overall >= self.confidence_threshold:
                reason = "confidence_threshold_reached"
                break
            if feedback.entropy <= self.entropy_threshold * 0.3:
                reason = "attention_entropy_minimized"
                break
            if it > 1 and dist < self.convergence_epsilon:
                reason = "generation_converged"
                break

            # --- 4. Reranking + 5. Refinement (prep for next iteration) ---
            pruned = self.foe.context_pruning(retrieved, feedback.doc_scores, feedback.low_attention_docs)
            reranked = self.foe.hybrid_rerank(pruned, feedback.doc_scores, conf.overall)
            if self.foe.needs_query_expansion(conf.overall):
                top_terms = pq.entities[:3] if pq.entities else pq.tokens[:3]
                current_query = self.qp.expand(query, top_terms)

        final_ctx_len = sum(len(d.text) for d, _ in retrieved) or 1
        return SOARResult(
            final_response=trace[-1].response,
            trace=trace,
            converged_reason=reason,
            total_latency_s=time.time() - t_start,
            context_compression_ratio=final_ctx_len / first_round_ctx_len,
        )


class BaselineRAGPipeline:
    """Single-shot retrieve-then-generate, no attention feedback, no
    reranking, no iteration. Used as the paper's 'static retrieval'
    comparison point in evaluation."""

    def __init__(self, knowledge_base, generator, top_k: int = 5):
        self.kb = knowledge_base
        self.generator = generator
        self.top_k = top_k

    def run(self, query: str) -> SOARResult:
        t0 = time.time()
        q_emb = self.kb.embed_query(query)
        retrieved = self.kb.search(q_emb, top_k=self.top_k)
        gen = self.generator.generate(query, [d.text for d, _ in retrieved])
        trace = [IterationTrace(
            iteration=1, query_used=query,
            retrieved_doc_ids=[d.doc_id for d, _ in retrieved],
            response=gen.text, confidence=float("nan"), attention_entropy=float("nan"),
            top_k=self.top_k, latency_s=time.time() - t0,
        )]
        return SOARResult(
            final_response=gen.text, trace=trace, converged_reason="single_shot",
            total_latency_s=time.time() - t0, context_compression_ratio=1.0,
        )
