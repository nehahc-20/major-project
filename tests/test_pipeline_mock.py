"""
Offline tests (no model downloads) verifying the SOAR-RAG control flow
and math -- attention feedback mining, hybrid reranking, confidence
estimation, and the iterative pipeline -- run correctly end to end.
Run with: pytest tests/ -v
"""
import numpy as np
from src.query_processing import QueryProcessor
from src.attention_feedback import AttentionFeedbackMiner
from src.confidence import ConfidenceEstimator
from src.reranking import FeedbackOptimizationEngine
from src.pipeline import SOARRAGPipeline, BaselineRAGPipeline
from tests.mocks import MockKnowledgeBase, MockGenerator


CORPUS = [
    "The capital of France is Paris, a city on the Seine river.",
    "Bananas are a good source of potassium and grow in tropical climates.",
    "The Eiffel Tower was completed in 1889 for the World's Fair in Paris.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
]


def test_query_processor_basic():
    qp = QueryProcessor(knowledge_base=None, use_spacy=False)
    pq = qp.process("What is the capital of France and its population?")
    assert "capital" in pq.tokens
    assert "the" not in pq.tokens  # stopword removed
    score = qp.complexity_score(pq)
    assert 0.0 <= score <= 1.0


def test_attention_feedback_scores_documents():
    kb = MockKnowledgeBase(CORPUS)
    gen = MockGenerator()
    result = gen.generate("capital of France", CORPUS)
    miner = AttentionFeedbackMiner()
    feedback = miner.mine(result.attentions, result.doc_token_spans)
    assert len(feedback.doc_scores) == len(CORPUS)
    # doc 0 was given all the attention mass by MockGenerator
    best_doc = max(feedback.doc_scores, key=feedback.doc_scores.get)
    assert best_doc == 0
    assert feedback.entropy >= 0.0


def test_confidence_estimator_bounds():
    est = ConfidenceEstimator()
    report = est.estimate(
        token_logprobs=[-0.1, -0.05, -0.2],
        attn_entropy=0.3, max_possible_entropy=2.0,
        retrieval_sims=[0.9, 0.85, 0.7],
    )
    assert 0.0 <= report.overall <= 1.0
    assert report.token_prob_conf > 0.5  # low negative logprobs -> high prob


def test_hybrid_reranking_orders_by_score():
    kb = MockKnowledgeBase(CORPUS)
    retrieved = kb.search(kb.embed_query("capital of France"), top_k=4)
    doc_scores = {0: 0.7, 1: 0.05, 2: 0.2, 3: 0.05}
    foe = FeedbackOptimizationEngine()
    reranked = foe.hybrid_rerank(retrieved, doc_scores, confidence=0.8)
    assert reranked[0].hybrid_score >= reranked[-1].hybrid_score
    k = foe.dynamic_top_k(query_complexity=0.9, attention_feedback_score=0.8)
    assert foe.min_k <= k <= foe.max_k


def test_context_pruning_never_empties():
    foe = FeedbackOptimizationEngine()
    kb = MockKnowledgeBase(CORPUS)
    retrieved = kb.search(kb.embed_query("bananas"), top_k=4)
    pruned = foe.context_pruning(retrieved, {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1}, low_attention_idx=[0, 1, 2, 3])
    assert len(pruned) >= 1


def test_soar_pipeline_runs_and_converges():
    kb = MockKnowledgeBase(CORPUS)
    gen = MockGenerator(answer="Paris is the capital of France.")
    pipeline = SOARRAGPipeline(kb, gen, max_iterations=3, confidence_threshold=0.3)
    result = pipeline.run("What is the capital of France?")
    assert result.final_response
    assert len(result.trace) >= 1
    assert result.converged_reason in {
        "confidence_threshold_reached", "attention_entropy_minimized",
        "generation_converged", "max_iterations_reached",
    }
    assert result.context_compression_ratio > 0


def test_baseline_pipeline_single_shot():
    kb = MockKnowledgeBase(CORPUS)
    gen = MockGenerator(answer="Paris.")
    baseline = BaselineRAGPipeline(kb, gen, top_k=3)
    result = baseline.run("capital of France")
    assert len(result.trace) == 1
    assert result.final_response == "Paris."
