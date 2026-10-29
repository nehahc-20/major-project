"""
Evaluation metrics matching paper III.M: Exact Match (EM), F1, retrieval
accuracy, hallucination rate, context compression ratio, response latency.
"""
from __future__ import annotations
import re
import string
from collections import Counter
from typing import List


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold_answers: List[str]) -> int:
    pred = normalize_answer(prediction)
    return int(any(pred == normalize_answer(g) for g in gold_answers))


def f1_score(prediction: str, gold_answers: List[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    best = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def retrieval_accuracy(retrieved_texts: List[str], gold_answers: List[str]) -> int:
    """Answer-in-context hit@k: 1 if any gold answer string appears
    (normalized) inside the retrieved context, else 0. Standard proxy
    metric when gold passage IDs aren't available (NQ-open, TriviaQA)."""
    joined = normalize_answer(" ".join(retrieved_texts))
    return int(any(normalize_answer(g) in joined for g in gold_answers if g))


def hallucination_flag(prediction: str, retrieved_texts: List[str]) -> int:
    """Heuristic faithfulness check: flags a response as a likely
    hallucination if none of its non-trivial content words appear anywhere
    in the retrieved context. Cheap proxy for the paper's 'rate of
    hallucination' metric -- swap in an NLI/entailment model for a more
    rigorous version if you have GPU budget to spare."""
    context = normalize_answer(" ".join(retrieved_texts))
    pred_tokens = [t for t in normalize_answer(prediction).split() if len(t) > 3]
    if not pred_tokens:
        return 0
    grounded = sum(1 for t in pred_tokens if t in context)
    coverage = grounded / len(pred_tokens)
    return int(coverage < 0.3)


def context_compression_ratio(final_ctx_len: int, first_round_ctx_len: int) -> float:
    return final_ctx_len / max(first_round_ctx_len, 1)
