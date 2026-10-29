"""
Produces an ILLUSTRATIVE results/ folder that demonstrates the SOAR-RAG
MECHANISM (why attention-guided reranking + dynamic top-K can beat a
fixed-small-K baseline) using a fully scripted, deterministic mock model.

This is NOT a real LLM and NOT a real accuracy benchmark. It is a
constructed scenario, built by hand, to preview the exact output format
(columns, summary table, plot) that `python run_demo.py --eval` /
`python -m eval.run_eval` produce with a REAL model on a REAL dataset --
and to make concrete what "attention feedback reduces missed-context
errors" actually looks like in the metrics.

For real numbers, run on your machine / Colab (needs a Hugging Face
model download -- not runnable in a network-restricted sandbox):
    python run_demo.py --eval --n 10
    python -m eval.run_eval --datasets hotpotqa --n 50 --device cuda
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List
import torch

from src.knowledge_base import Document
from src.generation import GenerationResult
from src.pipeline import SOARRAGPipeline, BaselineRAGPipeline
from eval.metrics import exact_match, f1_score, retrieval_accuracy, hallucination_flag


# --- Scripted scenario -------------------------------------------------
# Each question has a corpus where the GOLD document is deliberately
# ranked 4th by "similarity" -- realistic when a query's wording doesn't
# closely match the gold passage's wording, which is exactly when static
# top-3 retrieval misses relevant context in practice.
SCENARIOS = [
    {
        "dataset": "demo_geography",
        "question": "When was the Eiffel Tower's construction finished?",
        "gold_answers": ["1889"],
        # ranked by descending mock similarity; gold doc is 4th
        "ranked_docs": [
            "The Louvre Museum in Paris houses the Mona Lisa and thousands of other works of art.",
            "Notre-Dame Cathedral is a medieval Catholic cathedral on the Île de la Cité in Paris.",
            "The Seine river flows through Paris and empties into the English Channel.",
            "The Eiffel Tower's construction was completed in 1889, in time for the World's Fair.",
            "Paris is divided into 20 administrative arrondissements.",
        ],
        "wrong_answer": "The available context does not specify an exact completion date.",
    },
    {
        "dataset": "demo_geography",
        "question": "Which mountain range contains the world's highest peak?",
        "gold_answers": ["Himalayas", "the Himalayas"],
        "ranked_docs": [
            "The Alps are a major mountain range in Europe, stretching across eight countries.",
            "The Andes form the longest continental mountain range in the world, running through South America.",
            "The Rocky Mountains extend through western North America.",
            "Mount Everest, Earth's highest peak, is located in the Himalayas on the border of Nepal and Tibet.",
            "The Atlas Mountains stretch across Morocco, Algeria, and Tunisia.",
        ],
        "wrong_answer": "Based on the retrieved context, this cannot be determined with confidence.",
    },
    {
        "dataset": "demo_science",
        "question": "What two Nobel Prize categories did Marie Curie win in?",
        "gold_answers": ["Physics and Chemistry", "physics and chemistry"],
        "ranked_docs": [
            "Albert Einstein developed the theory of relativity, revolutionizing physics.",
            "The Nobel Prize was established by Alfred Nobel's will in 1895.",
            "Rosalind Franklin's X-ray diffraction work contributed to understanding DNA's structure.",
            "Marie Curie won the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911.",
            "Nobel Prizes are awarded annually in Stockholm and Oslo.",
        ],
        "wrong_answer": "The retrieved context does not clearly state which categories.",
    },
    {
        "dataset": "demo_science",
        "question": "What are the two main outputs of photosynthesis?",
        "gold_answers": ["glucose and oxygen", "glucose", "oxygen"],
        "ranked_docs": [
            "Cellular respiration breaks down glucose to release energy in the form of ATP.",
            "Chlorophyll is the pigment that gives plants their green color and absorbs light.",
            "Plant roots absorb water and minerals from the soil.",
            "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
            "Stomata are small pores on leaves that regulate gas exchange.",
        ],
        "wrong_answer": "The context provided doesn't clearly specify the outputs.",
    },
]


class ScriptedKnowledgeBase:
    """Returns a hand-fixed ranking for a single scenario's docs -- lets us
    control exactly which documents a given top_k includes, instead of
    depending on a real (or hash-based) embedding's behavior."""

    def __init__(self, ranked_texts: List[str]):
        self.documents = [Document(doc_id=i, text=t) for i, t in enumerate(ranked_texts)]
        # descending fake similarity scores
        self._sims = [1.0 - 0.15 * i for i in range(len(ranked_texts))]

    def embed_query(self, query: str):
        return None  # unused; search() ignores it

    def search(self, _query_embedding, top_k: int = 5):
        k = min(top_k, len(self.documents))
        return [(self.documents[i], self._sims[i]) for i in range(k)]


class ScriptedGenerator:
    """Answers correctly if the gold document is among the docs it was
    given, otherwise gives a hedged wrong answer -- modeling what actually
    happens when relevant context is missing (a hedge / near-miss), not a
    fabricated confident hallucination."""

    def __init__(self, gold_doc_text: str, correct_answer: str, wrong_answer: str):
        self.gold_doc_text = gold_doc_text
        self.correct_answer = correct_answer
        self.wrong_answer = wrong_answer

    def generate(self, query: str, documents: List[str]) -> GenerationResult:
        has_gold = self.gold_doc_text in documents
        text = self.correct_answer if has_gold else self.wrong_answer

        spans, cursor = [], 0
        for d in documents:
            span_len = max(len(d.split()), 1)
            spans.append((cursor, cursor + span_len))
            cursor += span_len
        prompt_len = max(cursor, 4)

        # attention concentrated on the gold doc if present, else spread
        # thin across everything (models real "unsure where to look" behavior)
        n_layers, n_heads = 2, 2
        attn = torch.zeros(1, n_heads, 1, prompt_len)
        if has_gold:
            gold_idx = documents.index(self.gold_doc_text)
            s, e = spans[gold_idx]
            attn[0, :, 0, s:e] = 1.0 / max(e - s, 1)
        else:
            attn[0, :, 0, :] = 1.0 / prompt_len
        attentions = tuple(attn.clone() for _ in range(n_layers))

        logprobs = [-0.05, -0.08, -0.1] if has_gold else [-0.9, -1.1, -0.7]
        gen_ids = torch.randint(0, 100, (5,))
        return GenerationResult(
            text=text, generated_token_ids=gen_ids,
            input_token_ids=torch.arange(prompt_len), attentions=attentions,
            token_logprobs=logprobs, doc_token_spans=spans,
        )


def run_scenario(scn: dict) -> dict:
    kb = ScriptedKnowledgeBase(scn["ranked_docs"])
    gold_doc = scn["ranked_docs"][3]  # gold is 4th-ranked, by construction
    gen = ScriptedGenerator(gold_doc, scn["gold_answers"][0], scn["wrong_answer"])

    baseline = BaselineRAGPipeline(kb, gen, top_k=3)   # misses the gold doc (rank 4)
    soar = SOARRAGPipeline(kb, gen, max_iterations=2, confidence_threshold=0.3, top_k_default=5)  # catches it

    b_res = baseline.run(scn["question"])
    b_ctx = [kb.documents[i].text for i in b_res.trace[0].retrieved_doc_ids]

    s_res = soar.run(scn["question"])
    s_ctx = [kb.documents[i].text for i in s_res.trace[-1].retrieved_doc_ids]

    gold = scn["gold_answers"]
    return {
        "dataset": scn["dataset"], "corpus_mode": "SCRIPTED_MECHANISM_DEMO",
        "question": scn["question"],
        "baseline_em": exact_match(b_res.final_response, gold),
        "baseline_f1": f1_score(b_res.final_response, gold),
        "baseline_retrieval_acc": retrieval_accuracy(b_ctx, gold),
        "baseline_hallucination": hallucination_flag(b_res.final_response, b_ctx),
        "baseline_latency_s": b_res.total_latency_s,
        "soar_em": exact_match(s_res.final_response, gold),
        "soar_f1": f1_score(s_res.final_response, gold),
        "soar_retrieval_acc": retrieval_accuracy(s_ctx, gold),
        "soar_hallucination": hallucination_flag(s_res.final_response, s_ctx),
        "soar_latency_s": s_res.total_latency_s,
        "soar_context_compression": s_res.context_compression_ratio,
        "soar_iterations": len(s_res.trace),
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["dataset", "corpus_mode"]).agg(
        baseline_EM=("baseline_em", "mean"), soar_EM=("soar_em", "mean"),
        baseline_F1=("baseline_f1", "mean"), soar_F1=("soar_f1", "mean"),
        baseline_RetrievalAcc=("baseline_retrieval_acc", "mean"), soar_RetrievalAcc=("soar_retrieval_acc", "mean"),
        baseline_HallucinationRate=("baseline_hallucination", "mean"), soar_HallucinationRate=("soar_hallucination", "mean"),
        baseline_LatencyS=("baseline_latency_s", "mean"), soar_LatencyS=("soar_latency_s", "mean"),
        soar_ContextCompression=("soar_context_compression", "mean"),
        soar_AvgIterations=("soar_iterations", "mean"),
    ).round(3)


def main():
    out_dir = Path(__file__).parent.parent / "results_demo"
    out_dir.mkdir(exist_ok=True)

    rows = [run_scenario(s) for s in SCENARIOS]
    full = pd.DataFrame(rows)
    full.to_csv(out_dir / "all_raw.csv", index=False)
    summary = summarize(full)
    summary.to_csv(out_dir / "summary.csv")
    print(summary)

    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "DISCLAIMER": "Generated with a hand-scripted deterministic model "
                           "(demo/mock_eval_demo.py), NOT a real LLM. This is a "
                           "constructed scenario illustrating the retrieval-depth "
                           "mechanism, not a real accuracy benchmark. Run "
                           "run_demo.py --eval or eval/run_eval.py with a real "
                           "model for genuine results.",
            "results": summary.reset_index().to_dict(orient="records"),
        }, f, indent=2)

    try:
        import matplotlib.pyplot as plt
        metrics = [("EM", "baseline_EM", "soar_EM"), ("F1", "baseline_F1", "soar_F1"),
                   ("RetrievalAcc", "baseline_RetrievalAcc", "soar_RetrievalAcc"),
                   ("HallucinationRate", "baseline_HallucinationRate", "soar_HallucinationRate")]
        fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))
        for ax, (title, bcol, scol) in zip(axes, metrics):
            summary[[bcol, scol]].plot(kind="bar", ax=ax, legend=(title == "EM"), color=["#6b7280", "#f2a623"])
            ax.set_title(title)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("")
        fig.suptitle("SCRIPTED MECHANISM DEMO -- illustrates why deeper/reranked retrieval helps; not a real benchmark", fontsize=10, color="crimson")
        plt.tight_layout()
        plt.savefig(out_dir / "comparison_demo.png", dpi=150)
        print(f"\nPlot saved to {out_dir / 'comparison_demo.png'}")
    except ImportError:
        pass

    print(f"\nWritten to {out_dir}/. This is a SCRIPTED MECHANISM DEMO, not a real benchmark.")
    print("For real numbers: python run_demo.py --eval --n 10   (on your machine, needs internet + model download)")


if __name__ == "__main__":
    main()
