"""
Runs baseline RAG vs SOAR-RAG across the 4 corpora and produces a
comparison table + bar charts, matching paper III.M/III.N.

Usage (inside Colab/Kaggle, GPU runtime):
    python -m eval.run_eval --datasets nq hotpotqa triviaqa bioasq --n 100
"""
from __future__ import annotations
import argparse
import time
import json
import pandas as pd
from pathlib import Path

from src.knowledge_base import KnowledgeBase
from src.generation import LLMGenerator
from src.pipeline import SOARRAGPipeline, BaselineRAGPipeline
from eval.metrics import exact_match, f1_score, retrieval_accuracy, hallucination_flag
from data.loaders import DATASET_LOADERS


def build_kb_for_dataset(samples, embedding_model="sentence-transformers/all-MiniLM-L6-v2", device="cuda"):
    contexts = [s["context"] for s in samples if s.get("context")]
    if not contexts:
        # NQ/TriviaQA ship without gold context; use the answers+questions
        # pool itself as a lightweight open-domain corpus stand-in unless
        # you wire in a real Wikipedia dump index (recommended for a
        # resume-grade run -- see README "Scaling to real Wikipedia").
        contexts = [f"{s['question']} {' '.join(s['answers'])}" for s in samples]
    kb = KnowledgeBase(embedding_model=embedding_model, device=device)
    kb.build(contexts)
    return kb


def evaluate_dataset(name: str, samples, generator, device="cuda", corpus_mode="real_context"):
    kb = build_kb_for_dataset(samples, device=device)
    baseline = BaselineRAGPipeline(kb, generator, top_k=5)
    soar = SOARRAGPipeline(kb, generator)

    rows = []
    for i, s in enumerate(samples):
        q, gold = s["question"], s["answers"]
        print(f"  [{name}] {i+1}/{len(samples)}: {q[:70]}", flush=True)

        t0 = time.time()
        b_res = baseline.run(q)
        b_latency = time.time() - t0
        b_ctx = [kb.documents[i].text for i in b_res.trace[0].retrieved_doc_ids]

        t0 = time.time()
        s_res = soar.run(q)
        s_latency = time.time() - t0
        s_ctx = [kb.documents[i].text for i in s_res.trace[-1].retrieved_doc_ids]

        rows.append({
            "dataset": name, "corpus_mode": corpus_mode, "question": q,
            "baseline_em": exact_match(b_res.final_response, gold),
            "baseline_f1": f1_score(b_res.final_response, gold),
            "baseline_retrieval_acc": retrieval_accuracy(b_ctx, gold),
            "baseline_hallucination": hallucination_flag(b_res.final_response, b_ctx),
            "baseline_latency_s": b_latency,
            "soar_em": exact_match(s_res.final_response, gold),
            "soar_f1": f1_score(s_res.final_response, gold),
            "soar_retrieval_acc": retrieval_accuracy(s_ctx, gold),
            "soar_hallucination": hallucination_flag(s_res.final_response, s_ctx),
            "soar_latency_s": s_latency,
            "soar_context_compression": s_res.context_compression_ratio,
            "soar_iterations": len(s_res.trace),
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["dataset", "corpus_mode"]).agg(
        baseline_EM=("baseline_em", "mean"), soar_EM=("soar_em", "mean"),
        baseline_F1=("baseline_f1", "mean"), soar_F1=("soar_f1", "mean"),
        baseline_RetrievalAcc=("baseline_retrieval_acc", "mean"), soar_RetrievalAcc=("soar_retrieval_acc", "mean"),
        baseline_HallucinationRate=("baseline_hallucination", "mean"), soar_HallucinationRate=("soar_hallucination", "mean"),
        baseline_LatencyS=("baseline_latency_s", "mean"), soar_LatencyS=("soar_latency_s", "mean"),
        soar_ContextCompression=("soar_context_compression", "mean"),
        soar_AvgIterations=("soar_iterations", "mean"),
    ).round(3)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["nq", "hotpotqa", "triviaqa", "bioasq"])
    ap.add_argument("--n", type=int, default=20, help="samples per dataset (keep small for a laptop/Colab T4; raise once you've confirmed it runs)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cuda", help="'cuda' auto-falls-back to CPU if no GPU is found (see LLMGenerator)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quantize", default=None, choices=[None, "4bit", "8bit"],
                     help="GPU only, needs bitsandbytes. Cuts VRAM ~2-4x for bigger models on a T4/L4.")
    ap.add_argument("--bioasq-local-path", default=None, help="path to a real BioASQ task-B JSON export; omit to use the honestly-labeled PubMedQA fallback")
    args = ap.parse_args()

    Path(args.out).mkdir(exist_ok=True)
    generator = LLMGenerator(model_name=args.model, device=args.device, quantize=args.quantize)
    print(f"Model loaded on device: {generator.device}" + (f" ({args.quantize})" if args.quantize else ""))

    all_rows = []
    source_labels = {}
    for name in args.datasets:
        print(f"=== {name} ===")
        kwargs = {"n": args.n}
        if name == "bioasq" and args.bioasq_local_path:
            kwargs["local_path"] = args.bioasq_local_path
        samples, source_label, corpus_mode = DATASET_LOADERS[name](**kwargs)
        source_labels[name] = source_label
        if corpus_mode == "synthetic_qa_pool":
            print(f"  NOTE: {name} has no gold passages in this config; using a synthetic "
                  f"QA-pool corpus. Retrieval/EM/F1 here are NOT comparable to real "
                  f"open-domain retrieval -- see data/loaders.py docstring.")
        if source_label != name:
            print(f"  NOTE: '{name}' results are actually sourced from '{source_label}'.")
        df = evaluate_dataset(source_label, samples, generator, device=args.device, corpus_mode=corpus_mode)
        df.to_csv(f"{args.out}/{source_label}_raw.csv", index=False)
        all_rows.append(df)

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(f"{args.out}/all_raw.csv", index=False)
    summary = summarize(full)
    summary.to_csv(f"{args.out}/summary.csv")
    print(summary)

    with open(f"{args.out}/summary.json", "w") as f:
        json.dump({
            "source_labels": source_labels,
            "results": summary.reset_index().to_dict(orient="records"),
        }, f, indent=2)
    print(f"\nResults written to {args.out}/. Dataset source labels: {source_labels}")


if __name__ == "__main__":
    main()
