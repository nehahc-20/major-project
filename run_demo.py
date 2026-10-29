"""
One-command entry point: run a live demo query, then a small real eval
on whichever datasets you ask for, with CPU-safe defaults out of the box.

    python run_demo.py                       # demo query only, CPU, tiny model
    python run_demo.py --eval                # + a real 5-sample eval on hotpotqa
    python run_demo.py --eval --datasets hotpotqa nq --n 10
    python run_demo.py --eval --device cuda --model Qwen/Qwen2.5-1.5B-Instruct --n 50

No numbers are printed here that weren't just computed in this process --
this script does not read or fabricate results, it only runs the real
pipeline and reports what comes out.
"""
from __future__ import annotations
import argparse
import sys


DEMO_CORPUS = [
    "The Eiffel Tower was completed in 1889 in Paris, France, as the entrance "
    "arch for the 1889 World's Fair. It was designed by the engineer "
    "Gustave Eiffel's company.",
    "The Great Wall of China is a series of fortifications built across the "
    "historical northern borders of China, with major construction phases "
    "during the Ming dynasty.",
    "Marie Curie was a physicist and chemist who conducted pioneering "
    "research on radioactivity. She was the first woman to win a Nobel Prize.",
    "Photosynthesis is the process by which green plants convert light "
    "energy, water, and carbon dioxide into glucose and oxygen.",
]


def run_demo_query(model_name: str, device: str, quantize: str = None, adapter_path: str = None,
                    scratch_checkpoint: str = None):
    from src.knowledge_base import KnowledgeBase
    from src.pipeline import SOARRAGPipeline, BaselineRAGPipeline

    if scratch_checkpoint:
        from scratch_llm.soar_adapter import ScratchLLMGenerator
        print(f"Loading from-scratch model checkpoint: {scratch_checkpoint} ...")
        generator = ScratchLLMGenerator(scratch_checkpoint, device=device)
        print(f"Scratch model on device: {generator.device}. Expect noticeably weaker answers "
              f"than Qwen -- this model's weights are entirely your own training run, at a "
              f"fraction of the scale/data of a pretrained LLM.")
    else:
        from src.generation import LLMGenerator
        print(f"Loading model {model_name} ...")
        generator = LLMGenerator(model_name=model_name, device=device, quantize=quantize, adapter_path=adapter_path)
        print(f"Model on device: {generator.device}" + (f" ({quantize})" if quantize else "")
              + (f" + LoRA adapter: {adapter_path}" if adapter_path else ""))

    kb = KnowledgeBase()
    kb.build(DEMO_CORPUS)

    query = "When and where was the Eiffel Tower completed, and who designed it?"
    print(f"\nQuery: {query}\n")

    baseline = BaselineRAGPipeline(kb, generator, top_k=3)
    b_res = baseline.run(query)
    print("--- Baseline RAG (single pass) ---")
    print(b_res.final_response)

    soar = SOARRAGPipeline(kb, generator)
    s_res = soar.run(query)
    print(f"\n--- SOAR-RAG ({len(s_res.trace)} feedback iterations, "
          f"converged: {s_res.trace[-1].confidence:.2f} confidence) ---")
    print(s_res.final_response)

    return generator


def run_small_eval(generator, datasets, n, out_dir, bioasq_local_path):
    from eval.run_eval import evaluate_dataset, summarize
    from data.loaders import DATASET_LOADERS
    from pathlib import Path
    import pandas as pd
    import json

    Path(out_dir).mkdir(exist_ok=True)
    all_rows = []
    source_labels = {}
    for name in datasets:
        print(f"\n=== eval: {name} (n={n}) ===")
        kwargs = {"n": n}
        if name == "bioasq" and bioasq_local_path:
            kwargs["local_path"] = bioasq_local_path
        samples, source_label, corpus_mode = DATASET_LOADERS[name](**kwargs)
        source_labels[name] = source_label
        if corpus_mode == "synthetic_qa_pool":
            print(f"  NOTE: {name} has no gold passages here; using a synthetic "
                  f"QA-pool corpus. Don't cite these numbers as real open-domain "
                  f"retrieval -- see data/loaders.py.")
        if source_label != name:
            print(f"  NOTE: '{name}' results are actually sourced from '{source_label}'.")
        df = evaluate_dataset(source_label, samples, generator, device=generator.device, corpus_mode=corpus_mode)
        df.to_csv(f"{out_dir}/{source_label}_raw.csv", index=False)
        all_rows.append(df)

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(f"{out_dir}/all_raw.csv", index=False)
    summary = summarize(full)
    summary.to_csv(f"{out_dir}/summary.csv")
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump({"source_labels": source_labels,
                    "results": summary.reset_index().to_dict(orient="records")}, f, indent=2)

    print("\n=== Summary ===")
    print(summary)
    print(f"\nWritten to {out_dir}/. Source labels: {source_labels}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                     help="small by default so this runs on CPU in reasonable time; "
                          "pass a bigger model once you've confirmed it works")
    ap.add_argument("--device", default="cpu", help="'cuda' if you have a GPU")
    ap.add_argument("--quantize", default=None, choices=[None, "4bit", "8bit"],
                     help="GPU only, needs bitsandbytes. Cuts VRAM ~2-4x, lets bigger models fit on a free-tier GPU.")
    ap.add_argument("--adapter_path", default=None,
                     help="path to a LoRA adapter from train/train_lora.py, to run the fine-tuned generator instead of the base model")
    ap.add_argument("--scratch_checkpoint", default=None,
                     help="path to a from-scratch model from scratch_llm/train.py -- overrides --model/--quantize/--adapter_path, uses your own pretrained model instead of Qwen")
    ap.add_argument("--eval", action="store_true", help="also run a small real eval after the demo query")
    ap.add_argument("--datasets", nargs="+", default=["hotpotqa"],
                     help="hotpotqa has real gold passages and is the most meaningful "
                          "one to run without extra setup -- see data/loaders.py")
    ap.add_argument("--n", type=int, default=5, help="samples per dataset; keep small on CPU")
    ap.add_argument("--out", default="results")
    ap.add_argument("--bioasq-local-path", default=None)
    args = ap.parse_args()

    generator = run_demo_query(args.model, args.device, args.quantize, args.adapter_path, args.scratch_checkpoint)

    if args.eval:
        run_small_eval(generator, args.datasets, args.n, args.out, args.bioasq_local_path)
    else:
        print("\n(Run with --eval to also compute real EM/F1/hallucination-rate numbers "
              "on a small sample -- e.g. `python run_demo.py --eval --n 10`.)")


if __name__ == "__main__":
    main()
