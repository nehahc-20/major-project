"""
Compares the BASE model vs the LoRA fine-tuned model on a held-out
HotpotQA split, using the exact same EM/F1 metrics as eval/run_eval.py.
This produces the actual evidence behind a "fine-tuning improved accuracy
by X points" claim -- run it, don't assert it.

Usage:
    python -m train.evaluate_finetune --adapter_path checkpoints/soar-lora --n 50
"""
from __future__ import annotations
import argparse
import json
import torch
import pandas as pd
from pathlib import Path

from train.data_prep import examples_from_hotpotqa, PROMPT_TEMPLATE
from eval.metrics import exact_match, f1_score


@torch.inference_mode()
def generate_answer(model, tokenizer, context: str, question: str, device: str, max_new_tokens: int = 64) -> str:
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    gen_ids = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def evaluate(model, tokenizer, examples, device: str, label: str) -> pd.DataFrame:
    rows = []
    for i, ex in enumerate(examples):
        pred = generate_answer(model, tokenizer, ex.context, ex.question, device)
        rows.append({
            "question": ex.question, "gold": ex.answer, "prediction": pred,
            "em": exact_match(pred, [ex.answer]), "f1": f1_score(pred, [ex.answer]),
        })
        if (i + 1) % 10 == 0:
            print(f"[{label}] {i+1}/{len(examples)} evaluated")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter_path", required=True, help="output dir from train_lora.py")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=50, help="held-out examples, disjoint split from training")
    ap.add_argument("--out", default="results_finetune")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    Path(args.out).mkdir(exist_ok=True)

    print("Loading held-out examples (validation split -- not seen during training) ...")
    examples = examples_from_hotpotqa(n=args.n, split="validation")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading base model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    base_model.eval()

    print("Evaluating BASE model ...")
    base_df = evaluate(base_model, tokenizer, examples, device, "base")
    base_df.to_csv(Path(args.out) / "base_predictions.csv", index=False)

    print(f"Loading LoRA adapter from {args.adapter_path} ...")
    tuned_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    tuned_model.eval()

    print("Evaluating FINE-TUNED model ...")
    tuned_df = evaluate(tuned_model, tokenizer, examples, device, "fine-tuned")
    tuned_df.to_csv(Path(args.out) / "finetuned_predictions.csv", index=False)

    summary = {
        "n_examples": len(examples),
        "base_EM": round(base_df["em"].mean(), 4), "base_F1": round(base_df["f1"].mean(), 4),
        "finetuned_EM": round(tuned_df["em"].mean(), 4), "finetuned_F1": round(tuned_df["f1"].mean(), 4),
        "EM_delta": round(tuned_df["em"].mean() - base_df["em"].mean(), 4),
        "F1_delta": round(tuned_df["f1"].mean() - base_df["f1"].mean(), 4),
    }
    with open(Path(args.out) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Base vs Fine-tuned ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nFull predictions saved to {args.out}/*.csv")


if __name__ == "__main__":
    main()
