"""
Pretrains the from-scratch GPT (scratch_llm/model.py) on Wikipedia-style
text. This is the actual "build an LLM" step: random-init weights, real
gradient descent, a tokenizer trained on the same corpus, from nothing.

Usage (GPU strongly recommended -- pretraining from scratch is far more
compute-hungry than fine-tuning an existing model):
    python -m scratch_llm.train --n_articles 3000 --steps 3000 --out checkpoints/scratch-gpt

Realistic expectations: a model this size (a few million to a few tens of
millions of parameters, trained on a few thousand Wikipedia paragraphs for
a few thousand steps) will produce grammatical-ish fragments and pick up
local word statistics, NOT coherent factual answers -- GPT-2 itself needed
~1.5B parameters and 40GB of text to get good, and modern LLMs are 1000x
that. This step exists to make "I built an LLM" true, not to compete with
Qwen -- that's what train/train_lora.py (fine-tuning Qwen) is for.
"""
from __future__ import annotations
import argparse
import math
import time
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from scratch_llm.model import GPT, GPTConfig
from scratch_llm.tokenizer import ScratchTokenizer
from scratch_llm.data import load_wikitext_corpus, ChunkedLMDataset


def cosine_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.inference_mode()
def sample(model, tokenizer, device, prompt: str = "The", max_new_tokens: int = 40) -> str:
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    gen_ids, _, _ = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40,
                                    eos_token_id=tokenizer.eos_token_id)
    model.train()
    return tokenizer.decode(gen_ids[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_articles", type=int, default=3000, help="WikiText-103 paragraphs to pretrain on")
    ap.add_argument("--vocab_size", type=int, default=8000)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--n_embd", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--max_lr", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=3e-5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--sample_every", type=int, default=500)
    ap.add_argument("--out", default="checkpoints/scratch-gpt")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and device == "cpu":
        print("WARNING: no CUDA device found -- pretraining on CPU will be extremely slow. "
              "Use a Colab/Kaggle GPU runtime for a real run.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.n_articles} WikiText-103 paragraphs ...")
    texts = load_wikitext_corpus(n_articles=args.n_articles)
    print(f"Loaded {len(texts)} paragraphs, {sum(len(t) for t in texts):,} characters")

    print(f"Training a BPE tokenizer (vocab_size={args.vocab_size}) on this corpus ...")
    tokenizer = ScratchTokenizer()
    tokenizer.train(texts, vocab_size=args.vocab_size, save_dir=str(out_dir / "tokenizer"))

    dataset = ChunkedLMDataset(texts, tokenizer, block_size=args.block_size)
    print(f"{len(dataset)} training chunks of {args.block_size} tokens each")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size, block_size=args.block_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout,
    )
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=0.1, betas=(0.9, 0.95))

    print(f"\nPretraining for {args.steps} steps on {device} ...")
    step = 0
    losses = []
    t0 = time.time()
    model.train()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            lr = cosine_lr(step, args.steps, args.warmup_steps, args.max_lr, args.min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids, labels=labels)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())
            if step % args.log_every == 0:
                recent = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
                ppl = math.exp(min(recent, 20))
                elapsed = time.time() - t0
                print(f"step {step}/{args.steps}  loss={recent:.4f}  ppl={ppl:.1f}  lr={lr:.2e}  ({elapsed:.0f}s elapsed)")
            if step > 0 and step % args.sample_every == 0:
                text = sample(model, tokenizer, device, prompt="The history of")
                print(f"  [sample @ step {step}]: {text!r}")

            step += 1

    torch.save({"model_state_dict": model.state_dict(), "config": config.__dict__}, out_dir / "model.pt")
    final_loss = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
    with open(out_dir / "training_log.json", "w") as f:
        json.dump({
            "steps": args.steps, "final_loss": final_loss, "final_perplexity": math.exp(min(final_loss, 20)),
            "n_params": sum(p.numel() for p in model.parameters()),
            "n_training_chunks": len(dataset), "config": config.__dict__,
        }, f, indent=2)

    print(f"\nDone. Checkpoint + tokenizer saved to {out_dir}/")
    print(f"Final loss: {final_loss:.4f}  (perplexity: {math.exp(min(final_loss, 20)):.1f})")
    print(f"Try it: python -m scratch_llm.generate_text --checkpoint {out_dir} --prompt \"The capital of\"")
    print(f"Or plug it into SOAR-RAG: python run_demo.py --scratch_checkpoint {out_dir}")


if __name__ == "__main__":
    main()
