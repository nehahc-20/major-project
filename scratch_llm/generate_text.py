"""
Loads a scratch_llm checkpoint and generates text -- a quick sanity check
independent of the RAG pipeline, so you can see what your pretrained
model actually produces before wiring it into anything.

Usage:
    python -m scratch_llm.generate_text --checkpoint checkpoints/scratch-gpt --prompt "The history of"
"""
from __future__ import annotations
import argparse
import torch
from pathlib import Path

from scratch_llm.model import GPT, GPTConfig
from scratch_llm.tokenizer import ScratchTokenizer


def load_checkpoint(checkpoint_dir: str, device: str):
    ckpt = torch.load(Path(checkpoint_dir) / "model.pt", map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    tokenizer = ScratchTokenizer(str(Path(checkpoint_dir) / "tokenizer"))
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", default="The history of")
    ap.add_argument("--max_new_tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_checkpoint(args.checkpoint, device)

    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long).to(device)
    gen_ids, logprobs, _ = model.generate(
        prompt_ids, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k, eos_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(gen_ids[0].tolist()))


if __name__ == "__main__":
    main()
