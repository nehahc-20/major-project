"""
Fine-tunes an open-weight causal LM with LoRA (or QLoRA when --quantize is
set) on grounded-QA examples, so the SOAR-RAG generator gets genuinely
better at answering from the retrieved context -- this is real gradient
descent on real weights, not a wrapper around a bigger model.

Usage (Colab/Kaggle GPU recommended; runs on CPU too, just slowly):
    python -m train.train_lora --n 300 --epochs 2 --out checkpoints/soar-lora
    python -m train.train_lora --n 300 --quantize 4bit --model Qwen/Qwen2.5-3B-Instruct

Output: a LoRA adapter directory (a few MB, NOT a full model copy) that
src/generation.py can load on top of the base model via --adapter_path.
"""
from __future__ import annotations
import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from train.data_prep import examples_from_hotpotqa, GroundedQADataset, collate_fn


def build_model_and_tokenizer(model_name: str, device: str, quantize: str = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if quantize in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig
        if quantize == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
            )
        else:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" and quant_config is None else torch.float32,
        quantization_config=quant_config,
    )
    if quant_config is not None:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    else:
        model = model.to(device)
    return model, tokenizer


def wrap_with_lora(model, r: int = 8, alpha: int = 16, dropout: float = 0.05,
                    target_modules=None):
    from peft import LoraConfig, get_peft_model, TaskType
    # sensible default target modules for the Llama/Mistral/Qwen family
    # (all use these attention projection names); override for other archs
    target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, bias="none",
    )
    model = get_peft_model(model, config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.3f}%) "
          f"-- this is the actual point of LoRA: you're training <1% of the weights.")
    return model


def train_loop(model, tokenizer, dataset, device, epochs=2, batch_size=4, lr=2e-4,
                grad_accum_steps=4, log_every=10):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()

    step = 0
    loss_history = []
    for epoch in range(epochs):
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            if (i + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            loss_history.append(outputs.loss.item())
            step += 1
            if step % log_every == 0:
                recent = sum(loss_history[-log_every:]) / len(loss_history[-log_every:])
                print(f"epoch {epoch+1}/{epochs} step {step}: loss={recent:.4f}")
    return loss_history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quantize", default=None, choices=[None, "4bit", "8bit"])
    ap.add_argument("--n", type=int, default=300, help="number of HotpotQA training examples")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--out", default="checkpoints/soar-lora")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and device == "cpu":
        print("WARNING: no CUDA device found, falling back to CPU. This will be very slow "
              "for anything beyond a tiny smoke test -- use a GPU runtime for a real run.")

    print(f"Loading base model {args.model} on {device} (quantize={args.quantize}) ...")
    model, tokenizer = build_model_and_tokenizer(args.model, device, args.quantize)
    model = wrap_with_lora(model, r=args.lora_r)

    print(f"Loading {args.n} HotpotQA training examples ...")
    examples = examples_from_hotpotqa(n=args.n)
    dataset = GroundedQADataset(examples, tokenizer)

    print("Training ...")
    loss_history = train_loop(
        model, tokenizer, dataset, device,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"\nSaved LoRA adapter to {args.out}/ "
          f"(first-step loss {loss_history[0]:.3f} -> last-step loss {loss_history[-1]:.3f})")
    print("Next: python -m train.evaluate_finetune --adapter_path", args.out)


if __name__ == "__main__":
    main()
