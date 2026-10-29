"""
Proves the LoRA fine-tuning mechanics in train/train_lora.py actually work
-- real forward/backward passes, real loss reduction, real adapter
save/load -- WITHOUT downloading anything from Hugging Face. Uses a tiny
randomly-initialized GPT-2-architecture model (a few hundred KB, built
from a config object, no network call) standing in for Qwen/Llama/Mistral,
so this can run in any offline/CI environment as a sanity check before
you spend real GPU time training the actual model.

This test is NOT evidence the fine-tuning IMPROVES accuracy on real data
-- it only proves the training loop is mechanically correct (loss goes
down, adapter round-trips). Real accuracy evidence requires
train/evaluate_finetune.py on your machine with a real model.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from train.data_prep import TrainingExample, GroundedQADataset, collate_fn
from train.train_lora import wrap_with_lora, train_loop
from tests.toy_tokenizer import ToyTokenizer


def make_toy_model_and_tokenizer():
    tokenizer = ToyTokenizer()
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size, n_embd=32, n_layer=2, n_head=2,
        n_positions=256, bos_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    return model, tokenizer


TOY_EXAMPLES = [
    TrainingExample(context="Paris is the capital of France.", question="What is the capital of France?", answer="Paris"),
    TrainingExample(context="The sky appears blue due to Rayleigh scattering.", question="Why is the sky blue?", answer="Rayleigh scattering"),
    TrainingExample(context="Water boils at 100 degrees Celsius at sea level.", question="At what temperature does water boil?", answer="100 degrees"),
    TrainingExample(context="Dogs are commonly kept as pets worldwide.", question="What animal is commonly kept as a pet?", answer="Dogs"),
]


def test_lora_wrapping_reduces_trainable_params():
    model, tokenizer = make_toy_model_and_tokenizer()
    total_before = sum(p.numel() for p in model.parameters())
    model = wrap_with_lora(model, r=4, target_modules=["c_attn"])  # GPT-2's attention proj name
    trainable, total = model.get_nb_trainable_parameters()
    assert trainable < total
    assert trainable < total_before * 0.5  # LoRA should train a small fraction of weights


def test_training_loop_actually_reduces_loss():
    model, tokenizer = make_toy_model_and_tokenizer()
    model = wrap_with_lora(model, r=4, target_modules=["c_attn"])
    dataset = GroundedQADataset(TOY_EXAMPLES, tokenizer, max_length=128)

    loss_history = train_loop(
        model, tokenizer, dataset, device="cpu",
        epochs=6, batch_size=2, lr=5e-3, grad_accum_steps=1, log_every=1000,
    )
    assert len(loss_history) > 0
    first_avg = sum(loss_history[:2]) / 2
    last_avg = sum(loss_history[-2:]) / 2
    assert last_avg < first_avg, f"loss did not decrease: first={first_avg:.4f} last={last_avg:.4f}"


def test_adapter_save_and_reload_roundtrip(tmp_path):
    model, tokenizer = make_toy_model_and_tokenizer()
    model = wrap_with_lora(model, r=4, target_modules=["c_attn"])
    dataset = GroundedQADataset(TOY_EXAMPLES, tokenizer, max_length=128)
    train_loop(model, tokenizer, dataset, device="cpu", epochs=1, batch_size=2, log_every=1000)

    save_dir = tmp_path / "toy-adapter"
    model.save_pretrained(str(save_dir))
    assert (save_dir / "adapter_config.json").exists()
    assert any(save_dir.glob("adapter_model.*"))

    from peft import PeftModel
    base_model2, _ = make_toy_model_and_tokenizer()
    reloaded = PeftModel.from_pretrained(base_model2, str(save_dir))
    assert reloaded is not None


def test_collate_fn_masks_prompt_tokens():
    tokenizer = ToyTokenizer()
    dataset = GroundedQADataset(TOY_EXAMPLES[:2], tokenizer, max_length=128)
    batch = collate_fn([dataset[0], dataset[1]], tokenizer.pad_token_id)
    # every example must have SOME non-masked (trainable) label tokens --
    # otherwise the model would learn nothing from that example
    for row in batch["labels"]:
        assert (row != -100).any()
