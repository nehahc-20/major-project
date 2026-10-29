"""
Builds instruction-tuning examples for fine-tuning the SOAR-RAG generator:
(context + question) -> grounded answer, in the exact prompt format
src/generation.py already uses at inference time, so a fine-tuned model
drops straight into the existing pipeline with zero prompt-format drift.

Each example is tokenized with the prompt tokens masked out of the loss
(label = -100) so the model is only trained to predict the answer, not to
re-predict the context it was given -- standard instruction-tuning setup.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import torch
from torch.utils.data import Dataset


PROMPT_TEMPLATE = (
    "Answer the question using ONLY the context below. Be concise and factual.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)


@dataclass
class TrainingExample:
    context: str
    question: str
    answer: str

    def prompt(self) -> str:
        return PROMPT_TEMPLATE.format(context=self.context, question=self.question)


def examples_from_hotpotqa(n: int = 500, split: str = "train") -> List[TrainingExample]:
    """Real training data: HotpotQA ships gold supporting-fact passages, so
    unlike NQ/TriviaQA this gives genuine (context, question, answer)
    triples rather than a synthetic stand-in corpus."""
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split=f"{split}[:{n}]", trust_remote_code=True)
    out = []
    for row in ds:
        ctx = " ".join(" ".join(sents) for sents in row["context"]["sentences"])
        ctx = ctx[:1500]  # keep prompts short enough to train quickly on modest GPUs
        out.append(TrainingExample(context=ctx, question=row["question"], answer=row["answer"]))
    return out


class GroundedQADataset(Dataset):
    """Tokenizes (prompt, answer) pairs with prompt tokens masked out of
    the loss. Works with any causal-LM tokenizer (Qwen/Llama/Mistral/etc.)."""

    def __init__(self, examples: List[TrainingExample], tokenizer, max_length: int = 768):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        prompt = ex.prompt()
        full_text = prompt + " " + ex.answer + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

        # Right-truncation (the default) cuts from the END -- which is
        # exactly where the answer lives. That silently drops the answer,
        # masks every label as -100, and produces a NaN loss for that
        # example (caught by tests/test_lora_training.py). Truncate from
        # the LEFT instead (drop the earliest part of the context) so the
        # question+answer at the tail is always preserved.
        if len(full_ids) > self.max_length:
            excess = len(full_ids) - self.max_length
            full_ids = full_ids[excess:]
            prompt_len_for_mask = max(len(prompt_ids) - excess, 0)
        else:
            prompt_len_for_mask = len(prompt_ids)

        labels = list(full_ids)
        prompt_len = min(prompt_len_for_mask, len(labels))
        # Safety net for a pathological max_length smaller than the answer
        # itself: always leave at least the final token trainable so a
        # batch never silently contributes an all-masked (NaN-loss) row.
        prompt_len = min(prompt_len, max(len(labels) - 1, 0))
        for i in range(prompt_len):
            labels[i] = -100  # don't train the model to reproduce the context/question

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids, labels, attention_mask = [], [], []
    for item in batch:
        pad_len = max_len - item["input_ids"].shape[0]
        input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        labels.append(torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        attention_mask.append(torch.cat([torch.ones_like(item["input_ids"]), torch.zeros(pad_len, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }
