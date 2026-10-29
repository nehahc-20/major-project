"""
A minimal from-scratch tokenizer (character-level, fixed small vocab) used
ONLY by the offline LoRA training test. Exists so that test doesn't need
to download a real tokenizer from Hugging Face Hub -- everything about it
is constructed locally, no network calls.
"""
from __future__ import annotations
from typing import List


class ToyTokenizer:
    def __init__(self):
        chars = sorted(set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?'\n"
        ))
        self.char_to_id = {c: i + 3 for i, c in enumerate(chars)}  # 0,1,2 reserved
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.vocab_size = len(chars) + 3

    def _encode_str(self, text: str) -> List[int]:
        return [self.char_to_id.get(c, self.unk_token_id) for c in text]

    def __call__(self, text, add_special_tokens: bool = True, truncation: bool = False,
                 max_length: int = None, return_tensors=None):
        ids = self._encode_str(text)
        if truncation and max_length:
            ids = ids[:max_length]
        result = {"input_ids": ids}
        if return_tensors == "pt":
            import torch
            result = {"input_ids": torch.tensor([ids], dtype=torch.long),
                       "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
        return result

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        chars = []
        for i in ids:
            i = int(i)
            if i in (self.pad_token_id, self.eos_token_id, self.unk_token_id) and skip_special_tokens:
                continue
            chars.append(self.id_to_char.get(i, ""))
        return "".join(chars)

    def save_pretrained(self, path):
        pass  # no-op: this tokenizer only exists for the offline mechanics test
