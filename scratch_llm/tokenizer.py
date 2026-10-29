"""
Trains a byte-level BPE tokenizer FROM SCRATCH on your own corpus -- no
pretrained vocab download, unlike AutoTokenizer.from_pretrained(). This is
the other half of "building an LLM": the model needs a vocabulary that
was learned from the same data it's about to be pretrained on.
"""
from __future__ import annotations
from pathlib import Path
from typing import List
import torch


class ScratchTokenizer:
    PAD, EOS, UNK = "<pad>", "<eos>", "<unk>"

    def __init__(self, tokenizer_path: str = None):
        from tokenizers import ByteLevelBPETokenizer
        self._tok = ByteLevelBPETokenizer()
        self._loaded_path = None
        if tokenizer_path is not None:
            self.load(tokenizer_path)

    def train(self, texts: List[str], vocab_size: int = 8000, min_frequency: int = 2,
              save_dir: str = None):
        """Trains directly from an in-memory list of strings (writes a temp
        file internally -- `tokenizers` needs a file path to train from)."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            for t in texts:
                f.write(t.replace("\n", " ") + "\n")
            tmp_path = f.name
        try:
            self._tok.train(
                files=[tmp_path], vocab_size=vocab_size, min_frequency=min_frequency,
                special_tokens=[self.PAD, self.EOS, self.UNK],
            )
        finally:
            os.unlink(tmp_path)
        if save_dir:
            self.save(save_dir)
        return self

    def save(self, save_dir: str):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self._tok.save_model(save_dir)
        self._loaded_path = save_dir

    def load(self, tokenizer_path: str):
        from tokenizers import ByteLevelBPETokenizer
        vocab_file = str(Path(tokenizer_path) / "vocab.json")
        merges_file = str(Path(tokenizer_path) / "merges.txt")
        self._tok = ByteLevelBPETokenizer(vocab_file, merges_file)
        self._loaded_path = tokenizer_path
        return self

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        return self._tok.token_to_id(self.PAD)

    @property
    def eos_token_id(self) -> int:
        return self._tok.token_to_id(self.EOS)

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text).ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [e.ids for e in self._tok.encode_batch(texts)]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        ids = [int(i) for i in ids]
        if skip_special_tokens:
            special_ids = {self.pad_token_id, self.eos_token_id, self._tok.token_to_id(self.UNK)}
            ids = [i for i in ids if i not in special_ids]
        return self._tok.decode(ids)

    def __call__(self, text: str, return_tensors: str = None):
        ids = self.encode(text)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}
