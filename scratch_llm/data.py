"""
Wikipedia-style pretraining corpus + fixed-length chunking for causal LM
training. Uses WikiText-103 (real Wikipedia article text, cleaned) via the
`datasets` library -- genuine factual/encyclopedic text, matching what a
RAG system's knowledge base actually looks like, as opposed to a toy
story corpus. Needs internet access (Hugging Face Hub) to download -- not
runnable in a network-restricted sandbox; run on your machine/Colab.
"""
from __future__ import annotations
from typing import List
import torch
from torch.utils.data import Dataset


def load_wikitext_corpus(n_articles: int = 2000, min_chars: int = 200) -> List[str]:
    """Pulls real Wikipedia article paragraphs from WikiText-103. Filters
    out WikiText's markup lines (headers like '= Title =') and very short
    fragments so the pretraining corpus is genuine prose."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    texts = []
    for row in ds:
        line = row["text"].strip()
        if len(line) < min_chars or line.startswith("="):
            continue
        texts.append(line)
        if len(texts) >= n_articles:
            break
    return texts


class ChunkedLMDataset(Dataset):
    """Tokenizes a corpus, concatenates it into one long stream (standard
    GPT pretraining setup), and slices it into fixed-length (input, target)
    windows shifted by one token -- the actual "predict the next token"
    training objective."""

    def __init__(self, texts: List[str], tokenizer, block_size: int = 256):
        self.block_size = block_size
        all_ids: List[int] = []
        for t in texts:
            all_ids.extend(tokenizer.encode(t))
            all_ids.append(tokenizer.eos_token_id)
        self.data = torch.tensor(all_ids, dtype=torch.long)
        self.n_chunks = max((len(self.data) - 1) // block_size, 0)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.block_size
        chunk = self.data[start: start + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return {"input_ids": x, "labels": y}
