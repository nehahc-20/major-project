"""
Wraps the from-scratch GPT (scratch_llm/model.py) so it satisfies the same
interface as src/generation.py's LLMGenerator -- `.generate(query,
documents) -> GenerationResult`. This is what lets SOARRAGPipeline and
BaselineRAGPipeline (src/pipeline.py) run against YOUR pretrained model
with zero changes to the pipeline code: the pipeline only ever calls
`generator.generate(...)`, and doesn't know or care whether that's Qwen
via Hugging Face or a transformer you trained from nothing.
"""
from __future__ import annotations
from typing import List
import math
import torch
from pathlib import Path

from src.generation import GenerationResult  # reuse the same result dataclass as LLMGenerator
from scratch_llm.model import GPT, GPTConfig
from scratch_llm.tokenizer import ScratchTokenizer


class ScratchLLMGenerator:
    """Drop-in replacement for src.generation.LLMGenerator, backed by a
    scratch_llm checkpoint instead of a Hugging Face model. Expect
    noticeably weaker answers than Qwen -- this model has, at best, tens
    of millions of parameters trained on a few thousand paragraphs, versus
    Qwen's ~500M-3B parameters trained on trillions of tokens. That gap is
    real and expected; the point of this class is that the weights
    answering your questions are entirely your own gradient descent, not
    someone else's pretraining run."""

    def __init__(self, checkpoint_dir: str, device: str = "cuda", max_new_tokens: int = 64):
        self.device = device if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(Path(checkpoint_dir) / "model.pt", map_location=self.device, weights_only=False)
        self.config = GPTConfig(**ckpt["config"])
        self.model = GPT(self.config).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.tokenizer = ScratchTokenizer(str(Path(checkpoint_dir) / "tokenizer"))
        self.max_new_tokens = max_new_tokens

    def _build_prompt_ids(self, query: str, documents: List[str]):
        header_ids = self.tokenizer.encode(
            "Answer the question using ONLY the context below. Be concise and factual.\n\nContext:\n"
        )
        all_ids = list(header_ids)
        doc_spans = []
        for i, doc in enumerate(documents):
            tag_ids = self.tokenizer.encode(f"[DOC{i}] ")
            doc_ids = self.tokenizer.encode(doc + "\n")
            all_ids += tag_ids
            start = len(all_ids)
            all_ids += doc_ids
            end = len(all_ids)
            doc_spans.append((start, end))
        tail_ids = self.tokenizer.encode(f"\nQuestion: {query}\nAnswer:")
        all_ids += tail_ids

        # this model's context window is small (block_size, e.g. 256) --
        # keep-the-tail truncation so the question always survives, same
        # fix applied in train/data_prep.py for the same underlying reason
        if len(all_ids) > self.config.block_size:
            excess = len(all_ids) - self.config.block_size
            all_ids = all_ids[excess:]
            doc_spans = [(max(s - excess, 0), max(e - excess, 0)) for s, e in doc_spans]
        return all_ids, doc_spans

    def generate(self, query: str, documents: List[str]) -> GenerationResult:
        input_ids, doc_spans = self._build_prompt_ids(query, documents)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        gen_full, logprobs, attentions = self.model.generate(
            input_tensor, max_new_tokens=self.max_new_tokens,
            temperature=0.7, top_k=40, eos_token_id=self.tokenizer.eos_token_id,
        )
        gen_ids = gen_full[0][len(input_ids):]
        text = self.tokenizer.decode(gen_ids.tolist())

        return GenerationResult(
            text=text.strip(),
            generated_token_ids=gen_ids,
            input_token_ids=torch.tensor(input_ids, dtype=torch.long),
            attentions=tuple(attentions) if attentions else None,
            token_logprobs=logprobs,
            doc_token_spans=doc_spans,
        )
