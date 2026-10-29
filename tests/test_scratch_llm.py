"""
Proves the from-scratch GPT (scratch_llm/) actually works end to end and
plugs into the real SOAR-RAG pipeline -- not just "compiles", but: trains
with real gradient descent and the loss goes down, generates text, and
its attention weights are consumable by the SAME AttentionFeedbackMiner
used for Qwen/Llama, running inside the SAME SOARRAGPipeline with zero
pipeline code changes.

Uses a tiny local corpus (not Wikipedia -- WikiText-103 needs a download,
not available in this offline test) so this runs anywhere with no network.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from scratch_llm.model import GPT, GPTConfig
from scratch_llm.tokenizer import ScratchTokenizer
from scratch_llm.soar_adapter import ScratchLLMGenerator
from src.pipeline import SOARRAGPipeline, BaselineRAGPipeline
from tests.mocks import MockKnowledgeBase

TOY_CORPUS = [
    "Paris is the capital of France, known for the Eiffel Tower and the Louvre museum.",
    "The Eiffel Tower was completed in 1889 for the World's Fair in Paris.",
    "Mount Everest is the tallest mountain on Earth, located in the Himalayas.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "Photosynthesis converts sunlight into chemical energy inside plant cells.",
] * 15


def _train_tiny_checkpoint(tmp_path):
    tokenizer = ScratchTokenizer()
    tokenizer.train(TOY_CORPUS, vocab_size=300, save_dir=str(tmp_path / "tokenizer"))

    config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=64, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(config)

    ids = []
    for t in TOY_CORPUS:
        ids += tokenizer.encode(t + " ")
    block = config.block_size
    chunks = [ids[i:i + block + 1] for i in range(0, len(ids) - block - 1, block)]
    xb = torch.tensor([c[:-1] for c in chunks], dtype=torch.long)
    yb = torch.tensor([c[1:] for c in chunks], dtype=torch.long)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(40):
        out = model(xb, labels=yb)
        opt.zero_grad(); out["loss"].backward(); opt.step()
        losses.append(out["loss"].item())

    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir()
    torch.save({"model_state_dict": model.state_dict(), "config": config.__dict__}, ckpt_dir / "model.pt")
    tokenizer.save(str(ckpt_dir / "tokenizer"))
    return ckpt_dir, losses


def test_pretraining_actually_reduces_loss(tmp_path):
    _, losses = _train_tiny_checkpoint(tmp_path)
    first_avg = sum(losses[:5]) / 5
    last_avg = sum(losses[-5:]) / 5
    assert last_avg < first_avg, f"loss did not decrease: first={first_avg:.3f} last={last_avg:.3f}"


def test_checkpoint_loads_and_generates(tmp_path):
    ckpt_dir, _ = _train_tiny_checkpoint(tmp_path)
    gen = ScratchLLMGenerator(str(ckpt_dir), device="cpu", max_new_tokens=15)
    result = gen.generate("What is the capital of France?", [TOY_CORPUS[0], TOY_CORPUS[1]])
    assert isinstance(result.text, str) and len(result.text) > 0
    assert result.attentions is not None and len(result.attentions) == 2  # n_layer=2
    assert len(result.doc_token_spans) == 2


def test_scratch_generator_runs_inside_real_soar_pipeline(tmp_path):
    """The actual integration proof: SOARRAGPipeline (unmodified, the same
    class used with Qwen) runs correctly against a from-scratch model."""
    ckpt_dir, _ = _train_tiny_checkpoint(tmp_path)
    gen = ScratchLLMGenerator(str(ckpt_dir), device="cpu", max_new_tokens=15)
    kb = MockKnowledgeBase(TOY_CORPUS)

    pipeline = SOARRAGPipeline(kb, gen, max_iterations=2, confidence_threshold=0.3)
    result = pipeline.run("What is the capital of France?")
    assert result.final_response
    assert len(result.trace) >= 1
    assert result.trace[0].attention_entropy >= 0.0  # attention mining ran without error

    baseline = BaselineRAGPipeline(kb, gen, top_k=2)
    b_result = baseline.run("What is the tallest mountain?")
    assert b_result.final_response
