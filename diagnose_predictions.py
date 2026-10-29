"""
Quick diagnostic: prints the model's actual predicted text next to the
gold answer for a handful of HotpotQA questions, so you can SEE why EM/F1
are near zero -- instead of just staring at the aggregate score.

Run from the project root (same folder as run_demo.py):
    python diagnose_predictions.py
"""
from data.loaders import load_hotpotqa
from src.knowledge_base import KnowledgeBase
from src.generation import LLMGenerator
from src.pipeline import SOARRAGPipeline, BaselineRAGPipeline

N = 5  # just a handful, this is for reading not scoring

print("Loading data...")
samples, source_label, corpus_mode = load_hotpotqa(n=N)

print("Loading model (this is the slow part, same as run_demo.py)...")
generator = LLMGenerator(model_name="Qwen/Qwen2.5-0.5B-Instruct", device="cpu")

contexts = [s["context"] for s in samples]
kb = KnowledgeBase(embedding_model="sentence-transformers/all-MiniLM-L6-v2", device="cpu")
kb.build(contexts)

baseline = BaselineRAGPipeline(kb, generator, top_k=5)
soar = SOARRAGPipeline(kb, generator)

for i, s in enumerate(samples):
    q, gold = s["question"], s["answers"]
    b_res = baseline.run(q)
    s_res = soar.run(q)

    print("\n" + "=" * 80)
    print(f"Q{i+1}: {q}")
    print(f"GOLD ANSWER:      {gold}")
    print(f"BASELINE PREDICTS: {b_res.final_response!r}")
    print(f"SOAR PREDICTS:     {s_res.final_response!r}")
