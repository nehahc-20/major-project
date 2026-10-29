# SOAR-RAG: Self-Optimizing Attention-Guided Retrieval-Augmented Generation

Implementation of the **SOAR-RAG** framework described in *"Self-Optimizing
Retrieval-Augmented Generation System Using LLM Attention Feedback"* — a RAG
pipeline that uses the generator's own cross-attention weights as a feedback
signal to re-rank retrieved documents, prune irrelevant context, and adjust
retrieval depth **without any fine-tuning**.

```
Q → R0 → G0 → Af → Rf → Gf
```
Query → first-round retrieval → first-round generation → attention feedback →
refined retrieval → optimized response, looped until convergence.

## Quickest way to run it (one command)
```bash
pip install -r requirements.txt
python run_demo.py                          # demo query, CPU, small model
python run_demo.py --eval --n 10            # + a real 10-sample eval on HotpotQA
python run_demo.py --device cuda --quantize 4bit --model Qwen/Qwen2.5-3B-Instruct --eval --n 30
```
Defaults are chosen so this works with **no GPU and no manual dataset setup**:
CPU device, a small model (`Qwen2.5-0.5B-Instruct`), and HotpotQA as the eval
dataset because it's the only one of the four that ships real gold/distractor
passages without extra work (see "About the four datasets" below). The third
example shows the efficiency path once you have a GPU: `--quantize 4bit` uses
`bitsandbytes` NF4 quantization to cut VRAM ~2-4x vs fp16, which is what lets
a 3B model (usually needs an A100) fit comfortably on a free-tier Colab T4.

## Have I actually run this and gotten real numbers?
**No, not yet — and I'm not going to pretend otherwise.** Producing genuine
accuracy numbers requires downloading a model from Hugging Face, which needs
real internet access to huggingface.co. See `VERIFICATION.md` for the full,
dated log of everything that has actually been checked. What *is* verified
as of this version (re-confirmed from a clean install on 2026-08-15, see
`VERIFICATION.md`):
- `pytest tests/` — **14/14 passing** (not just the mock-pipeline subset —
  this now includes the LoRA training mechanics and from-scratch-pretraining
  tests too), exercises the full attention-mining → reranking → confidence →
  convergence control flow against a mock model
- Device auto-detection (`src/generation.py`) — confirmed to fall back to
  CPU safely when `--device cuda` is requested but no GPU is present
- BioASQ fallback labeling (`data/loaders.py`) — confirmed the
  `bioasq_pubmedqa_fallback` label propagates correctly through
  `run_eval.py`/`run_demo.py` into every output file, so a fallback run can
  never be mistaken for real BioASQ results
- `npm run build` — frontend compiles clean
- `demo/mock_eval_demo.py` — a hand-scripted (not random, not a real LLM)
  scenario proving the *mechanism*: a fixed top-3 baseline misses a
  relevant document ranked 4th, SOAR-RAG's larger/rerank-aware retrieval
  catches it, and the resulting `results_demo/summary.csv` +
  `results_demo/comparison_demo.png` show what that looks like in the same
  metric columns the real eval produces. Every file it writes says
  `SCRIPTED_MECHANISM_DEMO` in the `corpus_mode` column and carries a
  disclaimer in `summary.json` — treat it as a format preview, not a result.

**Run `python run_demo.py --eval` on your own machine to get real numbers.**
That's the one step that has to happen outside this environment.

## Why this is different from a standard RAG project
Most student RAG projects retrieve once and generate once. This one closes
the loop: it inspects **which tokens the LLM actually attended to** while
answering, scores each retrieved document by how much attention it received,
prunes the ones it ignored, dynamically resizes top-K based on query
complexity, and repeats until the model's confidence stabilizes or its
answer stops changing. That requires running an **open-weight local model**
(Llama/Mistral/Qwen family) rather than an API model, because attention
weights and token log-probabilities are internals that closed APIs
(OpenAI, Groq, Anthropic) never expose.

## Efficiency choices, and what they trade off
| Choice | Where | Effect | Trade-off |
|---|---|---|---|
| fp16 on GPU | `src/generation.py` | ~2x less memory/latency vs fp32 | none at inference time |
| `--quantize 4bit`/`8bit` | `src/generation.py` | ~2-4x less VRAM | small accuracy risk (~1-2 F1 pts typical for NF4); GPU-only |
| `attn_implementation="eager"` | `src/generation.py` | slower than fused SDPA attention | required — SOAR-RAG needs real `output_attentions`, which SDPA doesn't reliably return |
| Query-embedding LRU cache | `src/knowledge_base.py` | skips re-encoding repeated queries across iterations/eval runs | memory for the cache (default 256 entries) |
| IVF FAISS index above 5k chunks | `src/knowledge_base.py` | sub-linear search on large KBs | slightly lower recall than exact search; irrelevant below the threshold (uses exact `IndexFlatIP`) |
| `torch.inference_mode()` | `src/generation.py` | marginal speedup over `no_grad()` | none |

The `attn_implementation="eager"` line is the one **not** to "optimize away" —
switching to `"sdpa"` for speed will silently break the attention-feedback
mechanism this whole project is built around.

## Architecture
| Component | File | Paper section |
|---|---|---|
| Query Processing | `src/query_processing.py` | III.C |
| Knowledge Base / Retrieval | `src/knowledge_base.py` | III.D, III.K |
| Generation (attention + logprob capture) | `src/generation.py` | III.E |
| Attention-Based Feedback Mining | `src/attention_feedback.py` | III.F |
| Feedback Optimization Engine (pruning, expansion, dynamic-K) | `src/reranking.py` | III.G, III.H |
| Confidence Estimation | `src/confidence.py` | III.J |
| Iterative orchestration loop | `src/pipeline.py` | III.A, III.I |
| Evaluation (EM/F1/retrieval-acc/hallucination/compression/latency) | `eval/` | III.M, III.N |
| Live API | `app/api.py` | — |

## Building an LLM from scratch (not fine-tuning — actual pretraining)
This is the real thing: a transformer architecture written from nothing
(`scratch_llm/model.py`), a byte-level BPE tokenizer trained on your own
corpus (not downloaded), and randomly-initialized weights updated purely
by your own gradient descent on Wikipedia-style text. No Qwen, no
pretrained starting point anywhere in this path.

```bash
# pretrain (GPU strongly recommended -- this is far more compute-hungry than fine-tuning)
python -m scratch_llm.train --n_articles 3000 --steps 3000 --out checkpoints/scratch-gpt

# sanity-check what it generates on its own
python -m scratch_llm.generate_text --checkpoint checkpoints/scratch-gpt --prompt "The history of"

# plug it into the real SOAR-RAG pipeline in place of Qwen
python run_demo.py --scratch_checkpoint checkpoints/scratch-gpt
```
`scratch_llm/soar_adapter.py` wraps it to match the exact interface
`src/generation.py`'s `LLMGenerator` uses, so `SOARRAGPipeline` runs
against it completely unmodified — same attention-feedback mining, same
reranking, same convergence logic, just pointed at weights that are
entirely yours.

**Set expectations honestly, both for yourself and for anyone you show
this to:** a model this size (a few million to tens of millions of
parameters, trained on a few thousand Wikipedia paragraphs) will produce
grammatical-ish fragments and pick up local word statistics — it will
**not** produce coherent factual answers. GPT-2 needed 1.5B parameters and
40GB of text to get good; modern LLMs are ~1000x that again. This step
exists to make "I built an LLM" literally true, not to compete with Qwen
on answer quality — for actually-good answers in your pipeline, use the
Qwen + LoRA fine-tuning path above. Both are legitimate, they just prove
different things: this proves you can build the architecture; the LoRA
path proves you can meaningfully improve a real model's task accuracy.

**Verified here, before spending real GPU time:** `pytest
tests/test_scratch_llm.py` — trains the tokenizer, trains the tiny model
for real (asserts loss decreases: ~5.7 → ~1.6 in one run here), generates
text, and runs it through the actual unmodified `SOARRAGPipeline` end to
end, confirming attention capture and document-span mapping both work.
3/3 passing, on top of the project's other 11 tests (14/14 total).

## Fine-tuning the generator (real training, real weight updates)
This is the part you can honestly call "I trained a model" — it's LoRA
fine-tuning: real gradient descent on real weights, starting from a
pretrained model instead of random initialization (see "What training
actually means" below for why that distinction matters).

```bash
python -m train.train_lora --n 300 --epochs 2 --out checkpoints/soar-lora
# GPU + QLoRA for a bigger model:
python -m train.train_lora --n 300 --quantize 4bit --model Qwen/Qwen2.5-3B-Instruct

# then, the actual evidence -- base model vs fine-tuned, same held-out set:
python -m train.evaluate_finetune --adapter_path checkpoints/soar-lora --n 50

# plug the fine-tuned generator into the SOAR-RAG pipeline:
python run_demo.py --adapter_path checkpoints/soar-lora
```
Trains on HotpotQA (context, question, answer) triples, in the exact prompt
format `src/generation.py` uses at inference time, so the adapter drops
straight into your existing pipeline. `train/evaluate_finetune.py` runs
BOTH the base model and the fine-tuned model on an identical held-out split
and reports the EM/F1 delta — that delta is your real, defensible resume
number, not an assumption.

**Verified here, before you spend GPU time on it:** `pytest
tests/test_lora_training.py` runs the actual training loop (forward pass,
backward pass, optimizer step, adapter save/load) against a tiny
randomly-initialized model — no downloads needed — and asserts the loss
genuinely decreases. This caught a real bug during development: naive
right-truncation on a long context was silently cutting off the answer
tokens, masking an entire training example to `-100` and producing NaN
loss. Fixed in `train/data_prep.py` (left-truncation instead, so the
answer at the tail is always preserved) — worth knowing this class of bug
exists if you ever change `max_length` or the prompt template.

### What "training" actually means here — read this before writing your resume line
"I built an LLM like ChatGPT/Claude" is not an accurate claim for this or
almost any other project — those took trillions of tokens of data,
thousands of GPUs for months, and tens of millions of dollars. Nobody
outside a handful of labs pretrains a model from scratch.
**What actually happened here is real training** — LoRA fine-tuning updates
actual weights via actual gradient descent, it just starts from an
already-pretrained model rather than random noise. That's what "I trained
a model" means in the vast majority of ML job postings and resumes.
Accurate, defensible phrasing:
- "Fine-tuned Qwen2.5 using LoRA on a custom grounded-QA dataset, measuring
  a +X F1 / +X EM improvement over the base model on held-out HotpotQA"
- "Trained a low-rank adapter (<3% of total parameters) to improve
  context-grounded answer accuracy in a RAG pipeline"

Avoid: "built/trained an LLM from scratch", "built a model like ChatGPT" —
these are one follow-up question away from falling apart in an interview.

## Quickstart
### 1. Verify the logic offline (no GPU/downloads needed)
```bash
pip install -r requirements.txt
pytest tests/ -v
```
This runs the full pipeline against a hash-based mock encoder/generator so
you can confirm the control flow (retrieval → attention mining → pruning →
reranking → convergence) is correct before spending GPU time.

### 2. Run for real on Colab/Kaggle (GPU)
Open `notebooks/SOAR_RAG_Colab.ipynb`, set the runtime to GPU, and run all
cells. It installs deps, loads an open-weight model (`Qwen2.5-1.5B-Instruct`
by default — swap to a 7B model if you have an A100/L4), runs a live demo
query with the trace printed iteration-by-iteration, then runs the full
evaluation across NQ / HotpotQA / TriviaQA / BioASQ and plots baseline vs
SOAR-RAG.

### 3. Run the evaluation from the command line
```bash
python -m eval.run_eval --datasets nq hotpotqa triviaqa bioasq --n 100 \
    --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --out results
```
Outputs `results/summary.csv` (aggregate metrics per dataset) and
per-dataset raw CSVs.

### 4. Serve it as a live API + React console
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
# POST /ingest  {"texts": [...]}   -> builds the knowledge base + loads the model
# POST /query    {"query": "..."}   -> runs SOAR-RAG only
# POST /compare  {"query": "..."}   -> runs baseline RAG + SOAR-RAG, for side-by-side comparison
```
Then, in a second terminal:
```bash
cd frontend && npm install && npm run dev
```
Opens a console at `localhost:5173` with two tabs, both hitting the same
backend:
- **Chat** — a plain chat interface for asking questions against the
  knowledge base. Every answer carries the pipeline's own confidence score
  (Eq. 14, blended token-probability + attention-consistency +
  retrieval-consistency) as a visible badge, plus a collapsible list of the
  documents it actually drew the most attention from. Low-confidence answers
  are flagged instead of stated flatly — the point is to surface when the
  model doesn't know, not to hide it.
- **Compare** — the original research console: baseline (single-shot) RAG
  next to SOAR-RAG on the same question, with per-document attention bars
  for every feedback iteration, so you can see the reranking loop working.

Since the backend needs a GPU, the usual setup is running `app/api.py`
inside the Colab notebook (exposed via ngrok/Colab port-forwarding) and
pointing the frontend's `VITE_API_URL` at that URL. See `frontend/README.md`.

A note on claims: this project can honestly support "SOAR-RAG reduces
hallucination and improves accuracy **relative to baseline single-pass RAG
on the same base model**" — that's the paper's actual thesis, and the
`/compare` endpoint plus `eval/run_eval.py` give you real numbers to back it.
It can't honestly support "beats ChatGPT/Claude/Gemini" — those are
much larger, differently-trained systems, and no eval in this repo tests
that. Keep resume/report claims to the comparison this project actually
measures.

## Notes on the four datasets
Every loader now returns `(samples, source_label, corpus_mode)` so results
can never silently misrepresent what was actually tested — `run_eval.py`
and `run_demo.py` both write `source_label` and `corpus_mode` into
`results/summary.json`/`summary.csv` alongside the metrics.

- **NQ-open / TriviaQA** ship without gold passages, so `data/loaders.py`
  builds a lightweight in-memory corpus from the QA pairs themselves
  (`corpus_mode: "synthetic_qa_pool"`). This makes retrieval close to
  trivial — the answer is inside the "document" being searched for — so
  EM/F1/retrieval-accuracy on these two are **not comparable** to real
  open-domain numbers; treat them as a pipeline sanity-check, not a
  resume-grade result. For a stronger claim ("evaluated on real
  open-domain retrieval"), swap in a real Wikipedia passage index (e.g. the
  DPR Wikipedia FAISS index) as the knowledge base — `KnowledgeBase` works
  unchanged, just point `build()` at real Wikipedia passages.
- **HotpotQA** ships with real distractor contexts (`corpus_mode:
  "real_context"`), used directly — this is the most meaningful dataset to
  report results from without extra setup, and `run_demo.py` defaults to it.
- **BioASQ** requires registration to download officially. Without a local
  export, `load_bioasq()` falls back to PubMedQA and labels results
  `source_label: "bioasq_pubmedqa_fallback"` rather than `"bioasq"` — so a
  results file can never be read as claiming real BioASQ numbers when it
  isn't. Pass `--bioasq-local-path <file>` (both `run_eval.py` and
  `run_demo.py`) with a real BioASQ task-B export to get true BioASQ results.

## What to put in your resume / project writeup
- "Built a closed-loop RAG system that uses transformer cross-attention as
  a feedback signal to re-rank and prune retrieved context, with dynamic
  top-K retrieval and confidence-gated iteration — no fine-tuning required."
- Report the `results/summary.csv` numbers: EM/F1 delta over a static-RAG
  baseline, hallucination-rate reduction, and context-compression ratio.
- Mention the offline test suite (`pytest tests/`) as evidence of validated
  control flow, separate from model-dependent results.

## Repo layout
```
soar-rag/
├── run_demo.py             # one-command entry point: demo query + optional small eval, CPU-safe defaults
├── scratch_llm/              # from-scratch pretraining: model.py, tokenizer.py, data.py, train.py, soar_adapter.py
├── train/                   # LoRA fine-tuning: train_lora.py, evaluate_finetune.py, data_prep.py
├── src/                  # core framework (see table above)
├── eval/                 # metrics + baseline-vs-SOAR comparison runner
├── data/loaders.py        # NQ / HotpotQA / TriviaQA / BioASQ loaders (honest source labels)
├── demo/mock_eval_demo.py  # scripted (non-LLM) mechanism demo -> results_demo/, format preview only
├── results_demo/            # output of the command above, committed so you can see the format immediately
├── app/api.py             # FastAPI demo service
├── frontend/                # React + Vite console (Chat tab + attention-bar Compare tab)
├── notebooks/              # Colab notebook (real GPU run)
├── tests/                  # offline unit tests (mocked model, no downloads) + LoRA training mechanics tests
├── configs/config.yaml
└── requirements.txt
```


cd "/Users/nehahc/Downloads/soar-rag 4/frontend"
cp .env.example .env
npm install
npm run dev

cd "/Users/nehahc/Downloads/soar-rag 4"
source .venv/bin/activate
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false uvicorn app.api:app --host 0.0.0.0 --port 8000 --workers 1

