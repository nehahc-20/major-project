# Verification Log

This file records what was actually checked, when, and how — separate from
the aspirational README so results and mechanism-verification claims never
get mixed up. Every line below was run, not assumed.

## 2026-08-15 — Sandbox verification (CPU, no GPU, no internet to huggingface.co)

**Environment constraints (be aware of these when reading this log):**
this sandbox cannot reach `huggingface.co`, so no real model weights could
be downloaded and no GPU was available. Everything below is either (a) pure
control-flow / unit-level verification that doesn't need a real model, or
(b) static code review of the two previously-unconfirmed patches. **The
real accuracy eval (`run_demo.py --eval` / `eval/run_eval.py`) still has
not been run against a real model as of this log** — that step requires an
environment with internet access to Hugging Face (your own machine or
Colab), and is the one thing this log does not change.

### 1. Full test suite — re-run clean, from a fresh environment
```
$ pip install torch transformers peft numpy pytest
$ pytest tests/ -v
================= 14 passed, 6 warnings in 10.95s =================
```
All 14 tests genuinely pass (`test_pipeline_mock.py` ×7, `test_scratch_llm.py`
×3, `test_lora_training.py` ×4) — confirms the README's "14/14" claim with a
clean install, not a stale cache. Warnings are harmless (`peft` fan_in_fan_out
notice for Conv1D layers, a `tokenizers` deprecation notice) — no test-logic
warnings.

### 2. Device auto-detection (`src/generation.py:55`) — confirmed correct
```python
self.device = device if torch.cuda.is_available() else "cpu"
```
Tested directly in a CPU-only sandbox:
```
requested='cuda' -> resolved='cpu'   # safe fallback, no crash
requested='cpu'  -> resolved='cpu'   # respects explicit choice
```
This was the open item flagged as "unconfirmed" — it's correct. Requesting
`--device cuda` on a machine without a GPU degrades gracefully to CPU rather
than raising a CUDA error. `run_demo.py --device` defaults to `"cpu"`, so the
one-command quickstart never assumes a GPU exists.

### 3. BioASQ fallback labeling (`data/loaders.py:63-87`) — confirmed correct
Traced the label from source through to disk:
- No `--bioasq-local-path` given → `load_bioasq()` pulls PubMedQA and returns
  `source_label = "bioasq_pubmedqa_fallback"`.
- A real local export given → returns `source_label = "bioasq"`.
- Both `run_eval.py` and `run_demo.py` use `source_label` (never the literal
  string `"bioasq"`) for the output CSV filename, the `summary.json`
  `source_labels` dict, and the printed console note. A fallback run
  therefore cannot silently end up in a file or report claiming to be real
  BioASQ — confirmed by reading the write path, not just the loader.

### 4. Frontend build — confirmed clean
```
$ npm install && npm run build
✓ 34 modules transformed.
✓ built in 1.34s
```
Matches the README's "frontend compiles clean" claim.

### 5. Mock mechanism demo — re-read, not re-run (no model needed, deterministic)
`demo/mock_eval_demo.py` output already in `results_demo/` is hand-scripted,
not a byproduct of a real model — confirmed every row in
`results_demo/*.csv` still carries `corpus_mode = SCRIPTED_MECHANISM_DEMO`
and `results_demo/summary.json` still carries its disclaimer. This has not
changed and should not be cited as an accuracy result.

## 2026-08-15 — Real HotpotQA run + root-cause fix (n=30, CPU, Qwen2.5-0.5B-Instruct)

**Real numbers obtained for the first time** (on the person's own machine,
not this sandbox): baseline_EM=0.0, soar_EM=0.0, baseline_F1=0.033,
soar_F1=0.027, soar_ContextCompression=1.768, soar_AvgIterations=2.9.

**Root-caused via manual inspection of raw predictions** (`diagnose_predictions.py`,
5 sample questions, predicted text read against gold text) — two distinct
problems found, not one:

1. **Genuine model-capability gap.** Qwen2.5-0.5B-Instruct, even with the
   correct context in front of it, answers confidently and *wrongly* on
   multi-hop questions (e.g. answered "No" to a yes/no question whose gold
   answer was "yes", inventing specific fabricated supporting "facts" not
   present in the retrieved context). This is a real, reportable finding —
   0.5B is likely below the capability threshold needed for this task — not
   a bug.
2. **Real pipeline bug, now fixed**: `src/generation.py`'s `build_prompt()`
   was building a raw completion-style prompt string instead of using the
   tokenizer's chat template. Qwen2.5-*-Instruct models are fine-tuned to
   expect `<|im_start|>system/user/assistant` formatted input; feeding them
   a raw string is out-of-distribution for an instruct model. Observed
   direct evidence: on at least one question, the SOAR pipeline's
   "answer" was actually leaked prompt/instruction text (something like
   "You are an AI assistant... I will now provide the answer according to
   the guidelines above") rather than a real answer — the model was
   hallucinating the chat structure it expected but wasn't given.

   **Fix applied**: `build_prompt()` now uses `tokenizer.apply_chat_template()`
   when the tokenizer has one, with a system message instructing short,
   direct answers. The character-span logic that the attention-attribution
   mechanism (Eq. 7/8) depends on is preserved by locating the unchanged
   `user_content` substring inside the final templated prompt and shifting
   spans by that offset — verified correct with a standalone span test
   (mock chat template, asserted each document's text is recoverable from
   its shifted span in the final prompt — passed). Full `pytest tests/`
   suite re-run after the change: still 14/14 passing.

   **This fix has not yet been re-run against the real model in an eval**
   (this sandbox still has no Hugging Face access) — the `n=30` numbers
   above are from *before* this fix. Re-running `run_demo.py --eval --n 30`
   after pulling this update is the next real step, and the EM/F1 numbers
   should be re-measured — don't assume they'll change until you've
   actually seen the new run.

## What this verification does NOT cover — still your responsibility

- **No real EM/F1/hallucination numbers exist yet.** This log fixes the
  *confidence* in the code being correct; it does not produce results.
  Run one of:
  ```bash
  python run_demo.py --eval --n 10                     # quick sanity check
  python run_demo.py --eval --n 30                     # CPU, 0.5B model, HotpotQA (recommended baseline)
  python -m eval.run_eval --datasets hotpotqa --n 100 --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --out results
  ```
  on a machine with internet access to Hugging Face.
- **BioASQ**: to get a real (non-fallback) BioASQ result, you still need to
  register and download a task-B export yourself, then pass
  `--bioasq-local-path <file>`. Nothing here can do that step for you.
- **LoRA fine-tuning deltas** (`train/evaluate_finetune.py`) — the training
  loop mechanics are verified (loss decreases, adapter save/load round-trips),
  but the actual base-vs-fine-tuned F1/EM delta requires a real training run.

## Bottom line for your writeup

You can now honestly say, with evidence:
- "14/14 tests pass, verifying attention-mining → reranking → confidence →
  convergence control flow, LoRA training mechanics, and from-scratch
  pretraining, all against a clean install."
- "Device fallback and BioASQ fallback-labeling logic were verified correct
  by direct testing/code trace, not left as an assumption."
- "Frontend build verified clean."

You still need real numbers before you can say anything about accuracy,
hallucination reduction, or F1/EM deltas — that step is unchanged and has
to happen where you have GPU + internet access.
