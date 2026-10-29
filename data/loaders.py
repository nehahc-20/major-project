"""
Dataset loaders for the four corpora used in the paper's Experiment Design
(III.M): Natural Questions, HotpotQA, TriviaQA, BioASQ.

Uses the `datasets` library (Hugging Face). Requires internet access
(works on Colab/Kaggle/local-with-internet) -- not runnable inside a
network-restricted sandbox.

Every loader returns a 3-tuple: (samples, source_label, corpus_mode)

    samples      -- list of {"question": str, "answers": [str, ...], "context": str}
    source_label -- the actual dataset this data came from. Usually equals
                     the loader's own name, EXCEPT bioasq without a local
                     export, which is honestly labeled "bioasq_pubmedqa_fallback"
                     rather than silently reported as "bioasq".
    corpus_mode  -- "real_context": each sample ships a genuine gold/distractor
                     passage (HotpotQA, BioASQ-with-local-file, PubMedQA fallback).
                     Retrieval accuracy and EM/F1 here reflect real open-book QA.
                   "synthetic_qa_pool": the sample has no gold passage (NQ,
                     TriviaQA in this config), so the loader builds a stand-in
                     corpus where each "document" is the question paired with
                     its own answer. This makes retrieval close to trivial
                     (the answer is *in* the document being searched for) and
                     will inflate EM/F1/retrieval-accuracy relative to a real
                     open-domain setup. Report these numbers as "synthetic
                     corpus, not real open-domain retrieval" -- swap in a real
                     Wikipedia passage index (see README) before citing these
                     as comparable to the real_context datasets.
"""
from __future__ import annotations
from typing import List, Dict, Tuple


def load_natural_questions(n: int = 300) -> Tuple[List[Dict], str, str]:
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/nq_open", split=f"validation[:{n}]")
    out = []
    for row in ds:
        out.append({"question": row["question"], "answers": row["answer"], "context": ""})
    return out, "nq", "synthetic_qa_pool"


def load_hotpotqa(n: int = 300) -> Tuple[List[Dict], str, str]:
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split=f"validation[:{n}]", trust_remote_code=True)
    out = []
    for row in ds:
        ctx = " ".join(" ".join(sents) for sents in row["context"]["sentences"])
        out.append({"question": row["question"], "answers": [row["answer"]], "context": ctx})
    return out, "hotpotqa", "real_context"


def load_triviaqa(n: int = 300) -> Tuple[List[Dict], str, str]:
    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=f"validation[:{n}]")
    out = []
    for row in ds:
        aliases = row["answer"]["aliases"] + [row["answer"]["value"]]
        out.append({"question": row["question"], "answers": aliases, "context": ""})
    return out, "triviaqa", "synthetic_qa_pool"


def load_bioasq(n: int = 200, local_path: str = None) -> Tuple[List[Dict], str, str]:
    """BioASQ requires registration to download officially; if you have a
    local JSON export, pass local_path (expects BioASQ task-B format).
    Otherwise this falls back to a small PubMedQA subset as a stand-in
    biomedical QA set, so the pipeline still runs end to end -- but the
    source_label makes that substitution explicit rather than silent."""
    if local_path:
        import json
        with open(local_path) as f:
            data = json.load(f)
        out = []
        for q in data.get("questions", [])[:n]:
            snippets = " ".join(s.get("text", "") for s in q.get("snippets", []))
            answers = q.get("exact_answer", [])
            if isinstance(answers, list) and answers and isinstance(answers[0], list):
                answers = [a for group in answers for a in group]
            out.append({"question": q["body"], "answers": answers or [""], "context": snippets})
        return out, "bioasq", "real_context"
    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split=f"train[:{n}]")
    out = []
    for row in ds:
        ctx = " ".join(row["context"]["contexts"])
        out.append({"question": row["question"], "answers": [row["long_answer"]], "context": ctx})
    return out, "bioasq_pubmedqa_fallback", "real_context"


DATASET_LOADERS = {
    "nq": load_natural_questions,
    "hotpotqa": load_hotpotqa,
    "triviaqa": load_triviaqa,
    "bioasq": load_bioasq,
}
