"""
FastAPI service exposing SOAR-RAG as a live endpoint -- good for a demo
link on your resume/portfolio. Run with:
    uvicorn app.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from fastapi.middleware.cors import CORSMiddleware

from src.knowledge_base import KnowledgeBase
from src.generation import LLMGenerator
from src.pipeline import SOARRAGPipeline, BaselineRAGPipeline

app = FastAPI(title="SOAR-RAG", description="Self-Optimizing Attention-Guided RAG")

# Dev CORS: allow the local Vite frontend (http://localhost:5173) to call this
# API directly. Tighten allow_origins before deploying anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state = {"kb": None, "generator": None, "pipeline": None, "baseline": None}


class IngestRequest(BaseModel):
    texts: List[str]
    sources: Optional[List[str]] = None


class QueryRequest(BaseModel):
    query: str
    verbose: bool = False


@app.on_event("startup")
def startup():
    # Loaded lazily via /ingest in typical use; kept out of startup so the
    # API boots fast and you choose the model/knowledge base at request time.
    pass


@app.post("/ingest")
def ingest(req: IngestRequest, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
           device: str = None, quantize: str = None):
    import torch as _torch
    if device is None:
        device = "cuda" if _torch.cuda.is_available() else "cpu"
    kb = KnowledgeBase(device=device)
    kb.build(req.texts, sources=req.sources)
    generator = LLMGenerator(model_name=model_name, device=device, quantize=quantize)
    _state["kb"], _state["generator"] = kb, generator
    _state["pipeline"] = SOARRAGPipeline(kb, generator)
    _state["baseline"] = BaselineRAGPipeline(kb, generator, top_k=5)
    return {"status": "ok", "num_chunks": len(kb.documents)}

import math

def _clean_float(val):
    if val is None:
        return None
    try:
        fval = float(val)
        return None if math.isnan(fval) else fval
    except (ValueError, TypeError):
        return None

def _serialize(result, kb):
    return {
        "response": result.final_response,
        "converged_reason": result.converged_reason,
        "total_latency_s": result.total_latency_s,
        "context_compression_ratio": result.context_compression_ratio,
        "trace": [
            {
                "iteration": t.iteration, "query_used": t.query_used,
                "retrieved_doc_ids": t.retrieved_doc_ids,
                "retrieved_texts": [kb.documents[i].text for i in t.retrieved_doc_ids],
                "response": t.response,
                "confidence": _clean_float(t.confidence), "attention_entropy": _clean_float(t.attention_entropy),
                "top_k": t.top_k, "latency_s": t.latency_s,
                "doc_attention_scores": {k: _clean_float(v) for k, v in t.doc_attention_scores.items()} if hasattr(t, 'doc_attention_scores') and t.doc_attention_scores else {},
            } for t in result.trace
        ],
    }


@app.post("/query")
def query(req: QueryRequest):
    """Runs SOAR-RAG only."""
    if _state["pipeline"] is None:
        return {"error": "Call /ingest first to build the knowledge base."}
    result = _state["pipeline"].run(req.query, verbose=req.verbose)
    return _serialize(result, _state["kb"])


@app.post("/compare")
def compare(req: QueryRequest):
    """Runs baseline (single-shot) RAG and SOAR-RAG on the same query so the
    frontend can show the attention-feedback loop's effect side by side."""
    if _state["pipeline"] is None:
        return {"error": "Call /ingest first to build the knowledge base."}
    baseline_result = _state["baseline"].run(req.query)
    soar_result = _state["pipeline"].run(req.query, verbose=req.verbose)
    return {
        "baseline": _serialize(baseline_result, _state["kb"]),
        "soar": _serialize(soar_result, _state["kb"]),
    }


@app.get("/health")
def health():
    return {"status": "ok", "kb_loaded": _state["kb"] is not None}
