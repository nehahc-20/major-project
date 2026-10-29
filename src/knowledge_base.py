"""
Knowledge Base construction: semantic chunking, embedding, FAISS indexing.
Corresponds to paper section III.K (Structure of External Knowledge Base).
"""
from __future__ import annotations
import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


@dataclass
class Document:
    doc_id: int
    text: str
    source: Optional[str] = None
    metadata: dict = field(default_factory=dict)


def semantic_chunk(text: str, max_chars: int = 500, overlap: int = 50) -> List[str]:
    """Sentence-aware chunking: splits on sentence boundaries, then packs
    sentences into chunks up to max_chars, with a small overlap to preserve
    context across chunk edges (paper: 'semantic chunking')."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current, current_len = [], [], 0
    for sent in sentences:
        if not sent:
            continue
        if current_len + len(sent) > max_chars and current:
            chunk_text = " ".join(current)
            chunks.append(chunk_text)
            tail = chunk_text[-overlap:] if overlap else ""
            current, current_len = ([tail] if tail else []), len(tail)
        current.append(sent)
        current_len += len(sent)
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


class KnowledgeBase:
    """FAISS index (inner-product over L2-normalized embeddings == cosine,
    matches Eq. 3) plus the Document store used to map retrieval hits back
    to text and to attention spans."""

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu", query_cache_size: int = 256):
        if SentenceTransformer is None:
            raise ImportError("pip install sentence-transformers")
        self.embedding_model_name = embedding_model
        self.encoder = SentenceTransformer(embedding_model, device=device)
        self.dim = self.encoder.get_sentence_embedding_dimension()
        self.index = None
        self.documents: List[Document] = []
        # SOARRAGPipeline re-embeds the same or near-identical query text
        # across iterations (original query, expanded query re-derived
        # each loop, etc.), and eval runs frequently re-issue queries with
        # small variations -- an LRU cache on exact-string queries avoids
        # redundant encoder forward passes for the (common) exact-repeat case.
        self._query_cache_size = query_cache_size
        self._query_cache: "dict[str, np.ndarray]" = {}
        self._query_cache_order: List[str] = []

    def build(self, texts: List[str], sources: Optional[List[str]] = None,
              chunk: bool = True, batch_size: int = 64, use_ivf_above: int = 5000):
        if faiss is None:
            raise ImportError("pip install faiss-cpu (or faiss-gpu)")
        docs: List[Document] = []
        for i, t in enumerate(texts):
            src = sources[i] if sources else None
            pieces = semantic_chunk(t) if chunk else [t]
            for p in pieces:
                docs.append(Document(doc_id=len(docs), text=p, source=src))
        self.documents = docs
        embeddings = self.encoder.encode(
            [d.text for d in docs], batch_size=batch_size,
            show_progress_bar=True, normalize_embeddings=True,
        ).astype("float32")

        n = len(docs)
        if n > use_ivf_above:
            # IndexFlatIP is exact but O(n) per query -- fine up to a few
            # thousand chunks, but scanning tens of thousands of vectors
            # per retrieval call gets slow. IVF trades a small amount of
            # recall for large speedups on bigger corpora (real Wikipedia-
            # scale KBs) by only searching a subset of clusters.
            nlist = max(int(n ** 0.5), 8)
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
            self.index.nprobe = min(16, nlist)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        return self

    def embed_query(self, query: str) -> np.ndarray:
        cached = self._query_cache.get(query)
        if cached is not None:
            return cached
        emb = self.encoder.encode([query], normalize_embeddings=True).astype("float32")
        if len(self._query_cache) >= self._query_cache_size:
            oldest = self._query_cache_order.pop(0)
            self._query_cache.pop(oldest, None)
        self._query_cache[query] = emb
        self._query_cache_order.append(query)
        return emb

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.index is None:
            raise RuntimeError("KnowledgeBase.build() must be called first")
        sims, idxs = self.index.search(query_embedding, top_k)
        results = []
        for sim, idx in zip(sims[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.documents[idx], float(sim)))
        return results

    def save(self, path: str):
        import pickle
        faiss.write_index(self.index, path + ".faiss")
        with open(path + ".docs.pkl", "wb") as f:
            pickle.dump({"documents": self.documents, "model": self.embedding_model_name}, f)

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        import pickle
        with open(path + ".docs.pkl", "rb") as f:
            meta = pickle.load(f)
        kb = cls(embedding_model=meta["model"], device=device)
        kb.documents = meta["documents"]
        kb.index = faiss.read_index(path + ".faiss")
        return kb
