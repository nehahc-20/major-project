"""
Query Processing Module (paper III.C): normalization, stopword removal,
semantic normalization, NER, and query embedding (Eq. 2: E_q = f_enc(Q)).
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional

_STOPWORDS = set("""a an the is are was were be been being of to in on at for
with and or but if then else so as by from this that these those it its
""".split())


@dataclass
class ProcessedQuery:
    raw: str
    normalized: str
    tokens: List[str]
    entities: List[str]
    embedding: Optional["object"] = None  # np.ndarray, set by QueryProcessor.embed


class QueryProcessor:
    def __init__(self, knowledge_base=None, use_spacy: bool = True):
        """`knowledge_base` supplies the encoder used for Eq. 2 (E_q = f_enc(Q))
        so the query and the documents live in the same embedding space."""
        self.kb = knowledge_base
        self._nlp = None
        if use_spacy:
            try:
                import spacy
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    self._nlp = None  # model not downloaded; fall back gracefully
            except ImportError:
                self._nlp = None

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9']+", text.lower())

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t not in _STOPWORDS]

    def extract_entities(self, text: str) -> List[str]:
        if self._nlp is not None:
            return [ent.text for ent in self._nlp(text).ents]
        # lightweight fallback: capitalized multi-word spans
        return re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b", text)

    def normalize(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text.strip())
        return text

    def process(self, query: str) -> ProcessedQuery:
        normalized = self.normalize(query)
        tokens = self.remove_stopwords(self.tokenize(normalized))
        entities = self.extract_entities(normalized)
        pq = ProcessedQuery(raw=query, normalized=normalized, tokens=tokens, entities=entities)
        if self.kb is not None:
            pq.embedding = self.kb.embed_query(normalized)
        return pq

    def complexity_score(self, pq: ProcessedQuery) -> float:
        """Heuristic query-complexity signal c_q used in dynamic Top-K (Eq. 11):
        combines token count, entity count, and presence of multi-hop cues
        ('and', 'compare', 'difference', wh-words + numeric constraints)."""
        n_tokens = max(len(pq.tokens), 1)
        n_entities = len(pq.entities)
        multi_hop_markers = sum(
            1 for w in ["compare", "difference", "relationship", "both", "and", "versus"]
            if w in pq.normalized.lower()
        )
        raw = 0.5 * min(n_tokens / 12, 1.0) + 0.35 * min(n_entities / 3, 1.0) + 0.15 * min(multi_hop_markers / 2, 1.0)
        return float(min(max(raw, 0.0), 1.0))

    def expand(self, query: str, delta_terms: List[str]) -> str:
        """Q' = Q + delta_Q (Eq. 10)."""
        if not delta_terms:
            return query
        return query.strip() + " " + " ".join(delta_terms)
