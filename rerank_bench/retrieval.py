"""First-stage pieces shared by the benchmark: pinned models, query parsing and fusion."""

import re
from collections import defaultdict
from dataclasses import dataclass

BGE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BGE_RERANKER = ("BAAI/bge-reranker-base", "2cfc18c9415c912f9d8155881c133215df768a70")
MINILM_RERANKER = (
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    "233902d25c440f23af6f7d6e94d2946bac0bee0a",
)


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float


def literal_query(question: str) -> str:
    """Quote complete literal terms and phrases, with explicit OR semantics."""
    if not isinstance(question, str):
        raise ValueError("The question must be text")
    terms = re.findall(r"\w+(?:[._:/-]\w+)*", question)
    unique = {}
    for term in terms:
        unique.setdefault(term.casefold(), term)
    return " OR ".join(f'"{term}"' for term in unique.values())


def reciprocal_rank_fusion(*rankings: list[Hit], constant: int = 60, limit: int = 50) -> list[Hit]:
    if type(constant) is not int or constant < 0 or type(limit) is not int or limit < 1:
        raise ValueError("Fusion requires a nonnegative rank constant and positive limit")
    scores = defaultdict(float)
    for ranking in rankings:
        unique_ids = dict.fromkeys(hit.doc_id for hit in ranking)
        for rank, doc_id in enumerate(unique_ids, start=1):
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError("Fusion requires nonempty document IDs")
            scores[doc_id] += 1 / (constant + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [Hit(doc_id, score) for doc_id, score in ordered[:limit]]
