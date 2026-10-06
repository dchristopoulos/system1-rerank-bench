"""Rerank hybrid candidates on a BEIR dataset and report standard ranking metrics.

`run` retrieves candidates for every test query and scores them with the local rerankers.
`report` reads one or more rankings files and computes nDCG@10, recall@10 and MRR@10.
Only query text and document title/text reach a model; qrels and query metadata never do.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import sqlite3
import time
from pathlib import Path
from statistics import mean, median, quantiles

from rerank_bench.retrieval import (
    BGE_MODEL,
    BGE_QUERY_PREFIX,
    BGE_RERANKER,
    BGE_REVISION,
    MINILM_RERANKER,
    Hit,
    literal_query,
    reciprocal_rank_fusion,
)

DEPTH = 50


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(folder: Path):
    """The corpus, the test queries with at least one positive judgment, and graded qrels."""
    corpus = {}
    for line in (folder / "corpus.jsonl").read_text().splitlines():
        row = json.loads(line)
        corpus[row["_id"]] = (row.get("title", "") + " " + row["text"]).strip()
    qrels = {}
    with (folder / "qrels" / "test.tsv").open() as file:
        for row in csv.DictReader(file, delimiter="\t"):
            if int(row["score"]) > 0:
                qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    queries = {}
    for line in (folder / "queries.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["_id"] in qrels:
            queries[row["_id"]] = row["text"]
    if queries.keys() != qrels.keys() or any(d not in corpus for q in qrels.values() for d in q):
        raise ValueError("Judgments, queries and corpus do not line up")
    return corpus, dict(sorted(queries.items())), qrels


def ranking_metrics(ranked: list[str], judged: dict[str, int], *, k: int = 10) -> dict:
    """nDCG with linear graded gain (trec_eval ndcg_cut), recall and reciprocal rank at k."""
    if len(set(ranked)) != len(ranked):
        raise ValueError("A ranking cannot repeat a document")
    if not judged or any(type(grade) is not int or grade < 1 for grade in judged.values()):
        raise ValueError("A query needs at least one positive integer judgment")
    top = ranked[:k]
    dcg = sum(judged.get(doc, 0) / math.log2(rank + 1) for rank, doc in enumerate(top, 1))
    ideal = sum(
        grade / math.log2(rank + 1)
        for rank, grade in enumerate(sorted(judged.values(), reverse=True)[:k], 1)
    )
    first = next((rank for rank, doc in enumerate(top, 1) if doc in judged), None)
    return {
        "ndcg": dcg / ideal,
        "recall": sum(doc in judged for doc in top) / len(judged),
        "reciprocal_rank": 1 / first if first else 0.0,
    }


def paired_difference(left: list[float], right: list[float], *, resamples=10000, seed=0) -> dict:
    """Mean of right minus left with a paired percentile bootstrap over queries."""
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Paired comparison needs the same queries, at least two")
    deltas = [b - a for a, b in zip(left, right)]
    rng = random.Random(seed)
    draws = [mean(rng.choices(deltas, k=len(deltas))) for _ in range(resamples)]
    bounds = quantiles(draws, n=40, method="inclusive")
    return {"difference": mean(deltas), "interval_95": [bounds[0], bounds[-1]]}


def run(args):
    import numpy as np
    from sentence_transformers import CrossEncoder, SentenceTransformer

    corpus, queries, _ = load_dataset(args.dataset)
    ids = list(corpus)
    lexical = sqlite3.connect(":memory:")
    lexical.execute(
        "CREATE VIRTUAL TABLE docs USING fts5(id UNINDEXED, body, tokenize='unicode61')"
    )
    lexical.executemany("INSERT INTO docs VALUES (?, ?)", corpus.items())
    embedder = SentenceTransformer(
        BGE_MODEL,
        revision=BGE_REVISION,
        device=args.device,
        cache_folder=str(args.embedding_cache),
        local_files_only=True,
    )
    limit = embedder.max_seq_length
    lengths = [
        len(t) for t in embedder.tokenizer(list(corpus.values()), truncation=False)["input_ids"]
    ]

    def embed(texts):
        return embedder.encode(
            texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True
        )

    vectors = embed(list(corpus.values()))
    query_vectors = embed([BGE_QUERY_PREFIX + text for text in queries.values()])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "recipe": "beir-screen-v1",
        "dataset": args.dataset.name,
        "inputs_sha256": {
            name: sha(args.dataset / name)
            for name in ["corpus.jsonl", "queries.jsonl", "qrels/test.tsv"]
        },
        "documents": len(corpus),
        "queries": len(queries),
        "depth": DEPTH,
        "retrieval": "SQLite FTS5 BM25 (unicode61, OR terms) + BGE-small exact cosine, RRF constant 60",
        "embedder": [BGE_MODEL, BGE_REVISION],
        "documents_over_embedder_limit": sum(length > limit for length in lengths),
        "rerankers": {"minilm": MINILM_RERANKER, "bge": BGE_RERANKER},
        "reranker_max_length": 512,
        "truncation": "Embedder and rerankers truncate at 512 tokens, as in standard BEIR runs.",
        "device": args.device,
        "script_sha256": sha(Path(__file__)),
    }
    candidates = {}
    with (args.output_dir / "rankings.jsonl").open("x") as output:

        def save(arm, query_id, ranked, seconds=0.0):
            row = {"arm": arm, "query_id": query_id, "seconds": seconds, "ranking": ranked}
            output.write(json.dumps(row) + "\n")

        for (query_id, text), vector in zip(queries.items(), query_vectors, strict=True):
            rows = lexical.execute(
                "SELECT id, bm25(docs) AS s FROM docs WHERE docs MATCH ? ORDER BY s, id LIMIT ?",
                (literal_query(text), DEPTH),
            ).fetchall()
            bm25 = [Hit(doc, score) for doc, score in rows]
            scores = vectors @ vector
            order = sorted(range(len(ids)), key=lambda row: (-float(scores[row]), ids[row]))[:DEPTH]
            dense = [Hit(ids[row], float(scores[row])) for row in order]
            hybrid = reciprocal_rank_fusion(bm25, dense, constant=60, limit=DEPTH)
            for arm, hits in [("bm25", bm25), ("dense", dense), ("hybrid", hybrid)]:
                save(arm, query_id, [hit.doc_id for hit in hits])
            candidates[query_id] = [hit.doc_id for hit in hybrid]
        for arm, (name, revision) in [("minilm", MINILM_RERANKER), ("bge", BGE_RERANKER)]:
            model = CrossEncoder(
                name,
                revision=revision,
                device=args.device,
                cache_folder=str(args.reranker_cache),
                max_length=512,
                trust_remote_code=False,
                local_files_only=True,
            )
            first = next(iter(queries))
            model.predict(
                [(queries[first], corpus[doc]) for doc in candidates[first][:16]]
            )  # warmup
            for query_id, docs in candidates.items():
                started = time.perf_counter()
                scores = np.asarray(
                    model.predict([(queries[query_id], corpus[doc]) for doc in docs], batch_size=16)
                ).reshape(-1)
                if not np.isfinite(scores).all():
                    raise ValueError("Reranker returned a non-finite score")
                order = sorted(zip(docs, scores.tolist()), key=lambda item: (-item[1], item[0]))
                save(arm, query_id, [doc for doc, _ in order], time.perf_counter() - started)
            del model
            print(arm, "scored", len(candidates), flush=True)
    with (args.output_dir / "candidates.jsonl").open("x") as output:
        for query_id, docs in candidates.items():
            output.write(json.dumps({"query_id": query_id, "candidates": docs}) + "\n")
    manifest["rankings_sha256"] = sha(args.output_dir / "rankings.jsonl")
    manifest["candidates_sha256"] = sha(args.output_dir / "candidates.jsonl")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def merge_attempts(rows):
    """Final ranking per arm and query from every attempt, in file order.

    A success is never replaced by a later failure. A query whose attempts all failed keeps an
    empty ranking, which scores zero. Rows that were never submitted are ignored.
    """
    rankings, seconds, failed = {}, {}, {}
    for row in rows:
        status = row.get("status", "ok")
        if status == "not_submitted":
            continue
        done = rankings.setdefault(row["arm"], {})
        unresolved = failed.setdefault(row["arm"], set())
        if status == "ok":
            done[row["query_id"]] = [
                item[0] if isinstance(item, list) else item for item in row["ranking"]
            ]
            unresolved.discard(row["query_id"])
            if row.get("seconds"):
                seconds.setdefault(row["arm"], []).append(row["seconds"])
        elif row["query_id"] not in done:
            done[row["query_id"]] = []
            unresolved.add(row["query_id"])
    return rankings, seconds, {arm: len(queries) for arm, queries in failed.items()}


def report(args):
    _, queries, qrels = load_dataset(args.dataset)
    rankings, seconds, failed = merge_attempts(
        json.loads(line) for file in args.rankings for line in file.read_text().splitlines()
    )
    # Only queries every arm attempted are scored, so a partial arm shrinks the set for all.
    cohort = sorted(set(queries).intersection(*(set(done) for done in rankings.values())))
    per_query = {
        arm: {name: [] for name in ["ndcg", "recall", "reciprocal_rank"]} for arm in rankings
    }
    for arm, done in rankings.items():
        for query_id in cohort:
            for name, value in ranking_metrics(done[query_id], qrels[query_id]).items():
                per_query[arm][name].append(value)
    summary = {
        "dataset": args.dataset.name,
        "queries": len(cohort),
        "at_10": {arm: {k: mean(v) for k, v in m.items()} for arm, m in per_query.items()},
        "median_seconds_per_query": {arm: median(values) for arm, values in seconds.items()},
        "failed_queries": failed,
        "paired_ndcg_at_10": {
            f"{right}-minus-{left}": paired_difference(
                per_query[left]["ndcg"], per_query[right]["ndcg"]
            )
            for left, right in (pair.split(":") for pair in args.compare)
        },
        "rankings_sha256": {str(file): sha(file) for file in args.rankings},
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--dataset", type=Path, required=True)
    run_parser.add_argument("--embedding-cache", type=Path, required=True)
    run_parser.add_argument("--reranker-cache", type=Path, required=True)
    run_parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    run_parser.add_argument("--output-dir", type=Path, required=True)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--dataset", type=Path, required=True)
    report_parser.add_argument("--rankings", type=Path, nargs="+", required=True)
    report_parser.add_argument(
        "--compare", nargs="*", default=[], help="baseline:candidate arm pairs"
    )
    report_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args) if args.command == "run" else report(args)


if __name__ == "__main__":
    main()
