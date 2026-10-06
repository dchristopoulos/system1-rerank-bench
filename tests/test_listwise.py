import io
import json
import unittest
from types import SimpleNamespace

from rerank_bench.arms import collect, luna_ranker, parse_ranking

ALIASES = {"P1": "a", "P2": "b", "P3": "c"}


class ParseRankingTests(unittest.TestCase):
    def test_complete_ranking_maps_aliases_in_listed_order(self):
        ranked, repairs = parse_ranking('{"ranking": ["P3", "P1", "P2"]}', ALIASES, ["a", "b", "c"])
        self.assertEqual(ranked, ["c", "a", "b"])
        self.assertEqual(repairs, {"unknown_ids": 0, "repeated_ids": 0, "omitted_passages": 0})

    def test_omitted_passages_follow_in_retrieval_order_and_bad_ids_are_counted(self):
        ranked, repairs = parse_ranking('{"ranking": ["P2", "P9", "P2"]}', ALIASES, ["a", "b", "c"])
        self.assertEqual(ranked, ["b", "a", "c"])
        self.assertEqual(repairs, {"unknown_ids": 1, "repeated_ids": 1, "omitted_passages": 2})

    def test_unusable_outputs_are_rejected(self):
        for text in ['{"ranking": []}', '{"ranking": ["P9"]}', '{"order": ["P1"]}', '["P1"]', "P1"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_ranking(text, ALIASES, ["a", "b", "c"])


class Stream:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __iter__(self):
        return iter(self.events)


def completed(text):
    return {
        "type": "response.completed",
        "response": {
            "status": "completed",
            "usage": {"input_tokens": 9, "output_tokens": 1, "total_tokens": 10},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
        },
    }


QUERY = SimpleNamespace(id="q", question="Which GPU?")
CORPUS = {"a": "reviews", "b": "A100"}
LUNA = {"arms": ["luna"], "model": "m", "timeout_seconds": 5, "reasoning_effort": "low"}


def luna_client(streams, calls):
    def create(**request):
        calls.append(request)
        return Stream(streams[len(calls) - 1])

    return SimpleNamespace(responses=SimpleNamespace(create=create))


class CollectTests(unittest.TestCase):
    def run_luna(self, streams, questions):
        calls, output = [], io.StringIO()
        cohort = [
            (SimpleNamespace(id=f"q{n}", question="Which GPU?"), ["a", "b"])
            for n in range(questions)
        ]
        collect("luna", cohort, luna_ranker(luna_client(streams, calls), LUNA, CORPUS), output)
        return [json.loads(line) for line in output.getvalue().splitlines()], calls

    def test_ranking_is_saved_and_references_never_enter_the_request(self):
        rows, calls = self.run_luna([[completed('{"ranking": ["P2", "P1"]}')]], 1)
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual([identity for identity, _ in rows[0]["ranking"]], ["b", "a"])
        self.assertEqual(rows[0]["usage"]["input_tokens"], 9)
        self.assertEqual(
            calls[0]["input"][0]["content"],
            "Question: Which GPU?\n\nPassages:\n\n[P1]\nreviews\n\n[P2]\nA100",
        )

    def test_three_failures_in_a_row_stop_collection_and_keep_every_row(self):
        failed = [{"type": "response.failed"}]
        rows, calls = self.run_luna([failed, failed, failed], 4)
        self.assertEqual([row["status"] for row in rows], ["failed"] * 3 + ["not_submitted"])
        self.assertEqual(len(calls), 3)

    def test_a_success_resets_the_failure_count(self):
        failed, good = [{"type": "response.failed"}], [completed('{"ranking": ["P1"]}')]
        rows, _ = self.run_luna([failed, failed, good, failed], 4)
        self.assertEqual([row["status"] for row in rows], ["failed", "failed", "ok", "failed"])


if __name__ == "__main__":
    unittest.main()
