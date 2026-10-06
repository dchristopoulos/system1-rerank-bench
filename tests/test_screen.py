import math
import unittest

from rerank_bench.screen import merge_attempts, paired_difference, ranking_metrics


class RankingMetricTests(unittest.TestCase):
    def test_graded_gain_uses_all_judged_documents_for_the_ideal(self):
        # Judged: a=2, b=1, z=1 (z never retrieved). Ranking puts b first, a third.
        result = ranking_metrics(["b", "x", "a"], {"a": 2, "b": 1, "z": 1})
        dcg = 1 / math.log2(2) + 2 / math.log2(4)
        ideal = 2 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
        self.assertAlmostEqual(result["ndcg"], dcg / ideal)
        self.assertAlmostEqual(result["ndcg"], 0.6388, places=4)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertEqual(result["reciprocal_rank"], 1.0)

    def test_documents_below_the_cutoff_earn_nothing(self):
        result = ranking_metrics(["x", "y", "a"], {"a": 1}, k=2)
        self.assertEqual(result, {"ndcg": 0.0, "recall": 0.0, "reciprocal_rank": 0.0})

    def test_an_empty_ranking_scores_zero(self):
        self.assertEqual(ranking_metrics([], {"a": 1})["ndcg"], 0.0)

    def test_malformed_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "repeat"):
            ranking_metrics(["a", "a"], {"a": 1})
        with self.assertRaisesRegex(ValueError, "judgment"):
            ranking_metrics(["a"], {})
        with self.assertRaisesRegex(ValueError, "judgment"):
            ranking_metrics(["a"], {"a": 0})


class PairedDifferenceTests(unittest.TestCase):
    def test_constant_improvement_has_a_degenerate_interval(self):
        result = paired_difference([0.1, 0.5, 0.9], [0.3, 0.7, 1.1], resamples=200)
        self.assertAlmostEqual(result["difference"], 0.2)
        for bound in result["interval_95"]:
            self.assertAlmostEqual(bound, 0.2)

    def test_unpaired_lengths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same queries"):
            paired_difference([0.1, 0.2], [0.1])


class MergeAttemptsTests(unittest.TestCase):
    def test_a_retry_repairs_a_failure_and_a_later_failure_never_replaces_a_success(self):
        rows = [
            {"arm": "x", "query_id": "q1", "status": "failed", "seconds": 9.0},
            {"arm": "x", "query_id": "q2", "status": "ok", "ranking": [["a", 0.9]], "seconds": 2.0},
            {"arm": "x", "query_id": "q3", "status": "failed"},
            {"arm": "x", "query_id": "q4", "status": "not_submitted"},
            {"arm": "x", "query_id": "q1", "status": "ok", "ranking": ["b"], "seconds": 4.0},
            {"arm": "x", "query_id": "q2", "status": "failed"},
        ]
        rankings, seconds, failed = merge_attempts(rows)
        self.assertEqual(rankings, {"x": {"q1": ["b"], "q2": ["a"], "q3": []}})
        self.assertEqual(seconds, {"x": [2.0, 4.0]})
        self.assertEqual(failed, {"x": 1})


if __name__ == "__main__":
    unittest.main()
