import unittest
from types import SimpleNamespace

from rerank_bench.arms import batched_ranker, decision_ranker, relevance

REGISTRATION = {"model": "m", "decision": {"relevant": {"type": "choice"}}}
CORPUS = {"a": "alpha", "b": "beta", "c": "gamma"}
QUERY = SimpleNamespace(id="q", question="which?")


class DecisionRankerTests(unittest.TestCase):
    def test_one_body_per_passage_ranked_by_yes_probability_then_id(self):
        seen = []

        def decide(bodies):
            seen.extend(bodies)
            return [0.2, 0.9, 0.2], {"input_tokens": 7}

        ranked, extra = decision_ranker(decide, REGISTRATION, CORPUS)(QUERY, ["c", "b", "a"])
        self.assertEqual(ranked, [["b", 0.9], ["a", 0.2], ["c", 0.2]])
        self.assertEqual(extra, {"input_tokens": 7})
        self.assertEqual(
            [body["state"] for body in seen][0], {"query": "which?", "passage": "gamma"}
        )
        self.assertTrue(all(body["questions"] == REGISTRATION["decision"] for body in seen))

    def test_missing_or_out_of_range_probabilities_are_rejected(self):
        for scores in ([0.5], [0.1, 1.2], [0.1, None]):
            with self.assertRaisesRegex(ValueError, "invalid probabilities"):
                decision_ranker(lambda bodies, s=scores: (s, {}), REGISTRATION, CORPUS)(
                    QUERY, ["a", "b"]
                )


class RelevanceTests(unittest.TestCase):
    def test_yes_probability_for_a_choice_and_true_probability_for_a_noul(self):
        answer = {"probabilities": {"no": 0.3, "yes": 0.7}}
        self.assertEqual(relevance({"type": "choice"}, answer), 0.7)
        self.assertEqual(relevance({"type": "noul"}, {"noul": 0.4}), 0.4)

    def test_expected_rubric_level_scaled_to_one(self):
        question = {"type": "score", "criteria": ["a", "b", "c", "d"]}
        # Expected level = 1*0.2 + 2*0.3 + 3*0.4 = 2.0 of a maximum 3.
        answer = {"probabilities": {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4}}
        self.assertAlmostEqual(relevance(question, answer), 2 / 3)
        self.assertAlmostEqual(relevance(question, {"score": 1.5}), 0.5)


class BatchedRankerTests(unittest.TestCase):
    REGISTRATION = {
        "model": "m",
        "passages_per_call": 2,
        "usd_per_million_input_tokens": 2.0,
        "decision": {
            "relevant": {"type": "noul", "instructions": "Is the passage relevant to the query?"}
        },
    }

    def test_groups_passages_names_each_in_its_question_and_merges_scores(self):
        bodies, replies = [], [{"p01": 0.1, "p02": 0.8}, {"p01": 0.5}]

        def post(body):
            bodies.append(body)
            answers = {alias: {"noul": p} for alias, p in replies[len(bodies) - 1].items()}
            return {"answers": answers, "usage": {"input_tokens": 1000}}

        ranked, extra = batched_ranker(post, self.REGISTRATION, CORPUS)(QUERY, ["a", "b", "c"])
        self.assertEqual(ranked, [["b", 0.8], ["c", 0.5], ["a", 0.1]])
        self.assertEqual(bodies[0]["state"]["passages"], {"p01": "alpha", "p02": "beta"})
        self.assertEqual(bodies[1]["state"]["passages"], {"p01": "gamma"})
        self.assertEqual(
            bodies[0]["questions"]["p02"]["instructions"], "Is passage p02 relevant to the query?"
        )
        # 2,000 tokens at 2 USD per million, no billed amount reported.
        self.assertEqual(extra["input_tokens"], 2000)
        self.assertAlmostEqual(extra["cost_usd"], 0.004)

    def test_billed_cost_is_preferred(self):
        def post(body):
            answers = {alias: {"noul": 0.5} for alias in body["questions"]}
            return {"answers": answers, "usage": {"input_tokens": 1000, "cost": 0.01}}

        _, extra = batched_ranker(post, self.REGISTRATION, CORPUS)(QUERY, ["a", "b", "c"])
        self.assertAlmostEqual(extra["cost_usd"], 0.02)

    def test_wording_that_cannot_name_each_passage_is_rejected(self):
        registration = {
            **self.REGISTRATION,
            "decision": {"relevant": {"instructions": "Relevant?"}},
        }
        with self.assertRaisesRegex(ValueError, "the passage"):
            batched_ranker(lambda body: {}, registration, CORPUS)

    def test_a_paid_response_without_usage_is_rejected(self):
        def post(body):
            return {"answers": {alias: {"noul": 0.5} for alias in body["questions"]}}

        registration = {**self.REGISTRATION, "max_spend_usd": 1.0}
        with self.assertRaisesRegex(ValueError, "no usage"):
            batched_ranker(post, registration, CORPUS)(QUERY, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
