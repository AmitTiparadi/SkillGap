import unittest
from unittest.mock import Mock

from skillgap.config import Settings
from skillgap.graph.nodes import Nodes
from skillgap.graph.workflow import build_workflow
from skillgap.llm import JSONResult, SupportDecision


class RetryTests(unittest.TestCase):
    def run_graph(self, limit):
        llm = Mock()
        llm.structured.return_value = JSONResult(SupportDecision(
            supports=True, citation="Invented line", explanation="Invalid covered claim"))
        embeddings = Mock()
        embeddings.search.return_value = [{"line": "Built Python APIs", "score": 0.9}]
        nodes = Nodes(Settings(max_retries=limit), Mock(), llm, embeddings)
        requirement = {"id": "0:0", "job_id": 1, "job_index": 0,
                       "text": "Python", "canonical_key": "python",
                       "importance": "required", "source_quote": "Python"}
        nodes.parse = lambda state: {"requirements": [requirement], "lines": ["Built Python APIs"],
            "evidence": {}, "grades": {}, "retry_counters": {}, "retry_pending": [], "warnings": []}
        nodes.rank = lambda state: {"ranked_gaps": []}
        nodes.recommend = lambda state: {"recommendations": []}
        nodes.report = lambda state: {"report": {"grades": state["grades"]}}
        result = build_workflow(nodes).invoke({})
        return result, embeddings, llm

    def test_two_retries_then_partial_and_finish(self):
        state, embeddings, llm = self.run_graph(2)
        self.assertEqual(state["retry_counters"]["0:0"], 2)
        self.assertEqual(state["grades"]["0:0"]["category"], "partial")
        self.assertEqual(state["retry_pending"], [])
        self.assertEqual(llm.structured.call_count, 3)
        self.assertEqual([call.args[2] for call in embeddings.search.call_args_list], [3, 6, 9])

    def test_zero_retries_still_finishes(self):
        state, embeddings, llm = self.run_graph(0)
        self.assertEqual(embeddings.search.call_count, 1)
        self.assertEqual(state["retry_pending"], [])
