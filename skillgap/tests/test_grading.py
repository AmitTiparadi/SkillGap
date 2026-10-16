import unittest

from skillgap.config import Settings
from skillgap.graph.nodes import grade_evidence
from skillgap.llm import SupportDecision


class GradeTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.line = "Built a Flask REST API with tests."
        self.lines = [self.line]

    def grade(self, score, decision=None):
        return grade_evidence([{"score": score, "line": self.line}], self.lines, self.settings, decision)

    def test_threshold_boundaries(self):
        self.assertEqual(self.grade(self.settings.low_similarity - 0.0001)["category"], "missing")
        self.assertEqual(self.grade(self.settings.low_similarity)["category"], "partial")
        self.assertEqual(self.grade(self.settings.high_similarity - 0.0001)["category"], "partial")
        self.assertEqual(self.grade(self.settings.high_similarity)["category"], "partial")
        good = SupportDecision(supports=True, citation=self.line, explanation="Exact support")
        self.assertEqual(self.grade(self.settings.high_similarity, good)["category"], "covered")

    def test_invalid_citation_is_downgraded_and_flagged(self):
        for citation in ("", "Invented production experience", self.line + " "):
            with self.subTest(citation=citation):
                decision = SupportDecision(supports=True, citation=citation, explanation="Claim")
                grade = self.grade(0.95, decision)
                self.assertEqual(grade["category"], "partial")
                self.assertTrue(grade["invalid_covered_citation"])

    def test_high_score_cannot_override_negative_support(self):
        decision = SupportDecision(supports=False, citation=self.line, explanation="Different technology")
        self.assertEqual(self.grade(0.99, decision)["category"], "partial")

    def test_no_candidates_and_missing_have_no_support_citation(self):
        self.assertEqual(grade_evidence([], self.lines, self.settings)["category"], "missing")
        self.assertIsNone(self.grade(0.1)["citation"])

    def test_ongoing_study_cannot_prove_completed_degree(self):
        line = "Final-year Computer Science Engineering student"
        decision = SupportDecision(supports=True, citation=line, explanation="Incorrect model confirmation")
        grade = grade_evidence([{"line": line, "score": 0.9}], [line], self.settings,
                               decision, "Computer science degree")
        self.assertEqual(grade["category"], "partial")
        self.assertFalse(grade["invalid_covered_citation"])
