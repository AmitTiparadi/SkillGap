import unittest

from skillgap.graph.nodes import grounded_key, rank_gaps


def req(key, job, importance="required", suffix=""):
    return {"id": f"{job}:{key}:{suffix}", "job_index": job, "text": key,
            "canonical_key": key, "importance": importance}


class RankingTests(unittest.TestCase):
    def test_frequency_then_required_and_distinct_jobs(self):
        items = [req("docker", 0, "preferred"), req("docker", 0, "preferred", "repeat"),
                 req("docker", 1, "preferred"), req("aws", 0), req("sql", 0, "preferred")]
        grades = {item["id"]: {"category": "missing"} for item in items}
        gaps = rank_gaps(items, grades)
        self.assertEqual([gap["canonical_key"] for gap in gaps], ["docker", "aws", "sql"])
        self.assertEqual(gaps[0]["frequency"], 2)

    def test_covered_requirements_are_not_gaps(self):
        items = [req("python", 0), req("sql", 0)]
        grades = {items[0]["id"]: {"category": "covered"}, items[1]["id"]: {"category": "partial"}}
        self.assertEqual([gap["canonical_key"] for gap in rank_gaps(items, grades)], ["sql"])

    def test_empty_requirements(self):
        self.assertEqual(rank_gaps([], {}), [])

    def test_unrelated_model_key_cannot_merge_api_and_pandas(self):
        api = grounded_key("rest_api", "Build REST APIs with Python and Flask")
        pandas = grounded_key("rest_api", "Analyze tabular data with Python and pandas")
        self.assertNotEqual(api, pandas)
        self.assertNotEqual(grounded_key("sql", "Write SQL queries"),
                            grounded_key("sql", "Write SQL queries and design PostgreSQL databases"))
        self.assertNotEqual(grounded_key("experience", "Three years of professional experience"),
                            grounded_key("experience", "Seven years of professional experience"))
