"""Foundation checks run offline with Python's built-in unittest runner."""

from pathlib import Path
import sqlite3
import unittest
from uuid import uuid4

from skillgap.config import PROJECT_ROOT, Settings
from skillgap.db import Database, cache_key, input_hash


class FoundationTests(unittest.TestCase):
    def setUp(self):
        # Keep test artifacts inside the workspace, including in restricted sandboxes.
        test_root = PROJECT_ROOT / "data" / "test-temp"
        test_root.mkdir(parents=True, exist_ok=True)
        # Ordinary files avoid restrictive Windows ACLs created by mkdtemp.
        path = test_root / f"{uuid4().hex}.sqlite3"
        for suffix in ("", "-wal", "-shm"):
            self.addCleanup(Path(str(path) + suffix).unlink, missing_ok=True)
        self.db = Database(path)
        self.db.initialize()

    def test_local_default_and_threshold_validation(self):
        self.assertEqual(Settings().llm_provider, "ollama")
        self.assertEqual(Settings().max_retries, 2)
        with self.assertRaises(ValueError):
            Settings(low_similarity=0.8, high_similarity=0.7)
        with self.assertRaises(ValueError):
            Settings(llm_provider="api")

    def test_schema_is_idempotent(self):
        self.db.initialize()
        with self.db.connection() as connection:
            tables = {row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )}
            self.assertTrue({"resumes", "job_descriptions", "requirements",
                             "embeddings", "reports", "report_jobs"} <= tables)
            self.assertEqual(connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)

    def test_parse_cache_changes_with_model_and_reuses_identical_input(self):
        digest = input_hash(b"Python developer")
        key = cache_key({"model": "local-v1", "parser": "1"})
        first = self.db.save_resume(digest, key, "resume.txt", "Python developer", [])
        second = self.db.save_resume(digest, key, "renamed.txt", "Python developer", [])
        self.assertEqual(first, second)
        self.assertEqual(self.db.get_parse("resume", digest, key)["extraction"], [])
        changed_key = cache_key({"model": "local-v2", "parser": "1"})
        self.assertIsNone(self.db.get_parse("resume", digest, changed_key))
        self.assertEqual(cache_key({"a": 1, "b": 2}), cache_key({"b": 2, "a": 1}))

    def test_job_write_rolls_back_invalid_requirements(self):
        bad = [{"text": "Python", "canonical_key": "python",
                "importance": "invalid", "source_quote": "Python"}]
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.save_job("job-hash", "parse-key", "Python", bad)
        self.assertIsNone(self.db.get_parse("job", "job-hash", "parse-key"))

    def test_embedding_and_report_round_trip(self):
        self.db.save_embedding("line-hash", "model-a", [0.1, 0.2])
        self.assertEqual(self.db.get_embedding("line-hash", "model-a"), [0.1, 0.2])
        self.assertIsNone(self.db.get_embedding("line-hash", "model-b"))
        resume_id = self.db.save_resume("resume-hash", "parse-key", "resume.txt", "Python", [])
        requirements = [{"text": "Python", "canonical_key": "python",
                         "importance": "required", "source_quote": "Python"}]
        job_id = self.db.save_job("job-hash", "parse-key", "Python", requirements)
        self.assertEqual(len(self.db.get_requirements(job_id)), 1)
        report = {"covered": [], "partial": [], "missing": [], "recommendations": []}
        self.db.save_report("analysis-hash", resume_id, [job_id], {"provider": "ollama"}, report)
        self.assertEqual(self.db.get_report("analysis-hash"), report)
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.save_report("invalid", 9999, [job_id], {}, report)


if __name__ == "__main__":
    unittest.main()
