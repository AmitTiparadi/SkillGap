from pathlib import Path
from uuid import uuid4
import unittest
from unittest.mock import Mock, patch
from filelock import FileLock
from langsmith.run_helpers import get_tracing_context

from skillgap.config import PROJECT_ROOT, Settings
from skillgap.db import Database
from skillgap.graph.nodes import Nodes
from skillgap.graph.workflow import analyze
from skillgap.llm import JSONResult, JobRequirement, Recommendation, ResumeEntry, SupportDecision


class WorkflowTests(unittest.TestCase):
    def test_concurrent_analysis_is_rejected_before_model_calls(self):
        path = PROJECT_ROOT / "data" / "test-temp" / f"{uuid4().hex}.sqlite3"
        lock_path = Path(str(path) + ".analysis.lock")
        self.addCleanup(lock_path.unlink, missing_ok=True)
        nodes = Nodes(Settings(db_path=path), Mock(), Mock(), Mock())
        with FileLock(str(lock_path)):
            with self.assertRaisesRegex(ValueError, "Another SkillGap analysis"):
                analyze(b"Python", "resume.txt", ["Python"], nodes=nodes)
        nodes.llm.structured.assert_not_called()

    def test_report_citations_recommendations_and_storage(self):
        path = PROJECT_ROOT / "data" / "test-temp" / f"{uuid4().hex}.sqlite3"
        for suffix in ("", "-wal", "-shm"):
            self.addCleanup(Path(str(path) + suffix).unlink, missing_ok=True)
        self.addCleanup(Path(str(path) + ".analysis.lock").unlink, missing_ok=True)
        database = Database(path)
        database.initialize()
        llm = Mock()
        line = "Built Python APIs"
        def respond(message, schema, fallback, validator=None):
            self.assertFalse(get_tracing_context()["enabled"])
            if schema == list[ResumeEntry]:
                value = [ResumeEntry(kind="experience", name="Python", source_quote=line)]
            elif schema == list[JobRequirement]:
                value = [JobRequirement(text=key, canonical_key=key.lower(), importance="required", source_quote=key)
                         for key in ["Python", "Docker"]]
            elif schema == SupportDecision:
                value = SupportDecision(supports=True, citation=line, explanation="Exact evidence")
            else:
                value = Recommendation(kind="project", title="Containerize an API", what_to_do="Build a Dockerfile and run API tests inside the container.",
                    why_it_helps="Shows Docker practice", estimated_hours=8, resource_type="Official Docker tutorial",
                    resume_example="Containerized an API after completing the project.")
            if validator:
                value = validator(value)
            return JSONResult(value)
        llm.structured.side_effect = respond
        embeddings = Mock()
        embeddings.search.side_effect = lambda query, lines, k: [{"line": line, "score": 0.9 if query == "Python" else 0.1}]
        settings = Settings(db_path=path)
        nodes = Nodes(settings, database, llm, embeddings)
        with patch.dict("os.environ", {"LANGSMITH_TRACING": "true"}):
            report = analyze(line.encode(), "resume.txt", ["Python Docker"], nodes=nodes)
        self.assertEqual(report["covered"][0]["citation"], line)
        self.assertEqual(report["missing"][0]["canonical_key"], "docker")
        self.assertEqual(report["recommendations"][0]["gap_key"], "docker")
        self.assertFalse(report["incomplete"])
        with database.connection() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM reports").fetchone()[0], 1)

    def test_failed_job_is_visible_and_not_cached(self):
        path = PROJECT_ROOT / "data" / "test-temp" / f"{uuid4().hex}.sqlite3"
        for suffix in ("", "-wal", "-shm"):
            self.addCleanup(Path(str(path) + suffix).unlink, missing_ok=True)
        self.addCleanup(Path(str(path) + ".analysis.lock").unlink, missing_ok=True)
        database = Database(path)
        database.initialize()
        llm = Mock()
        llm.structured.return_value = JSONResult([], used_fallback=True)
        nodes = Nodes(Settings(db_path=path), database, llm, Mock())
        report = analyze(b"Built Python APIs", "resume.txt", ["Python required"], nodes=nodes)
        self.assertTrue(report["incomplete"])
        self.assertGreaterEqual(len(report["warnings"]), 2)
        with database.connection() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM job_descriptions").fetchone()[0], 0)
