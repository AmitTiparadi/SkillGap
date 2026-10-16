from pathlib import Path
from uuid import uuid4
import unittest

from skillgap.config import PROJECT_ROOT, Settings
from skillgap.db import Database
from skillgap.embeddings import EmbeddingService, cosine_similarity


class PairEncoder:
    """Controlled vectors test retrieval separately from model quality."""
    def __init__(self):
        self.calls = 0

    def encode(self, sentences, **kwargs):
        self.calls += 1
        return [[1.0, 0.0] if "Python" in text else [0.0, 1.0] for text in sentences]


class EmbeddingTests(unittest.TestCase):
    def test_cosine_pairs(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1)
        self.assertEqual(cosine_similarity([1, 0], [0, 1]), 0)
        self.assertEqual(cosine_similarity([0, 0], [1, 0]), 0)
        with self.assertRaises(ValueError):
            cosine_similarity([1], [1, 2])

    def test_retrieval_and_cache(self):
        folder = PROJECT_ROOT / "data" / "test-temp"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{uuid4().hex}.sqlite3"
        for suffix in ("", "-wal", "-shm"):
            self.addCleanup(Path(str(path) + suffix).unlink, missing_ok=True)
        database = Database(path)
        database.initialize()
        encoder = PairEncoder()
        service = EmbeddingService(Settings(), database, encoder)
        lines = ["Built Python APIs", "Built Power BI dashboards"]
        self.assertEqual(service.search("Python", lines, 2)[0]["line"], lines[0])
        self.assertEqual(service.search("Power BI", lines, 2)[0]["line"], lines[1])
        calls = encoder.calls
        service.search("Python", lines, 2)
        self.assertEqual(encoder.calls, calls)
