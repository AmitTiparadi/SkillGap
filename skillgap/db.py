"""SQLite schema and cache helpers, using only Python's standard library.

Each operation opens a short-lived connection, suitable for Streamlit reruns.
Parameterized SQL keeps resume text out of SQL syntax. JSON retains structured
LLM output while relational requirement rows support cross-job analysis.
"""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator


SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS resumes (
    id INTEGER PRIMARY KEY,
    input_hash TEXT NOT NULL,
    parse_key TEXT NOT NULL,
    filename TEXT NOT NULL,
    plain_text TEXT NOT NULL,
    extraction_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(input_hash, parse_key)
);
CREATE TABLE IF NOT EXISTS job_descriptions (
    id INTEGER PRIMARY KEY,
    input_hash TEXT NOT NULL,
    parse_key TEXT NOT NULL,
    plain_text TEXT NOT NULL,
    extraction_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(input_hash, parse_key)
);
CREATE TABLE IF NOT EXISTS requirements (
    id INTEGER PRIMARY KEY,
    job_description_id INTEGER NOT NULL
        REFERENCES job_descriptions(id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK(position >= 0),
    requirement_text TEXT NOT NULL,
    canonical_key TEXT NOT NULL,
    importance TEXT NOT NULL CHECK(importance IN ('required', 'preferred')),
    source_quote TEXT NOT NULL,
    UNIQUE(job_description_id, position)
);
CREATE INDEX IF NOT EXISTS idx_requirements_canonical ON requirements(canonical_key);
CREATE TABLE IF NOT EXISTS embeddings (
    input_hash TEXT NOT NULL,
    model_key TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK(dimensions > 0),
    vector_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(input_hash, model_key)
);
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY,
    analysis_hash TEXT NOT NULL UNIQUE,
    resume_id INTEGER NOT NULL REFERENCES resumes(id),
    settings_json TEXT NOT NULL,
    report_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS report_jobs (
    report_id INTEGER NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    job_description_id INTEGER NOT NULL REFERENCES job_descriptions(id),
    PRIMARY KEY(report_id, job_description_id)
);
"""


def input_hash(value: bytes | str) -> str:
    """Hash exact input bytes; never log the underlying personal text."""
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def cache_key(metadata: dict[str, Any]) -> str:
    """Stable fingerprint for model, prompt, parser, or analysis settings.

    Pass only non-secret metadata: API keys never belong in cached settings.
    Include provider/model for parsing and model/revision for embeddings.
    """
    return input_hash(json.dumps(metadata, sort_keys=True, ensure_ascii=False))


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Commit successful operations and roll back failures as one unit."""
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise RuntimeError(f"Unsupported database schema version: {version}")
            # WAL lets report reads coexist with a writer. Keep all DB sidecars local.
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(SCHEMA)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def get_parse(self, kind: str, digest: str, parse_key: str) -> dict | None:
        table = self._parse_table(kind)
        with self.connection() as connection:
            row = connection.execute(
                f"SELECT * FROM {table} WHERE input_hash = ? AND parse_key = ?",
                (digest, parse_key),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["extraction"] = json.loads(result.pop("extraction_json"))
        return result

    @staticmethod
    def _parse_table(kind: str) -> str:
        # Only these hard-coded identifiers can be interpolated into SQL.
        if kind not in ("resume", "job"):
            raise ValueError("kind must be 'resume' or 'job'")
        return {"resume": "resumes", "job": "job_descriptions"}[kind]

    def save_resume(self, digest: str, parse_key: str, filename: str,
                    plain_text: str, extraction: Any) -> int:
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO resumes(input_hash, parse_key, filename, plain_text, extraction_json)
                   VALUES (?, ?, ?, ?, ?) ON CONFLICT(input_hash, parse_key) DO NOTHING""",
                (digest, parse_key, filename, plain_text, _json(extraction)),
            )
            return connection.execute(
                "SELECT id FROM resumes WHERE input_hash = ? AND parse_key = ?",
                (digest, parse_key),
            ).fetchone()[0]

    def save_job(self, digest: str, parse_key: str, plain_text: str,
                 requirements: list[dict[str, str]]) -> int:
        """Save extraction and its requirement rows atomically.

        Each validated requirement needs text, canonical_key, importance,
        and source_quote. Canonical keys will group equivalent gaps later.
        """
        with self.connection() as connection:
            cursor = connection.execute(
                """INSERT INTO job_descriptions(input_hash, parse_key, plain_text, extraction_json)
                   VALUES (?, ?, ?, ?) ON CONFLICT(input_hash, parse_key) DO NOTHING""",
                (digest, parse_key, plain_text, _json(requirements)),
            )
            job_id = connection.execute(
                "SELECT id FROM job_descriptions WHERE input_hash = ? AND parse_key = ?",
                (digest, parse_key),
            ).fetchone()[0]
            if cursor.rowcount:
                connection.executemany(
                    """INSERT INTO requirements(job_description_id, position, requirement_text,
                       canonical_key, importance, source_quote) VALUES (?, ?, ?, ?, ?, ?)""",
                    [(job_id, position, item["text"], item["canonical_key"],
                      item["importance"], item["source_quote"])
                     for position, item in enumerate(requirements)],
                )
            return job_id

    def get_requirements(self, job_id: int) -> list[dict]:
        with self.connection() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT * FROM requirements WHERE job_description_id = ? ORDER BY position",
                (job_id,),
            )]

    def get_embedding(self, digest: str, model_key: str) -> list[float] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT vector_json FROM embeddings WHERE input_hash = ? AND model_key = ?",
                (digest, model_key),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def save_embedding(self, digest: str, model_key: str, vector: list[float]) -> None:
        if not vector:
            raise ValueError("Cannot cache an empty embedding")
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO embeddings(input_hash, model_key, dimensions, vector_json)
                   VALUES (?, ?, ?, ?) ON CONFLICT(input_hash, model_key) DO NOTHING""",
                (digest, model_key, len(vector), _json(vector)),
            )

    def save_report(self, analysis_hash: str, resume_id: int, job_ids: list[int],
                    settings: dict, report: dict) -> int:
        """Persist report and job links together; settings must omit secrets."""
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO reports(analysis_hash, resume_id, settings_json, report_json)
                   VALUES (?, ?, ?, ?) ON CONFLICT(analysis_hash) DO NOTHING""",
                (analysis_hash, resume_id, _json(settings), _json(report)),
            )
            report_id = connection.execute(
                "SELECT id FROM reports WHERE analysis_hash = ?", (analysis_hash,),
            ).fetchone()[0]
            connection.executemany(
                "INSERT OR IGNORE INTO report_jobs(report_id, job_description_id) VALUES (?, ?)",
                [(report_id, job_id) for job_id in job_ids],
            )
            return report_id

    def get_report(self, analysis_hash: str) -> dict | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT report_json FROM reports WHERE analysis_hash = ?", (analysis_hash,),
            ).fetchone()
        return json.loads(row[0]) if row else None
