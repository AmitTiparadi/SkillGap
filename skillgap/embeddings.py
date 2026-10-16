"""Local sentence embeddings and deterministic cosine retrieval."""

import math
import json
import subprocess
import sys
import os
from typing import Protocol
from pathlib import Path

from skillgap.config import Settings
from skillgap.db import Database, cache_key, input_hash


class Encoder(Protocol):
    def encode(self, sentences: list[str], **kwargs): ...


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Embedding vectors must have matching nonzero dimensions")
    if not all(math.isfinite(value) for value in left + right):
        raise ValueError("Embedding vectors must be finite")
    denominator = math.sqrt(sum(x * x for x in left) * sum(y * y for y in right))
    if denominator == 0:
        return 0.0
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(left, right)) / denominator))


class EmbeddingService:
    def __init__(self, settings: Settings, database: Database, encoder: Encoder | None = None):
        self.settings = settings
        self.database = database
        self.encoder = encoder
        self.model_key = cache_key({"model": settings.embedding_model,
                                    "revision": settings.embedding_revision,
                                    "normalization": "l2", "implementation": "1"})

    def _load_encoder(self) -> Encoder:
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer

            prefetched = self.database.path.parent / "models" / "prefetched" / self.model_key
            local_snapshot = (prefetched / "modules.json").is_file()
            self.encoder = SentenceTransformer(
                str(prefetched) if local_snapshot else self.settings.embedding_model,
                revision=None if local_snapshot else self.settings.embedding_revision,
                local_files_only=local_snapshot or self.settings.embedding_local_files_only,
                cache_folder=str(self.database.path.parent / "models"),
                trust_remote_code=False,
                device="cpu",
            )
        return self.encoder

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode cache misses as one batch, preserving the original order."""
        unique = list(dict.fromkeys(texts))
        vectors = {text: self.database.get_embedding(input_hash(text), self.model_key)
                   for text in unique}
        missing = [text for text in unique if vectors[text] is None]
        if missing:
            if self.encoder is not None:
                encoded = self.encoder.encode(missing, normalize_embeddings=True,
                                              show_progress_bar=False, convert_to_numpy=True)
            else:
                # PyTorch remains resident after unloading an encoder. A short
                # local worker releases all its memory before Ollama grading.
                # Batch all cache misses so one analysis needs only one worker.
                payload = {"texts": missing, "model": self.settings.embedding_model,
                           "revision": self.settings.embedding_revision,
                           "offline": self.settings.embedding_local_files_only,
                           "db_path": str(self.database.path)}
                process = subprocess.run(
                    [sys.executable, "-m", "skillgap.embedding_worker"],
                    input=json.dumps(payload), capture_output=True, text=True,
                    encoding="utf-8", timeout=self.settings.embedding_timeout_seconds,
                    cwd=str(Path(__file__).resolve().parent.parent),
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                if process.returncode:
                    raise RuntimeError("Local embedding worker failed. Check model files, dependencies, and available memory.")
                encoded = json.loads(process.stdout)
            if len(encoded) != len(missing):
                raise ValueError("Encoder returned an unexpected vector count")
            for text, vector in zip(missing, encoded):
                values = [float(value) for value in vector]
                cosine_similarity(values, values)  # Validate before persistence.
                self.database.save_embedding(input_hash(text), self.model_key, values)
                vectors[text] = values
        return [vectors[text] for text in texts]

    def search(self, query: str, lines: list[str], top_k: int) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not lines:
            return []
        vectors = self.embed([query, *lines])
        candidates = [{"line": line, "score": cosine_similarity(vectors[0], vector)}
                      for line, vector in zip(lines, vectors[1:])]
        return sorted(candidates, key=lambda item: -item["score"])[:top_k]
