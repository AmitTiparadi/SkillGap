"""Short-lived local embedding worker. Resume text travels only over local pipes."""
from dataclasses import replace
import json
from pathlib import Path
import sys

from skillgap.config import Settings
from skillgap.db import Database
from skillgap.embeddings import EmbeddingService


def main():
    payload = json.load(sys.stdin)
    settings = replace(Settings(), embedding_model=payload["model"],
                       embedding_revision=payload["revision"],
                       embedding_local_files_only=payload["offline"],
                       db_path=Path(payload["db_path"]))
    service = EmbeddingService(settings, Database(settings.db_path))
    vectors = service._load_encoder().encode(payload["texts"], normalize_embeddings=True,
                                             show_progress_bar=False, convert_to_numpy=True)
    json.dump(vectors.tolist(), sys.stdout, allow_nan=False)


if __name__ == "__main__":
    main()
