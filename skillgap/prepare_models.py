"""Download the pinned MiniLM weights without any resume data.

Optional bootstrap for restricted/proxied environments:
    python -m skillgap.prepare_models
Only the project's configured MiniLM snapshot is supported here; other models
use SentenceTransformer's normal loader. Downloads stream to disk, not RAM.
"""
import json
from pathlib import Path

import httpx

from skillgap.config import load_settings
from skillgap.db import Database
from skillgap.embeddings import EmbeddingService


def main():
    settings = load_settings()
    if settings.embedding_model != "sentence-transformers/all-MiniLM-L6-v2":
        raise ValueError("This optional bootstrap supports all-MiniLM-L6-v2 only")
    if settings.embedding_local_files_only:
        raise ValueError("Disable SKILLGAP_OFFLINE to download weights")
    service = EmbeddingService(settings, Database(settings.db_path))
    folder = settings.db_path.parent / "models" / "prefetched" / service.model_key
    files = ["config.json", "config_sentence_transformers.json", "sentence_bert_config.json",
             "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
             "vocab.txt", "1_Pooling/config.json", "model.safetensors", "modules.json"]
    with httpx.Client(trust_env=False, follow_redirects=True, timeout=120) as client:
        for name in files:
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_file():
                continue
            url = f"https://huggingface.co/{settings.embedding_model}/resolve/{settings.embedding_revision}/{name}"
            temporary = path.with_suffix(path.suffix + ".part")
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for chunk in response.iter_bytes():
                        output.write(chunk)
            temporary.replace(path)
            print(f"Downloaded {name}")
    print(f"Local embedding snapshot ready: {folder}")


if __name__ == "__main__":
    main()
