"""Central configuration. Importing this module never contacts a service."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Explicit settings shared by the UI and workflow.

    API mode is opt-in. The future LLM wrapper must only use api_base_url
    when llm_provider == 'api'; a local failure must never switch providers.
    """

    llm_provider: Literal["ollama", "api"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:1.5b"
    ollama_num_gpu: int = 0
    ollama_keep_alive: str = "0"
    ollama_batch_size: int = 64
    api_base_url: str = "https://api.openai.com/v1"
    api_model: str = ""
    api_key: str = field(default="", repr=False)
    # Load model weights once; subsequent inference runs on the machine.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    embedding_local_files_only: bool = False
    # Calibrated on the included MiniLM pairs, not universal probabilities.
    high_similarity: float = 0.49
    low_similarity: float = 0.30
    max_retries: int = 2
    top_n_gaps: int = 5
    match_top_k: int = 3
    retry_top_k_increment: int = 3
    structured_output_retries: int = 1
    llm_timeout_seconds: int = 120
    ollama_context_tokens: int = 8192
    max_output_tokens: int = 2048
    max_extracted_entries: int = 50
    resume_chunk_bytes: int = 1600
    resume_entries_per_chunk: int = 8
    embedding_timeout_seconds: int = 240
    max_resume_bytes: int = 5_000_000
    max_resume_characters: int = 30_000
    max_job_characters: int = 20_000
    max_jobs: int = 10
    parser_version: str = "1"
    prompt_version: str = "5"
    db_path: Path = PROJECT_ROOT / "data" / "skillgap.sqlite3"

    def __post_init__(self) -> None:
        if self.llm_provider not in ("ollama", "api"):
            raise ValueError("llm_provider must be 'ollama' or 'api'")
        if not 0 <= self.low_similarity < self.high_similarity <= 1:
            raise ValueError("Require 0 <= low_similarity < high_similarity <= 1")
        if self.max_retries < 0 or self.structured_output_retries < 0:
            raise ValueError("Retry limits must be nonnegative")
        if self.resume_chunk_bytes < 256 or self.resume_entries_per_chunk < 1:
            raise ValueError("Resume chunks require at least 256 bytes and one entry")
        if min(self.top_n_gaps, self.match_top_k, self.retry_top_k_increment,
               self.llm_timeout_seconds, self.ollama_batch_size,
               self.embedding_timeout_seconds, self.max_output_tokens,
               self.max_extracted_entries) < 1:
            raise ValueError("Counts and timeout must be positive")
        if self.llm_provider == "api" and not (self.api_model and self.api_key):
            raise ValueError("API mode requires SKILLGAP_API_MODEL and SKILLGAP_API_KEY")
        local_url = urlparse(self.ollama_base_url)
        if local_url.scheme not in ("http", "https") or local_url.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("Ollama mode requires a loopback URL to keep resumes local")
        if "cloud" in self.ollama_model.lower():
            raise ValueError("Use a local Ollama model, not a cloud model")
        if min(self.max_resume_bytes, self.max_resume_characters, self.max_job_characters, self.max_jobs) < 1:
            raise ValueError("Input limits must be positive")


def load_settings() -> Settings:
    """Read environment overrides at startup, keeping credentials out of Git."""
    return Settings(
        llm_provider=os.getenv("SKILLGAP_LLM_PROVIDER", "ollama"),
        ollama_base_url=os.getenv("SKILLGAP_OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("SKILLGAP_OLLAMA_MODEL", "qwen2.5:1.5b"),
        ollama_num_gpu=int(os.getenv("SKILLGAP_OLLAMA_NUM_GPU", "0")),
        ollama_keep_alive=os.getenv("SKILLGAP_OLLAMA_KEEP_ALIVE", "0"),
        api_base_url=os.getenv("SKILLGAP_API_BASE_URL", "https://api.openai.com/v1"),
        api_model=os.getenv("SKILLGAP_API_MODEL", ""),
        api_key=os.getenv("SKILLGAP_API_KEY", ""),
        embedding_model=os.getenv("SKILLGAP_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        embedding_revision=os.getenv("SKILLGAP_EMBEDDING_REVISION", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
        embedding_local_files_only=os.getenv("SKILLGAP_OFFLINE", "0") == "1",
        db_path=Path(os.getenv("SKILLGAP_DB_PATH", str(PROJECT_ROOT / "data" / "skillgap.sqlite3"))),
    )
