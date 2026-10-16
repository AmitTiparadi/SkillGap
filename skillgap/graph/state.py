"""Typed state: inputs and artifacts, never open database/model connections."""
from typing import Literal, TypedDict


class Requirement(TypedDict):
    id: str
    job_id: int
    job_index: int
    text: str
    canonical_key: str
    importance: Literal["required", "preferred"]
    source_quote: str


class Candidate(TypedDict):
    line: str
    score: float


class Grade(TypedDict):
    category: Literal["covered", "partial", "missing"]
    score: float
    citation: str | None
    explanation: str
    invalid_covered_citation: bool


class SkillGapState(TypedDict, total=False):
    resume_bytes: bytes
    resume_filename: str
    job_descriptions: list[str]
    resume_text: str
    resume_entries: list[dict]
    resume_id: int | None
    job_ids: list[int]
    requirements: list[Requirement]
    lines: list[str]
    evidence: dict[str, list[Candidate]]
    grades: dict[str, Grade]
    retry_counters: dict[str, int]
    retry_pending: list[str]
    active_requirements: list[str]
    ranked_gaps: list[dict]
    recommendations: list[dict]
    warnings: list[str]
    incomplete: bool
    report: dict
    report_id: int | None
