"""Small, grounded resume requests for local models with limited context.

The original resume lines remain the evidence used by the match/grade nodes.
Chunking only changes how the optional structured inventory is collected.
"""
from dataclasses import dataclass
import json

from skillgap.config import Settings
from skillgap.llm import LLMClient, ResumeFact, prompt
from skillgap.parsers import resume_lines


def resume_chunks(lines: list[str], budget: int) -> list[list[dict]]:
    """Bound serialized UTF-8 input, including line IDs and JSON overhead.

    Long paragraphs can span requests, but keep their original line ID. No
    text is dropped, and citations always resolve to the original full line.
    """
    if budget < 256:
        raise ValueError("Chunk budget must be at least 256 bytes")
    chunks, current, used = [], [], 2
    for line_id, line in enumerate(lines):
        remaining = line
        while remaining:
            low, high = 1, min(len(remaining), budget - 100)
            while low < high:
                middle = (low + high + 1) // 2
                if len(remaining[:middle].encode("utf-8")) <= budget - 100:
                    low = middle
                else:
                    high = middle - 1
            piece, remaining = remaining[:low], remaining[low:]
            unit = {"line_id": line_id, "text": piece}
            size = len(json.dumps(unit, ensure_ascii=False).encode("utf-8")) + 1
            # Quotes/backslashes also consume serialized bytes. Split again
            # when escaping makes an otherwise short fragment exceed budget.
            while size + 2 > budget:
                cut = max(1, len(piece) // 2)
                remaining = piece[cut:] + remaining
                piece = piece[:cut]
                unit["text"] = piece
                size = len(json.dumps(unit, ensure_ascii=False).encode("utf-8")) + 1
            if current and used + size > budget:
                chunks.append(current)
                current, used = [], 2
            current.append(unit)
            used += size
    if current:
        chunks.append(current)
    return chunks


@dataclass
class ResumeExtraction:
    entries: list[dict]
    total_chunks: int
    failed_chunks: int
    reasons: list[str]


def extract_resume_entries(text: str, llm: LLMClient, settings: Settings) -> ResumeExtraction:
    lines = resume_lines(text)
    chunks = resume_chunks(lines, settings.resume_chunk_bytes)
    entries, reasons, failed, seen = [], [], 0, set()
    for chunk in chunks:
        def validate(items):
            for item in items:
                # Validate against the supplied fragment, not another chunk.
                supplied = [unit["text"] for unit in chunk if unit["line_id"] == item.line_id]
                name = " ".join(item.name.casefold().split())
                if not name or not any(name in " ".join(part.casefold().split()) for part in supplied):
                    raise ValueError("Resume entry is not grounded in its supplied line")
            return items

        result = llm.structured(
            prompt("parse_resume", {"lines": chunk, "maximum_entries": settings.resume_entries_per_chunk}),
            list[ResumeFact], [], validate, max_items=settings.resume_entries_per_chunk,
        )
        if result.used_fallback:
            failed += 1
            reasons.append(result.failure_reason or "validation_failed")
            continue
        # Recheck injected/mock clients too; never index an unvalidated line ID.
        for item in validate(result.value):
            key = (item.kind, item.name.casefold(), item.line_id)
            if key not in seen:
                seen.add(key)
                entries.append({"kind": item.kind, "name": item.name,
                                "source_quote": lines[item.line_id]})
    return ResumeExtraction(entries, len(chunks), failed, sorted(set(reasons)))
