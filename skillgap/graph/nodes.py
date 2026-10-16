"""Workflow nodes with injected services, allowing offline mocked tests."""

from dataclasses import asdict
import json
import re

from skillgap.config import Settings
from skillgap.db import Database, cache_key, input_hash
from skillgap.embeddings import EmbeddingService
from skillgap.graph.state import SkillGapState
from skillgap.llm import JobRequirement, LLMClient, Recommendation, ResumeEntry, SupportDecision, prompt
from skillgap.parsers import DocumentError, extract_text, resume_lines


def canonicalize(value: str) -> str:
    """Normalize spelling punctuation, without guessing that tools are equivalent."""
    return re.sub(r"\s+", " ", re.sub(r"[_-]+", " ", value.strip().casefold()))


def grounded_key(key: str, text: str) -> str:
    """Guard grouping against invented/reused labels from small models.

    Key words must appear in the requirement (allowing simple plurals). When
    they do not, keep a normalized requirement instead of merging unrelated
    gaps. Explicit tool combinations and years remain distinct qualifiers.
    This is a conservative lexical check, not a complete skill ontology.
    """
    clean_text = re.sub(r"^(required|preferred)\s*:\s*", "", text.strip(), flags=re.I)
    def tokens(value):
        return {word.rstrip("s") for word in re.findall(r"[a-z0-9]+", value.casefold())}
    key = canonicalize(key)
    if not tokens(key) <= tokens(clean_text):
        key = re.sub(r"[^\w+#. ]", "", canonicalize(clean_text)).strip(" .")
    tools = [tool for tool in ("python", "flask", "django", "pandas", "sql", "postgresql", "sqlite",
                              "docker", "aws", "linux", "bash", "power bi", "matplotlib", "git")
             if re.search(r"\b" + re.escape(tool) + r"\b", clean_text, re.I)]
    qualifiers = re.findall(r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:\+\s*)?years?\b", clean_text, re.I)
    if len(tools) > 1:
        key += " | tools: " + ", ".join(sorted(tools))
    if qualifiers:
        key += " | tenure: " + ", ".join(qualifiers).casefold()
    return key


def cited_entries(items, source: str):
    """Reject hallucinated extraction quotes, including whitespace-only quotes."""
    for item in items:
        if not item.source_quote.strip() or item.source_quote not in source:
            raise ValueError("Extraction citation is not present in the input")
    return items


def grade_evidence(candidates: list[dict], lines: list[str], settings: Settings,
                   decision: SupportDecision | None = None, requirement_text: str = "") -> dict:
    """Pure grading policy: scores retrieve evidence; only confirmation covers.

    Boundary scores are inclusive: low is partial, high is eligible for
    covered. Missing means no retrieved resume evidence, not no human ability.
    """
    best = candidates[0] if candidates else {"score": 0.0, "line": ""}
    score, line = best["score"], best["line"]
    base = {"score": score, "invalid_covered_citation": False}
    if not candidates or score < settings.low_similarity:
        return {**base, "category": "missing", "citation": None,
                "explanation": "No resume line met the lower similarity threshold."}
    citation = line if line and line in lines else None
    if score < settings.high_similarity:
        return {**base, "category": "partial", "citation": citation,
                "explanation": "Related evidence met the lower threshold, but not the high threshold."}
    if decision and decision.supports:
        ongoing_study = re.search(r"\bstudent\b|\bpursuing\b|\bexpected\b|final[- ]year|in progress", line, re.I)
        degree_required = re.search(r"\bdegree\b|bachelor|\bdiploma\b", requirement_text, re.I)
        graduated = re.search(r"\bgraduated\b|\bawarded\b|\bearned\b|\bcompleted\b", line, re.I)
        if degree_required and ongoing_study and not graduated:
            return {**base, "category": "partial", "citation": citation,
                    "explanation": "The resume describes ongoing study, not a completed degree; the LLM confirmation was downgraded."}
        valid = bool(decision.citation.strip()) and decision.citation == line and line in lines
        return {**base, "category": "covered" if valid else "partial",
                "citation": decision.citation if valid else citation,
                "explanation": decision.explanation if valid else "Covered claim had an invalid citation; downgraded to partial.",
                "invalid_covered_citation": not valid}
    return {**base, "category": "partial", "citation": citation,
            "explanation": decision.explanation if decision else "High similarity was not confirmed by the grader; review this related evidence."}


def rank_gaps(requirements: list[dict], grades: dict[str, dict]) -> list[dict]:
    """Count distinct jobs, not repeated mentions inside a single posting."""
    gap_keys = {canonicalize(item["canonical_key"]) for item in requirements
                if grades[item["id"]]["category"] in ("partial", "missing")}
    grouped = {}
    for item in requirements:
        key = canonicalize(item["canonical_key"])
        if key not in gap_keys:
            continue
        group = grouped.setdefault(key, {"canonical_key": key, "text": item["text"],
            "job_indices": set(), "required_jobs": set(), "requirement_ids": [], "categories": set()})
        group["job_indices"].add(item["job_index"])
        if item["importance"] == "required":
            group["required_jobs"].add(item["job_index"])
        group["requirement_ids"].append(item["id"])
        group["categories"].add(grades[item["id"]]["category"])
    gaps = []
    for group in grouped.values():
        gaps.append({**group, "frequency": len(group["job_indices"]),
                     "required_count": len(group["required_jobs"]),
                     "importance": "required" if group["required_jobs"] else "preferred",
                     "job_indices": sorted(group["job_indices"]),
                     "required_jobs": sorted(group["required_jobs"]),
                     "categories": sorted(group["categories"])})
    return sorted(gaps, key=lambda gap: (-gap["frequency"], -bool(gap["required_count"]), gap["canonical_key"]))


class Nodes:
    def __init__(self, settings: Settings, database: Database,
                 llm: LLMClient, embeddings: EmbeddingService):
        self.settings = settings
        self.db = database
        self.llm = llm
        self.embeddings = embeddings

    def parse(self, state: SkillGapState) -> dict:
        settings = self.settings
        data = state["resume_bytes"]
        jobs = list(dict.fromkeys(job.strip() for job in state["job_descriptions"] if job.strip()))
        if not jobs or len(jobs) > settings.max_jobs:
            raise DocumentError(f"Provide between 1 and {settings.max_jobs} distinct job descriptions.")
        if len(data) > settings.max_resume_bytes:
            raise DocumentError("Resume exceeds the configured upload size limit.")
        if any(len(job) > settings.max_job_characters for job in jobs):
            raise DocumentError("A job description exceeds the configured text limit.")
        provider_model = settings.ollama_model if settings.llm_provider == "ollama" else settings.api_model
        metadata = {"provider": settings.llm_provider, "model": provider_model,
                    "endpoint": settings.ollama_base_url if settings.llm_provider == "ollama" else settings.api_base_url,
                    "parser": settings.parser_version, "prompt": settings.prompt_version,
                    "context_tokens": settings.ollama_context_tokens}
        resume_key = cache_key({**metadata, "template": input_hash(prompt("parse_resume", {})),
                                "format": state["resume_filename"].rsplit(".", 1)[-1].lower()})
        digest = input_hash(data)
        cached = self.db.get_parse("resume", digest, resume_key)
        warnings = list(state.get("warnings", []))
        incomplete = False
        if cached:
            text, entries, resume_id = cached["plain_text"], cached["extraction"], cached["id"]
        else:
            text = extract_text(data, state["resume_filename"])
            if len(text) > settings.max_resume_characters:
                raise DocumentError("Resume text exceeds the configured context limit. Upload a shorter version.")
            result = self.llm.structured(prompt("parse_resume", {"resume": text, "exact_lines": resume_lines(text)}), list[ResumeEntry], [],
                                         lambda items: cited_entries(items, text))
            entries = [item.model_dump() for item in result.value]
            resume_id = None
            if result.used_fallback:
                warnings.append("Resume extraction failed; grades use raw resume lines. This report is incomplete.")
                incomplete = True
            else:
                resume_id = self.db.save_resume(digest, resume_key, state["resume_filename"], text, entries)
        requirements = []
        job_ids = []
        known_keys = []
        for index, job in enumerate(jobs):
            job_key = cache_key({**metadata, "template": input_hash(prompt("parse_job", {})),
                                 "known_keys": sorted(set(known_keys))})
            job_digest = input_hash(job)
            cached_job = self.db.get_parse("job", job_digest, job_key)
            if cached_job:
                extraction, job_id = cached_job["extraction"], cached_job["id"]
            else:
                def validate(items):
                    cited_entries(items, job)
                    if not items:
                        raise ValueError("No requirements extracted")
                    for item in items:
                        item.canonical_key = grounded_key(item.canonical_key, item.text)
                        if not item.canonical_key:
                            raise ValueError("Empty canonical key")
                    return items

                result = self.llm.structured(prompt("parse_job", {"job": job, "known_keys": known_keys}),
                                             list[JobRequirement], [], validate)
                extraction = [item.model_dump() for item in result.value]
                if result.used_fallback:
                    warnings.append(f"Job {index + 1} extraction failed; no requirements were graded for this job.")
                    incomplete = True
                    continue
                # Small models sometimes emit the same requirement twice with
                # conflicting tags. Count it once, retaining required priority.
                unique = {}
                for item in extraction:
                    key = canonicalize(item["text"])
                    if key not in unique or item["importance"] == "required":
                        unique[key] = item
                extraction = list(unique.values())
                job_id = self.db.save_job(job_digest, job_key, job, extraction)
            job_ids.append(job_id)
            for position, item in enumerate(extraction):
                requirements.append({**item, "id": f"{index}:{position}", "job_id": job_id, "job_index": index})
                known_keys.append(item["canonical_key"])
        return {"resume_text": text, "resume_entries": entries, "resume_id": resume_id,
                "job_descriptions": jobs, "job_ids": job_ids, "requirements": requirements,
                "lines": resume_lines(text), "warnings": warnings, "incomplete": incomplete,
                "evidence": {}, "grades": {}, "retry_counters": {}, "retry_pending": []}

    def match(self, state: SkillGapState) -> dict:
        evidence = dict(state.get("evidence", {}))
        counters = dict(state.get("retry_counters", {}))
        pending = state.get("retry_pending", [])
        targets = pending or [item["id"] for item in state["requirements"]]
        if targets:
            # One batch avoids launching a model worker for every requirement.
            self.embeddings.embed([item["text"] for item in state["requirements"] if item["id"] in targets] + state["lines"])
        for requirement in state["requirements"]:
            key = requirement["id"]
            if key not in targets:
                continue
            if pending:
                counters[key] = counters.get(key, 0) + 1
            top_k = self.settings.match_top_k + counters.get(key, 0) * self.settings.retry_top_k_increment
            # Broaden with more candidate lines, keeping the original query so
            # similarity scores stay comparable across retries.
            evidence[key] = self.embeddings.search(requirement["text"], state["lines"], top_k)
        return {"evidence": evidence, "retry_counters": counters, "active_requirements": targets}

    def grade(self, state: SkillGapState) -> dict:
        grades = dict(state.get("grades", {}))
        pending = []
        warnings = list(state.get("warnings", []))
        incomplete = state.get("incomplete", False)
        targets = state.get("active_requirements", [])
        for requirement in state["requirements"]:
            key = requirement["id"]
            if key not in targets:
                continue
            candidates = state["evidence"].get(key, [])
            decision = None
            if candidates and candidates[0]["score"] >= self.settings.high_similarity:
                payload = {"requirement": requirement["text"], "top_line": candidates[0]["line"]}
                if state["retry_counters"].get(key, 0):
                    payload["other_candidates"] = candidates[1:]
                result = self.llm.structured(prompt("grade", payload), SupportDecision, None)
                decision = result.value
                if result.used_fallback:
                    warnings.append(f"Job {requirement['job_index'] + 1}: grader unavailable for {requirement['text']}.")
                    incomplete = True
            grades[key] = grade_evidence(candidates, state["lines"], self.settings, decision, requirement["text"])
            if grades[key]["invalid_covered_citation"]:
                if state["retry_counters"].get(key, 0) < self.settings.max_retries:
                    pending.append(key)
                else:
                    warnings.append(f"Citation retry limit reached for {requirement['text']}; retained partial grade.")
        return {"grades": grades, "retry_pending": pending, "warnings": warnings, "incomplete": incomplete}

    def rank(self, state: SkillGapState) -> dict:
        return {"ranked_gaps": rank_gaps(state["requirements"], state["grades"])}

    def recommend(self, state: SkillGapState) -> dict:
        recommendations = []
        warnings = list(state["warnings"])
        incomplete = state.get("incomplete", False)
        for gap in state["ranked_gaps"][:self.settings.top_n_gaps]:
            fallback = Recommendation(
                kind="study", title=f"Practice {gap['text']}",
                what_to_do=f"Follow one official introductory tutorial for {gap['text']}; complete three exercises and publish a short worked example with a README explaining the results.",
                why_it_helps="Creates demonstrable practice evidence. Professional tenure or credentials still require separate experience or qualification.",
                estimated_hours=8, resource_type="Official introductory documentation and practice exercises",
                resume_example=f"After completion only: Completed guided practice in {gap['text']} and documented three worked exercises.",
            )
            # Recommendations need the gap and grades, not personal contact details.
            result = self.llm.structured(prompt("recommend", {"gap": gap}), Recommendation, fallback)
            if re.search(r"\byears?\b|\bdegree\b|bachelor|citizenship|security clearance", gap["text"], re.I):
                result.value.why_it_helps += " This task builds related evidence but cannot satisfy professional tenure, degree, citizenship, or clearance requirements."
            recommendations.append({**result.value.model_dump(), "gap_key": gap["canonical_key"],
                                    "requirement_ids": gap["requirement_ids"], "used_fallback": result.used_fallback})
            if result.used_fallback:
                warnings.append(f"Recommendation for {gap['text']} uses a generic fallback; review it before starting.")
                incomplete = True
        return {"recommendations": recommendations, "warnings": warnings, "incomplete": incomplete}

    def report(self, state: SkillGapState) -> dict:
        report = {"covered": [], "partial": [], "missing": [],
                  "ranked_gaps": state["ranked_gaps"], "recommendations": state["recommendations"],
                  "warnings": list(dict.fromkeys(state["warnings"])),
                  "incomplete": state.get("incomplete", False),
                  "job_count": len(state["job_descriptions"]),
                  "retry_counters": state["retry_counters"],
                  "resume_entries": state["resume_entries"]}
        for requirement in state["requirements"]:
            grade = state["grades"][requirement["id"]]
            report[grade["category"]].append({**requirement, **grade,
                                            "candidates": state["evidence"][requirement["id"]]})
        settings = asdict(self.settings)
        settings.pop("api_key")
        settings["db_path"] = str(settings["db_path"])
        report["settings"] = settings
        analysis_hash = cache_key({"resume": input_hash(state["resume_bytes"]),
                                   "jobs": [input_hash(job) for job in state["job_descriptions"]],
                                   "settings": settings,
                                   "requirements": state["requirements"], "report": report})
        # A failed resume parse has no cached resume row. Return its report to
        # the UI, but never make that failure look like a reusable successful parse.
        report_id = None
        if state["resume_id"] is not None:
            report_id = self.db.save_report(analysis_hash, state["resume_id"], state["job_ids"], settings, report)
        return {"report": report, "report_id": report_id}
