"""Run real local services; never replace unavailable models with fake results.

Use the same synthetic resume for condensed requirements from two real roles:
    python -m skillgap.evaluation.run_evaluation
Outputs contain synthetic personal data only and can be committed for a demo.
"""
import json
from pathlib import Path

from skillgap.config import load_settings
from skillgap.graph.workflow import analyze, create_services


def main():
    settings = load_settings()
    folder = Path(__file__).resolve().parent
    resume = (folder.parent / "examples" / "resume.txt").read_bytes()
    nodes = create_services(settings)
    lines = ["Built a Flask REST API for a student library with SQLite and unit tests.",
             "Analyzed a public sales CSV using pandas and SQL queries.",
             "Created Matplotlib charts to explain monthly sales trends."]
    pairs = {}
    queries = ["Build REST APIs with Python and Flask", "Analyze data with pandas", "Write SQL queries",
               "Advanced PostgreSQL performance tuning", "Docker container experience", "Build Power BI dashboards"]
    nodes.embeddings.embed([*queries, *lines])
    for query in queries:
        pairs[query] = nodes.embeddings.search(query, lines, 3)
    (folder / "pairs.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    print("Embedding pairs saved", flush=True)
    for name in ("backend", "analyst"):
        job = (folder / f"{name}.txt").read_text(encoding="utf-8")
        # Source metadata identifies the fixture; grade only its requirements.
        job = job[job.index("Required:"):]
        report = analyze(resume, "synthetic-resume.txt", [job], settings, nodes)
        (folder / f"{name}-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(name, {category: len(report[category]) for category in ("covered", "partial", "missing")},
              "incomplete:", report["incomplete"], flush=True)


if __name__ == "__main__":
    main()
