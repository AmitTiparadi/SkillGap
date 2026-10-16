"""LangGraph wiring: only invalid covered citations trigger retrieval retries."""

from langgraph.graph import END, START, StateGraph
from filelock import FileLock, Timeout
from langsmith import tracing_context

from skillgap.config import Settings, load_settings
from skillgap.db import Database
from skillgap.embeddings import EmbeddingService
from skillgap.graph.nodes import Nodes
from skillgap.graph.state import SkillGapState
from skillgap.llm import LLMClient


def route_after_grade(state: SkillGapState) -> str:
    return "match" if state.get("retry_pending") else "rank"


def build_workflow(nodes: Nodes):
    builder = StateGraph(SkillGapState)
    for name in ("parse", "match", "grade", "rank", "recommend", "report"):
        builder.add_node(name, getattr(nodes, name))
    builder.add_edge(START, "parse")
    builder.add_edge("parse", "match")
    builder.add_edge("match", "grade")
    builder.add_conditional_edges("grade", route_after_grade, {"match": "match", "rank": "rank"})
    builder.add_edge("rank", "recommend")
    builder.add_edge("recommend", "report")
    builder.add_edge("report", END)
    return builder.compile()


def create_services(settings: Settings) -> Nodes:
    database = Database(settings.db_path)
    database.initialize()
    return Nodes(settings, database, LLMClient(settings), EmbeddingService(settings, database))


def analyze(resume_bytes: bytes, resume_filename: str, jobs: list[str],
            settings: Settings | None = None, nodes: Nodes | None = None) -> dict:
    settings = settings or (nodes.settings if nodes else load_settings())
    nodes = nodes or create_services(settings)
    # Prevent CLI and UI analyses using the same database from overlapping
    # their embedding/LLM memory peaks. Native file locks release on exceptions.
    lock = FileLock(str(settings.db_path) + ".analysis.lock", timeout=0)
    try:
        # Override inherited tracing environment settings: document state must
        # not be uploaded to a tracing service in either provider mode.
        with lock, tracing_context(enabled=False):
            app = build_workflow(nodes)
            # Pending requirements retry together per graph round. Each round
            # increments their counters, so this bound scales with retries.
            state = app.invoke({"resume_bytes": resume_bytes, "resume_filename": resume_filename,
                                "job_descriptions": jobs},
                               config={"recursion_limit": 12 + 2 * settings.max_retries})
            return state["report"]
    except Timeout as error:
        raise ValueError("Another SkillGap analysis is running. Wait for it to finish and try again.") from error
