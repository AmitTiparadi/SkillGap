import json
from pathlib import Path
from uuid import uuid4
import unittest
from unittest.mock import Mock

import httpx

from skillgap.config import PROJECT_ROOT, Settings
from skillgap.db import Database
from skillgap.graph.nodes import Nodes
from skillgap.llm import JSONResult, JobRequirement, LLMClient, ResumeEntry, SupportDecision


class LLMTests(unittest.TestCase):
    def test_json_failure_retries_once_then_falls_back(self):
        client = LLMClient(Settings())
        client._request = Mock(return_value="not JSON")
        with self.assertLogs("skillgap.llm", level="WARNING"):
            result = client.structured("JSON only", SupportDecision, None)
        self.assertTrue(result.used_fallback)
        self.assertEqual(client._request.call_count, 2)

    def test_schema_failure_then_valid_json(self):
        client = LLMClient(Settings())
        client._request = Mock(side_effect=[json.dumps({"items": {"supports": "yes"}}),
            json.dumps({"items": {"supports": True, "citation": "Built Python APIs", "explanation": "Exact support"}})])
        with self.assertLogs("skillgap.llm", level="WARNING"):
            result = client.structured("JSON only", SupportDecision, None)
        self.assertFalse(result.used_fallback)
        self.assertTrue(result.value.supports)

    def test_local_endpoint_never_falls_back_to_api(self):
        urls = []
        def handler(request):
            urls.append(str(request.url))
            if request.url.path == "/api/show":
                return httpx.Response(200, json={})
            return httpx.Response(500)
        client = LLMClient(Settings(), httpx.MockTransport(handler))
        with self.assertLogs("skillgap.llm", level="WARNING"):
            result = client.structured("JSON only", SupportDecision, None)
        self.assertTrue(result.used_fallback)
        self.assertEqual(urls, ["http://localhost:11434/api/show"] + ["http://localhost:11434/api/chat"] * 2)

    def test_cloud_alias_is_blocked_before_resume_transmission(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"remote_model": "remote-model"})
        client = LLMClient(Settings(), httpx.MockTransport(handler))
        with self.assertLogs("skillgap.llm", level="WARNING"):
            result = client.structured("PRIVATE RESUME", SupportDecision, None)
        self.assertTrue(result.used_fallback)
        self.assertTrue(all(request.url.path == "/api/show" for request in requests))
        self.assertTrue(all(b"PRIVATE RESUME" not in request.content for request in requests))

    def test_parse_mock_and_cache(self):
        path = PROJECT_ROOT / "data" / "test-temp" / f"{uuid4().hex}.sqlite3"
        for suffix in ("", "-wal", "-shm"):
            self.addCleanup(Path(str(path) + suffix).unlink, missing_ok=True)
        database = Database(path)
        database.initialize()
        llm = Mock()
        llm.structured.side_effect = [
            JSONResult([ResumeEntry(kind="experience", name="Python", source_quote="Built Python APIs")]),
            JSONResult([JobRequirement(text="Python", canonical_key="python", importance="required", source_quote="Python")]),
        ]
        nodes = Nodes(Settings(), database, llm, Mock())
        state = {"resume_bytes": b"Built Python APIs", "resume_filename": "resume.txt", "job_descriptions": ["Python"]}
        first = nodes.parse(state)
        second = nodes.parse(state)
        self.assertEqual(first["requirements"], second["requirements"])
        self.assertEqual(llm.structured.call_count, 2)

    def test_remote_ollama_url_is_rejected(self):
        with self.assertRaises(ValueError):
            Settings(ollama_base_url="https://remote.example")

    def test_list_schema_definitions_are_rooted_correctly(self):
        bodies = []
        def handler(request):
            if request.url.path == "/api/show":
                return httpx.Response(200, json={})
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": json.dumps({"items": [
                {"kind": "skill", "name": "Python", "source_quote": "Python"}]})}})
        result = LLMClient(Settings(), httpx.MockTransport(handler)).structured("Python", list[ResumeEntry], [])
        self.assertFalse(result.used_fallback)
        self.assertIn("ResumeEntry", bodies[0]["format"]["$defs"])
        self.assertEqual(bodies[0]["format"]["properties"]["items"]["items"]["$ref"], "#/$defs/ResumeEntry")

    def test_opt_in_api_uses_json_object_and_auth_header(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"items": {
                "supports": True, "citation": "Python", "explanation": "Exact mention"}})}}]})
        settings = Settings(llm_provider="api", api_model="test-model", api_key="test-only-key")
        result = LLMClient(settings, httpx.MockTransport(handler)).structured("Python", SupportDecision, None)
        self.assertFalse(result.used_fallback)
        self.assertEqual(requests[0].headers["Authorization"], "Bearer test-only-key")
        self.assertEqual(json.loads(requests[0].content)["response_format"], {"type": "json_object"})
