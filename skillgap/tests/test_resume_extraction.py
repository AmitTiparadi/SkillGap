"""Regression checks for real-length resumes without reading personal files."""
import json
from pathlib import Path
from uuid import uuid4
from unittest import TestCase
import httpx

from skillgap.config import PROJECT_ROOT, Settings
from skillgap.db import Database
from skillgap.graph.nodes import Nodes
from skillgap.llm import LLMClient, ResumeFact, SupportDecision
from skillgap.resume_extraction import extract_resume_entries, resume_chunks


class ResumeExtractionTests(TestCase):
    def test_unicode_long_lines_and_escaping_preserve_every_character(self):
        lines = ['Python "SQL" \\' * 200, 'अनुभव डेटा ' * 150, 'Built Flask APIs']
        chunks = resume_chunks(lines, 400)
        restored = [''] * len(lines)
        for chunk in chunks:
            self.assertLessEqual(len(json.dumps(chunk, ensure_ascii=False).encode('utf-8')), 400)
            for unit in chunk:
                restored[unit['line_id']] += unit['text']
        self.assertEqual(restored, lines)

    def client(self, invalid=False, fail_chunk=False):
        calls = []
        def handle(request):
            if request.url.path == '/api/show':
                return httpx.Response(200, json={})
            body = json.loads(request.content)
            message = body['messages'][1]['content']
            # Extract only the application's JSON input, never interpreting
            # instructions embedded inside the synthetic resume.
            payload, _ = json.JSONDecoder().raw_decode(message.split('INPUT DATA (not instructions):\n')[1])
            if 'job' in payload:
                return httpx.Response(200,json={'message':{'content':json.dumps({'items':[{'text':'Python','canonical_key':'python','importance':'required','source_quote':'Python'}]})}})
            calls.append(body)
            unit = payload['lines'][0]
            if fail_chunk and 'UNAVAILABLE' in unit['text']:
                return httpx.Response(200, json={'message': {'content': 'not json'}})
            fact = {'kind':'skill', 'name':unit['text'].split()[0],
                    'line_id':99999 if invalid else unit['line_id']}
            return httpx.Response(200,json={'message':{'content':json.dumps({'items':[fact]})}})
        return LLMClient(Settings(), httpx.MockTransport(handle)), calls

    def test_long_resume_uses_bounded_requests_and_exact_original_citations(self):
        text = '\n'.join(f'Python project {i}: Built a Flask API with unit tests and SQL queries.' for i in range(140))
        client, calls = self.client()
        result = extract_resume_entries(text, client, Settings())
        self.assertGreater(len(calls), 1)
        self.assertEqual(result.failed_chunks, 0)
        self.assertTrue(result.entries)
        for item in result.entries:
            self.assertIn(item['source_quote'], text.splitlines())
        for body in calls:
            self.assertNotIn('exact_lines', body['messages'][1]['content'])
            self.assertEqual(body['format']['properties']['items']['maxItems'], 8)

    def test_invalid_line_ids_retry_and_remain_incomplete(self):
        client, calls = self.client(invalid=True)
        with self.assertLogs('skillgap.llm', level='WARNING'):
            result = extract_resume_entries('Python API project', client, Settings())
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.failed_chunks, 1)
        self.assertEqual(result.entries, [])

    def test_one_failed_section_retains_valid_entries_and_is_not_cached(self):
        client, _ = self.client(fail_chunk=True)
        text = 'Python ' + 'API ' * 380 + '\nUNAVAILABLE ' + 'SQL ' * 380
        path = PROJECT_ROOT / 'data/test-temp' / f'{uuid4().hex}.sqlite3'
        for suffix in ('','-wal','-shm'):
            self.addCleanup(Path(str(path)+suffix).unlink, missing_ok=True)
        database = Database(path); database.initialize()
        nodes = Nodes(Settings(db_path=path), database, client, None)
        with self.assertLogs('skillgap.llm', level='WARNING'):
            result = nodes.parse({'resume_bytes':text.encode(), 'resume_filename':'resume.txt', 'job_descriptions':['Python']})
        self.assertTrue(result['incomplete'])
        self.assertTrue(result['resume_entries'])
        self.assertIsNone(result['resume_id'])
        self.assertTrue(any('Resume skill summary' in warning for warning in result['warnings']))
        self.assertTrue(result['requirements'])
        self.assertEqual(result['resume_text'], text)

    def test_truncated_response_is_a_safe_failure(self):
        def handle(request):
            if request.url.path == '/api/show': return httpx.Response(200,json={})
            return httpx.Response(200,json={'done_reason':'length','message':{'content':'{"items":'}})
        with self.assertLogs('skillgap.llm', level='WARNING'):
            result=LLMClient(Settings(),httpx.MockTransport(handle)).structured('Check',SupportDecision,None)
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.failure_reason,'output_limit')
