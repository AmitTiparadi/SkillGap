from unittest import TestCase
from unittest.mock import patch
from fastapi.testclient import TestClient
from skillgap.app import app
from skillgap.config import Settings

class UITests(TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.config = patch('skillgap.app.load_settings', return_value=Settings())
        self.config.start()
        self.addCleanup(self.config.stop)

    def post(self, **data):
        return self.client.post('/api/analyze', files={'resume': ('resume.txt', b'Python developer')}, data={'jobs': 'Python required', **data})

    def test_page_assets_and_private_config(self):
        response = self.client.get('/')
        self.assertIn('Make your next', response.text)
        self.assertEqual(response.headers['cache-control'], 'no-store')
        self.assertIn("default-src 'self'", response.headers['content-security-policy'])
        for asset in ('styles.css', 'app.js'):
            self.assertEqual(self.client.get('/assets/' + asset).status_code, 200)
        self.assertEqual(set(self.client.get('/api/config').json()), {'model'})
        self.assertIn('resume', self.client.get('/api/sample').json())

    @patch('skillgap.app.analyze', return_value={'covered': []})
    def test_default_provider_local_and_jobs_split(self, analyze):
        self.assertEqual(self.post(jobs='Python\n---\nSQL').status_code, 200)
        args = analyze.call_args.args
        self.assertEqual(args[2], ['Python', 'SQL'])
        self.assertEqual(args[3].llm_provider, 'ollama')

    @patch('skillgap.app.analyze', return_value={})
    def test_api_requires_explicit_consent(self, analyze):
        self.assertEqual(self.post(provider='api', api_model='test', api_key='secret').status_code, 400)
        analyze.assert_not_called()
        self.assertEqual(self.post(provider='api', api_model='test', api_key='secret', api_consent='true').status_code, 200)
        self.assertEqual(analyze.call_args.args[3].llm_provider, 'api')

    @patch('skillgap.app.analyze')
    def test_foreign_origin_and_empty_jobs_rejected(self, analyze):
        response = self.client.post('/api/analyze', headers={'Origin': 'https://foreign.example'})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.post(jobs='').status_code, 422)
        analyze.assert_not_called()

    @patch('skillgap.app.analyze', side_effect=RuntimeError('private payload'))
    def test_failure_does_not_expose_payload(self, analyze):
        response = self.post()
        self.assertEqual(response.status_code, 500)
        self.assertNotIn('private payload', response.text)

