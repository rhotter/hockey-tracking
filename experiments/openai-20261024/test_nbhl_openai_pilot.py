"""Exercise spending, resume and abstention using a fake transport; no API calls."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_nbhl_openai_pilot as runner


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'evidence'
        self.root.mkdir()
        self.out = Path(self.temp.name)/'output'
        (self.root/'sample.jpg').write_bytes(b'fake JPEG for mocked transport')
        self.protocol = {'models': [{'id': 'model-a', 'input_usd_per_million': 10,
                                    'output_usd_per_million': 50}],
                         'budget_usd': 1, 'max_output_tokens': 100,
                         'max_input_tokens': 20000, 'image_detail': 'original',
                         'reasoning_effort': 'high'}
        self.save_fixture()
        self.key_patch = patch.dict(os.environ, {'OPENAI_API_KEY': 'test-only-fake-value'})
        self.key_patch.start(); self.addCleanup(self.key_patch.stop)

    def save_fixture(self):
        runner.atomic_json(self.root/'protocol.json', self.protocol)
        image = {'path': 'sample.jpg', 'sha256': runner.sha(self.root/'sample.jpg')}
        rows = [{'id': 'edge/'+task, 'task': task, 'clip': 'test', 'edge_id': 'edge',
                 'prompt': 'Compare images', 'images': [image, image]} for task in ['multi', 'reverse']]
        manifest = {'phase': 'development-pilot', 'selection_uses_reference_labels': False,
                    'protocol_sha256': runner.sha(self.root/'protocol.json'),
                    'clips': [{'clip': 'test', 'edges': 1}], 'requests': rows}
        runner.atomic_json(self.root/'manifest.json', manifest)

    def response(self, status='completed'):
        return {'status': status, 'model': 'model-a', 'id': 'response-test',
                'usage': {'input_tokens': 100, 'output_tokens': 10},
                'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps({
                    'decision': 'uncertain', 'a_consistent': 'yes', 'b_consistent': 'uncertain',
                    'evidence_type': 'insufficient', 'evidence': 'Digits are unreadable.'})}]}]}

    def fake_api(self, endpoint, body, key):
        return {'input_tokens': 100} if endpoint.endswith('input_tokens') else self.response()

    def run_fixture(self, api=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.run(self.root, self.out, api=api or self.fake_api)

    def test_success_and_resume_never_rebill(self):
        ledger = self.run_fixture()
        self.assertEqual(len(ledger['requests']), 2)
        self.assertTrue(all(r['parsed']['decision'] == 'uncertain' for r in ledger['requests'].values()))
        def forbidden(*args):
            self.fail('A completed request must not be sent again')
        self.assertEqual(self.run_fixture(forbidden), ledger)

    def test_budget_stops_before_generation(self):
        self.protocol['budget_usd'] = .001; self.save_fixture()
        def api(endpoint, body, key):
            self.assertEqual(endpoint, 'responses/input_tokens')
            return {'input_tokens': 100}
        ledger = self.run_fixture(api)
        self.assertEqual(ledger['stop_reason'], 'budget')
        self.assertEqual(ledger['requests'], {})

    def test_interrupted_request_keeps_reservation_and_is_not_retried(self):
        calls = []
        def broken(endpoint, body, key):
            if endpoint.endswith('input_tokens'):
                return {'input_tokens': 100}
            calls.append(endpoint)
            raise TimeoutError('simulated timeout')
        with self.assertRaises(TimeoutError):
            self.run_fixture(broken)
        before = json.loads((self.out/'ledger.json').read_text())
        first = before['requests']['model-a/edge/multi']
        self.assertEqual(first['status'], 'unknown_billing')
        self.assertEqual(first['cost_ceiling_usd'], .00625)
        after = self.run_fixture()
        self.assertEqual(after['requests']['model-a/edge/multi'], first)
        self.assertEqual(len(after['requests']), 2)

    def test_both_models_share_budget(self):
        self.protocol['models'].append(self.protocol['models'][0] | {'id': 'model-b'})
        self.protocol['budget_usd'] = .0081; self.save_fixture()
        ledger = self.run_fixture()
        self.assertEqual(len(ledger['requests']), 2)
        self.assertEqual({r['model'] for r in ledger['requests'].values()}, {'model-a', 'model-b'})
        self.assertEqual(ledger['stop_reason'], 'budget')

    def test_corrupted_evidence_is_rejected_before_network(self):
        (self.root/'sample.jpg').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.run_fixture(lambda *args: self.fail('No network after hash mismatch'))

    def test_changed_protocol_cannot_reuse_ledger(self):
        self.run_fixture()
        self.protocol['budget_usd'] = 2; self.save_fixture()
        with self.assertRaisesRegex(AssertionError, 'changed'):
            self.run_fixture()

    def test_refusal_incomplete_and_invalid_outputs_abstain(self):
        self.assertIsNone(runner.parsed_answer(self.response('incomplete')))
        value = self.response()
        value['output'][0]['content'].append({'type': 'refusal', 'refusal': 'test'})
        self.assertIsNone(runner.parsed_answer(value))
        value['output'][0]['content'] = [{'type': 'output_text', 'text': '{"decision":"same"}'}]
        self.assertIsNone(runner.parsed_answer(value))

    def test_path_traversal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            runner.image_path(self.root, {'path': '../secret', 'sha256': ''})


if __name__ == '__main__':
    unittest.main()
