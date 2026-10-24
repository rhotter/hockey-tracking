import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nbhl_openai_video_baseline as b


def answer():
    return {'summary':'test','players':[], 'frames':[{'frame':0,'objects':[]},{'frame':14,'objects':[]}],
            'events':[], 'possession':[], 'scoreboard':[], 'limitations':[]}


class DirectVideoTests(unittest.TestCase):
    def test_sampling_includes_last_and_requested_frames(self):
        inputs, outputs=b.sample_indices({'frame_count':450,'input_step_frames':15,'output_step_frames':75})
        self.assertEqual(len(inputs),31)
        self.assertEqual(outputs,[0,75,150,225,300,375,449])
        self.assertTrue(set(outputs)<=set(inputs))

    def test_missing_frames_and_duplicate_identity_are_not_repaired(self):
        value=answer();value['frames'][0]['objects']=[{'player_id':'a','bbox':[0,0,100,200]}]*2
        errors=b.validate(value,{'output_frames':[0,7,14],'frame_count':15})
        self.assertIn('missing or unexpected output frames',errors)
        self.assertIn('same player twice in one frame',errors)
        self.assertIn('unknown player in frame',errors)

    def test_incomplete_is_not_success(self):
        self.assertIsNone(b.parse({'status':'incomplete','output':[]}))
        self.assertEqual(b.validate(None,{}),['No completed structured answer'])

    def test_input_is_only_images_timestamps_and_task(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'frame.jpg';p.write_bytes(b'test-image')
            clip={'clip':'x','frame_count':15,'input_step_frames':15,'output_frames':[0,14],
                  'images':[{'frame':0,'path':'frame.jpg','sha256':b.sha(p),'width':1920,'height':1080}]}
            payload=b.make_payload(clip,{'id':'gpt-6-astra'}, {'reasoning_effort':'high','max_output_tokens':24576},root)
            self.assertNotIn('tools',payload)
            self.assertEqual([c['type'] for c in payload['input'][0]['content']],['input_text','input_text','input_image'])
            self.assertEqual(payload['input'][0]['content'][-1]['detail'],'original')

    def test_dispatched_failure_remains_reserved_and_is_not_retried(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,{'OPENAI_API_KEY':'test-only'}):
            root=Path(folder);out=root/'run'
            (root/'protocol.json').write_text('{}');(root/'manifest.json').write_text('{}')
            protocol={'budget_usd':10,'max_input_tokens':250000,'max_output_tokens':24576,
                      'reasoning_effort':'high','models':[{'id':'gpt-6-astra','input_usd_per_million':10,'output_usd_per_million':50}]}
            manifest={'clips':[{'clip':'x','frame_count':15,'input_step_frames':15,'output_frames':[0,14],'images':[]}]}
            calls=[]
            def api(endpoint,body,key):
                calls.append(endpoint)
                if endpoint.endswith('input_tokens'):return {'input_tokens':100}
                raise RuntimeError('simulated timeout')
            with self.assertRaises(RuntimeError):b.run(root,out,protocol,manifest,api=api)
            ledger=b.run(root,out,protocol,manifest,api=api)
            self.assertEqual(calls,['responses/input_tokens','responses'])
            record=ledger['requests']['gpt-6-astra/x']
            self.assertEqual(record['status'],'unknown_billing')
            self.assertGreater(record['cost_ceiling_usd'],1)

    def test_budget_stop_precedes_generation(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,{'OPENAI_API_KEY':'test-only'}):
            root=Path(folder);(root/'protocol.json').write_text('{}');(root/'manifest.json').write_text('{}')
            protocol={'budget_usd':.001,'max_input_tokens':250000,'max_output_tokens':24576,
                      'reasoning_effort':'high','models':[{'id':'gpt-6-astra','input_usd_per_million':10,'output_usd_per_million':50}]}
            manifest={'clips':[{'clip':'x','frame_count':15,'input_step_frames':15,'output_frames':[0,14],'images':[]}]}
            def api(endpoint,body,key):
                self.assertEqual(endpoint,'responses/input_tokens');return {'input_tokens':100}
            ledger=b.run(root,root/'run',protocol,manifest,api=api)
            self.assertEqual(ledger['stop_reason'],'budget');self.assertEqual(ledger['requests'],{})


if __name__=='__main__':unittest.main()
