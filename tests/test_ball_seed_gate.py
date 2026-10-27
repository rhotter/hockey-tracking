"""Reject false-object seeds without quietly relabeling or moving their tracks."""
from pathlib import Path
import importlib.util
import unittest

path = Path(__file__).resolve().parents[1]/'experiments/ball-temporal-20261026/gate_detector_seeds.py'
spec = importlib.util.spec_from_file_location('ball_seed_gate', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SeedGateTests(unittest.TestCase):
    def test_scoreboard_uncertain_and_incomplete_seeds_cannot_enter_tracker(self):
        seeds = {'seeds':[{'frame':f, 'pixel':[100+f,200], 'support':'unclear',
            'evidence':'Candidate.', 'source_sha256':str(f)} for f in range(5)]}
        tracks = {'tracks':[{'seed_frame':f, 'seed_pixel':[100+f,200],
            'rows':[{'frame':f+1, 'pixel':[105+f,203]}]} for f in range(5)]}
        ledger = {'requests':{
            '0':{'status':'completed','answer':{'samples':[
                {'frame':0,'verdict':'non_ball','support':'unclear','reason':'Scoreboard digit.'},
                {'frame':1,'verdict':'uncertain','support':'unclear','reason':'Hidden.'},
                {'frame':2,'verdict':'visible_ball','support':'held','reason':'Separate ball in hand.'}]}},
            '1':{'status':'incomplete','answer':None},
            '2':{'status':'withheld_after_uncertain_request'}}}
        accepted, kept, reviews = module.gate(seeds, tracks, ledger)
        self.assertEqual([r['frame'] for r in accepted], [2])
        self.assertEqual(accepted[0]['pixel'], seeds['seeds'][2]['pixel'])
        self.assertEqual(accepted[0]['support'], 'held')
        self.assertEqual(kept, [tracks['tracks'][2]])
        self.assertEqual([r['verdict'] for r in reviews],
                         ['non_ball','uncertain','visible_ball','not_verified','not_verified'])


if __name__ == '__main__': unittest.main()
