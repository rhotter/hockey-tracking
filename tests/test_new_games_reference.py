"""The reference must not turn ambiguous team matches into identity truth."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'experiments/new-games-20261025/scripts/seal_reference.py'
spec = importlib.util.spec_from_file_location('new_game_reference', path)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def observation(key, frame, identity=None, team='orange', clip='one'):
    return {'id': key, 'clip': clip, 'frame': frame, 'identity': identity, 'team': team}


class ReferenceTests(unittest.TestCase):
    def test_unknown_same_team_is_uncertain_across_time(self):
        rows = [observation('a', 0), observation('b', 15)]
        self.assertEqual(reference.pairs_for(rows)[0]['truth'], 'uncertain')

    def test_local_groups_do_not_connect_across_clips(self):
        rows = [observation('a', 0, 'O1'), observation('b', 15, 'O1', clip='two')]
        self.assertEqual(reference.pairs_for(rows), [])

    def test_covisibility_proves_difference_but_names_alone_do_not(self):
        rows = [observation('a', 0, 'A'), observation('b', 0, 'B'),
                observation('c', 15, 'A'), observation('d', 30, 'B'),
                observation('e', 45, 'C')]
        pairs = {(p['a'], p['b']): p['truth'] for p in reference.pairs_for(rows)}
        self.assertEqual(pairs['a', 'c'], 'same')
        self.assertEqual(pairs['c', 'd'], 'different')
        self.assertEqual(pairs['c', 'e'], 'uncertain')

    def test_duplicate_identity_within_frame_is_rejected(self):
        with self.assertRaises(AssertionError):
            reference.pairs_for([observation('a', 0, 'A'), observation('b', 0, 'A')])


if __name__ == '__main__':
    unittest.main()
