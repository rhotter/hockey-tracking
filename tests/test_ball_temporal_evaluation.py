"""Guard the experiment's localization accounting and held-out image identity."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1]/'experiments/ball-temporal-20261026/evaluate.py'
spec = importlib.util.spec_from_file_location('temporal_evaluation', path)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class TemporalEvaluationTests(unittest.TestCase):
    point = {'threshold': .5, 'appearance_power': 1.}

    def frame(self, frame, pixel=None):
        return {'clip': 'test', 'frame': frame, 'source_sha256': str(frame),
                'candidates': [] if pixel is None else [{'pixel': pixel, 'score': .9, 'appearance_score': .9}]}

    def label(self, frame, visibility='visible'):
        return {'clip': 'test', 'frame': frame, 'visibility': visibility,
                'pixel': [10,10] if visibility == 'visible' else None, 'source_sha256': str(frame)}

    def test_wrong_location_counts_false_positive_and_missed_ball(self):
        report = evaluation.evaluate([self.frame(0,[100,100])], [self.label(0)], self.point)
        self.assertEqual((report['tp'],report['fp'],report['fn']), (0,1,1))

    def test_unknown_does_not_become_negative(self):
        report = evaluation.evaluate([self.frame(0,[10,10]),self.frame(1,[100,100])],
                                     [self.label(0),self.label(1,'unknown')], self.point)
        self.assertEqual((report['tp'],report['fp'],report['fn']), (1,0,0))
        self.assertEqual(report['outputs_on_unknown'], 1)

    def test_absent_frame_penalizes_spurious_output(self):
        report = evaluation.evaluate([self.frame(0,[10,10])], [self.label(0,'absent')], self.point)
        self.assertEqual(report['false_on_absent'], 1)

    def test_mismatched_or_missing_source_fails(self):
        frame = self.frame(0,[10,10]); frame['source_sha256']='different'
        with self.assertRaisesRegex(AssertionError, 'Image mismatch'):
            evaluation.evaluate([frame], [self.label(0)], self.point)
        with self.assertRaisesRegex(AssertionError, 'Missing inference'):
            evaluation.evaluate([], [self.label(0)], self.point)

    def test_absent_prediction_is_a_miss(self):
        report = evaluation.evaluate([self.frame(0)], [self.label(0)], self.point)
        self.assertEqual((report['missing'],report['fn'],report['fp']), (1,1,0))


if __name__ == '__main__': unittest.main()
