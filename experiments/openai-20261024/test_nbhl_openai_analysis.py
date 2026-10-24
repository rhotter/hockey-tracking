import unittest
from analyze_nbhl_openai_pilot import summarize_model


def row(edge, task, value):
    return {'id': edge+'/'+task, 'parsed': {'decision': value, 'a_consistent': 'yes',
            'b_consistent': 'yes', 'evidence_type': 'jersey_number'}}


class ComparisonTests(unittest.TestCase):
    def test_abstaining_on_every_case_is_not_decisive_coverage(self):
        rows = [row('a', task, 'uncertain') for task in ['multi', 'reverse']]
        result = summarize_model('test', rows, {r['id'] for r in rows})
        self.assertEqual(result['order_agreement'], 1)
        self.assertEqual(result['stable_decisive_edges'], 0)
        self.assertEqual(result['symmetric_strict_same_edges'], 0)
        self.assertIsNone(result['identity_accuracy'])

    def test_missing_and_invalid_are_separate_from_model_uncertainty(self):
        rows = [row('a', 'multi', 'uncertain'), {'id':'a/reverse','parsed':None}]
        result = summarize_model('test', rows, {'a/multi','a/reverse','b/multi','b/reverse'})
        self.assertEqual(result['missing_requests'], 2)
        self.assertEqual(result['invalid_or_unfinished_requests'], 1)
        self.assertEqual(result['valid_order_comparisons'], 0)
        self.assertEqual(result['forward_decisions'], {'uncertain':1})

    def test_direct_flip_is_distinct_from_abstention_change(self):
        rows = [row('a','multi','same'), row('a','reverse','different'),
                row('b','multi','same'), row('b','reverse','uncertain')]
        result = summarize_model('test',rows,{r['id'] for r in rows})
        self.assertEqual(result['order_disagreement'], 2)
        self.assertEqual(result['direct_same_different_flips'], 1)

    def test_symmetric_support_requires_evidence_in_both_directions(self):
        rows = [row('a',task,'same') for task in ['multi','reverse']]
        rows[1]['parsed']['evidence_type'] = 'uniform_only'
        result = summarize_model('test',rows,{r['id'] for r in rows})
        self.assertEqual(result['stable_decisive_edges'], 1)
        self.assertEqual(result['symmetric_strict_same_edges'], 0)


if __name__ == '__main__':
    unittest.main()
