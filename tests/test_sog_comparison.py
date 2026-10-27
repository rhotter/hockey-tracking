"""Event matching must retain disagreements and resolve nearby rebounds fairly."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_sog_terra import match_events, usage_priced_cost


def event(name, seconds, team='red', clip='game', status='sog', outcome='save'):
    return dict(id=name, clip_seconds=seconds, team_color=team, clip=clip,
                automatic_status=status, outcome=outcome)


class SogComparisonTests(unittest.TestCase):
    def test_cache_reads_and_writes_are_not_double_charged(self):
        usage = {'input_tokens': 1000, 'output_tokens': 100,
            'input_tokens_details': {'cached_tokens': 300, 'cache_write_tokens': 500}}
        pricing = {'input_usd_per_million': 1, 'cached_input_usd_per_million': .05, 'output_usd_per_million': 5}
        self.assertAlmostEqual(usage_priced_cost(usage, pricing), (200+15+625+500)/1e6)

    def test_mixed_cache_lifetimes_use_separate_write_rates(self):
        usage = {'input_tokens': 1000, 'output_tokens': 0, 'input_tokens_details': {
            'cached_tokens': 100, 'cache_write_tokens': 800,
            'cache_write_tokens_details': {'ttl_30m_tokens': 600, 'ttl_12h_tokens': 200}}}
        pricing = {'input_usd_per_million': 1, 'cached_input_usd_per_million': .05, 'output_usd_per_million': 5}
        self.assertAlmostEqual(usage_priced_cost(usage, pricing), (100+5+750+500)/1e6)

    def test_impossible_cache_accounting_is_rejected(self):
        usage = {'input_tokens': 100, 'output_tokens': 0, 'input_tokens_details': {'cached_tokens': 80, 'cache_write_tokens': 50}}
        with self.assertRaises(ValueError):
            usage_priced_cost(usage, {})

    def test_rebound_alignment_maximizes_number_of_distinct_matches(self):
        # Nearest-first would consume the second reference with the first query.
        report = match_events([event('a', 0), event('b', 2)], [event('c', 1.5), event('d', 3.5)])
        self.assertEqual([(p['reference_id'], p['candidate_id']) for p in report['pairs']], [('a', 'c'), ('b', 'd')])

    def test_conflicting_calls_still_match(self):
        report = match_events([event('a', 4)], [event('b', 4.2, status='not_sog', outcome='miss')])
        self.assertEqual(report['matched'], 1)
        self.assertEqual(report['same_count_category'], 0)
        self.assertEqual(report['same_outcome'], 0)

    def test_other_teams_and_recordings_never_match(self):
        report = match_events([event('a', 4)], [event('b', 4, team='white'), event('c', 4, clip='other')])
        self.assertEqual(report['matched'], 0)
        self.assertEqual(report['reference_only'], ['a'])
        self.assertEqual(report['candidate_only'], ['b', 'c'])

    def test_lower_timing_error_breaks_equal_cardinality_tie(self):
        report = match_events([event('a', 1), event('b', 2.5)], [event('c', 2.3)])
        self.assertEqual(report['pairs'][0]['reference_id'], 'b')


if __name__ == '__main__':
    unittest.main()
