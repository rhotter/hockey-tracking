"""Recompute the completed OpenAI comparisons from saved, hash-checked inputs."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/openai-20261024'))
from analyze_nbhl_openai_pilot import analyze as reconnect
from analyze_nbhl_openai_video_baseline import analyze as video


def read(path):
    return json.loads(path.read_text())


def main():
    base = ROOT/'web/data/reliability/openai-20261024'
    summary = read(base/'summary.json')
    for name, expected in summary['input_hashes'].items():
        assert hashlib.sha256((base/name).read_bytes()).hexdigest() == expected, name
    r = base/'reconnect'
    rec = reconnect(r, r/'ledger.json', r/'qwen-baselines.json')
    assert rec == read(r/'comparison.json')
    v = base/'video'
    vid = video(v/'manifest.json', v/'ledger.json',
                ROOT/'web/data/reliability/tracking-four-hour-final-reference-v1.json',
                ROOT/'web/data/reliability/sam3-20261024/sampled-predictions', v/'protocol.json')
    assert vid == read(v/'comparison.json')
    assert summary['reference_sha256'] == vid['reference_sha256']
    assert summary['independent_human_ground_truth'] is False
    assert summary['event_and_stat_accuracy'] is None
    assert summary['reconnection_accuracy_measured'] is False
    assert summary['reconnection'] == rec['models']
    for model, shown in zip(vid['models'], summary['short_clip_results']):
        rows = [c for c in model['clips'] if c['status']=='completed']
        assert len(rows) == 2 and shown['model'] == model['model']
        counts = {k:sum(r['counts'][k] for r in rows) for k in rows[0]['counts']}
        assert counts == shown['counts']
        assert (counts['matched_observations'],counts['correct'],counts['wrong'],counts['unknown_same']) == (96,69,0,0)
    rows = vid['pipeline']['clips'][:2]
    assert {k:sum(r['counts'][k] for r in rows) for k in rows[0]['counts']} == summary['short_clip_pipeline']
    failed = [r for r in read(v/'ledger.json')['requests'].values() if r['status']=='incomplete']
    assert len(failed)==2
    for row, shown in zip(failed, summary['long_clip_failures']):
        assert row['parsed'] is None and row['clip']=='pittsburgh-stress-0900'
        assert row['model']==shown['model']
        assert row['incomplete_details']['reason']==shown['reason']=='max_output_tokens'
        assert row['usage']['output_tokens']==shown['output_tokens']==24576
    cost = sum(m['conservative_list_cost_usd'] for m in rec['models'] if m['model'].startswith('gpt-'))+sum(m['cost_ceiling_usd'] for m in vid['models'])
    assert abs(cost-summary['cost_ceiling_usd'])<1e-9
    print(json.dumps({'reconnection_requests':260,'valid_reconnection_responses':259,
                      'direct_video_requests':6,'completed_direct_video_requests':4,
                      'short_same_player_checks_per_model':69,'long_output_limit_failures':2,
                      'conservative_api_cost_usd':round(cost,5)}))


if __name__=='__main__':
    main()
