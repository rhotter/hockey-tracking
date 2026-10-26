"""Verify published ball states, covariance, provenance and deterministic output."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hockey_tracking.ball_context import infer, normalized_homography, project


def main():
    source=ROOT/'web/data/overhead-20261025/replay.json'
    folder=ROOT/'web/data/ball-context-20261026'
    replay=json.loads(source.read_text());result=json.loads((folder/'replay.json').read_text())
    report=json.loads((folder/'report.json').read_text())
    assert report['input_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    assert report['algorithm_sha256']==hashlib.sha256((ROOT/'hockey_tracking/ball_context.py').read_bytes()).hexdigest()
    assert not report['independent_ground_truth'] and not result['uncertainty_calibrated']
    assert len(result['frames'])==4500
    regenerated=infer(replay)
    assert json.loads(json.dumps(regenerated['config']))==result['config']
    for actual,expected in zip(regenerated['frames'],result['frames']):
        assert actual.keys()==expected.keys()
        for key in actual:
            if key in ['xy','pixel','covariance','pixel_covariance','model_radius_px']:
                np.testing.assert_allclose(actual[key],expected[key],rtol=1e-9,atol=1e-11)
            else:assert actual[key]==expected[key]
    counts={state:0 for state in ['detected','inferred','unknown']}
    for f,row in enumerate(result['frames']):
        assert row['frame']==f;counts[row['state']]+=1
        if row['state']=='unknown':
            assert 'xy' not in row and 'pixel' not in row
            continue
        assert replay['frames'][f]['camera_valid']
        assert np.all(np.linalg.eigvalsh(row['covariance'])>0)
        assert np.all(np.linalg.eigvalsh(row['pixel_covariance'])>0)
        H=normalized_homography(replay['frames'][f],replay['relative_bounds'])
        np.testing.assert_allclose(project(H,np.array(row['pixel'])*[1920,1080]),row['xy'],atol=1e-10)
        for anchor in row['evidence_frames']:assert replay['frames'][anchor]['ball']
        if row['state']=='detected':
            np.testing.assert_allclose(row['xy'],replay['frames'][f]['ball']['xy'],atol=1e-10)
        else:
            assert not replay['frames'][f]['ball']
            assert row['model_radius_px']<=100
            first=min(row['evidence_frames']+[f]);last=max(row['evidence_frames']+[f])
            assert all(replay['frames'][k]['camera_valid'] for k in range(first,last+1))
    assert counts==report['variants']['context']['states']
    assert counts=={'detected':106,'inferred':62,'unknown':4332}
    print(json.dumps({'frames':4500,'states':counts,'deterministic_reproduction':True,'uncertainty_calibrated':False}))


if __name__=='__main__':main()
