"""Check published overhead coordinates, coverage, exclusions and provenance."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/overhead-20261025/scripts'))
from geometry import floor_point

def main():
    base=ROOT/'web/data/overhead-20261025';read=lambda name:json.loads((base/name).read_text())
    replay=read('replay.json');report=read('report.json');review=read('player-review.json');manifest=read('export-manifest.json');assets=json.loads((ROOT/'site-assets.json').read_text())
    for name,digest in manifest['files'].items():
        actual=hashlib.sha256((ROOT/'web'/name).read_bytes()).hexdigest()if name.startswith('data/')else assets['files'][name]
        assert actual==digest,name
    assert replay['duration_s']==300 and replay['fps']==15 and len(replay['frames'])==4500
    assert not replay['metric_distances_allowed']and not report['independent_human_ground_truth']
    selected=[];mapped=0;camera=0;ball=0;last={};bounds=replay['relative_bounds']
    for f,row in enumerate(replay['frames']):
        assert row['frame']==f and row['camera_valid']==(row['image_to_floor']is not None)
        camera+=row['camera_valid'];ball+=row['ball']is not None
        if 466<=f<=515:assert not row['camera_valid']and row['ball']is None
        ids=[o['player_id']for o in row['objects']];assert len(ids)==len(set(ids))
        for o in row['objects']:
            if o['player_id']==replay['default_player']:
                selected.append(o);assert any(a['track_id']==o['track_id']and a['first_frame']<=f<=a['last_frame']for a in review['associations'])
                mapped+=o['xy']is not None
            if o['xy']is not None:
                assert row['camera_valid']and o['foot']and o['segment']is not None
                reconstructed=floor_point(row['image_to_floor'],np.array(o['foot'])*[1920,1080],bounds)
                assert reconstructed is not None and np.max(np.abs(np.array(reconstructed)-o['xy']))<.0001
                assert all(0<=v<=1 for v in o['xy'])
                prior=last.get(o['player_id'])
                if prior and prior[1]['segment']==o['segment']:
                    assert f==prior[0]+1 and o['track_id']==prior[1]['track_id']
                last[o['player_id']]=(f,o)
            if not row['camera_valid']:assert o['xy']is None
    assert camera==report['camera_valid_frames']and mapped==report['selected_mapped_frames']and ball==report['ball_emitted_frames']and len(selected)==report['selected_observed_frames']
    assert len({o['segment']for o in selected if o['segment']is not None})==report['selected_segments']
    print(json.dumps({'frames':4500,'camera_valid':camera,'selected_mapped':mapped,'ball_estimates':ball,'assets_verified':len(manifest['files'])}))
if __name__=='__main__':main()
