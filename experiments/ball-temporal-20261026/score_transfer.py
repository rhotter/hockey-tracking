"""Score automatic neighboring-frame SAM prompts using the fixed transfer policy."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = ROOT/'build/ball-temporal-20261026'
sys.path.insert(0,str(ROOT))
from hockey_tracking.ball_refinement import valid_mask
from evaluate import evaluate, top_candidate


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol_path = HERE/'transfer-protocol.json'
    protocol = read(protocol_path)
    labels = read(HERE/'fresh-reference.json')['rows'] + read(HERE/'extra-reference.json')['rows']
    selection = read(HERE/'results/development-selection.json')['variants']['yolo-full1920']['operating_point']
    predictions = {name:[] for name in ['yolo','sam_consensus','sam_appearance','hybrid_consensus']}
    input_hashes = {str(protocol_path.relative_to(ROOT)):sha(protocol_path)}
    diagnostics = []
    for clip in sorted({r['clip'] for r in labels}):
        paths = [BUILD/'yolo-full1920'/f'{clip}.json',
                 BUILD/f'tracking-seeds-transfer-{clip}-gpu.json',
                 BUILD/f'short-sam3-transfer-{clip}-gpu.json']
        detector,seeds,tracks = [read(p) for p in paths]
        assert detector['complete'] and tracks['complete']
        assert tracks['seed_file_sha256'] == sha(paths[1])
        assert not tracks['protocol']['offload_state_to_cpu']
        input_hashes.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
        by_seed = {t['seed_frame']:t for t in tracks['tracks']}
        for f in protocol['targets']:
            row = detector['frames'][f]
            assert row['frame'] == f
            baseline = top_candidate(row,**selection)
            valid = []
            for sf in [f+o for o in protocol['seed_offsets']]:
                track = by_seed.get(sf)
                if not track or not track['seed_mask_valid']: continue
                target = next((r for r in track['rows'] if r['frame']==f),None)
                if valid_mask(target): valid.append(target)
            consensus = None
            agreement = math.dist(valid[0]['pixel'],valid[1]['pixel']) if len(valid)==2 else None
            if agreement is not None and agreement <= protocol['agreement_radius_px']:
                consensus = [(valid[0]['pixel'][i]+valid[1]['pixel'][i])/2 for i in range(2)]
            verified = [r for r in valid if r['appearance_score']>=.2]
            appearance = max(verified,key=lambda r:r['appearance_score'])['pixel'] if verified else None
            baseline_pixel = baseline['pixel'] if baseline else None
            outputs = {'yolo':baseline_pixel,'sam_consensus':consensus,'sam_appearance':appearance,
                       'hybrid_consensus':consensus if consensus is not None else baseline_pixel}
            for name,pixel in outputs.items():
                # Score=1 is an inclusion flag for this already-selected output,
                # not a detector score or calibrated confidence.
                predictions[name].append({'clip':clip,'frame':f,'source_sha256':row['source_sha256'],
                    'candidates':[{'pixel':pixel,'score':1.}] if pixel is not None else []})
            diagnostics.append({'clip':clip,'frame':f,'valid_neighbor_tracks':len(valid),
                                'agreement_px':agreement,'outputs':outputs})
    result = {'created_at':datetime.now(timezone.utc).isoformat(), 'protocol':protocol,
              'input_sha256':input_hashes, 'models':{}, 'rows':diagnostics}
    for name,frames in predictions.items():
        result['models'][name] = {f'at_{radius}px':evaluate(frames,labels,
            {'threshold':.5,'appearance_power':0},radius) for radius in [12,25]}
    dest = HERE/'results/transfer-evaluation.json'
    assert not dest.exists(), 'Do not overwrite scored transfer results.'
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({name:{k:v for k,v in r['at_25px'].items() if k!='details'}
                      for name,r in result['models'].items()},indent=2))


if __name__ == '__main__': main()
