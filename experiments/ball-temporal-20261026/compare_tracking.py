"""Compare conditional tracking at other reviewed seed frames, never own prompts."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math

HERE=Path(__file__).resolve().parent
BUILD=HERE.parents[1]/'build/ball-temporal-20261026'


def compare(path):
    data=json.loads(path.read_text())
    assert data['complete']
    references={t['seed_frame']:t['seed_pixel'] for t in data['tracks']}
    rows=[]
    for track in data['tracks']:
        for row in track['rows']:
            target=row['frame'];seed=track['seed_frame']
            if target not in references or target==seed:continue
            valid=(track['seed_mask_valid'] and row['visible'] and row['shape_valid']
                   and row['inside_crop'] and row['inside_image'])
            error=math.dist(row['pixel'],references[target]) if valid else None
            direction='1' if target>seed else '-1'
            guide=track.get('motion_guides',{}).get(direction)
            assert guide!=target, 'Scored target cannot guide its crop.'
            rows.append({'seed':seed,'frame':target,'error_px':error,
                         'guide_frame':guide,'verdict':'missing' if error is None else
                         'correct' if error<=12 else 'wrong'})
    return {'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'seed_file_sha256':data['seed_file_sha256'],'weights_sha256':data['weights_sha256'],
            'valid_seeds':sum(t['seed_mask_valid'] for t in data['tracks']),
            'seeds':len(data['tracks']),'comparisons':len(rows),
            'counts':{name:sum(r['verdict']==name for r in rows) for name in ['correct','wrong','missing']},
            'rows':rows},references


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/tracking-ablations.json')
    args=parser.parse_args()
    variants={}
    paths=[(kind,BUILD/f'short-{kind}-dense-gpu.json') for kind in
           ['sam3','sam3-box6','sam3-box14','sam3-point','sam3-point-negative',
            'sam3-mask3','sam3-mask5','sam2-box10']]
    paths.extend((name,BUILD/f'short-sam3-box6-dense-{name}-gpu.json') for name in ['wide','close','motion'])
    expected=None
    for name,path in paths:
        if not path.exists():continue
        result,references=compare(path)
        if expected is None:expected=references
        assert references==expected,'Variants must use identical seed coordinates.'
        variants[name]=result
    report={'created_at':datetime.now(timezone.utc).isoformat(),'threshold_px':12,
            'scope':'Same29 assistant-audited LLM sightings;70 directed comparisons predict another seed frame, never the prompt itself. Correlated development diagnostic, not independent accuracy. The motion variant additionally uses one sighting on the opposite side of the prompt for crop guidance. Mask prompts force the seed mask by construction, so seed validity is not an accuracy result.',
            'variants':variants}
    destination=HERE/args.output
    assert not destination.exists(),'Use a new filename for an amended comparison.'
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v['counts'] for k,v in variants.items()},indent=2))


if __name__=='__main__':main()
