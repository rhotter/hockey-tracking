"""Build a candidate replay and a deterministic output-review sample."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT))
from hockey_tracking.ball_refinement import refine_sequence, RefinementConfig

BUILD = ROOT/'build/ball-temporal-20261026'


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batches', default='dense-full-gpu')
    parser.add_argument('--name', default='first')
    parser.add_argument('--tracker', default='sam3', choices=['sam3','sam3-box6'])
    parser.add_argument('--config', default='fusion-config.json')
    args = parser.parse_args()
    config_path = HERE/args.config
    config = RefinementConfig(**read(config_path)['config'])
    inputs = [ROOT/'web/data/overhead-20261025/replay.json',
              ROOT/'web/data/ball-sam-20261026/positions.json',
              BUILD/'rgb-expanded/replay.json', config_path]
    rink,baseline,detector = [read(p)for p in inputs[:3]]
    seeds,tracks = [],[]
    for batch in args.batches.split(','):
        seed_path = BUILD/f'tracking-seeds-{batch}.json'
        track_path = BUILD/f'short-{args.tracker}-{batch}.json'
        seed,track = read(seed_path),read(track_path)
        assert track['seed_file_sha256']==sha(seed_path)
        seeds.append(seed);tracks.append(track);inputs.extend([seed_path,track_path])
    result,report = refine_sequence(rink,baseline,seeds,tracks,detector['frames'],config,
                                    excluded_frames=range(466,516))
    report['created_at']=datetime.now(timezone.utc).isoformat()
    report['input_sha256']={str(p.relative_to(ROOT)):sha(p)for p in inputs}
    report['implementation_sha256']=sha(ROOT/'hockey_tracking/ball_refinement.py')
    changes=[]
    for f in report['changed_frames']:
        a,b=baseline['frames'][f],result['frames'][f]
        distance=math.dist([a['pixel'][0]*1920,a['pixel'][1]*1080],
                           [b['pixel'][0]*1920,b['pixel'][1]*1080])if a.get('pixel')and b.get('pixel')else None
        changes.append({'frame':f,'before_state':a['state'],'state':b['state'],
                        'pixel':[b['pixel'][0]*1920,b['pixel'][1]*1080],
                        'before_pixel':[a['pixel'][0]*1920,a['pixel'][1]*1080]if a.get('pixel')else None,
                        'change_px':distance,'reason':b['reason']})
    seed_frames={r['frame'] for stream in seeds for r in stream['seeds']}
    candidates=[r for r in changes if r['state']=='tracked' and r['frame'] not in seed_frames and
                (r['change_px'] is None or r['change_px']>=8)]
    # The sample is fixed from the output distribution, before visual review.
    count=min(60,len(candidates))
    audit=[candidates[round(i*(len(candidates)-1)/max(1,count-1))]for i in range(count)]
    report['audit_selection']={'method':'60 evenly spaced changed non-seed tracked outputs, minimum8px displacement; post-prediction diagnostic, not independent ground truth.',
                               'population':len(candidates),'rows':audit}
    report['changes']=changes
    destination=BUILD/args.name;destination.mkdir(parents=True,exist_ok=True)
    for name,data in [('positions.json',result),('report.json',report)]:
        (destination/name).write_text(json.dumps(data,separators=(',',':'))+'\n')
    print(json.dumps({k:report[k]for k in ['seed_count','valid_seed_masks','accepted_seeds','before_states','after_states','changes_by_source']},indent=2))
    print('Review population',len(candidates),'sample',len(audit))


if __name__ == '__main__': main()
