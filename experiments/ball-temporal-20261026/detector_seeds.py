"""Find additional ball sightings using agreement of three heatmap variants."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BUILD=ROOT/'build/ball-temporal-20261026'
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    models=['rgb-expanded','temporal3-expanded','motion3-expanded','motion3-synthetic']
    paths={name:BUILD/name/'replay.json' for name in models}
    datasets={name:read(path)['frames'] for name,path in paths.items()}
    baseline=read(ROOT/'web/data/ball-sam-20261026/positions.json')['frames']
    existing=[seed for batch in ['dense-full-gpu','dense-discovery-gpu']
              for seed in read(BUILD/f'tracking-seeds-{batch}.json')['seeds']]
    proposals=[]
    for f in range(4500):
        if 466<=f<516 or baseline[f]['state']=='detected':continue
        peaks=[]
        for name,frames in datasets.items():
            if not frames[f]['candidates']:continue
            p=max(frames[f]['candidates'],key=lambda c:c['score']*c['appearance_score'])
            if p['score']>=.01 and p['appearance_score']>=.8:peaks.append(dict(p,model=name))
        groups=[]
        for p in peaks:
            group={q['model']:q for q in peaks if math.dist(p['pixel'],q['pixel'])<=6}
            if len(group)>=3 and max(q['score']for q in group.values())>=.5:
                groups.append((len(group),sum(q['score']for q in group.values()),p,group))
        if not groups:continue
        _,_,point,group=max(groups,key=lambda g:g[:2])
        if any(abs(s['frame']-f)<=2 and math.dist(s['pixel'],point['pixel'])<=16 for s in existing):continue
        proposals.append({'frame':f,'pixel':point['pixel'],'appearance_score':point['appearance_score'],
            'models':sorted(group),'sum_score':sum(p['score']for p in group.values()),
            'source':'detector_consensus','support':'unclear',
            'evidence':'Three or more heatmap variants agree on this ball candidate; SAM checks its motion.',
            'source_sha256':datasets['rgb-expanded'][f]['source_sha256']})
    seeds=[]
    for row in sorted(proposals,key=lambda r:(-len(r['models']),-r['sum_score'],r['frame'])):
        if all(abs(row['frame']-s['frame'])>=3 for s in seeds):seeds.append(row)
    seeds.sort(key=lambda r:r['frame'])
    protocol={'created_at':datetime.now(timezone.utc).isoformat(),
        'selection':'Top peak per model by score times appearance. At least3 models within6px; every vote score>=.01 and appearance>=.8; at least one score>=.5. Skip existing visual anchors and nearby sequence sightings; minimum seed spacing3frames.',
        'input_sha256':{str(path.relative_to(ROOT)):sha(path)for path in paths.values()},
        'source':'detector_consensus','reference_labels_used':False,
        'limits':'Added after inspecting residual development failures. Models share training data and are correlated. No independent accuracy claim.',
        'proposals':len(proposals),'seeds':len(seeds)}
    result={'batch':'dense-detector-gpu','pilot':False,'seeds':seeds,'samples':proposals,
            'half_window_frames':6,'crop_size':[768,576],'offload_state_to_cpu':False,
            'selection_protocol':protocol}
    path=BUILD/'tracking-seeds-dense-detector-gpu.json'
    assert not path.exists(),'Seed selection is sealed.'
    path.write_text(json.dumps(result,indent=2)+'\n')
    (HERE/'detector-seed-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({'proposals':len(proposals),'seeds':len(seeds)}))


if __name__=='__main__':main()
