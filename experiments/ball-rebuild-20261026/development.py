"""Development-only trajectory ablations. Reserved review frames are not read."""
from pathlib import Path
import json,sys,math
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from hockey_tracking.ball_path import select_path,PathConfig,fuse_candidates,filter_person_parts,filter_markings
from evaluate import center,score


def run(folder):
    replay=json.loads((ROOT/'web/data/overhead-20261025/replay.json').read_text())
    labels=[]
    for name in ['review','targeted']:
        labels.extend(r for r in json.loads((Path(__file__).parent/f'{name}-labels.json').read_text())['rows'] if r['split']=='development')
    audit=json.loads((Path(__file__).parent/'development-audit.json').read_text())['rows']
    good=[{'frame':r['frame'],'center':r['pixel'],'visibility':'visible'} for r in audit if r['verdict']=='visible_ball']
    people_file=folder/('people-pose-tiled.json' if (folder/'people-pose-tiled.json').exists() else 'people-pose.json')
    people=json.loads(people_file.read_text())['frames'] if people_file.exists() else None
    variants={}
    for name in ['full1280','full1920','full1920-fast','tiles640','heatmap-rgb-dense','heatmap-temporal3-dense']:
        if (folder/f'{name}.json').exists():variants[name]=json.loads((folder/f'{name}.json').read_text())['frames']
    if all(k in variants for k in ['full1280','full1920','tiles640']):
        variants['fusion']=fuse_candidates([variants[k] for k in ['full1280','full1920','tiles640']])
        if (folder/'appearance.json').exists():variants['fusion-lines']=filter_markings(variants['fusion'],json.loads((folder/'appearance.json').read_text())['frames'])
    result=[]
    for name,raw in variants.items():
        for use_people in ([False,True] if people else [False]):
            candidates=filter_person_parts(raw,people) if use_people else raw
            for threshold in [.08,.15,.3]:
                config=PathConfig(evidence_floor=threshold)
                # Score only development, though offline path optimization may use
                # adjacent unlabeled future imagery, just as deployment does.
                path=select_path(candidates,replay,config,excluded_frames=range(466,516))
                pred=[center(b) if b else None for b in path['detections']]
                measured=score(labels,[pred[r['frame']] for r in labels]);audited=score(good,[pred[r['frame']] for r in good])
                wrong=sum(pred[r['frame']] is not None and math.dist(pred[r['frame']],r['pixel'])<20 for r in audit if r['verdict']=='wrong_object')
                record={'variant':name,'people':use_people,'person_context':people_file.name,'threshold':threshold,'development':measured,'audit_positive':audited,
                        'repeated_known_false_objects':wrong,'known_false_object_samples':sum(r['verdict']=='wrong_object' for r in audit),'proposed_frames':sum(p is not None for p in pred)}
                result.append(record)
                stem=f'{name}-people{int(use_people)}-p{threshold}'
                (folder/(stem+'-path.json')).write_text(json.dumps(path,separators=(',',':'))+'\n')
                print(stem,'review',measured['tp'],measured['fp'],measured['fn'],'audit',audited['tp'],audited['fp'],audited['fn'],'known wrong',wrong,'outputs',record['proposed_frames'])
    (folder/'development-scores.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':run(Path(sys.argv[1]))
