"""Exploratory fixed-rule ensemble of frozen detector outputs."""
from pathlib import Path
from datetime import datetime, timezone
import itertools
import hashlib
import json
import math
from evaluate import evaluate

HERE=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol=read(HERE/'ensemble-protocol.json')
    labels=read(HERE/'fresh-reference.json')['rows']+read(HERE/'extra-reference.json')['rows']
    models=protocol['models'];points={name:{} for name in models}
    hashes={}
    for scope in ['fresh','extra']:
        path=HERE/f'results/{scope}-evaluation.json';report=read(path);hashes[path.name]=sha(path)
        for name in models:
            for row in report['models'][name]['at_25px']['details']:
                points[name][(row['clip'],row['frame'])]=row['pixel']
    rows=[];predictions={name:[] for name in protocol['rules']}
    for label in labels:
        key=(label['clip'],label['frame'])
        available=[name for name in models if points[name][key] is not None]
        groups=[]
        for size in range(2,len(available)+1):
            for group in itertools.combinations(available,size):
                if all(math.dist(points[a][key],points[b][key])<=protocol['agreement_radius_px']
                       for a,b in itertools.combinations(group,2)):
                    groups.append(group)
        chosen=max(groups,key=lambda g:(len(g),'yolo-full1920' in g)) if groups else None
        consensus=[sum(points[name][key][i] for name in chosen)/len(chosen) for i in range(2)] if chosen else None
        fallback=points['yolo-full1920'][key]
        for name in predictions:
            point=consensus if consensus is not None else fallback if name=='majority_else_yolo' else None
            predictions[name].append({'clip':key[0],'frame':key[1],'source_sha256':label['source_sha256'],
                'candidates':[{'pixel':point,'score':1.}] if point is not None else []})
        rows.append({'clip':key[0],'frame':key[1],'voters':chosen,'consensus':consensus})
    result={'created_at':datetime.now(timezone.utc).isoformat(),'protocol':protocol,'input_sha256':hashes,
            'models':{name:{f'at_{radius}px':evaluate(frames,labels,{'threshold':.5,'appearance_power':0},radius)
                           for radius in [12,25]} for name,frames in predictions.items()},'rows':rows}
    path=HERE/'results/ensemble-evaluation.json';assert not path.exists()
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({name:{k:v for k,v in r['at_25px'].items() if k!='details'}
                      for name,r in result['models'].items()},indent=2))


if __name__=='__main__':main()
