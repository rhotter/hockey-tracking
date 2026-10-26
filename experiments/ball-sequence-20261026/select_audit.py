"""Deterministic, conditional review samples from new estimates."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def select(data,baseline):
    context=[]
    for start in range(0,4500,150):
        for offset in [45,105]:
            target=start+offset
            eligible=[f for f in range(max(0,target-15),min(4500,target+16))if data['frames'][f]['state']=='context'and baseline['frames'][f]['state']=='unknown']
            if eligible:context.append(min(eligible,key=lambda f:(abs(f-target),f)))
    tracked=[r['frame']for r in data['frames']if r['state']=='tracked']
    points=sorted({tracked[i]for i in np.linspace(0,len(tracked)-1,min(20,len(tracked))).round().astype(int)})if tracked else[]
    return {'context':context,'tracked':points}

if __name__=='__main__':
    path=Path(sys.argv[1]);data=json.loads(path.read_text());baseline=json.loads((ROOT/'web/data/ball-rebuild-20261026/replay.json').read_text())
    result=select(data,baseline);result['replay_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    destination=path.parent/'audit-selection.json';destination.write_text(json.dumps(result,indent=2)+'\n')
    for k in ['context','tracked']:print(k,','.join(map(str,result[k])))
