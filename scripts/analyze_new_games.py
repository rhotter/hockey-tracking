"""Describe reconnection gains and residual failures without changing predictions."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'web/data/reliability/new-games-20261025'


def compute(data=DATA):
    result=json.loads((data/'results.json').read_text())
    raw=[];linked=[]
    for clip in result['clips']:
        for method,rows in [('existing-sam2-local',raw),('existing-sam2-osnet-qwen2b',linked)]:
            report=json.loads((data/'reports'/(clip['clip']+'-'+method+'.json')).read_text())
            rows.extend(report['pairs'])
    old={p['id']:p for p in raw}
    changed=[]
    for p in linked:
        before=old[p['id']]['predicted_same'];after=p['predicted_same']
        if before is not after:
            changed.append({k:p[k]for k in ['id','clip','a','b','truth']}|{'before':before,'after':after})
        if before is True:assert after is True
    return {
        'same_pairs_recovered':sum(p['truth']=='same'and p['after']is True for p in changed),
        'wrong_pair_joins_added':sum(p['truth']=='different'and p['after']is True for p in changed),
        'unverified_pair_joins_added':sum(p['truth']=='uncertain'and p['after']is True for p in changed),
        'remaining_same_pairs_missing_endpoint':sum(p['truth']=='same'and p['predicted_same']is None for p in linked),
        'remaining_same_pairs_different_ids':sum(p['truth']=='same'and p['predicted_same']is False for p in linked),
        'changes':changed,
        'scope':'Post-test description of frozen predictions. No parameter selection, reference changes or additional inference.'
    }


if __name__=='__main__':
    import shutil
    result=compute()
    (DATA/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    shutil.copy2(ROOT/'experiments/new-games-20261025/data/diagnostic-scene-review.json',DATA/'diagnostic-scene-review.json')
    print(json.dumps({k:v for k,v in result.items()if k!='changes'}))
