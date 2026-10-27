"""Score the frozen disagreement policy without relabeling its evaluation images."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
from evaluate import evaluate

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol=read(HERE/'adjudication-protocol.json')
    ledger_path=ROOT/'build/ball-temporal-20261026/adjudication-ledger.json'
    ledger=read(ledger_path)
    assert ledger['protocol']==protocol
    labels=read(HERE/'fresh-reference.json')['rows']+read(HERE/'extra-reference.json')['rows']
    points={}
    for scope in ['fresh','extra']:
        path=HERE/f'results/{scope}-evaluation.json'
        assert sha(path)==protocol['input_sha256'][scope]
        for row in read(path)['models']['yolo-full1920']['at_25px']['details']:
            points[(row['clip'],row['frame'])]=row['pixel']
    reviews=[]
    for request in ledger['requests'].values():
        key=(request['clip'],request['frame']);before=points[key]
        answer=request.get('answer')
        if request['status']=='completed' and answer:
            choice=answer['choice']
            if choice.startswith('candidate'):
                points[key]=next(c['pixel'] for c in request['candidates'] if c['id']==int(choice[-1]))
            elif choice=='neither':
                xy=answer['alternative_xy_1000']
                points[key]=[xy[0]*request['width']/1000,xy[1]*request['height']/1000] if xy else None
        label=next(r for r in labels if (r['clip'],r['frame'])==key)
        assert request['source_sha256'][str(key[1])]==label['source_sha256']
        reviews.append({'clip':key[0],'frame':key[1],'status':request['status'],
            'answer':answer,'before_pixel':before,'pixel':points[key],
            'source_sha256':label['source_sha256'],'estimated_usd':request.get('estimated_usd',request.get('reserved_usd',0))})
    frames=[{'clip':r['clip'],'frame':r['frame'],'source_sha256':r['source_sha256'],
             'candidates':[{'pixel':points[(r['clip'],r['frame'])],'score':1.}] if points[(r['clip'],r['frame'])] is not None else []} for r in labels]
    result={'created_at':datetime.now(timezone.utc).isoformat(),'protocol':protocol,
        'ledger_sha256':sha(ledger_path),'estimated_usd':sum(r['estimated_usd'] for r in reviews),
        'reviews':reviews,'models':{'yolo_with_sequence_adjudication':{
            f'at_{radius}px':evaluate(frames,labels,{'threshold':.5,'appearance_power':0},radius) for radius in [12,25]}}}
    path=HERE/'results/adjudication-evaluation.json';assert not path.exists()
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result['models']['yolo_with_sequence_adjudication']['at_25px'].items() if k!='details'},indent=2))


if __name__=='__main__':main()
