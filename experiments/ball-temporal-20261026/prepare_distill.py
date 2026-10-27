"""Seal teacher-data splits and unlabeled transfer proposals before head training."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
from evaluate import top_candidate

HERE=Path(__file__).resolve().parent
BUILD=HERE.parents[1]/'build/ball-temporal-20261026'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')

def main():
    assert not (HERE/'distill-protocol.json').exists(),'Protocol is sealed.'
    rows=[]
    for source in ['detector','yolo']:
        for r in read(HERE/f'results/{source}-seed-verification.json')['rows']:
            if r['verdict']not in ['visible_ball','non_ball']:continue
            f=r['frame']
            split='train' if f<2694 else 'development' if 2706<=f<3594 else 'test' if f>=3606 else 'purged'
            if split=='purged':continue
            rows.append({'frame':f,'pixel':r['pixel'],'source_sha256':r['source_sha256'],
                'label':int(r['verdict']=='visible_ball'),'split':split,'teacher_reason':r['reason'],'proposal_source':source})
    write(HERE/'distill-labels.json',{'scope':'Model-generated sequence verdicts, not human labels; chronological split with12frame purges.','rows':rows})
    point=read(HERE/'results/development-selection.json')['variants']['yolo-full1920']['operating_point'];transfer=[]
    for scope in ['fresh','extra','final-check']:
        keys={(r['clip'],r['frame'])for r in read(HERE/f'{scope}-reference.json')['rows']}
        for r in read(BUILD/f'yolo-full1920/{scope}.json')['frames']:
            if (r['clip'],r['frame'])not in keys:continue
            candidate=top_candidate(r,**point)
            transfer.append({k:r[k]for k in ['clip','frame','source_sha256']}|{'pixel':candidate['pixel']if candidate else None})
    assert len(transfer)==150
    write(HERE/'results/distill-transfer-inputs.json',{'selection':'Previously frozen YOLO top candidate at the150fixed review times; no reference coordinates.','rows':transfer})
    protocol={'created_at':datetime.now(timezone.utc).isoformat(),'warmstart_sha256':'88d18c801ca34dfbe1decdbc4b506acfccad3f43a02e507b49f232722f00700a',
        'variants':['original','image-head','temporal-head'],'encoder':'Frozen SpatialBallVerifier ResNet18 stride4;64px and192px crops resized128; center plus global pooled features.',
        'temporal_input':'Camera-affine-aligned frames -3,0,+3; concatenate center features and signed neighbor differences; initialize added weights to zero.',
        'training':'Only classifier heads updated. Balanced16positive+16negative batches,8steps per epoch; AdamW weight_decay.01; no labels from evaluation games.',
        'epochs':48,'seed':61,'learning_rate':.0001,'thresholds':[.02,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99],
        'selection':'Development F1, then fewer false positives; threshold ties prefer higher threshold; epoch ties retain earlier epoch. Compare image and temporal heads using this development criterion, simplicity as final tie break.',
        'teacher_labels_sha256':sha(HERE/'distill-labels.json'),'transfer_inputs_sha256':sha(HERE/'results/distill-transfer-inputs.json'),
        'test_scope':'Later same-game proposals use teacher verdicts, not independent truth. Transfer uses previously opened fixed new-game points, exploratory only.'}
    write(HERE/'distill-protocol.json',protocol)
    print({'teacher_rows':len(rows),'transfer':len(transfer)})

if __name__=='__main__':main()
