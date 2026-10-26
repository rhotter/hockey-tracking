"""Fixed-radius, single-ball localization; never turn unknown visibility into negatives."""
import json
import math
from pathlib import Path


def center(box):
    return [(box[0]+box[2])/2, (box[1]+box[3])/2]


def score(labels, predictions, radius=12):
    tp=fp=fn=tn=unknown=0; errors=[]; details=[]
    for label,point in zip(labels,predictions,strict=True):
        visibility=label.get('visibility','visible')
        if visibility=='unknown': unknown+=1; continue
        target=label.get('center') or (label.get('ball') or {}).get('center')
        error=math.dist(point,target) if point is not None and target is not None else None
        hit=error is not None and error<=radius
        if visibility=='visible':
            tp+=hit;fn+=not hit;fp+=point is not None and not hit
            if hit:errors.append(error)
        else:
            fp+=point is not None;tn+=point is None
        details.append({'frame':label['frame'],'clip':label.get('clip'),'hit':hit,'error_px':error,'prediction':point,'target':target})
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'unknown_excluded':unknown,
            'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'mean_correct_error_px':sum(errors)/len(errors) if errors else None,'details':details}


def top1(rows,threshold):
    out=[]
    for row in rows:
        boxes=row['detections'];best=max(boxes,key=lambda b:b[4]) if boxes else None
        out.append(center(best) if best and best[4]>=threshold else None)
    return out


def multiview_regression(folder):
    from hockey_tracking.ball_path import fuse_candidates
    raw=[json.loads((folder/f'{name}-regression.json').read_text()) for name in ['full1280','full1920','tiles640']]
    fused=fuse_candidates([r['frames'] for r in raw])
    for r,s in zip(fused,raw[0]['frames']):r.update(frame=s['frame'],clip=s['clip'])
    return {'variant':'multi-view','weights_sha256':raw[0]['weights_sha256'],'frames':fused,
            'limits':'Single-frame multi-view detector comparison; does not include pose masks or temporal path selection.'}


def verified_regression(folder, verifier_name, threshold):
    from hockey_tracking.ball_path import filter_verified
    raw=multiview_regression(folder)
    verification=json.loads((folder/f'{verifier_name}-regression.json').read_text())
    filtered=filter_verified(raw['frames'],verification['frames'],threshold)
    for row,source in zip(filtered,raw['frames']):row.update(frame=source['frame'],clip=source['clip'])
    return {**raw,'variant':'verified-multi-view','frames':filtered,
            'verifier_weights_sha256':verification['weights_sha256'],'verification_threshold':threshold,
            'limits':'Single-frame detector plus crop verifier. No person/line masks or temporal path; this is a reused cross-game regression check.'}


def evaluate_regression(folder):
    labels=json.loads((Path(__file__).parent/'regression-labels.json').read_text())['rows']
    results={}
    for file in sorted(folder.glob('*-regression.json')):
        data=json.loads(file.read_text());rows=data['frames']
        # Crop-verifier probabilities are auxiliary gates, not box predictions.
        if 'variant' not in data:continue
        assert len(rows)==len(labels)
        assert all((a['frame'],a['clip'])==(b['frame'],b['clip']) for a,b in zip(rows,labels))
        dev=[i for i,r in enumerate(labels) if r['split']=='development']
        test=[i for i,r in enumerate(labels) if r['split']=='regression']
        thresholds=[.025,.05,.075,.1,.15,.2,.3,.4,.5,.6,.75,.9]
        options=[]
        for threshold in thresholds:
            pred=top1(rows,threshold)
            s=score([labels[i] for i in dev],[pred[i] for i in dev]);s.pop('details')
            options.append({'threshold':threshold,**s})
        best=max(options,key=lambda s:(s['f1'] or 0,-s['fp'],s['threshold']))
        pred=top1(rows,best['threshold'])
        groups={source:score([labels[i] for i in test if labels[i]['source_id']==source],
                           [pred[i] for i in test if labels[i]['source_id']==source])
                for source in sorted({labels[i]['source_id'] for i in test})}
        results[data['variant']]={'threshold':best['threshold'],'development':best,
                    'regression':score([labels[i] for i in test],[pred[i] for i in test]),
                    'by_source':groups,'operating_points':options,'weights_sha256':data['weights_sha256']}
    return results


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args();result=evaluate_regression(args.folder)
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    for k,r in result.items():print(k,'threshold',r['threshold'],'dev',r['development']['tp'],r['development']['fp'],
                                  'regression',r['regression']['tp'],r['regression']['fp'],r['regression']['fn'])
