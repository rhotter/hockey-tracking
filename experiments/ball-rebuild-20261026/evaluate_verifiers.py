"""Compare rejection gates on development and the explicitly reused diagnostic audit."""
import sys,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from hockey_tracking.ball_path import fuse_candidates,filter_verified,filter_person_parts,select_path,PathConfig
from hockey_tracking.ball_track import make_replay
from evaluate import score,top1,multiview_regression


def main(folder,appearance_name='appearance.json'):
    exp=Path(__file__).parent;replay=json.loads((ROOT/'web/data/overhead-20261025/replay.json').read_text())
    fused=fuse_candidates([json.loads((folder/f'{s}.json').read_text())['frames']for s in ['full1280','full1920','tiles640']])
    appearance=json.loads((folder/appearance_name).read_text())['frames'];people=json.loads((folder/'people-pose-tiled.json').read_text())['frames']
    shape_config={'min_area':100,'min_major_sigma':7,'min_axis_ratio':2,'max_distance':3} if appearance_name=='appearance-shape.json' else None
    labels=[]
    for name in ['review','targeted']:labels.extend(r for r in json.loads((exp/f'{name}-labels.json').read_text())['rows'] if r['split']=='development')
    audit=json.loads((exp/'output-audit-v1.json').read_text())['rows']
    good=[{'frame':r['frame'],'center':r['pixel'],'visibility':'visible'}for r in audit if r['verdict']=='visible_ball']
    negatives=[r for r in audit if r['verdict']=='wrong_location']
    regression_labels=json.loads((exp/'regression-labels.json').read_text())['rows'];reg_raw=multiview_regression(folder)['frames']
    dev_indices=[i for i,r in enumerate(regression_labels)if r['split']=='development'];results=[]
    for name in ['verifier','verifier-v2','verifier-v3','verifier-v4']:
        if appearance_name=='appearance-shape.json' and name!='verifier-v4':continue
        if not (folder/f'{name}.json').exists():continue
        verification=json.loads((folder/f'{name}.json').read_text())
        reg_verify=json.loads((folder/f'{name}-regression.json').read_text())['frames']
        for gate in [.001,.01,.05,.2,.5,.8,.9]:
            if appearance_name=='appearance-shape.json' and gate<.01:continue
            candidates=filter_person_parts(filter_verified(fused,verification['frames'],gate,appearance,shape_config),people)
            reg_candidates=filter_verified(reg_raw,reg_verify,gate)
            reg_dev=score([regression_labels[i] for i in dev_indices],[top1(reg_candidates,.1)[i]for i in dev_indices])
            for evidence in [.08,.15]:
                path=select_path(candidates,replay,PathConfig(evidence_floor=evidence),excluded_frames=range(466,516))
                data=make_replay(replay,path['detections']);points=[npixel(r) for r in data['frames']]
                s=score(labels,[points[r['frame']]for r in labels]);g=score(good,[points[r['frame']]for r in good])
                repeated=sum(points[r['frame']] is not None and math.dist(points[r['frame']],r['pixel'])<20 for r in negatives)
                record={'name':name,'gate':gate,'evidence':evidence,'appearance':appearance_name,'shape_config':shape_config,'development':s,'diagnostic_positive':g,'diagnostic_errors_repeated':repeated,
                        'old_development':reg_dev,'outputs':sum(p is not None for p in points),'weights_sha256':verification['weights_sha256']}
                results.append(record);stem=f'{name}-g{gate}-p{evidence}'
                (folder/f'{stem}-replay.json').write_text(json.dumps(data,separators=(',',':'))+'\n')
                print(stem,'dev',s['tp'],s['fp'],'diagnostic',g['tp'],g['fp'],'errors',repeated,'olddev',reg_dev['tp'],reg_dev['fp'],'outputs',record['outputs'],flush=True)
    (folder/'verifier-development-scores.json').write_text(json.dumps(results,indent=2)+'\n')


def npixel(r):return [r['pixel'][0]*1920,r['pixel'][1]*1080] if r.get('pixel') else None
if __name__=='__main__':main(Path(sys.argv[1]),sys.argv[2] if len(sys.argv)>2 else 'appearance.json')
