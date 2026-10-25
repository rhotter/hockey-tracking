"""Evaluate marked floor-line alignment, keeping withheld frames in the denominator."""
import json
import argparse
from pathlib import Path
import numpy as np
from geometry import usable

def evaluate(registration,reference):
    byframe={r['frame']:r for r in registration['keyframes']};results={}
    for method in registration['keyframes'][0]['methods']:
        rows=[];errors=[];marked=0
        for frame in reference['frames']:
            selected=byframe[frame['frame']]['methods'][method]['selected']
            H=np.asarray(selected['H'])if selected else None
            valid=usable(H);samples=[]
            for group in frame['marks']:
                line=np.array(reference['line_definitions'][group['line']],float)
                image_line=H.T@line if valid else None
                for x,y in group['points']:
                    error=float(abs(image_line@[x,y,1])/np.linalg.norm(image_line[:2]))if valid else None
                    marked+=1
                    if error is not None:errors.append(error)
                    samples.append({'line':group['line'],'pixel':[x,y],'error_px':error})
            rows.append({'frame':frame['frame'],'camera_valid':valid,'points':samples})
        results[method]={'marked_points':marked,'answered_points':len(errors),
            'median_error_px':float(np.median(errors))if errors else None,
            'p90_error_px':float(np.quantile(errors,.9))if errors else None,
            'maximum_error_px':float(max(errors))if errors else None,
            'within_10px':sum(e<=10 for e in errors),'within_20px':sum(e<=20 for e in errors),'frames':rows,
            'accepted_keyframes':sum(r['methods'][method]['selected']is not None and usable(r['methods'][method]['selected']['H'])for r in registration['keyframes']),
            'total_keyframes':len(registration['keyframes'])}
    return {'methods':results,'reference_status':reference['status'],
        'scope':f'Assistant-marked image-to-line alignment on {len(reference["frames"])} development frames. Measures consistency with approximate rink template, not surveyed floor-position error, identity accuracy or out-of-game generalization.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('registration',type=Path);p.add_argument('reference',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result=evaluate(json.loads(a.registration.read_text()),json.loads(a.reference.read_text()));a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{n:v for n,v in r.items()if n!='frames'}for k,r in result['methods'].items()},indent=2))
