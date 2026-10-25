"""Fit one global floor rectification from three calibration frames; reserve the rest."""
import copy
import numpy as np
from scipy.optimize import least_squares
from geometry import project,usable

FIT_FRAMES=[450,1800,2700]

def refine(variant,volume):
    from overhead import OUT,read,dump,sha
    from evaluate_geometry import evaluate
    raw=read(OUT/'registration.json');reference=read('/work/floor-line-review-v2.json')
    lookup={r['frame']:r for r in raw['keyframes']};observations=[]
    for frame in reference['frames']:
        if frame['frame']not in FIT_FRAMES:continue
        H=np.asarray(lookup[frame['frame']]['methods']['background']['selected']['H'])
        for group in frame['marks']:
            line=np.asarray(reference['line_definitions'][group['line']],float)
            for point in group['points']:
                if frame['frame']==1800 and group['line']=='center'and point==[451,600]:continue
                observations.append((H,line,point))
    def matrix(x):return np.array([[1+x[0],x[1],x[2]],[x[3],1+x[4],x[5]],[x[6],x[7],1.]])
    def residual(x):
        C=matrix(x);values=[]
        for H,line,point in observations:
            l=(C@H).T@line;values.append(float(l@[*point,1]/np.linalg.norm(l[:2])))
        # Preserve the manually defined center-circle origin and local y scale.
        values.extend((project(C,[[0,0]])[0]*1000).tolist())
        values.extend((x*2).tolist())
        return values
    result=least_squares(residual,np.zeros(8),loss='soft_l1',f_scale=3,
                         bounds=([-.15,-.1,-.05,-.1,-.15,-.05,-.05,-.05],[.15,.1,.05,.1,.15,.05,.05,.05]),max_nfev=1000)
    C=matrix(result.x);out=copy.deepcopy(raw)
    old_bounds=raw['calibration']['relative_bounds']
    ends=project(C,[[old_bounds['left'],0],[old_bounds['right'],0]])
    out['calibration']['relative_bounds']={'left':float(ends[0,0]),'right':float(ends[1,0]),'far':-1.,'near':1.}
    out['calibration']['image_to_relative_floor']=(C@np.asarray(raw['calibration']['image_to_relative_floor'])).tolist()
    out['calibration']['coordinate_definition']+=' Refined from three development frames; end extents transformed at the original mid-width. Ends remain approximate and near boundary retains symmetry assumption.'
    for row in out['keyframes']:
        base=row['methods']['background'];value=copy.deepcopy(base)
        if value['selected']:value['selected']['H']=(C@np.asarray(value['selected']['H'])).tolist()
        for candidate in value['candidates']:candidate['H']=(C@np.asarray(candidate['H'])).tolist()
        row['methods']['refined']=value
    out['global_floor_refinement']={'fit_frames':FIT_FRAMES,'fit_points':len(observations),
        'matrix':C.tolist(),'optimizer_success':bool(result.success),'base_registration_sha256':sha(OUT/'registration.json'),
        'reference_sha256':sha('/work/floor-line-review-v2.json'),
        'assumptions':'Blue lines parallel and equally spaced about center; far board/floor edge is straight. Relative coordinate system only; center-circle origin retained.',
        'method':'Robust global homography correction from raw paint/board marks on three frames. Other reference frames are withheld from the fit. Per-frame camera registration remains automatic.'}
    holdout=copy.deepcopy(reference);holdout['frames']=[f for f in reference['frames']if f['frame']not in FIT_FRAMES]
    fitref=copy.deepcopy(reference);fitref['frames']=[f for f in reference['frames']if f['frame']in FIT_FRAMES]
    report={'refinement':out['global_floor_refinement'],'heldout':evaluate(out,holdout),'fit':evaluate(out,fitref)}
    dump(OUT/'registration-final.json',out);dump(OUT/'geometry-final.json',report)
    return {'refinement':report['refinement'],'heldout':{k:{a:b for a,b in v.items()if a!='frames'}for k,v in report['heldout']['methods'].items()}}
