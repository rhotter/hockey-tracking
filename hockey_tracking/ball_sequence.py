"""Turn sequence-level ball hypotheses into a dense, explicitly inferred replay.

The model reasons over past and future video. Between its sampled states, a
controlled ball follows the identified carrier's observed motion when possible;
otherwise regions move in camera-compensated coordinates. Alternative locations
are retained. These are model estimates, not ground-truth or calibrated regions.
"""
from bisect import bisect_left
from copy import deepcopy
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from .ball_context import normalized_homography, project, jacobian


@dataclass(frozen=True)
class SequenceConfig:
    minimum_radius_px: float = 60.
    maximum_radius_px: float = 180.
    minimum_known_weight: float = .55
    minimum_dominant_weight: float = .4
    minimum_hypothesis_weight: float = .12
    maximum_sample_gap_frames: int = 9
    maximum_transition_px: float = 360.
    edge_hold_frames: int = 3


def camera_warp(replay, source_frame, target_frame, pixel):
    a=replay['frames'][source_frame].get('image_to_floor')
    b=replay['frames'][target_frame].get('image_to_floor')
    if a is None or b is None:return np.asarray(pixel,float)
    try:
        q=project(np.linalg.inv(np.asarray(b))@np.asarray(a),pixel)
        return q if np.isfinite(q).all() else np.asarray(pixel,float)
    except np.linalg.LinAlgError:return np.asarray(pixel,float)


def carrier_foot(replay,frame,track_id):
    if track_id is None:return None
    track_id=str(track_id).removeprefix('T')
    for person in replay['frames'][frame]['objects']:
        if str(person['track_id'])==track_id:
            if person.get('foot') is not None:return np.array(person['foot'])*[1920,1080]
            x1,_,x2,y2=person['bbox'];return np.array([(x1+x2)/2*1920,y2*1080])
    return None


def read_nodes(ledger):
    nodes={}
    for key,record in ledger['requests'].items():
        if record['status']!='completed' or not record.get('answer'):continue
        for row in record['answer']['samples']:
            f=row['frame']
            if f not in record['targets']:raise ValueError('Unexpected context target')
            if f in nodes:raise ValueError('Overlapping target states')
            node=deepcopy(row);node['window_id']=key
            for h in node['hypotheses']:
                h['pixel']=np.asarray(h['xy_1000'],float)*[1.92,1.08]
                if not np.isfinite(h['pixel']).all():raise ValueError('Invalid context coordinates')
            nodes[f]=node
    return nodes


def interpolate_hypotheses(replay,a,b,frame,config):
    af,bf=a['frame'],b['frame'];u=0 if af==bf else (frame-af)/(bf-af)
    aa=a['hypotheses'];bb=b['hypotheses']
    if not aa or not bb:return []
    ap=[camera_warp(replay,af,frame,h['pixel'])for h in aa]
    bp=[camera_warp(replay,bf,frame,h['pixel'])for h in bb]
    distances=np.linalg.norm(np.array(ap)[:,None,:]-np.array(bp)[None,:,:],axis=2)
    cost=distances.copy()
    for i,x in enumerate(aa):
        for j,y in enumerate(bb):
            if x['carrier_track_id'] and y['carrier_track_id'] and x['carrier_track_id']!=y['carrier_track_id']:cost[i,j]+=60
    pairs=zip(*linear_sum_assignment(cost));out=[]
    for i,j in pairs:
        x,y=aa[i],bb[j];distance=float(distances[i,j])
        if distance>config.maximum_transition_px:continue
        weight=float((1-u)*x['weight']+u*y['weight'])
        if weight<config.minimum_hypothesis_weight:continue
        center=(1-u)*ap[i]+u*bp[j];motion='camera_compensated_region'
        carrier=x['carrier_track_id']if x['carrier_track_id']==y['carrier_track_id']else None
        if carrier and a['play_state'] in ['carried','dead_ball'] and b['play_state'] in ['carried','dead_ball']:
            ca=carrier_foot(replay,af,carrier);cb=carrier_foot(replay,bf,carrier);cf=carrier_foot(replay,frame,carrier)
            if ca is not None and cb is not None and cf is not None:
                # Preserve the inferred stick/ball offset while following actual player motion.
                center=cf+(1-u)*(x['pixel']-ca)+u*(y['pixel']-cb);motion='carrier_motion'
        radius=max(config.minimum_radius_px,(1-u)*x['radius_px']+u*y['radius_px'])+u*(1-u)*distance*.35
        radius+=max(0,abs(frame-af)if af==bf else 0)*4
        if radius>config.maximum_radius_px or not np.isfinite(center).all() or np.any(center<0) or np.any(center>[1920,1080]):continue
        support=x['support']if x['support']==y['support']else'unclear'
        out.append({'pixel':center,'radius_px':float(radius),'weight':weight,'carrier_track_id':carrier,'support':support,'motion_source':motion})
    return sorted(out,key=lambda h:-h['weight'])


def map_hypothesis(replay,frame,h):
    p=h['pixel'];pcov=np.eye(2)*(h['radius_px']/2)**2;xy=cov=None
    row=replay['frames'][frame]
    if h['support']=='floor' and row.get('camera_valid') and row.get('image_to_floor') is not None:
        H=normalized_homography(row,replay['relative_bounds']);floor=project(H,p)
        if np.isfinite(floor).all() and np.all(floor>=0) and np.all(floor<=1):
            xy=floor.tolist();J=jacobian(H,p);cov=(J@pcov@J.T).tolist()
    return {**h,'pixel':(p/[1920,1080]).tolist(),'xy':xy,'covariance':cov,'pixel_covariance':pcov.tolist()}


def build_context_replay(replay,ledger,config=SequenceConfig(),excluded_frames=()):
    nodes=read_nodes(ledger);keys=sorted(nodes);excluded=set(excluded_frames);out=[]
    for frame in range(len(replay['frames'])):
        unknown={'frame':frame,'state':'unknown','reason':'sequence_context_unresolved'}
        if frame in excluded or not keys:
            out.append({**unknown,'reason':'edit_or_unavailable_context'});continue
        k=bisect_left(keys,frame)
        if k<len(keys) and keys[k]==frame:a=b=nodes[frame]
        elif 0<k<len(keys) and keys[k]-keys[k-1]<=config.maximum_sample_gap_frames:
            a,b=nodes[keys[k-1]],nodes[keys[k]]
            if any(f in excluded for f in range(a['frame'],b['frame']+1)):out.append(unknown);continue
        else:
            nearest=min(keys[max(0,k-1):min(len(keys),k+1)],key=lambda f:abs(f-frame))
            if abs(nearest-frame)>config.edge_hold_frames:out.append(unknown);continue
            a=b=nodes[nearest]
        hypotheses=interpolate_hypotheses(replay,a,b,frame,config)
        if not hypotheses or sum(h['weight']for h in hypotheses)<config.minimum_known_weight or hypotheses[0]['weight']<config.minimum_dominant_weight:
            out.append(unknown);continue
        mapped=[map_hypothesis(replay,frame,h)for h in hypotheses];best=mapped[0]
        nearest=a if abs(frame-a['frame'])<=abs(frame-b['frame'])else b
        out.append({'frame':frame,'state':'context','reason':'sequence_inference',
                    'pixel':best['pixel'],'xy':best['xy'],'covariance':best['covariance'],'pixel_covariance':best['pixel_covariance'],
                    'model_radius_px':best['radius_px'],'hypotheses':mapped,
                    'known_weight':sum(h['weight']for h in mapped),'play_state':nearest['play_state'],'visibility':nearest['visibility'],
                    'review_support':best['support'],'evidence_frames':sorted({a['frame'],b['frame']}),
                    'window_ids':sorted({a['window_id'],b['window_id']}),'evidence':nearest['transition_evidence'],
                    'nearest_evidence_s':min(abs(frame-a['frame']),abs(frame-b['frame']))/replay['fps']})
    return {'schema_version':1,'source_id':replay['source_id'],'start_s':replay['start_s'],'duration_s':replay['duration_s'],'fps':replay['fps'],
            'method':'Sequence-level ball-state hypotheses with carrier motion and camera compensation','uncertainty_calibrated':False,
            'independent_human_ground_truth':False,'config':asdict(config),'frames':out}


def merge_visual_and_context(visual,context):
    if len(visual['frames'])!=len(context['frames']):raise ValueError('Replay length mismatch')
    result=deepcopy(context)
    for f,row in enumerate(visual['frames']):
        if row['state']!='unknown':result['frames'][f]=deepcopy(row)
    result['method']='Visual anchors plus sequence-level contextual ball inference'
    return result


@dataclass(frozen=True)
class PointFusionConfig:
    appearance_threshold: float = .3
    standalone_appearance_threshold: float = .9
    standalone_seed_distance_frames: int = 12
    minimum_run_frames: int = 3
    maximum_step_px: float = 120.
    radius_px: float = 24.


def fuse_point_tracks(replay,visual,context,verified,config=PointFusionConfig()):
    """Refine context with visible, appearance-verified video tracks.

    Latent tracker coordinates are never promoted to observations. A weaker
    appearance proposal must agree with a contextual region. Isolated hits are
    removed, and an unresolved sequence needs strong appearance close to a seed.
    """
    result=merge_visual_and_context(visual,context);proposals={}
    for f,row in enumerate(verified['frames']):
        candidates=[];hypotheses=context['frames'][f].get('hypotheses',[])
        for p in row['proposals']:
            if not p['visible'] or not p['inside_image'] or not p['inside_crop'] or p['appearance_score']<config.appearance_threshold:continue
            matches=[h for h in hypotheses if np.linalg.norm(np.array(h['pixel'])*[1920,1080]-p['pixel'])<=h['radius_px']]
            standalone=p['appearance_score']>=config.standalone_appearance_threshold and abs(f-p['seed_frame'])<=config.standalone_seed_distance_frames
            if not matches and not standalone:continue
            support=max(matches,key=lambda h:h['weight'])['support']if matches else'unclear'
            candidates.append({**p,'support':support,'context_agreement':bool(matches)})
        if candidates:proposals[f]=max(candidates,key=lambda p:(p['appearance_score'],-abs(f-p['seed_frame'])))
    runs=[];run=[]
    for f,p in proposals.items():
        if run:
            previous=run[-1]
            distance=np.linalg.norm(camera_warp(replay,previous,f,proposals[previous]['pixel'])-p['pixel'])
            if f!=previous+1 or distance>config.maximum_step_px:runs.append(run);run=[]
        run.append(f)
    if run:runs.append(run)
    for run in runs:
        if len(run)<config.minimum_run_frames:continue
        for f in run:
            if visual['frames'][f]['state']!='unknown':continue
            p=proposals[f]
            mapped=map_hypothesis(replay,f,{'pixel':np.array(p['pixel']), 'radius_px':config.radius_px,'support':p['support']})
            result['frames'][f]={'frame':f,'state':'tracked','reason':'video_track_with_appearance_verification',
                'pixel':mapped['pixel'],'xy':mapped['xy'],'covariance':mapped['covariance'],'pixel_covariance':mapped['pixel_covariance'],
                'model_radius_px':config.radius_px,'review_support':p['support'],'seed_frame':p['seed_frame'],
                'appearance_score':p['appearance_score'],'context_agreement':p['context_agreement'],
                'play_state':context['frames'][f].get('play_state'),'window_ids':context['frames'][f].get('window_ids',[]),
                'evidence_frames':[p['seed_frame']],'nearest_evidence_s':abs(f-p['seed_frame'])/replay['fps']}
    result['point_fusion_config']=asdict(config)
    result['method']='Visual anchors, verified TAPNext++ video tracks, and sequence-level possession/pass hypotheses'
    return result
