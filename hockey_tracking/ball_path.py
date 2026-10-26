"""Offline single-ball candidate selection with an explicit missing state.

Unlike a threshold on isolated frames, this optimizes a camera-compensated path
through all candidates. Detection scores are evidence, not probabilities of a
correct trajectory. Missing states permit reinitialization without extrapolation.
"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class PathConfig:
    evidence_floor: float = .08
    birth_cost: float = .8
    motion_weight: float = .5
    motion_scale_px: float = 55.
    max_displacement_px: float = 180.
    torso_penalty: float = .7
    outside_penalty: float = 2.


def fuse_candidates(streams, radius_px=12., agreement_bonus=.3):
    """Merge nearby centers from different views; scores remain uncalibrated.

    Each source gets at most one vote, so duplicate tile proposals cannot inflate
    agreement. A bounded bonus rewards corroboration without multiplying scores
    as if correlated detector views were independent probabilities.
    """
    if not streams:return []
    lengths={len(s) for s in streams}
    if len(lengths)!=1:raise ValueError('Candidate streams must have equal lengths')
    result=[]
    for f in range(len(streams[0])):
        proposals=sorted([(np.array(b,float),i) for i,s in enumerate(streams)
                          for b in s[f]['detections']],key=lambda p:-p[0][4])
        clusters=[]
        for box,source in proposals:
            center=(box[:2]+box[2:4])/2
            match=next((c for c in clusters if np.linalg.norm(center-c['center'])<=radius_px),None)
            if match is None:clusters.append({'center':center,'boxes':{source:box}})
            elif source not in match['boxes']:match['boxes'][source]=box
        boxes=[]
        for cluster in clusters:
            members=np.array(list(cluster['boxes'].values()));weights=members[:,4]
            box=np.average(members,axis=0,weights=weights)
            box[4]=min(.999,float(weights.max())*(1+agreement_bonus*(len(members)-1)))
            box[5]=0;boxes.append(box.tolist())
        result.append({'frame':f,'detections':sorted(boxes,key=lambda b:-b[4])[:30]})
    return result


def filter_person_parts(detections, people, body_confidence=.25, joint_confidence=.3):
    """Reject visual hypotheses on people, including refs and spectators.

    This is a floor-ball prior. It can withhold a real airborne or hand-held ball
    and balls touching shoes. It must never be interpreted as possession evidence.
    """
    if len(detections)!=len(people):raise ValueError('Person/frame count mismatch')
    output=[]
    for f,(row,context) in enumerate(zip(detections,people)):
        kept=[]
        for b in row['detections']:
            x,y=(b[0]+b[2])/2,(b[1]+b[3])/2;blocked=False
            for person,joints in zip(context['boxes'],context['keypoints']):
                if person[4]<body_confidence:continue
                left,top,right,bottom=person[:4];w=right-left;h=bottom-top
                if left+.1*w<=x<=right-.1*w and top<=y<=bottom-.18*h:
                    blocked=True;break
                for ankle in joints[15:17]:
                    ax,ay,confidence=ankle
                    if confidence<joint_confidence:continue
                    # Ankles sit just above the shoe; a small ellipse covers it.
                    rx=max(10.,.08*h);ry=max(7.,.05*h)
                    if ((x-ax)/rx)**2+((y-ay-.025*h)/ry)**2<=1:
                        blocked=True;break
                if blocked:break
            if not blocked:kept.append(b)
        output.append({'frame':f,'detections':kept})
    return output


def filter_markings(detections, appearance):
    """Withhold a candidate embedded in an extended painted marking.

    Appearance must correspond to the same ordered fused candidate stream.
    This cue is specific to the current rink's red markings, not a ball label.
    """
    if len(detections)!=len(appearance):raise ValueError('Appearance/frame count mismatch')
    output=[]
    for row,features in zip(detections,appearance):
        if len(row['detections'])!=len(features['features']):raise ValueError('Appearance/candidate count mismatch')
        output.append({'frame':row['frame'],'detections':[b for b,a in zip(row['detections'],features['features']) if not a['extended_marking']]})
    return output


def filter_verified(detections, verification, threshold, appearance=None, shape_config=None):
    """Use an independently trained crop model as a rejection gate.

    It does not create observations or turn classifier scores into calibrated
    tracking probabilities. The ordered candidate stream must match inference.
    """
    if len(detections)!=len(verification):raise ValueError('Verifier/frame count mismatch')
    output=[]
    if appearance is not None and len(appearance)!=len(detections):raise ValueError('Appearance/frame count mismatch')
    for f,(row,verified) in enumerate(zip(detections,verification)):
        if len(row['detections'])!=len(verified['probabilities']):raise ValueError('Verifier/candidate count mismatch')
        flags=appearance[f]['features'] if appearance is not None else [{'extended_marking':False} for _ in row['detections']]
        if len(flags)!=len(row['detections']):raise ValueError('Appearance/candidate count mismatch')
        def elongated(a):
            if shape_config is None:return a.get('elongated_structure',False)
            return any(c['area']>=shape_config['min_area'] and c['major_sigma']>=shape_config['min_major_sigma']
                       and c['axis_ratio']>=shape_config['min_axis_ratio'] and c['distance']<=shape_config['max_distance']
                       for c in a.get('components',[]))
        output.append({'frame':row['frame'],'detections':[b for b,p,a in zip(row['detections'],verified['probabilities'],flags) if p>=threshold and not a['extended_marking'] and not elongated(a)]})
    return output


def project(H, p):
    p=np.asarray(p,float);q=np.c_[p,np.ones(len(p))]@np.asarray(H,float).T
    with np.errstate(divide='ignore',invalid='ignore'):return q[:,:2]/q[:,2:]


def select_path(detections,replay,config=PathConfig(),excluded_frames=()):
    if len(detections)!=len(replay['frames']):raise ValueError('Frame count mismatch')
    if not detections:return {'config':asdict(config),'detections':[],'frames':0}
    states=[];backs=[];scores=np.array([0.]);previous=None;excluded=set(excluded_frames)
    lo,hi,far,near=replay['relative_bounds']
    for f,(raw,row)in enumerate(zip(detections,replay['frames'])):
        # This recorded transition is black/edit graphics, not an occluded ball.
        ds=[] if f in excluded else raw['detections']
        ds=np.asarray(ds,float).reshape(-1,6)
        centers=(ds[:,:2]+ds[:,2:4])/2
        evidence=np.log(np.maximum(ds[:,4],1e-6)/config.evidence_floor)
        for j,center in enumerate(centers):
            x,y=center/[1920,1080]
            if any(o['bbox'][0]<=x<=o['bbox'][2] and o['bbox'][1]<=y<o['bbox'][3]-.2*(o['bbox'][3]-o['bbox'][1]) for o in row['objects']):
                evidence[j]-=config.torso_penalty
        H=row.get('image_to_floor')
        if H is not None and len(centers):
            floor=project(H,centers)
            outside=(floor[:,0]<lo-.2)|(floor[:,0]>hi+.2)|(floor[:,1]<far-.1)|(floor[:,1]>near+.1)|~np.isfinite(floor).all(axis=1)
            evidence[outside]-=config.outside_penalty
        costs=np.full((len(scores),len(ds)+1),-config.birth_cost)
        costs[:,0]=0
        if previous is not None and len(previous) and len(ds):
            old_centers=(previous[:,:2]+previous[:,2:4])/2
            oldH=replay['frames'][f-1].get('image_to_floor')
            if H is not None and oldH is not None:
                old_centers=project(np.linalg.inv(np.array(H))@oldH,old_centers)
            distances=np.linalg.norm(old_centers[:,None,:]-centers[None,:,:],axis=2)
            motion=-config.motion_weight*(distances/config.motion_scale_px)**2
            motion[~np.isfinite(distances)|(distances>config.max_displacement_px)]=-1e6
            costs[1:,1:]=motion
        values=scores[:,None]+costs
        back=values.argmax(axis=0);scores=values[back,np.arange(len(ds)+1)]+np.r_[0,evidence]
        # Subtract a common scalar for stability; this does not change paths.
        scores-=scores.max()
        states.append(ds);backs.append(back);previous=ds
    selected=[];state=int(scores.argmax())
    for f in range(len(states)-1,-1,-1):
        selected.append(states[f][state-1].tolist() if state else None)
        state=int(backs[f][state])
    selected.reverse()
    return {'config':asdict(config),'detections':selected,'frames':len(selected)}
