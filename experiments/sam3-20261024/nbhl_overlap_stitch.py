"""Join adjacent overlapping executions using only simultaneous geometric evidence.

This preserves automatic fragment hypotheses. It does not establish player identity.
"""
from collections import Counter,defaultdict

def stitch(chunks,min_matches=15,minimum_fraction=.8,minimum_iou=.8):
    import copy
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from nbhl_detection_metrics import iou
    assert chunks and chunks[0]['start_frame']==0
    chunks=sorted(chunks,key=lambda c:c['start_frame']);parent={};boundaries=[]
    for i,c in enumerate(chunks):
        for f,row in enumerate(c['frames']):
            assert row['frame']==f
            assert len({o['track_id'] for o in row['objects']})==len(row['objects'])
            for o in row['objects']:
                assert o['observed'] is True
                parent[(i,o['track_id'])]=(i,o['track_id'])
    def root(k):
        seen=set()
        while parent[k]!=k:
            assert k not in seen;seen.add(k);k=parent[k]
        return k
    for index in range(len(chunks)-1):
        a,b=chunks[index:index+2];lo=b['start_frame'];hi=min(a['start_frame']+len(a['frames']),b['start_frame']+len(b['frames']))
        assert lo<hi,'Chunks must overlap; no offscreen appearance-only boundary joining'
        votes=Counter();ca=Counter();cb=Counter();overlaps=defaultdict(list)
        for g in range(lo,hi):
            aa=a['frames'][g-a['start_frame']]['objects'];bb=b['frames'][g-b['start_frame']]['objects']
            ca.update(o['track_id'] for o in aa);cb.update(o['track_id'] for o in bb)
            if not aa or not bb:continue
            matrix=np.array([[iou(x['bbox'],y['bbox']) for y in bb] for x in aa])
            rows,cols=linear_sum_assignment(np.where(matrix>=minimum_iou,1-matrix,1e6))
            for x,y in zip(rows,cols):
                if matrix[x,y]<minimum_iou:continue
                key=(aa[x]['track_id'],bb[y]['track_id']);votes[key]+=1;overlaps[key].append(float(matrix[x,y]))
        choices_a=defaultdict(list);choices_b=defaultdict(list)
        for (x,y),n in votes.items():choices_a[x].append((n,y));choices_b[y].append((n,x))
        accepted=[]
        for (x,y),n in sorted(votes.items()):
            if n<min_matches or n/max(ca[x],cb[y])<minimum_fraction:continue
            # A unique mutual best prevents one old fragment from naming two new ones.
            if sum(v>=n for v,_ in choices_a[x])!=1 or sum(v>=n for v,_ in choices_b[y])!=1:continue
            ra=root((index,x));rb=root((index+1,y));assert ra!=rb
            parent[rb]=ra
            accepted.append({'left_id':x,'right_id':y,'matching_frames':n,'left_visible_overlap_frames':ca[x],
                             'right_visible_overlap_frames':cb[y],'support_fraction':n/max(ca[x],cb[y]),'mean_iou':sum(overlaps[x,y])/n})
        boundaries.append({'left_clip':a['clip'],'right_clip':b['clip'],'overlap_start_frame':lo,'overlap_end_frame_exclusive':hi,'accepted':accepted,'candidate_pairs':len(votes)})
    owners={}
    for i,c in enumerate(chunks):
        for f,row in enumerate(c['frames']):
            g=c['start_frame']+f;score=(min(f,len(c['frames'])-1-f),i)
            if g not in owners or score>owners[g][0]:owners[g]=(score,i,row)
    assert sorted(owners)==list(range(max(owners)+1)),'Uncovered source interval'
    labels={};frames=[]
    for g in sorted(owners):
        _,i,row=owners[g];objects=[]
        for o in row['objects']:
            key=root((i,o['track_id']));label=labels.setdefault(key,'T'+str(len(labels)+1))
            objects.append(copy.deepcopy(o)|{'track_id':label,'chunk_track_id':o['track_id'],'source_chunk':chunks[i]['clip']})
        assert len({o['track_id'] for o in objects})==len(objects),'Boundary stitching caused a simultaneous identity collision'
        frames.append({'frame':g,'source_time_s':round(g/15,4),'objects':objects})
    return {'frames':frames,'boundaries':boundaries,'fragments':len(labels),'labels_used_for_inference':False,
            'settings':{'min_matches':min_matches,'minimum_fraction':minimum_fraction,'minimum_iou':minimum_iou},
            'limits':'Same-time geometric overlap only. Fragment labels remain unverified; pre-existing within-chunk identity errors are retained. Overlap observations are selected from the chunk farther from its temporal edge. No interpolation.'}
