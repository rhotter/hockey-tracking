"""Render a selected visual ball path plus bounded, explicitly inferred gaps."""
import numpy as np
from .ball_context import Config, infer, normalized_homography, project, jacobian


def make_replay(replay, selected, gap_seconds=.2, max_inferred_radius_px=24.):
    if len(selected)!=len(replay['frames']):raise ValueError('Frame count mismatch')
    # Canonical numeric types keep persisted configuration hashes stable.
    gap_seconds=float(gap_seconds);max_inferred_radius_px=float(max_inferred_radius_px)
    anchors={}; visual={}
    for f,(box,row) in enumerate(zip(selected,replay['frames'])):
        if box is None:continue
        pixel=(np.array(box[:2])+np.array(box[2:4]))/2
        if not np.isfinite(pixel).all() or np.any(pixel<0) or np.any(pixel>[1920,1080]):continue
        xy=None;cov=None;mapping='unavailable'
        sigma=6.;pcov=np.eye(2)*sigma**2
        if row.get('camera_valid') and row.get('image_to_floor') is not None:
            H=normalized_homography(row,replay['relative_bounds']);floor=project(H,pixel)
            if np.isfinite(floor).all() and np.all(floor>=0) and np.all(floor<=1):
                xy=floor.tolist();J=jacobian(H,pixel);cov=(J@pcov@J.T).tolist();mapping='floor_estimate'
                anchors[f]={'pixel':(pixel/[1920,1080]).tolist(),'xy':xy,'confidence':float(box[4])}
            else:mapping='outside_rink'
        visual[f]={'frame':f,'state':'detected','reason':'visual_path','pixel':(pixel/[1920,1080]).tolist(),
                   'xy':xy,'covariance':cov,'pixel_covariance':pcov.tolist(),'model_radius_px':2*sigma,
                   'mapping_status':mapping,'evidence_frames':[f],'nearest_evidence_s':0,
                   'evidence_score':float(box[4])}
    result=infer(replay,anchors,Config(clear_gap_s=gap_seconds,occluded_gap_s=gap_seconds,
                                     max_radius_px=max_inferred_radius_px,coast=False,use_occlusion=False))
    for f,r in enumerate(result['frames']):
        if f in visual:result['frames'][f]=visual[f]
        elif r['state']=='inferred':r['mapping_status']='floor_estimate'
    result.update(method='multi-view visual path with person context',image_tracking_requires_floor_calibration=False)
    return result
