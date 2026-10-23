"""Equivalence trial: unchanged tracking with GPU-side mask bounding boxes."""
import json
from pathlib import Path
from nbhl_tracking_fourhour import sha, dump


def track(meta, features, picture_folder, checkpoint, size='large', margin=.1):
    import time
    import numpy as np
    import torch
    from scipy.optimize import linear_sum_assignment
    from sam2.build_sam import build_sam2_video_predictor
    from nbhl_detection_metrics import iou
    assert size in ['large', 'tiny']
    config = {'new_confidence': .65, 'continuation_confidence': .1, 'minimum_mask_iou': .3,
              'margin': margin, 'retire_frames': 15, 'maximum_live': 24, 'scene_resets': True,
              'output_requires_current_detection': True}
    predictor = build_sam2_video_predictor(
        f'configs/sam2.1/sam2.1_hiera_{"l" if size == "large" else "t"}.yaml',
        str(checkpoint), device='cuda', apply_postprocessing=False)
    predictor.non_overlap_masks = False
    from nbhl_mask_geometry import mask_box
    started = time.monotonic()
    rows, live, next_id = [], {}, 1
    counts = {'ambiguous': 0, 'retired': 0, 'rejected_seed': 0, 'suppressed_seed': 0, 'capacity': 0, 'scene_resets': 0}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        state = predictor.init_state(str(picture_folder), offload_video_to_cpu=True, offload_state_to_cpu=True)
        for f, row in enumerate(meta['frames']):
            if features['resets'][f]:
                predictor.reset_state(state)
                live.clear()
                counts['scene_resets'] += 1
            eligible = [(j, d) for j, d in enumerate(row['detections']) if d[4] >= config['continuation_confidence']]
            current = [d for _, d in eligible]
            masks = {}
            if live:
                for index, ids, logits in predictor.propagate_in_video(state, start_frame_idx=f, max_frame_num_to_track=0):
                    assert index == f and set(ids) == set(live)
                    masks = {tid: mask_box(logits[j, 0]) for j, tid in enumerate(ids)}
            usable = [tid for tid, box in masks.items() if box is not None]
            objects, assigned = [], set()
            if usable and current:
                overlap = np.array([[iou(masks[tid], d[:4]) for d in current] for tid in usable])
                aa, bb = linear_sum_assignment(-overlap)
                for a, b in zip(aa, bb):
                    score = float(overlap[a, b])
                    if score < config['minimum_mask_iou']:
                        continue
                    competitor = max(float(np.max(np.delete(overlap[a], b))) if len(current)>1 else 0.,
                                     float(np.max(np.delete(overlap[:, b], a))) if len(usable)>1 else 0.)
                    if score-competitor < margin:
                        counts['ambiguous'] += 1
                        continue
                    tid = usable[a]
                    live[tid] = f
                    assigned.add(b)
                    objects.append({'track_id': str(tid), 'bbox': current[b][:4], 'confidence': current[b][4],
                                    'observed': True, 'detection_index': eligible[b][0], 'mask_iou': score})
            for tid in list(live):
                if f-live[tid] > config['retire_frames']:
                    predictor.remove_object(state, tid)
                    del live[tid]
                    masks.pop(tid, None)
                    counts['retired'] += 1
            for j, d in enumerate(current):
                if j in assigned or d[4] < config['new_confidence']:
                    continue
                if any(box is not None and iou(box, d[:4]) >= .3 for box in masks.values()):
                    counts['suppressed_seed'] += 1
                    continue
                if len(live) >= config['maximum_live']:
                    counts['capacity'] += 1
                    continue
                tid = next_id
                next_id += 1
                _, ids, logits = predictor.add_new_points_or_box(state, frame_idx=f, obj_id=tid, box=np.asarray(d[:4], np.float32))
                box = mask_box(logits[ids.index(tid), 0])
                if box is None or iou(box, d[:4]) < .3:
                    predictor.remove_object(state, tid)
                    counts['rejected_seed'] += 1
                    continue
                live[tid] = f
                masks[tid] = box
                objects.append({'track_id': str(tid), 'bbox': d[:4], 'confidence': d[4], 'observed': True,
                                'detection_index': eligible[j][0], 'seed': True})
            assert len({o['track_id'] for o in objects}) == len(objects)
            rows.append({'frame': f, 'source_time_s': row['source_time_s'], 'objects': objects})
            if f % 150 == 0:
                print(json.dumps({'clip': meta['summary']['clip'], 'size': size, 'frame': f, 'seconds': time.monotonic()-started}), flush=True)
    result = {'clip': meta['summary']['clip'], 'setting': {'family': 'sam2', 'size': size, 'margin': margin},
              'config': config, 'frames': rows, 'diagnostics': counts, 'seconds': time.monotonic()-started,
              'raw_detection_sha256': meta['summary']['input_sha256'], 'weights_sha256': sha(checkpoint),
              'code_sha256': sha(__file__), 'labels_used_for_inference': False}
    del state, predictor
    torch.cuda.empty_cache()
    return result

