"""Cache SAM3's immutable per-frame memory eligibility decisions.

The pinned tracker creates a new score tensor when an object's removal changes
the batch score. Cache identity and threshold, not just the frame number. This
avoids repeatedly synchronizing a GPU scalar while retaining the complete
history and the original selection order.
"""
from types import MethodType


def install(tracker, verify=False):
    original = tracker.frame_filter
    stats = {'calls':0, 'hits':0, 'misses':0, 'verified_calls':0}

    def cached(self, output_dict, track_in_reverse, frame_idx, num_frames, r):
        stats['calls'] += 1
        if (frame_idx == 0 and not track_in_reverse) or (frame_idx == num_frames-1 and track_in_reverse):
            selected = []
        else:
            maximum = min(num_frames,self.max_obj_ptrs_in_encoder)
            must_include = frame_idx+1 if track_in_reverse else frame_idx-1
            indices = range(must_include,num_frames,r) if track_in_reverse else range(must_include,0,-r)
            selected = []
            for index in indices:
                out = output_dict['non_cond_frame_outputs'].get(index)
                if out is None or 'eff_iou_score' not in out:continue
                score = out['eff_iou_score']; previous = out.get('_nbhl_score_eligibility')
                if previous is None or previous[0] is not score or previous[1] != self.mf_threshold:
                    previous = (score,self.mf_threshold,bool(score > self.mf_threshold))
                    out['_nbhl_score_eligibility'] = previous
                    stats['misses'] += 1
                else:stats['hits'] += 1
                if previous[2]:selected.insert(0,index)
                if len(selected) >= maximum-1:break
            if must_include not in selected:selected.append(must_include)
        if verify:
            expected = original(output_dict,track_in_reverse,frame_idx,num_frames,r)
            assert selected == expected, (frame_idx,selected,expected)
            stats['verified_calls'] += 1
        return selected

    tracker.frame_filter = MethodType(cached,tracker)
    return stats
