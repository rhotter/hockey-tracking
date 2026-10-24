"""Move old SAM3 mask memories to CPU, retaining every memory and score.

SAM3 can reconsider old frames after removing an object from a batch. Dropping
old low-confidence memories would change that behavior. This leaves the complete
history available and uses the official memory reader's CPU->CUDA path.
"""


def offload_history(model,state,emitted_frame,keep_recent=32):
    cutoff=emitted_frame-keep_recent;stats=state.setdefault('_nbhl_offload_stats',{'records':0,'bytes':0})
    if cutoff<0:return stats
    for tracker_state in state['tracker_inference_states']:
        done=tracker_state.setdefault('_nbhl_offloaded',set())
        for storage in ['cond_frame_outputs','non_cond_frame_outputs']:
            for f,out in tracker_state['output_dict'][storage].items():
                key=(storage,f)
                if f>cutoff or key in done:continue
                changed=False
                for field in ['maskmem_features','pred_masks']:
                    tensor=out.get(field)
                    if tensor is not None and tensor.device.type=='cuda':
                        stats['bytes']+=tensor.numel()*tensor.element_size()
                        out[field]=tensor.to('cpu',non_blocking=False);changed=True
                if changed:
                    # Replace per-object views too; otherwise their old GPU
                    # storages stay alive even after the aggregate tensor moves.
                    model.tracker._add_output_per_object(tracker_state,f,out,storage)
                    stats['records']+=1
                done.add(key)
    return stats
