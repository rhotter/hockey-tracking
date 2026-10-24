"""SAM3 native association ablations and long continuous/overlap runs."""
from pathlib import Path
import json
import modal
from modal_campaign import ROOT,volume
from modal_nbhl_sam3_20261023 import image as old_image,REV,CODE,local_frames
BASE=Path('/data/ball-hockey/reliability-20261018');OUT=BASE/'sam3-hybrid-20261024-v1'
app=modal.App('sport-scans-sam3-extended-20261024')
image=old_image
for name in ['modal_nbhl_sam3_20261023.py','nbhl_sam3_lazy_frames.py','nbhl_sam3_state_offload.py','nbhl_sam3_visual_memory.py','nbhl_sam3_score_cache.py','nbhl_overlap_stitch.py','nbhl_tracking_reference_audit.py']:
    image=image.add_local_file(ROOT/'scripts'/name,'/work/'+name)
image=image.add_local_file(ROOT/'data/ball-hockey/sam3-hybrid-protocol-20261024.json','/work/hybrid-protocol.json')
image=image.add_local_file(ROOT/'data/ball-hockey/tracking-four-hour-reference-corrections-v1.json','/work/tracking-four-hour-reference-corrections-v1.json')
SETTINGS={'default':{},'association03':{'assoc_iou_thresh':.3},'recondition4':{'recondition_every_nth_frame':4},'recondition0':{'recondition_every_nth_frame':0},'alive8':{'max_trk_keep_alive':8,'init_trk_keep_alive':8}}
SETTINGS.update({'visual':{},'visual-person':{},'visual-hockey':{}})
SETTINGS.update({'visual-memory':{},'visual-memory-person':{},'visual-memory-hockey':{}})
SETTINGS.update({'adapt-margin0':{},'adapt-margin1':{},'adapt-margin2':{}})
SETTINGS.update({'adapt-control0':{},'adapt-control1':{},'adapt-control2':{}})
SETTINGS['adapt-person']={}
SETTINGS.update({'adapt-heads-person':{},'adapt-decoder-person':{}})
SETTINGS.update({'default-fast':{},'default-fast-verify':{}})


@app.function(image=image,cpu=4,memory=8192,timeout=600,volumes={'/data':volume})
def verify():
    import tempfile,shutil,torch,numpy as np
    from sam3.model.io_utils import load_video_frames_from_image_folder
    from nbhl_sam3_lazy_frames import Frames
    volume.reload();folder=Path(tempfile.mkdtemp(prefix='sam3-loader-check-'))
    source=Path('/data/sequences/gb-tracking-1800/img1')
    for i in range(20):shutil.copyfile(source/f'{i+1:06d}.jpg',folder/f'{i+1:06d}.jpg')
    eager,h,w=load_video_frames_from_image_folder(str(folder),1008,True,(.5,.5,.5),(.5,.5,.5),False)
    lazy=Frames(folder,1008,(.5,.5,.5),(.5,.5,.5))
    checks=[torch.equal(eager[i],lazy[i]) for i in [0,1,2,3,4,19,0,18,2]]
    assert all(checks) and torch.equal(lazy[-1],eager[-1]) and len(lazy.cache)<=4
    old=BASE/'sam3-20261023-v1/tracking/development/person/gb-tracking-1800/tracks.json'
    new=OUT/'native-settings/development/default/gb-tracking-1800/tracks.json'
    a=json.loads(old.read_text());b=json.loads(new.read_text());assert len(a['frames'])==len(b['frames'])==300
    differences=[];maximum_box=0.;maximum_score=0.
    for x,y in zip(a['frames'],b['frames']):
        xx={o['track_id']:o for o in x['objects']};yy={o['track_id']:o for o in y['objects']}
        if set(xx)!=set(yy):differences.append(x['frame'])
        for key in set(xx)&set(yy):
            maximum_box=max(maximum_box,float(np.max(np.abs(np.asarray(xx[key]['bbox'])-yy[key]['bbox']))))
            maximum_score=max(maximum_score,abs(xx[key]['confidence']-yy[key]['confidence']))
    result={'eager_and_lazy_tensor_checks':checks,'negative_index_equal':True,'cache':lazy.stats(),'output_frames':300,'frames_with_different_ids':differences,'maximum_box_coordinate_difference':maximum_box,'maximum_score_difference':maximum_score,'equivalent_ids_and_boxes':not differences and maximum_box<1e-4}
    result['history_offload_checks']=[]
    for clip in ['gb-tracking-1800','gb-reid-long-0900','long-plDp2kPtk_g']:
        old=BASE/'sam3-20261023-v1/tracking/development/person'/clip/'tracks.json'
        new=OUT/'native-settings/development/default-offload'/clip/'tracks.json'
        if not old.exists() or not new.exists():
            result['history_offload_checks'].append({'clip':clip,'status':'incomplete_original_or_optimized_run'});continue
        left=json.loads(old.read_text())['frames'];right=json.loads(new.read_text())['frames']
        assert len(left)==len(right)
        different=[a['frame'] for a,b in zip(left,right) if a!=b]
        result['history_offload_checks'].append({'clip':clip,'frames':len(left),'exactly_equal_rows':not different,'different_frames':different})
    (OUT/'runtime-verification.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit();return result


@app.function(image=image,cpu=2,memory=4096,timeout=300,volumes={'/data':volume})
def verify_score_cache():
    from nbhl_vlm_reconnect import dump,sha
    volume.reload();checks=[]
    for clip,variant in [('gb-tracking-1800','default-fast-verify'),('gb-reid-long-0900','default-fast')]:
        before=OUT/'native-settings/development/default-offload'/clip
        after=OUT/'native-settings/development'/(variant+'-offload')/clip
        left=json.loads((before/'tracks.json').read_text())['frames']
        right=json.loads((after/'tracks.json').read_text())['frames']
        assert len(left)==len(right)
        changed=[i for i,(a,b) in enumerate(zip(left,right)) if a!=b]
        assert not changed,(clip,changed[:10])
        old=json.loads((before/'completed.json').read_text())
        new=json.loads((after/'completed.json').read_text())
        checks.append({'clip':clip,'frames':len(left),'all_frame_rows_exactly_equal':True,
                       'before_sha256':sha(before/'tracks.json'),'after_sha256':sha(after/'tracks.json'),
                       'score_cache':new['settings']['memory_score_cache'],
                       'before_inference_seconds':old['inference_seconds'],'after_inference_seconds':new['inference_seconds'],
                       'timing_note':'Different cloud workers; verified control also executes original selector. Not a controlled speed benchmark.'})
    result={'checks':checks,'total_equivalent_frames':sum(c['frames'] for c in checks),
            'scores_and_memory_preserved':True}
    dump(OUT/'memory-score-cache-verification.json',result);volume.commit();return result


@app.function(image=image,cpu=8,memory=196608,gpu='H100',timeout=10800,max_containers=3,scaledown_window=10,volumes={'/data':volume})
def track(clip:str,variant:str='default',phase:str='development',delta:str='',offload:bool=False,score_cache:bool=False):
    import time,torch,resource,hashlib,gc
    from huggingface_hub import hf_hub_download
    from sam3.model_builder import build_sam3_video_predictor
    from nbhl_vlm_reconnect import datasets,dump,sha
    from nbhl_tracking_fourhour import evaluate
    from nbhl_tracking_reference_audit import corrected_report
    from nbhl_sam3_lazy_frames import install
    volume.reload();torch.set_num_threads(4)
    gc.collect();torch.cuda.empty_cache()
    dest=OUT/'native-settings'/phase/(variant+('-offload' if offload else '')+('-'+delta if delta else ''))/clip
    if (dest/'completed.json').exists():return json.loads((dest/'completed.json').read_text())
    if phase=='long':
        folder=BASE/'tracking-four-hour-v1/long-cache-trial'/clip
        meta=json.loads((folder/'metadata.json').read_text());frames=folder/'frames'
    else:meta,_,_,frames,_=next(r for r in datasets(phase,labels=True) if r[0]['summary']['clip']==clip)
    started=time.monotonic();staged=local_frames(frames,clip);print(json.dumps({'clip':clip,'stage':'frames_staged','seconds':time.monotonic()-started}),flush=True)
    checkpoint=hf_hub_download('facebook/sam3','sam3.pt',revision=REV,local_files_only=True)
    predictor=build_sam3_video_predictor(checkpoint_path=checkpoint,gpus_to_use=[0])
    actual={}
    if score_cache or variant.startswith('default-fast'):
        from nbhl_sam3_score_cache import install as install_score_cache
        actual['memory_score_cache']=install_score_cache(predictor.model.tracker,verify=variant=='default-fast-verify')
    for key,value in SETTINGS[variant].items():
        actual[key]={'before':getattr(predictor.model,key),'after':value};setattr(predictor.model,key,value)
    output_threshold=.5
    if delta:
        path=BASE/'sam3-finetune-20261024-v1'/delta/'best-delta.pt';weights=torch.load(path,weights_only=True,map_location='cpu')
        state=predictor.model.detector.state_dict()
        assert all(k in state and v.shape==state[k].shape for k,v in weights.items())
        missing,unexpected=predictor.model.detector.load_state_dict(weights,strict=False);assert not unexpected
        actual['fine_tune']={'variant':delta,'sha256':sha(path),'loaded_tensors':len(weights),'unchanged_tensors':len(missing)}
    if variant.startswith(('adapt-margin','adapt-control')):
        if variant.startswith('adapt-margin'):
            assert delta
            report=json.loads((BASE/'sam3-finetune-20261024-v1'/delta/'completed.json').read_text())
        else:
            assert not delta
            report=json.loads((BASE/'sam3-finetune-20261024-v1/selection.json').read_text())
            actual['matched_original_control']=True
        threshold=report['best']['validation']['selected']['confidence_threshold'];margin=int(variant[-1])*.1
        for key,value in [('score_threshold_detection',threshold),('new_det_thresh',min(.95,threshold+margin))]:
            actual[key]={'before':getattr(predictor.model,key),'after':value};setattr(predictor.model,key,value)
        output_threshold=threshold;actual['output_probability_threshold']=threshold
    restore,caches=install();rows={};height,width=meta['shape'];prompt='hockey player' if delta or variant.startswith('adapt-control') else 'person'
    if variant.startswith('adapt-') and variant.endswith('person'):
        assert delta
        prompt='person';actual['prompt_control']='Original person prompt and native thresholds; detector delta only.'
    inference_start=time.monotonic();offload_stats=None
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            response=predictor.handle_request({'type':'start_session','resource_path':str(staged),'offload_video_to_cpu':True});session=response['session_id']
            request={'type':'add_prompt','session_id':session,'frame_index':0,'text':prompt}
            if variant.startswith('visual'):
                eligible=[d for d in meta['frames'][0]['detections'] if d[4]>=.65 and d[3]-d[1]>=40 and d[0]>0 and d[1]>0 and d[2]<width and d[3]<height]
                if eligible:
                    chosen=max(eligible,key=lambda d:d[4]);x,y,u,v=chosen[:4]
                    request['bounding_boxes']=[[x/width,y/height,(u-x)/width,(v-y)/height]]
                    request['bounding_box_labels']=[1]
                    kind=variant.replace('-memory','')
                    prompt=None if kind=='visual' else 'person' if kind=='visual-person' else 'hockey player'
                    request['text']=prompt
                    actual['visual_exemplar']={'frame':0,'bbox':chosen[:4],'detector_confidence':chosen[4],'selection':'Highest-confidence complete initial-frame YOLO box, confidence>=0.65 and height>=40pixels; no reference labels.'}
                    if 'memory' in variant:
                        from nbhl_sam3_visual_memory import install as install_visual_memory
                        actual['visual_memory']=install_visual_memory(predictor.model.detector)
                else:actual['visual_exemplar']={'fallback':'No eligible initial box; person text used.'}
            initial=predictor.handle_request(request)
            actual['initial_prompt_output_count']=len(initial['outputs']['out_obj_ids'])
            del initial
            if variant=='visual-memory' and 'visual_memory' in actual:
                # The upstream gate tests text_prompt truthiness. Preserve the
                # visual text IDs/features while enabling later admissions.
                predictor._all_inference_states[session]['state']['text_prompt']='<persistent visual template>'
            for response in predictor.handle_stream_request({'type':'propagate_in_video','session_id':session,'propagation_direction':'forward','start_frame_index':0,'output_prob_thresh':output_threshold,'evict_cached_frame_outputs':True}):
                f=response['frame_index'];out=response['outputs'];objects=[]
                for tid,box,score in zip(out['out_obj_ids'],out['out_boxes_xywh'],out['out_probs']):
                    x,y,w,h=map(float,box)
                    if w>0 and h>0:objects.append({'track_id':'sam3-'+str(int(tid)),'bbox':[x*width,y*height,(x+w)*width,(y+h)*height],'confidence':float(score),'observed':True})
                rows[f]={'frame':f,'source_time_s':meta['frames'][f]['source_time_s'],'objects':objects}
                if offload:
                    from nbhl_sam3_state_offload import offload_history
                    state=predictor._all_inference_states[session]['state']
                    offload_stats=offload_history(predictor.model,state,f)
                if f%300==0:print(json.dumps({'clip':clip,'variant':variant,'delta':delta,'frame':f,'seconds':time.monotonic()-inference_start,'gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30}),flush=True)
            predictor.handle_request({'type':'close_session','session_id':session})
    finally:restore()
    assert len(rows)==len(meta['frames'])
    if 'visual_memory' in actual:assert actual['visual_memory']['encodings']==1 and actual['visual_memory']['calls']>=len(rows)
    result={'clip':clip,'frames':[rows[i] for i in range(len(rows))],'settings':actual,'prompt':prompt,'labels_used_for_inference':False,'code_revision':CODE,'weights_revision':REV}
    metrics=evaluate(meta,result)
    if phase=='development':
        corrections={r['pair_id']:r for r in json.loads(Path('/work/tracking-four-hour-reference-corrections-v1.json').read_text())['corrections']};metrics=corrected_report(metrics,corrections)
    dump(dest/'tracks.json',result);dump(dest/'metrics.json',metrics)
    report={k:v for k,v in metrics.items() if k!='pairs'}|{'phase':phase,'variant':variant,'delta':delta,'offload_history':offload_stats,'settings':actual,'total_seconds':time.monotonic()-started,'inference_seconds':time.monotonic()-inference_start,'frames':len(rows),'duration_s':len(rows)/15,'source_id':meta['summary']['source_id'],'tracks_sha256':sha(dest/'tracks.json'),'cache':caches[0].stats(),'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    dump(dest/'completed.json',report);volume.commit();return report


@app.function(image=image,cpu=8,memory=524288,gpu='H100',timeout=14400,max_containers=6,scaledown_window=10,volumes={'/data':volume})
def track_long(clip:str,variant:str='default',phase:str='long',delta:str='',offload:bool=True,score_cache:bool=True):
    """Same inference with enough guaranteed host RAM for retained mask history.

    A completed 315-second control used 301GiB RSS. The former 96GiB
    request depended on unreserved host memory and another input was killed.
    No output threshold, memory contents, or association rule changes here.
    """
    return track.local(clip,variant,phase,delta,offload,score_cache)


@app.local_entrypoint()
def main(phase:str='development',clips:str='gb-tracking-1800,gb-reid-long-0900',variants:str='association03,recondition4,recondition0,alive8',delta:str='',offload:bool=False,score_cache:bool=False):
    if phase=='verify':
        result=verify.remote();(ROOT/'notes/ball-hockey/sam3-runtime-verification-20261024.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return
    if phase=='verify-score-cache':
        result=verify_score_cache.remote();(ROOT/'notes/ball-hockey/sam3-20261024/memory-score-cache-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return
    jobs=[(c,v,phase,delta,offload,score_cache) for v in variants.split(',') for c in clips.split(',')]
    results=[]
    for r in track.starmap(jobs,order_outputs=False):
        results.append(r);print(json.dumps(r),flush=True)
        name=f'sam3-extended-{phase}-{variants.replace(",","_")}{"-offload" if offload else ""}{"-"+delta if delta else ""}-20261024.json'
        (ROOT/'notes/ball-hockey'/name).write_text(json.dumps(results,indent=2)+'\n')
