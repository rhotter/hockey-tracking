"""Frozen SAM3 detection and end-to-end tracking comparisons on NBHL only."""
from pathlib import Path
import json
import modal
from modal_campaign import ROOT, base_image, volume

BASE=Path('/data/ball-hockey/reliability-20261018')
OUT=BASE/'sam3-20261023-v1'
CODE='7f473cfd983d5eb43040ff608f66bff0d7bdcb53'
REV='3c879f39826c281e95690f02c7821c4de09afae7'
REV31='daa63191845a41281374e725f4c9e51c7a824460'
app=modal.App('sport-scans-nbhl-sam3-20261023')
image=(base_image.pip_install('ftfy==6.1.1','regex','decord','setuptools<81')
 .run_commands('git clone https://github.com/facebookresearch/sam3.git /opt/sam3',
               'git -C /opt/sam3 checkout '+CODE,'pip install --no-deps -e /opt/sam3')
 .env({'PYTHONPATH':'/opt/sam3:/opt/smp:/work','HF_HOME':'/data/cache/huggingface','HF_HUB_DISABLE_XET':'1'}))
for name in ['modal_campaign.py','nbhl_detection_metrics.py','nbhl_tracking_fourhour.py','nbhl_tracking_features_v2.py','nbhl_vlm_reconnect.py','nbhl_sam3_detector_supported.py']:
 image=image.add_local_file(ROOT/'scripts'/name,'/work/'+name)
image=image.add_local_file(ROOT/'data/ball-hockey/sam3-protocol-20261023.json','/work/protocol.json')
for source,folder in [('b8CM4rZexaY','canonical'),('1f0Y5BIr5Ps','heldout'),('5Qw_g9E7EqE','heldout')]:
 image=image.add_local_file(ROOT/'data/ball-hockey/annotations'/folder/(source+'.json'),'/work/annotations/'+source+'.json')

def dump(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')

def local_frames(folder, clip):
 # Byte-for-byte staging avoids repeated small-file reads from the cloud volume.
 import tempfile,shutil,concurrent.futures,hashlib
 target=Path(tempfile.mkdtemp(prefix='sam3-'+clip+'-'))
 paths=sorted(folder.glob('*.jpg'))
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
  list(pool.map(lambda p:shutil.copyfile(p,target/p.name),paths))
 for p in [paths[0],paths[-1]]:
  assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256((target/p.name).read_bytes()).digest()
 return target

@app.function(image=image,cpu=2,memory=8192,timeout=1800,secrets=[modal.Secret.from_name('huggingface',required_keys=['HF_TOKEN'])],volumes={'/data':volume})
def cache(version:str='sam3'):
 import os
 from huggingface_hub import hf_hub_download
 name,filename,revision=('facebook/sam3.1','sam3.1_multiplex.pt',REV31) if version=='sam3.1' else ('facebook/sam3','sam3.pt',REV)
 p=hf_hub_download(name,filename,revision=revision,token=os.environ['HF_TOKEN']);volume.commit();return p

@app.function(image=image,cpu=8,memory=49152,gpu='H100',timeout=3600,scaledown_window=10,volumes={'/data':volume})
def detection(phase:str,prompt:str='',threshold:float=.5):
 import time,hashlib,torch
 from PIL import Image
 from huggingface_hub import hf_hub_download
 from sam3.model_builder import build_sam3_image_model
 from sam3.model.sam3_image_processor import Sam3Processor
 from nbhl_detection_metrics import summarize,average_precision
 volume.reload();torch.set_num_threads(4)
 ckpt=hf_hub_download('facebook/sam3','sam3.pt',revision=REV,local_files_only=True)
 model=build_sam3_image_model(checkpoint_path=ckpt,load_from_HF=False)
 processor=Sam3Processor(model,confidence_threshold=.05)
 proto=json.loads(Path('/work/protocol.json').read_text())['detection']
 sources=[proto['development_source']] if phase=='development' else proto['regression_sources']
 prompts=proto['prompts'] if phase=='development' else [prompt]
 rows={p:[] for p in prompts};start=time.monotonic()
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  for source in sources:
   ann=json.loads(Path('/work/annotations',source+'.json').read_text())
   for row in ann['frames']:
    p=Path(row['image']);assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
    im=Image.open(p).convert('RGB');state=processor.set_image(im)
    for text in prompts:
     processor.reset_all_prompts(state)
     result=processor.set_text_prompt(state=state,prompt=text)
     boxes=result['boxes'].float().cpu().numpy();scores=result['scores'].float().cpu().numpy()
     pred=[{'bbox':b.tolist(),'confidence':float(s)}for b,s in zip(boxes,scores)]
     rows[text].append({'id':row['id'],'players':row['players'],'predictions':pred})
   print(json.dumps({'source':source,'frames':len(ann['frames']),'phase':phase}),flush=True)
 report={'phase':phase,'seconds':time.monotonic()-start,'code_revision':CODE,'weights_revision':REV,'prompts':{}}
 for text,values in rows.items():
  curve=[summarize(values,t) for t in proto['thresholds']]
  report['prompts'][text]={'ap50':average_precision(values),'curve':curve,'at_selected_threshold':summarize(values,threshold)}
  dump(OUT/'detection'/phase/(text.replace(' ','-')+'-predictions.json'),values)
 if phase=='development':
  _,text,row=max(( (r['f1'],r['precision'] or 0,r['confidence_threshold'],-i),p,r) for i,p in enumerate(prompts) for r in report['prompts'][p]['curve'])
  report['selected']={'prompt':text,'threshold':row['confidence_threshold'],'metrics':row}
 dump(OUT/'detection'/(phase+'.json'),report);volume.commit();return report

@app.function(image=image,cpu=8,memory=98304,gpu='H100',timeout=5400,max_containers=2,scaledown_window=10,volumes={'/data':volume})
def tracking(phase:str,clip:str,prompt:str,version:str='sam3',max_objects:int=64,geometry:bool=False):
 import time,torch,numpy as np
 from huggingface_hub import hf_hub_download
 from sam3.model_builder import build_sam3_video_predictor,build_sam3_predictor
 from nbhl_vlm_reconnect import datasets
 from nbhl_tracking_fourhour import evaluate
 volume.reload();torch.set_num_threads(4)
 meta,features,baseline,folder,_=next(r for r in datasets(phase,labels=True) if r[0]['summary']['clip']==clip)
 name,filename,revision=('facebook/sam3.1','sam3.1_multiplex.pt',REV31) if version=='sam3.1' else ('facebook/sam3','sam3.pt',REV)
 ckpt=hf_hub_download(name,filename,revision=revision,local_files_only=True)
 if version=='sam3.1':
  predictor=build_sam3_predictor(checkpoint_path=ckpt,version='sam3.1',max_num_objects=max_objects,multiplex_count=16,use_fa3=False,compile=False,warm_up=False,async_loading_frames=False)
  # The pinned shared session API passes this SAM3-only flag to SAM3.1.
  # Its false default has no effect; do not silently accept true offloading.
  init_state=predictor.model.init_state
  def compatible_init_state(*args,offload_state_to_cpu=False,**kwargs):
   assert offload_state_to_cpu is False
   return init_state(*args,**kwargs)
  predictor.model.init_state=compatible_init_state
 else:predictor=build_sam3_video_predictor(checkpoint_path=ckpt,gpus_to_use=[0])
 start=time.monotonic();height,width=meta['shape'];rows={};derived={'final-mask':{},'largest-component':{}}
 if geometry:
  import cv2
  cv2.setNumThreads(2)
 folder=local_frames(folder,clip)
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  response=predictor.handle_request({'type':'start_session','resource_path':str(folder),'offload_video_to_cpu':True})
  session=response['session_id']
  predictor.handle_request({'type':'add_prompt','session_id':session,'frame_index':0,'text':prompt})
  for response in predictor.handle_stream_request({'type':'propagate_in_video','session_id':session,'propagation_direction':'forward','start_frame_index':0,'output_prob_thresh':.5,'evict_cached_frame_outputs':True}):
   f=response['frame_index'];out=response['outputs'];objects=[]
   for tid,box,score in zip(out['out_obj_ids'],out['out_boxes_xywh'],out['out_probs']):
    x,y,w,h=map(float,box)
    if w<=0 or h<=0:continue
    objects.append({'track_id':version+'-'+str(int(tid)),'bbox':[x*width,y*height,(x+w)*width,(y+h)*height],'observed':True,'confidence':float(score)})
   rows[f]={'frame':f,'source_time_s':meta['frames'][f]['source_time_s'],'objects':objects}
   if geometry:
    alternative={name:[] for name in derived}
    for tid,mask,prob in zip(out['out_obj_ids'],out['out_binary_masks'],out['out_probs']):
     mask=np.asarray(mask,dtype=np.uint8);assert mask.shape==(height,width)
     if not mask.any():continue
     ys,xs=np.nonzero(mask)
     full_box=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]
     count,_,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
     assert count>1
     x,y,w,h,area=map(int,stats[1+stats[1:,cv2.CC_STAT_AREA].argmax()])
     for name,box in [('final-mask',full_box),('largest-component',[x,y,x+w,y+h])]:
      alternative[name].append({'track_id':version+'-'+str(int(tid)),'bbox':box,'observed':True,'confidence':float(prob)})
    for name in derived:derived[name][f]={'frame':f,'source_time_s':meta['frames'][f]['source_time_s'],'objects':alternative[name]}
   if f%75==0:print(json.dumps({'clip':clip,'prompt':prompt,'frame':f,'seconds':time.monotonic()-start}),flush=True)
  predictor.handle_request({'type':'close_session','session_id':session})
 assert len(rows)==len(meta['frames'])
 result={'clip':clip,'frames':[rows[i]for i in range(len(rows))],'seconds':time.monotonic()-start,'prompt':prompt,'labels_used_for_inference':False,'code_revision':CODE,'weights_revision':revision,'model':version,'max_objects':max_objects if version=='sam3.1' else None}
 score=evaluate(meta,result);score.update(seconds=result['seconds'],prompt=prompt,phase=phase,model=version,max_objects=result['max_objects'],baseline_raw_sam2=evaluate(meta,baseline)['counts'])
 variant=('tracking31'+(f'-cap{max_objects}' if max_objects!=64 else '')) if version=='sam3.1' else 'tracking'
 if geometry:
  target=OUT/(variant+'-geometry')/phase/prompt.replace(' ','-')/clip
  variants=[{'method':'original-boxes',**{k:v for k,v in score.items()if k!='pairs'}}]
  dump(target/'original-boxes/tracks.json',result);dump(target/'original-boxes/metrics.json',score)
  for name,values in derived.items():
   alt={**result,'frames':[values[i]for i in range(len(rows))],'box_recipe':name}
   metrics=evaluate(meta,alt);metrics.update(seconds=result['seconds'],prompt=prompt,phase=phase,model=version,max_objects=max_objects,box_recipe=name)
   dump(target/name/'tracks.json',alt);dump(target/name/'metrics.json',metrics)
   variants.append({'method':name,**{k:v for k,v in metrics.items()if k!='pairs'}})
  volume.commit();return {'clip':clip,'variants':variants}
 target=OUT/variant/phase/prompt.replace(' ','-')/clip
 dump(target/'tracks.json',result);dump(target/'metrics.json',score);volume.commit();return {k:v for k,v in score.items()if k!='pairs'}

@app.function(image=image,cpu=8,memory=98304,gpu='H100',timeout=5400,max_containers=2,scaledown_window=10,volumes={'/data':volume})
def supported(phase:str,clip:str):
 import torch
 from huggingface_hub import hf_hub_download
 from nbhl_vlm_reconnect import datasets
 from nbhl_tracking_fourhour import evaluate
 from nbhl_sam3_detector_supported import track
 volume.reload();torch.set_num_threads(4)
 meta,features,baseline,folder,_=next(r for r in datasets(phase,labels=True) if r[0]['summary']['clip']==clip)
 ckpt=hf_hub_download('facebook/sam3','sam3.pt',revision=REV,local_files_only=True)
 folder=local_frames(folder,clip)
 result=track(meta,features,folder,Path(ckpt),margin=baseline['config']['margin'])
 score=evaluate(meta,result);score.update(seconds=result['seconds'],phase=phase,baseline_raw_sam2=evaluate(meta,baseline)['counts'],config=result['config'])
 target=OUT/'tracking'/phase/'detector-supported'/clip
 dump(target/'tracks.json',result);dump(target/'metrics.json',score);volume.commit();return {k:v for k,v in score.items()if k!='pairs'}

@app.local_entrypoint()
def main(task:str='detection',phase:str='development',prompt:str='person',threshold:float=.5,clips:str='gb-tracking-1800,gb-reid-long-0900',version:str='sam3',max_objects:int=64,geometry:bool=False):
 if task=='cache':result=cache.remote(version)
 elif task=='detection':
  cache.remote();result=detection.remote(phase,prompt,threshold)
 elif task=='tracking':
  cache.remote(version);result=list(tracking.starmap([(phase,c,prompt,version,max_objects,geometry)for c in clips.split(',')]))
 elif task=='supported':result=list(supported.starmap([(phase,c)for c in clips.split(',')]))
 else:raise ValueError(task)
 prefix=('sam31'+(f'-cap{max_objects}' if max_objects!=64 else '')) if version=='sam3.1' else 'sam3'
 if geometry:prefix+='-geometry'
 path=ROOT/'notes/ball-hockey'/f'{prefix}-{task}-{phase}-{prompt.replace(" ","-")}-20261023.json';dump(path,result);print(json.dumps(result,indent=2))
