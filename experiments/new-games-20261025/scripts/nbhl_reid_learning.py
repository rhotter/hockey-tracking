"""Data preparation and pair-supervised OSNet adaptation. GPU execution on Modal."""
import json,hashlib,datetime
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def prepare_dataset(base,out):
    import numpy as np
    from PIL import Image
    from collections import Counter
    out.mkdir(parents=True,exist_ok=True);pictures=out/'crops';pictures.mkdir(exist_ok=True)
    protocol=json.loads(Path('/work/reid-finetune-protocol-v1.json').read_text())
    train=set(protocol['train_sources']);valid=set(protocol['validation_sources']);images={};pairs=[];input_hash={}
    def save(key,pic,sid,origin):
        path=pictures/(key+'.jpg')
        if not path.exists():pic.save(path,quality=95)
        images[key]={'path':str(path),'sha256':sha(path),'source_id':sid,'split':'train'if sid in train else'validation','origin':origin}
    short_path=Path('/work/identity-pairs.json');short=json.loads(short_path.read_text());input_hash['short']=sha(short_path)
    for row in short['observations']:
        sid=row['source_id'];assert sid in train|valid
        p=Path('/data/sequences')/row['clip']/'img1'/f'{row["frame"]+1:06}.jpg'
        im=Image.open(p).convert('RGB');save(row['id'],im.crop(tuple(row['bbox'])),sid,{'frame':str(p),'frame_sha256':sha(p),'bbox':row['bbox']})
    for r in short['pairs']:
        assert images[r['a']]['split']==images[r['b']]['split']
        pairs.append({k:r[k]for k in ['id','a','b','same']}|{'split':images[r['a']]['split'],'label_type':'reviewed','source_id':images[r['a']]['source_id']})
    for sample in ['long-track-audit-v1','sam2-long-audit-v1','guarded-long-audit-v1']:
        p=Path('/work')/(sample+'-review.json');review=json.loads(p.read_text());input_hash[sample]=sha(p)
        for sid in sorted(train|valid):
            ep=base/sample/sid/'evidence.json';data=json.loads(ep.read_text());assert sha(ep)==review['evidence_sha256'][sid]
            for row in data['cases']:
                truth=review['labels'][row['id']]['identity']
                if truth=='uncertain':continue
                assert truth in ['same','different']
                for side in ['a','b']:
                    key=row['id']+'-'+side;p=Path(row[side]['image']);assert sha(p)==row[side]['sha256']
                    save(key,Image.open(p).convert('RGB'),sid,{'image':str(p),'sha256':sha(p)})
                pairs.append({'id':row['id'],'a':row['id']+'-a','b':row['id']+'-b','same':truth=='same','split':'train'if sid in train else'validation','label_type':'reviewed','source_id':sid})
    # Optional weak labels: isolated, high-overlap 0.2s motion matches only in training recordings.
    # Kept separate from reviewed labels and never used as evaluation truth.
    tag='yolo11m-multi-s17-f10-e80-remap-v2-6f459b18'
    def overlap(a,b):
        xx=max(0,min(a[2],b[2])-max(a[0],b[0]));yy=max(0,min(a[3],b[3])-max(a[1],b[1]));i=xx*yy
        return i/max(1e-8,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i)
    clips=sorted({r['clip']for r in short['observations']if r['source_id']in train})
    for clip in clips:
        sid=next(r['source_id']for r in short['observations']if r['clip']==clip)
        dp=base/'tracking'/tag/'detection-cache'/clip/'detections.json';detections=json.loads(dp.read_text());input_hash['detections/'+clip]=sha(dp)
        for frame in range(0,len(detections)-3,15):
            a=[v for v in detections[frame]if v[4]>=.75];b=[v for v in detections[frame+3]if v[4]>=.75]
            if not a or not b:continue
            ov=np.array([[overlap(u,v)for v in b]for u in a]);accepted=[]
            for i in range(len(a)):
                j=int(ov[i].argmax())
                if ov[i,j]<.7 or int(ov[:,j].argmax())!=i:continue
                if (np.delete(ov[i],j)>.2).any()or(np.delete(ov[:,j],i)>.2).any():continue
                accepted.append((i,j))
            if not accepted:continue
            im1=Image.open(Path('/data/sequences')/clip/'img1'/f'{frame+1:06}.jpg').convert('RGB');im2=Image.open(Path('/data/sequences')/clip/'img1'/f'{frame+4:06}.jpg').convert('RGB')
            keys={}
            for i,j in accepted:
                ka=f'weak-{clip}-{frame:04}-{i}-a';kb=f'weak-{clip}-{frame:04}-{i}-b'
                for key,im,box in [(ka,im1,a[i]),(kb,im2,b[j])]:save(key,im.crop(tuple(box[:4])),sid,{'clip':clip,'weak_motion_pair':True})
                pairs.append({'id':ka,'a':ka,'b':kb,'same':True,'split':'train','source_id':sid,'label_type':'weak_motion'});keys[i]=(ka,kb)
            for ii,(i,j)in enumerate(accepted):
                for u,v in accepted[ii+1:]:
                    if overlap(a[i],a[u])<.05:
                        pairs.append({'id':keys[i][0]+'__'+keys[u][1],'a':keys[i][0],'b':keys[u][1],'same':False,'split':'train','source_id':sid,'label_type':'weak_motion'})
    assert not ({r['sha256']for r in images.values()if r['split']=='train'}&{r['sha256']for r in images.values()if r['split']=='validation'})
    data={'images':images,'pairs':pairs,'input_sha256':input_hash,'protocol_sha256':sha('/work/reid-finetune-protocol-v1.json')};dump(out/'dataset.json',data)
    result={'id':'nbhl-reid-finetune-data-v1','images':len(images),'pairs':len(pairs),'counts':dict(Counter(f'{r["split"]}/{r["label_type"]}/{r["same"]}'for r in pairs)),'dataset_sha256':sha(out/'dataset.json'),'input_sha256':input_hash,'protocol':protocol,'independent_human_ground_truth':False};dump(out/'dataset-summary.json',result);return result

def load_model(base,variant='generic',checkpoint=None,device='cuda'):
    import importlib.util,torch
    from huggingface_hub import hf_hub_download
    meta=json.loads((base/'osnet-pairs-v1/ain.json').read_text());code=base/'osnet-pairs-v1/osnet_ain.py';assert sha(code)==meta['code_sha256']
    spec=importlib.util.spec_from_file_location('nbhl_osnet_train',code);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    backbone=module.osnet_ain_x1_0(num_classes=1,pretrained=False)
    p=Path(hf_hub_download(meta['weights_repository'],meta['weights_file'],revision=meta['weights_revision'],local_files_only=True));assert sha(p)==meta['weights_sha256']
    state=torch.load(p,map_location='cpu',weights_only=True);state=state.get('state_dict',state);state={k.removeprefix('module.'):v for k,v in state.items()};state={k:v for k,v in state.items()if not k.startswith('classifier.')}
    missing,extra=backbone.load_state_dict(state,strict=False);assert set(missing)=={'classifier.weight','classifier.bias'}and not extra
    class Diagonal(torch.nn.Module):
        def __init__(self):
            super().__init__();self.log_weights=torch.nn.Parameter(torch.zeros(512))
        def forward(self,x):return x*torch.exp(self.log_weights.clamp(-2,2))
    class MetricModel(torch.nn.Module):
        def __init__(self):
            super().__init__();self.backbone=backbone;self.projection=Diagonal()if variant=='diagonal'else torch.nn.Linear(512,512,bias=False)
            if variant!='diagonal':
                with torch.no_grad():self.projection.weight.copy_(torch.eye(512))
        def forward(self,x):return torch.nn.functional.normalize(self.projection(self.backbone(x)),dim=1)
    model=MetricModel().to(device).eval()
    if checkpoint:model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
    return model,meta

def validation(model,pictures,rows,preprocess):
    import numpy as np,torch
    keys=sorted({r[k]for r in rows for k in ['a','b']});features={}
    with torch.inference_mode():
        for i in range(0,len(keys),64):
            batch=keys[i:i+64];z=model(torch.stack([preprocess(pictures[k])for k in batch]).cuda()).cpu().numpy();features.update(zip(batch,z))
    scored=[r|{'similarity':float(np.dot(features[r['a']],features[r['b']]))}for r in rows]
    positive=np.array([r['same']for r in rows]);scores=np.array([r['similarity']for r in scored]);thresholds=sorted(set([1.00001,*scores.tolist()]),reverse=True);options=[]
    for threshold in thresholds:
        pred=scores>=threshold;tp=int((pred&positive).sum());fp=int((pred&~positive).sum());tn=int((~pred&~positive).sum());fn=int((~pred&positive).sum());precision=tp/max(1,tp+fp)
        if fp==0:options.append((tp,threshold,{'threshold':threshold,'tp':tp,'fp':fp,'tn':tn,'fn':fn,'precision':precision,'recall':tp/max(1,tp+fn)}))
    best=max(options,key=lambda r:(r[0],r[1]))[2]
    order=np.argsort(-scores);truth=positive[order];ap=float(((np.cumsum(truth)/(np.arange(len(truth))+1))*truth).sum()/max(1,truth.sum()))
    return best|{'average_precision':ap,'pairs':scored}

def train_model(base,out,variant,volume):
    import torch,numpy as np,random,time
    from PIL import Image
    from torchvision import transforms as T
    assert variant in ['reviewed_head','reviewed_last','mixed_last','extended_last','extended_full','diagonal'];torch.set_num_threads(4);torch.manual_seed(17);np.random.seed(17);random.seed(17)
    folder=out/variant;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completed.json').exists():return json.loads((folder/'completed.json').read_text())
    dataset_path=out/('dataset-extended.json'if variant.startswith('extended_')else'dataset.json');data=json.loads(dataset_path.read_text());assert data['protocol_sha256']==sha('/work/reid-finetune-protocol-v1.json')
    if variant.startswith('extended_'):assert data['extension_protocol_sha256']==sha('/work/reid-finetune-extension-v1.json')
    uses_weak=variant in ['mixed_last','extended_last','extended_full']
    rows=[r for r in data['pairs']if r['split']=='train'and(r['label_type']=='reviewed'or uses_weak)];val=[r for r in data['pairs']if r['split']=='validation']
    keys=sorted({r[k]for r in rows+val for k in ['a','b']});pictures={}
    from concurrent.futures import ThreadPoolExecutor
    import io
    def read_picture(k):
        r=data['images'][k];payload=Path(r['path']).read_bytes();assert hashlib.sha256(payload).hexdigest()==r['sha256']
        return k,Image.open(io.BytesIO(payload)).convert('RGB')
    with ThreadPoolExecutor(max_workers=32)as pool:
        for n,(k,picture)in enumerate(pool.map(read_picture,keys),1):
            pictures[k]=picture
            if n%500==0 or n==len(keys):print(json.dumps({'variant':variant,'loaded_crops':n,'total_crops':len(keys)}),flush=True)
    normalize=T.Normalize([.485,.456,.406],[.229,.224,.225]);plain=T.Compose([T.Resize((256,128)),T.ToTensor(),normalize]);aug=T.Compose([T.Resize((280,140)),T.RandomCrop((256,128)),T.ColorJitter(.15,.15,.1,.015),T.RandomGrayscale(.05),T.ToTensor(),normalize,T.RandomErasing(p=.2,scale=(.02,.12),ratio=(.3,3.3))])
    model,meta=load_model(base,variant);teacher,_=load_model(base)
    for p in teacher.parameters():p.requires_grad=False
    for name,p in model.named_parameters():p.requires_grad=name.startswith('projection')or(variant=='extended_full'and name.startswith('backbone.')and not name.startswith('backbone.classifier'))or(variant not in ['reviewed_head','diagonal']and name.startswith(('backbone.conv4','backbone.conv5','backbone.fc')))
    params=[p for p in model.parameters()if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=.01 if variant=='diagonal'else 3e-4 if variant=='reviewed_head'else 1e-5 if variant=='extended_full'else 3e-5,weight_decay=1e-4)
    baseline=validation(model,pictures,val,plain);best_score=(baseline['tp'],baseline['average_precision']);best_step=0;best_val=baseline;torch.save(model.state_dict(),folder/'best.pt');history=[];began=time.monotonic()
    pools={(label,same):[r for r in rows if r['label_type']==label and r['same']==same]for label in ['reviewed','weak_motion']for same in [False,True]}
    for step in range(1,1201):
        batch=[]
        for j in range(32):
            label='weak_motion'if uses_weak and j>=24 else'reviewed';batch.append(random.choice(pools[label,j%2==0]))
        a=torch.stack([aug(pictures[r['a']])for r in batch]).cuda();b=torch.stack([aug(pictures[r['b']])for r in batch]).cuda();x=torch.cat([a,b]);z=model(x);za,zb=z.chunk(2)
        same=torch.tensor([r['same']for r in batch],device='cuda');similarity=(za*zb).sum(1);weights=torch.tensor([1. if r['label_type']=='reviewed'else .25 for r in batch],device='cuda')
        pair=torch.where(same,(1-similarity).square(),torch.relu(similarity-.25).square());loss=(pair*weights).sum()/weights.sum()
        with torch.no_grad():old=teacher(x)
        regularizer=(1-(z*old).sum(1)).mean();loss=loss+.05*regularizer
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,5);optimizer.step()
        if step%100==0:
            score=validation(model,pictures,val,plain);key=(score['tp'],score['average_precision']);entry={'step':step,'loss':float(loss.detach()),'validation':{k:v for k,v in score.items()if k!='pairs'},'seconds':time.monotonic()-began};history.append(entry)
            if key>best_score:best_score=key;best_step=step;best_val=score;torch.save(model.state_dict(),folder/'best.pt')
            dump(folder/'progress.json',{'variant':variant,'best_step':best_step,'history':history});volume.commit();print(json.dumps({'variant':variant,**entry}),flush=True)
    result={'variant':variant,'training_steps':1200,'selected_step':best_step,'train_pairs':len(rows),'validation_pairs':len(val),'dataset_sha256':sha(dataset_path),'code_sha256':sha(__file__),'optimizer_settings':{'optimizer':'AdamW','lr':optimizer.param_groups[0]['lr'],'weight_decay':1e-4,'batch_pairs':32,'distillation_weight':.05,'gradient_clip_norm':5,'seed':17},'weights_sha256':sha(folder/'best.pt'),'base_weights_sha256':meta['weights_sha256'],'validation_before':baseline,'validation_after':best_val,'history':history,'seconds':time.monotonic()-began,'trainable_parameters':sum(p.numel()for p in params),'selection':'Maximum same-pair recall at zero observed validation false matches, then pair average precision. Baseline step 0 remains eligible. Massachusetts only; no Great Britain input.','scope':'Pair-supervised identity adaptation; not an end-to-end detector or SAM2 fine-tune. Sparse assistant-reviewed labels and optionally separately identified weak motion pairs.'}
    dump(folder/'completed.json',result);volume.commit();return result

def prepare_extended(base,out):
    """Add explicitly weak 0.2s pairs from training-only long detection caches."""
    import gzip,subprocess,numpy as np
    from PIL import Image
    from collections import Counter
    data=json.loads((out/'dataset.json').read_text());pictures=out/'crops-extended';pictures.mkdir(exist_ok=True)
    protocol=json.loads(Path('/work/reid-finetune-extension-v1.json').read_text());sources=protocol['sources'];evidence={}
    def overlap(a,b):
        i=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));return i/max(1e-8,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i)
    for sid in sources:
        source=Path('/data/raw')/(sid+'.mp4');streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(source)]))['streams'];v=next(v for v in streams if v['codec_type']=='video');w,h=v['width'],v['height']
        for chunk in protocol['chunks']:
            folder=base/'long-tracking-v1'/sid/f'chunk-{chunk:03}';dp=folder/'detections.jsonl.gz';meta=json.loads((folder/'completed.json').read_text());evidence[f'{sid}/{chunk}']=sha(dp)
            with gzip.open(dp,'rt')as f:rows=[json.loads(s)for s in f]
            assert len(rows)==1800 and meta['start_s']==chunk*120
            detections=[[d for d,p in zip(r['detections'],r['role_probabilities'])if p>=meta['role_threshold']and d[4]>=.75]for r in rows]
            requested={}
            for f in range(0,1797,15):
                a,b=detections[f],detections[f+3]
                if not a or not b:continue
                ov=np.array([[overlap(x,y)for y in b]for x in a]);pairs=[]
                for i in range(len(a)):
                    j=int(ov[i].argmax())
                    if ov[i,j]>=.7 and int(ov[:,j].argmax())==i and not(np.delete(ov[i],j)>.2).any()and not(np.delete(ov[:,j],i)>.2).any():pairs.append((i,j))
                if pairs:requested[f]=(a,b,pairs)
            proc=subprocess.Popen(['ffmpeg','-v','error','-threads','2','-ss',str(chunk*120),'-i',str(source),'-t','120','-an','-vf','setpts=PTS-STARTPTS,fps=15:start_time=0,select=not(mod(n\\,3))','-vsync','0','-pix_fmt','rgb24','-f','rawvideo','-threads','2','pipe:1'],stdout=subprocess.PIPE);previous=None
            for f in range(0,1800,3):
                payload=proc.stdout.read(w*h*3);assert len(payload)==w*h*3
                if f in requested:previous=Image.fromarray(np.frombuffer(payload,np.uint8).reshape(h,w,3).copy())
                if f-3 not in requested:continue
                im=Image.fromarray(np.frombuffer(payload,np.uint8).reshape(h,w,3));a,b,pairs=requested[f-3];keys={}
                for i,j in pairs:
                    ka=f'weak-long-{sid}-{chunk}-{f-3}-{i}-a';kb=ka[:-1]+'b'
                    for key,frame,box in [(ka,previous,a[i]),(kb,im,b[j])]:
                        p=pictures/(key+'.jpg');frame.crop(tuple(box[:4])).save(p,quality=95);data['images'][key]={'path':str(p),'sha256':sha(p),'source_id':sid,'split':'train','origin':{'weak_motion_pair':True,'chunk':chunk,'frame':f-3 if key==ka else f,'bbox':box[:4]}}
                    data['pairs'].append({'id':ka,'a':ka,'b':kb,'same':True,'split':'train','source_id':sid,'label_type':'weak_motion'});keys[i]=(ka,kb)
                for n,(i,j)in enumerate(pairs):
                    for u,v in pairs[n+1:]:
                        if overlap(a[i],a[u])<.05:data['pairs'].append({'id':keys[i][0]+'__'+keys[u][1],'a':keys[i][0],'b':keys[u][1],'same':False,'split':'train','source_id':sid,'label_type':'weak_motion'})
            proc.stdout.close();assert proc.wait()==0;print(json.dumps({'source':sid,'chunk':chunk,'total_pairs':len(data['pairs'])}),flush=True)
    data['extension_protocol_sha256']=sha('/work/reid-finetune-extension-v1.json');data['input_sha256'].update(evidence);dump(out/'dataset-extended.json',data)
    result={'id':'nbhl-reid-extended-data-v1','images':len(data['images']),'pairs':len(data['pairs']),'counts':dict(Counter(f'{r["split"]}/{r["label_type"]}/{r["same"]}'for r in data['pairs'])),'dataset_sha256':sha(out/'dataset-extended.json'),'extension_protocol':protocol,'independent_human_ground_truth':False};dump(out/'dataset-extended-summary.json',result);return result
