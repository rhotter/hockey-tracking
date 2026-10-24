"""Box-supervised SAM3 detector adaptation on reviewed NBHL frames only."""
from pathlib import Path
import json
import modal
from modal_campaign import ROOT,volume
from modal_nbhl_sam3_20261023 import image as sam_image,REV
BASE=Path('/data/ball-hockey/reliability-20261018')
OUT=BASE/'sam3-finetune-20261024-v1'
app=modal.App('sport-scans-sam3-finetune-20261024')
image=sam_image.add_local_file(ROOT/'scripts/modal_nbhl_sam3_20261023.py','/work/modal_nbhl_sam3_20261023.py')
image=image.add_local_file(ROOT/'data/ball-hockey/sam3-finetune-protocol-20261024.json','/work/finetune-protocol.json')
image=image.add_local_file(ROOT/'data/ball-hockey/sam3-finetune-extension-20261024.json','/work/finetune-extension.json')
image=image.add_local_dir(ROOT/'data/ball-hockey/annotations/canonical-expanded','/work/finetune-annotations')
image=image.add_local_file(ROOT/'data/ball-hockey/sources.json','/work/sources.json')
image=image.add_local_file(ROOT/'scripts/nbhl_sam3_losses.py','/work/nbhl_sam3_losses.py')


def setup():
    import torch
    from huggingface_hub import hf_hub_download
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    torch.set_num_threads(4)
    checkpoint=hf_hub_download('facebook/sam3','sam3.pt',revision=REV,local_files_only=True)
    model=build_sam3_image_model(checkpoint_path=checkpoint,load_from_HF=False,enable_segmentation=False)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,Sam3Processor(model,confidence_threshold=.05)


def move(value,device):
    import torch
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,dict):return {k:move(v,device) for k,v in value.items()}
    if isinstance(value,list):return [move(v,device) for v in value]
    if isinstance(value,tuple):return tuple(move(v,device) for v in value)
    return value


@app.function(image=image,cpu=8,memory=49152,gpu='H100',timeout=3600,scaledown_window=10,volumes={'/data':volume})
def prepare():
    import hashlib,time,torch
    from PIL import Image,ImageOps
    from torchvision.transforms.v2.functional import to_image
    from nbhl_vlm_reconnect import sha,dump
    volume.reload();model,processor=setup();proto=json.loads(Path('/work/finetune-protocol.json').read_text())
    registry={s['id']:s for s in json.loads(Path('/work/sources.json').read_text())}
    folder=OUT/'features';folder.mkdir(parents=True,exist_ok=True);manifest=[];started=time.monotonic()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        language=move(model.backbone.forward_text([proto['prompt']],device='cuda'),'cpu')
        torch.save(language,folder/'language.pt')
        for source in proto['training_sources']+[proto['validation_source']]:
            split='train' if source in proto['training_sources'] else 'validation'
            assert registry[source]['sport']=='ball_hockey'
            if split=='train':assert registry[source]['training_eligible']
            annotation=Path('/work/finetune-annotations')/(source+'.json');data=json.loads(annotation.read_text())
            for frame in data['frames']:
                assert sha(frame['image'])==frame['sha256']
                original=Image.open(frame['image']).convert('RGB');w,h=original.size
                for flip in ([False,True] if split=='train' else [False]):
                    key=frame['id']+('-flip' if flip else '');path=folder/(key+'.pt')
                    boxes=torch.tensor([p['bbox'] for p in frame['players']],dtype=torch.float32).reshape(-1,4)
                    if flip:boxes[:,[0,2]]=w-boxes[:,[2,0]]
                    boxes/=torch.tensor([w,h,w,h])
                    if not path.exists():
                        picture=ImageOps.mirror(original) if flip else original
                        tensor=processor.transform(to_image(picture).to('cuda')).unsqueeze(0)
                        backbone=move(model.backbone.forward_image(tensor),'cpu')
                        torch.save({'backbone':backbone,'boxes':boxes},path)
                    manifest.append({'key':key,'split':split,'source':source,'image_sha256':frame['sha256'],'annotation_sha256':sha(annotation),'path':str(path),'width':w,'height':h,'players':frame['players'],'flip':flip,'sha256':sha(path)})
            volume.commit();print(json.dumps({'source':source,'features':len(manifest),'seconds':time.monotonic()-started}),flush=True)
    report={'protocol_sha256':sha('/work/finetune-protocol.json'),'language_sha256':sha(folder/'language.pt'),'frames':manifest,'seconds':time.monotonic()-started}
    dump(OUT/'features.json',report);volume.commit();return {'features':len(manifest),'seconds':report['seconds'],'manifest_sha256':sha(OUT/'features.json')}


from nbhl_sam3_losses import loss_for


@app.function(image=image,cpu=8,memory=98304,gpu='H100',timeout=10800,max_containers=3,scaledown_window=10,volumes={'/data':volume})
def train(variant:str='decoder-lr1e5',smoke:bool=False):
    import copy,random,time,math,shutil,tempfile,concurrent.futures,hashlib
    import numpy as np
    import torch
    from torchvision.ops import box_convert
    from nbhl_vlm_reconnect import sha,dump
    from nbhl_detection_metrics import summarize,average_precision
    volume.reload();proto=json.loads(Path('/work/finetune-protocol.json').read_text());manifest=json.loads((OUT/'features.json').read_text())
    assert manifest['protocol_sha256']==sha('/work/finetune-protocol.json')
    extension=json.loads(Path('/work/finetune-extension.json').read_text())
    spec=next(r for r in proto['variants']+extension['variants'] if r['id']==variant)
    vision_mode=variant.startswith('vision4');transformer_mode=variant.startswith('transformer')
    if vision_mode:
        # Pinned SAM3 uses an inference-only fused ViT MLP. Keep that exact
        # forward for validation; use differentiable Linear+GELU during training.
        import sam3.model.vitdet as vit_module
        fused=vit_module.addmm_act
        def differentiable_activation(activation,linear,x):
            if not torch.is_grad_enabled():return fused(activation,linear,x)
            if activation in [torch.nn.GELU,torch.nn.functional.gelu]:return torch.nn.functional.gelu(linear(x))
            if activation in [torch.nn.ReLU,torch.nn.functional.relu]:return torch.nn.functional.relu(linear(x))
            raise ValueError(activation)
        vit_module.addmm_act=differentiable_activation
    random.seed(17);np.random.seed(17);torch.manual_seed(17);torch.cuda.manual_seed_all(17)
    model,processor=setup();parameters={}
    for name,p in model.named_parameters():
        prefix='transformer.decoder.bbox_embed.' if variant.startswith('heads') else 'transformer.' if transformer_mode else 'transformer.decoder.'
        vision_parameter=vision_mode and any(name.startswith(f'backbone.vision_backbone.trunk.blocks.{b}.attn.{part}.') for b in range(28,32) for part in ['qkv','proj'])
        enable=name.startswith('dot_prod_scoring.') or name.startswith(prefix) or vision_parameter
        if enable:p.requires_grad_(True);parameters[name]=p
    assert parameters and not any(n.startswith('segmentation_head.') for n in parameters)
    if not vision_mode:assert not any(n.startswith('backbone.') for n in parameters)
    if not transformer_mode:assert not any(n.startswith('transformer.encoder.') for n in parameters)
    before={n:p.detach().cpu().clone() for n,p in parameters.items()}
    frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad and p.numel()<2000}
    def frozen_hash():
        digest=hashlib.sha256()
        for n,p in model.named_parameters():
            if not p.requires_grad:
                digest.update(n.encode());digest.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()
    frozen_before=frozen_hash()
    groups=[{'params':[p for n,p in parameters.items() if not n.startswith('backbone.')],'lr':spec['lr'],'base_lr':spec['lr']}]
    if vision_mode:groups.append({'params':[p for n,p in parameters.items() if n.startswith('backbone.')],'lr':spec['vision_lr'],'base_lr':spec['vision_lr']})
    optimizer=torch.optim.AdamW(groups,weight_decay=.0001)
    local=Path(tempfile.mkdtemp(prefix='sam3-ft-'));jobs=[r for r in manifest['frames'] if not smoke or r['split']=='validation' or r['key'] in [v['key'] for v in manifest['frames'] if v['split']=='train'][:4]]
    originals={}
    if vision_mode:
        for source in proto['training_sources']+[proto['validation_source']]:
            for frame in json.loads(Path('/work/finetune-annotations',source+'.json').read_text())['frames']:originals[frame['id']]=frame
    def stage(row):
        if vision_mode:
            original=originals[row['key'].removesuffix('-flip')];target=local/(row['key']+'.jpg')
            shutil.copyfile(original['image'],target);assert sha(target)==row['image_sha256'];return row|{'local_image':str(target)}
        target=local/(row['key']+'.pt');shutil.copyfile(row['path'],target);assert sha(target)==row['sha256'];return row|{'local':str(target)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:rows=list(pool.map(stage,jobs))
    language=move(torch.load(OUT/'features/language.pt',weights_only=False),'cuda')
    training=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='validation']
    folder=OUT/(('smoke-'+variant) if smoke else variant);folder.mkdir(parents=True,exist_ok=True)
    epochs=5 if smoke else spec.get('epochs',20);started=time.monotonic();history=[];best=None;steps=0;first_loss=None;last_loss=None;nonzero=False
    def forward(row):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if vision_mode:
                from PIL import Image,ImageOps
                from torchvision.transforms.v2.functional import to_image
                picture=Image.open(row['local_image']).convert('RGB')
                if row['flip']:picture=ImageOps.mirror(picture)
                tensor=processor.transform(to_image(picture).to('cuda')).unsqueeze(0)
                backbone=model.backbone.forward_image(tensor)
                boxes=torch.tensor([p['bbox'] for p in row['players']],dtype=torch.float32).reshape(-1,4)
                if row['flip']:boxes[:,[0,2]]=row['width']-boxes[:,[2,0]]
                boxes/=torch.tensor([row['width'],row['height']]*2)
            else:
                data=torch.load(row['local'],weights_only=False,map_location='cpu')
                backbone=move(data['backbone'],'cuda');boxes=data['boxes']
            backbone.update(language)
            out=model.forward_grounding(backbone_out=backbone,find_input=processor.find_stage,geometric_prompt=model._get_dummy_prompt(),find_target=None)
        return out,boxes.to('cuda')
    def validate():
        values=[]
        with torch.no_grad():
            for row in validation:
                out,_=forward(row);boxes=box_convert(out['pred_boxes'][0].float(),'cxcywh','xyxy')
                boxes*=torch.tensor([row['width'],row['height']]*2,device='cuda')
                scores=(out['pred_logits'][0].float().sigmoid()*out['presence_logit_dec'].float().sigmoid().reshape(-1)[0]).ravel()
                values.append({'id':row['key'],'players':row['players'],'predictions':[{'bbox':b,'confidence':s} for b,s in zip(boxes.cpu().tolist(),scores.cpu().tolist())]})
        curve=[summarize(values,t/10) for t in range(1,10)]
        chosen=max(curve,key=lambda r:(r['f1'],r['precision'] or 0,r['confidence_threshold']))
        return {'selected':chosen,'curve':curve,'ap50':average_precision(values)},values
    initial,_=validate();dump(folder/'initial-validation.json',initial)
    fixed_smoke_rows=training[:4]
    def smoke_loss():
        with torch.no_grad():return float(np.mean([float(loss_for(*forward(r))[0]) for r in fixed_smoke_rows]))
    smoke_before=smoke_loss() if smoke else None
    for epoch in range(epochs):
        random.shuffle(training);losses=[]
        for row in training:
            optimizer.zero_grad(set_to_none=True);out,target=forward(row);loss,parts=loss_for(out,target)
            assert torch.isfinite(loss),parts
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(list(parameters.values()),1.)
            assert torch.isfinite(norm)
            nonzero|=float(norm)>0
            optimizer.step();steps+=1;losses.append(float(loss.detach()))
            if first_loss is None:first_loss=losses[-1]
            last_loss=losses[-1]
            if steps%100==0:print(json.dumps({'variant':variant,'epoch':epoch+1,'step':steps,'loss':losses[-1],'seconds':time.monotonic()-started}),flush=True)
        for group in optimizer.param_groups:group['lr']=group['base_lr']*(.1+.9*.5*(1+math.cos(math.pi*(epoch+1)/epochs)))
        metrics,values=validate();entry={'epoch':epoch+1,'mean_loss':float(np.mean(losses)),'validation':metrics,'seconds':time.monotonic()-started};history.append(entry)
        objective=lambda x:(x['selected']['f1'],x['selected']['precision'] or 0,x['selected']['confidence_threshold'])
        if best is None or objective(metrics)>objective(best['validation']):
            best=entry;torch.save({n:p.detach().cpu() for n,p in parameters.items()},folder/'best-delta.pt');dump(folder/'best-validation-predictions.json',values)
        dump(folder/'progress.json',{'variant':variant,'best':best,'history':history});volume.commit()
        print(json.dumps({'variant':variant,**entry}),flush=True)
    changed=sum(not torch.equal(before[n],p.detach().cpu()) for n,p in parameters.items())
    vision_changed=sum(not torch.equal(before[n],p.detach().cpu()) for n,p in parameters.items() if n.startswith('backbone.'))
    if vision_mode:assert vision_changed>0
    assert changed>0 and nonzero
    current=dict(model.named_parameters());assert all(torch.equal(v,current[n].detach().cpu()) for n,v in frozen.items())
    frozen_after=frozen_hash();assert frozen_after==frozen_before
    smoke_after=smoke_loss() if smoke else None
    if smoke:assert smoke_after<smoke_before,(smoke_before,smoke_after)
    report={'variant':variant,'smoke':smoke,'best':best,'history':history,'initial':initial,'steps':steps,'seconds':time.monotonic()-started,'trainable_parameters':sum(p.numel() for p in parameters.values()),'trainable_tensors':len(parameters),'changed_tensors':changed,'frozen_small_tensors_checked':len(frozen),'nonzero_finite_gradients':nonzero,'first_loss':first_loss,'last_loss':last_loss,'protocol_sha256':sha('/work/finetune-protocol.json'),'features_manifest_sha256':sha(OUT/'features.json'),'delta_sha256':sha(folder/'best-delta.pt'),'trainable_names':list(parameters),'training_frames_with_flips':len(training)}
    report.update(frozen_parameters_before_sha256=frozen_before,frozen_parameters_after_sha256=frozen_after,smoke_fixed_loss_before=smoke_before,smoke_fixed_loss_after=smoke_after,vision_changed_tensors=vision_changed,extension_protocol_sha256=sha('/work/finetune-extension.json') if variant not in [r['id'] for r in proto['variants']] else None,vision_gradient_compatibility='Differentiable Linear+GELU replaces inference-only fused MLP only while gradients enabled. Validation retains official fused inference.' if vision_mode else None)
    report.update(gpu=torch.cuda.get_device_name(0),torch_version=str(torch.__version__))
    dump(folder/'completed.json',report);volume.commit();return report


@app.function(image=image,cpu=1,memory=2048,timeout=180,volumes={'/data':volume})
def select():
    import datetime
    from nbhl_vlm_reconnect import dump,sha
    volume.reload();assert not (OUT/'selection.json').exists(),'Immutable selection already exists'
    assert not (OUT/'regression.json').exists()
    proto=json.loads(Path('/work/finetune-protocol.json').read_text());extension=json.loads(Path('/work/finetune-extension.json').read_text());candidates=[];reports=[]
    for spec in proto['variants']+extension['variants']:
        path=OUT/spec['id']/'completed.json';r=json.loads(path.read_text());assert r['protocol_sha256']==sha('/work/finetune-protocol.json')
        if spec in extension['variants']:assert r['extension_protocol_sha256']==sha('/work/finetune-extension.json')
        if not candidates:candidates.append({'variant':'original','epoch':0,'validation':r['initial'],'delta_sha256':None})
        candidates.append({'variant':r['variant'],'epoch':r['best']['epoch'],'validation':r['best']['validation'],'delta_sha256':r['delta_sha256']})
        reports.append({'variant':r['variant'],'report_sha256':sha(path)})
    def key(r):
        m=r['validation']['selected'];return m['f1'],m['precision'] or 0,m['confidence_threshold'],r['variant']=='original'
    winner=max(candidates,key=key)
    result={'frozen_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':sha('/work/finetune-protocol.json'),'extension_protocol_sha256':sha('/work/finetune-extension.json'),'best':winner,'candidates':candidates,'reports':reports,'new_regression_scores_seen':False}
    dump(OUT/'selection.json',result);volume.commit();return result


@app.function(image=image,cpu=8,memory=49152,gpu=['H100','H200','A100-80GB'],timeout=1800,scaledown_window=10,volumes={'/data':volume})
def regression():
    import time,torch
    from PIL import Image
    from torchvision.transforms.v2.functional import to_image
    from torchvision.ops import box_convert
    from nbhl_vlm_reconnect import dump,sha
    from nbhl_detection_metrics import summarize,average_precision
    volume.reload();selection=json.loads((OUT/'selection.json').read_text());proto=json.loads(Path('/work/finetune-protocol.json').read_text())
    assert selection['protocol_sha256']==sha('/work/finetune-protocol.json')
    assert not (OUT/'regression.json').exists(),'Regression result already saved'
    reports=[]
    for spec in [selection['candidates'][0]]+([selection['best']] if selection['best']['variant']!='original' else []):
        model,processor=setup();variant=spec['variant']
        if variant!='original':
            path=OUT/variant/'best-delta.pt';assert sha(path)==spec['delta_sha256']
            delta=torch.load(path,weights_only=True,map_location='cpu');state=model.state_dict()
            assert all(k in state and v.shape==state[k].shape for k,v in delta.items())
            _,unexpected=model.load_state_dict(delta,strict=False);assert not unexpected
        rows=[];started=time.monotonic()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            language=model.backbone.forward_text([proto['prompt']],device='cuda')
            for source in proto['regression_sources']:
                ann=json.loads(Path('/work/annotations',source+'.json').read_text())
                for frame in ann['frames']:
                    assert sha(frame['image'])==frame['sha256'];picture=Image.open(frame['image']).convert('RGB')
                    tensor=processor.transform(to_image(picture).to('cuda')).unsqueeze(0)
                    backbone=model.backbone.forward_image(tensor);backbone.update(language)
                    out=model.forward_grounding(backbone_out=backbone,find_input=processor.find_stage,geometric_prompt=model._get_dummy_prompt(),find_target=None)
                    boxes=box_convert(out['pred_boxes'][0].float(),'cxcywh','xyxy')*torch.tensor([frame['width'],frame['height']]*2,device='cuda')
                    scores=(out['pred_logits'][0].float().sigmoid()*out['presence_logit_dec'].float().sigmoid().reshape(-1)[0]).ravel()
                    rows.append({'id':frame['id'],'source_id':source,'players':frame['players'],'predictions':[{'bbox':b,'confidence':s} for b,s in zip(boxes.cpu().tolist(),scores.cpu().tolist())]})
        threshold=spec['validation']['selected']['confidence_threshold']
        report={'variant':variant,'threshold':threshold,'metrics':summarize(rows,threshold),'ap50':average_precision(rows),'seconds':time.monotonic()-started,'selection_sha256':sha(OUT/'selection.json'),'sources':[{'source_id':source,'metrics':summarize([r for r in rows if r['source_id']==source],threshold)} for source in proto['regression_sources']]}
        dump(OUT/variant/'regression-predictions.json',rows);reports.append(report)
        del model,processor;torch.cuda.empty_cache()
    result={'selection_sha256':sha(OUT/'selection.json'),'results':reports,'gpu':torch.cuda.get_device_name(0),'torch_version':str(torch.__version__)};dump(OUT/'regression.json',result);volume.commit();return result


@app.local_entrypoint()
def main(task:str='prepare',variants:str='heads-lr1e4,decoder-lr1e5,decoder-lr5e5'):
    if task=='prepare':result=prepare.remote()
    elif task in ['train','smoke']:
        if variants=='vision4-lr1e6':result=[train.with_options(gpu='H200').remote(variants,task=='smoke')]
        else:result=list(train.starmap([(v,task=='smoke') for v in variants.split(',')]))
    elif task=='select':result=select.remote()
    elif task=='regression':result=regression.remote()
    else:raise ValueError(task)
    suffix='-'+variants.replace(',','_') if task in ['train','smoke'] else ''
    path=ROOT/'notes/ball-hockey'/f'sam3-finetune-{task}{suffix}-20261024.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'task':task,'saved':str(path)}))
