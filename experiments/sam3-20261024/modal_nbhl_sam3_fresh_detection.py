"""Frozen-model detection audit on eleven newly annotated Pittsburgh frames."""
from pathlib import Path
import json
import modal
from modal_campaign import ROOT,volume
from modal_nbhl_sam3_finetune import image as fine_image

BASE=Path('/data/ball-hockey/reliability-20261018')
FT=BASE/'sam3-finetune-20261024-v1'
CAMPAIGN=BASE/'sam3-hybrid-20261024-v1'
OUT=CAMPAIGN/'fresh-detection'
app=modal.App('sport-scans-sam3-fresh-detection-20261024')
image=fine_image
for name in ['modal_nbhl_sam3_finetune.py','nbhl_sam3_rink_filter.py']:
    image=image.add_local_file(ROOT/'scripts'/name,'/work/'+name)
for name in ['sam3-fresh-detection-audit-protocol-20261024.json','sam3-fresh-detection-audit-reference-20261024.json']:
    image=image.add_local_file(ROOT/'data/ball-hockey'/name,'/work/'+name)
image=image.add_local_file(ROOT/'data/ball-hockey/detector-heldout-freeze.json','/work/detector-heldout-freeze.json')


@app.function(image=image,cpu=8,memory=49152,gpu=['H100','H200','A100-80GB'],timeout=1800,scaledown_window=10,volumes={'/data':volume})
def run():
    import gc,time,base64,zlib,datetime
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.transforms.v2.functional import to_image
    from torchvision.ops import box_convert
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from huggingface_hub import hf_hub_download
    from modal_nbhl_sam3_finetune import setup,REV
    from nbhl_sam3_rink_filter import filter_row
    from nbhl_detection_metrics import summarize,average_precision
    from nbhl_vlm_reconnect import sha,dump
    volume.reload();started=time.monotonic()
    assert not (OUT/'report.json').exists()
    reference_path=Path('/work/sam3-fresh-detection-audit-reference-20261024.json')
    protocol_path=Path('/work/sam3-fresh-detection-audit-protocol-20261024.json')
    reference=json.loads(reference_path.read_text());assert reference['protocol_sha256']==sha(protocol_path)
    selection=json.loads((FT/'selection.json').read_text());rink=json.loads((CAMPAIGN/'rink-filter/selection.json').read_text())
    assert selection['frozen_at']<reference['sealed_at'] and rink['frozen_at']<reference['sealed_at']
    assert not reference['sam3_predictions_seen'] and not reference['rink_filter_predictions_seen']
    variants=[selection['candidates'][0],selection['best']];predictions={};results=[]
    for spec in variants:
        gc.collect();torch.cuda.empty_cache();model,processor=setup()
        if spec['variant']!='original':
            path=FT/spec['variant']/'best-delta.pt';assert sha(path)==spec['delta_sha256']
            delta=torch.load(path,map_location='cpu',weights_only=True);state=model.state_dict()
            assert all(k in state and v.shape==state[k].shape for k,v in delta.items())
            _,unexpected=model.load_state_dict(delta,strict=False);assert not unexpected
            del delta,state
        rows=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            language=model.backbone.forward_text(['hockey player'],device='cuda')
            for frame in reference['frames']:
                assert sha(frame['image'])==frame['sha256']
                picture=Image.open(frame['image']).convert('RGB')
                tensor=processor.transform(to_image(picture).to('cuda')).unsqueeze(0)
                backbone=model.backbone.forward_image(tensor);backbone.update(language)
                out=model.forward_grounding(backbone_out=backbone,find_input=processor.find_stage,geometric_prompt=model._get_dummy_prompt(),find_target=None)
                boxes=box_convert(out['pred_boxes'][0].float(),'cxcywh','xyxy')*torch.tensor([1920,1080]*2,device='cuda')
                scores=(out['pred_logits'][0].float().sigmoid()*out['presence_logit_dec'].float().sigmoid().reshape(-1)[0]).ravel()
                rows.append({'id':frame['id'],'players':frame['players'],'predictions':[{'bbox':b,'confidence':s} for b,s in zip(boxes.cpu().tolist(),scores.cpu().tolist())]})
        threshold=spec['validation']['selected']['confidence_threshold'];predictions[spec['variant']]=rows
        results.append({'variant':spec['variant'],'threshold':threshold,'metrics':summarize(rows,threshold),'ap50':average_precision(rows)})
        del model,processor,language,backbone,out,tensor,boxes,scores
    gc.collect();torch.cuda.empty_cache()
    checkpoint=hf_hub_download('facebook/sam3','sam3.pt',revision=REV,local_files_only=True)
    model=build_sam3_image_model(checkpoint_path=checkpoint,load_from_HF=False)
    setting=rink['best']['setting'];processor=Sam3Processor(model,confidence_threshold=.29)
    masks=[];filtered=[];fine={r['id']:r for r in predictions[selection['best']['variant']]}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for frame in reference['frames']:
            state=processor.set_image(Image.open(frame['image']).convert('RGB'))
            out=processor.set_text_prompt(state=state,prompt=setting['prompt'])
            scores=out['scores'].float().cpu().numpy();raw=out['masks'].detach().cpu().numpy().reshape(-1,1080,1920)
            chosen=raw[scores>=setting['mask_threshold']]
            union=np.any(chosen,axis=0) if len(chosen) else np.zeros((1080,1920),dtype=bool)
            mask=np.asarray(Image.fromarray(union).resize((252,252),Image.Resampling.NEAREST),dtype=bool)
            row,detail=filter_row(fine[frame['id']],mask,1920,1080,setting['dilation']);filtered.append(row)
            masks.append({'id':frame['id'],'mask_shape':[252,252],'width':1920,'height':1080,'frame_sha256':frame['sha256'],
                          'packed_bits_zlib_base64':base64.b64encode(zlib.compress(np.packbits(mask).tobytes())).decode(),'filter_diagnostics':detail})
    threshold=selection['best']['validation']['selected']['confidence_threshold']
    predictions['fine-tuned-rink']=filtered
    results.append({'variant':'fine-tuned-rink','threshold':threshold,'metrics':summarize(filtered,threshold),'ap50':average_precision(filtered),'setting':setting})
    meta_path=BASE/'tracking-four-hour-v1/long-cache-trial/pittsburgh-runtime-2385-315-absolute-gpu-bounds/metadata.json'
    meta=json.loads(meta_path.read_text());yolo=[]
    for frame in reference['frames']:
        det=meta['frames'][frame['frame']]['detections']
        yolo.append({'id':frame['id'],'players':frame['players'],'predictions':[{'bbox':d[:4],'confidence':d[4]} for d in det]})
    predictions['yolo-pipeline']=yolo
    results.append({'variant':'yolo-pipeline','threshold':.1,'metrics':summarize(yolo,.1),'ap50':average_precision(yolo),
                    'reference_bias':'Boxes were assisted by these proposals; this baseline is not independently annotated.'})
    dump(OUT/'predictions.json',predictions);dump(OUT/'masks.json',masks)
    report={'reference_sha256':sha(reference_path),'protocol_sha256':sha(protocol_path),'reference_sealed_at':reference['sealed_at'],
            'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fine_tune_selection_sha256':sha(FT/'selection.json'),
            'rink_selection_sha256':sha(CAMPAIGN/'rink-filter/selection.json'),'yolo_metadata_sha256':sha(meta_path),
            'predictions_sha256':sha(OUT/'predictions.json'),'masks_sha256':sha(OUT/'masks.json'),'results':results,
            'counts':reference['counts'],'selection_uses_this_audit':False,'gpu':torch.cuda.get_device_name(0),'seconds':time.monotonic()-started,
            'limits':reference['limits']}
    dump(OUT/'report.json',report);volume.commit();return report


@app.function(image=image,cpu=4,memory=8192,timeout=600,volumes={'/data':volume})
def review():
    from PIL import Image,ImageDraw,ImageFont
    from nbhl_detection_metrics import match
    from nbhl_vlm_reconnect import sha,dump
    volume.reload();report=json.loads((OUT/'report.json').read_text())
    reference=json.loads(Path('/work/sam3-fresh-detection-audit-reference-20261024.json').read_text())
    predictions=json.loads((OUT/'predictions.json').read_text())
    predictions={v:{r['id']:r for r in rows} for v,rows in predictions.items()}
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    labels=['Pretrained SAM3','Fine-tuned SAM3','Fine-tune + rink','YOLO (proposal-assisted reference)']
    files=[];dest=OUT/'public';dest.mkdir(parents=True,exist_ok=True)
    for sheet in range(4):
        frames=reference['frames'][sheet*3:(sheet+1)*3]
        canvas=Image.new('RGB',(3200,392*len(frames)),'white');draw=ImageDraw.Draw(canvas)
        for i,frame in enumerate(frames):
            assert sha(frame['image'])==frame['sha256'];picture=Image.open(frame['image']).convert('RGB').resize((640,360))
            for col,spec in enumerate([None]+report['results']):
                im=picture.copy();dr=ImageDraw.Draw(im)
                if spec is None:
                    ranked=[{'bbox':p['bbox'],'hit':True} for p in frame['players']]
                    title=f'{frame["source_time_s"]}s | {len(ranked)} reference players'
                else:
                    ranked,_=match([predictions[spec['variant']][frame['id']]])
                    ranked=[p for p in ranked if p['confidence']>=spec['threshold']]
                    tp=sum(p['hit'] for p in ranked);fp=len(ranked)-tp
                    title=f'{labels[col-1]} | {tp} found, {fp} false, {len(frame["players"])-tp} missed'
                for p in ranked:dr.rectangle([v/s*z for v,s,z in zip(p['bbox'],[1920,1080,1920,1080],[640,360,640,360])],outline='#008cff' if spec is None else '#00df55' if p['hit'] else '#ff3a35',width=2)
                x=col*640;y=i*392;draw.text((x+4,y+7),title,font=font,fill='black');canvas.paste(im,(x,y+32))
        path=dest/f'fresh-detection-review-{sheet+1}.jpg';canvas.save(path,quality=90)
        files.append({'name':path.name,'sha256':sha(path),'bytes':path.stat().st_size})
    value={'files':files,'frames':[f['id'] for f in reference['frames']],'selection':'Every fixed audit frame, in time order.','columns':['Reference']+labels}
    dump(OUT/'review.json',value);volume.commit();return value


@app.local_entrypoint()
def main(task='run'):
    result=run.remote() if task=='run' else baseline.remote() if task=='baseline' else review.remote()
    (ROOT/'notes/ball-hockey/sam3-20261024'/('fresh-detection-'+task+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


@app.function(image=image,cpu=2,memory=4096,timeout=300,volumes={'/data':volume})
def baseline():
    """Additional comparison at YOLO's already-frozen detector threshold."""
    import datetime
    from nbhl_detection_metrics import summarize,average_precision
    from nbhl_vlm_reconnect import sha,dump
    volume.reload();reference=json.loads(Path('/work/sam3-fresh-detection-audit-reference-20261024.json').read_text())
    path=Path('/work/detector-heldout-freeze.json');freeze=json.loads(path.read_text())
    spec=next(r for r in freeze['models'] if r['name']=='selected_finetuned')
    assert freeze['frozen_at']<reference['sealed_at']
    meta=json.loads((BASE/'tracking-four-hour-v1/long-cache-trial/pittsburgh-runtime-2385-315-absolute-gpu-bounds/metadata.json').read_text())
    assert spec['weights_sha256']==meta['summary']['weights_sha256']
    rows=json.loads((OUT/'predictions.json').read_text())['yolo-pipeline']
    threshold=spec['confidence_threshold']
    report={'variant':'yolo-frozen-detector','threshold':threshold,'metrics':summarize(rows,threshold),'ap50':average_precision(rows),
            'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'baseline_freeze_sha256':sha(path),
            'threshold_frozen_at':freeze['frozen_at'],'weights_sha256':spec['weights_sha256'],
            'predictions_sha256':sha(OUT/'predictions.json'),'reference_sha256':sha('/work/sam3-fresh-detection-audit-reference-20261024.json'),
            'added_after_audit_results':True,'threshold_selected_using_this_audit':False,
            'reason':'The primary audit used the tracking proposal threshold0.1. Also show the detector confidence0.65 frozen on October18, avoiding an unfair comparison against only low-confidence proposals. No threshold search or new inference.',
            'reference_bias':'The new boxes were assisted by YOLO proposals; this comparison is not independently annotated.'}
    dump(OUT/'baseline-control.json',report);volume.commit();return report
