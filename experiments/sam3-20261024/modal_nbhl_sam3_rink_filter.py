"""Post-regression hypothesis: pretrained SAM3 surface masks + frozen fine-tune."""
from pathlib import Path
import json
import modal
from modal_campaign import ROOT,volume
from modal_nbhl_sam3_20261023 import image as sam_image,REV,CODE

BASE=Path('/data/ball-hockey/reliability-20261018')
FT=BASE/'sam3-finetune-20261024-v1'
OUT=BASE/'sam3-hybrid-20261024-v1/rink-filter'
app=modal.App('sport-scans-sam3-rink-filter-20261024')
image=sam_image
for name in ['nbhl_sam3_rink_filter.py','modal_nbhl_sam3_20261023.py']:
    image=image.add_local_file(ROOT/'scripts'/name,'/work/'+name)
image=image.add_local_file(ROOT/'data/ball-hockey/sam3-rink-filter-protocol-20261024.json','/work/rink-filter-protocol.json')
for source in ['b8CM4rZexaY','1f0Y5BIr5Ps','5Qw_g9E7EqE']:
    kind='canonical-expanded' if source=='b8CM4rZexaY' else 'heldout'
    image=image.add_local_file(ROOT/'data/ball-hockey/annotations'/kind/(source+'.json'),'/work/rink-annotations/'+source+'.json')


@app.function(image=image,cpu=8,memory=49152,gpu=['H100','H200','A100-80GB'],timeout=2400,scaledown_window=10,volumes={'/data':volume})
def run(phase='development'):
    import time,datetime,base64,zlib
    import numpy as np
    import torch
    from PIL import Image
    from huggingface_hub import hf_hub_download
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from nbhl_sam3_rink_filter import filter_row
    from nbhl_detection_metrics import summarize,average_precision
    from nbhl_vlm_reconnect import dump,sha
    volume.reload();torch.set_num_threads(4);started=time.monotonic()
    protocol=Path('/work/rink-filter-protocol.json');proto=json.loads(protocol.read_text())
    frozen=json.loads((FT/'selection.json').read_text());variant=frozen['best']['variant']
    threshold=frozen['best']['validation']['selected']['confidence_threshold']
    source=FT/variant/('best-validation-predictions.json' if phase=='development' else 'regression-predictions.json')
    predictions=json.loads(source.read_text())
    ann={f['id']:f for p in Path('/work/rink-annotations').glob('*.json') for f in json.loads(p.read_text())['frames']}
    settings=[{'key':'unchanged','prompt':None,'mask_threshold':None,'dilation':None}]
    for prompt in proto['prompts']:
        for t in proto['mask_thresholds']:
            for d in proto['dilation_image_height_fractions']:
                settings.append({'key':f'{prompt}-{t}-{d}','prompt':prompt,'mask_threshold':t,'dilation':d})
    if phase=='regression':
        selection=json.loads((OUT/'selection.json').read_text())
        assert selection['protocol_sha256']==sha(protocol) and selection['improves_validation_f1']
        settings=[settings[0],selection['best']['setting']]
    else:
        assert not (OUT/'selection.json').exists()
    masks={};diagnostics=[]
    checkpoint=hf_hub_download('facebook/sam3','sam3.pt',revision=REV,local_files_only=True)
    model=build_sam3_image_model(checkpoint_path=checkpoint,load_from_HF=False)
    processor=Sam3Processor(model,confidence_threshold=.29)
    prompts=sorted({s['prompt'] for s in settings if s['prompt']})
    target=OUT/phase;target.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for index,row in enumerate(predictions):
            frame=ann[row['id']];assert sha(frame['image'])==frame['sha256']
            state=processor.set_image(Image.open(frame['image']).convert('RGB'))
            for prompt in prompts:
                processor.reset_all_prompts(state);result=processor.set_text_prompt(state=state,prompt=prompt)
                scores=result['scores'].float().cpu().numpy()
                raw=result['masks'].detach().cpu().numpy().reshape(-1,frame['height'],frame['width'])
                for t in sorted({s['mask_threshold'] for s in settings if s['prompt']==prompt}):
                    chosen=raw[scores>=t]
                    union=np.any(chosen,axis=0) if len(chosen) else np.zeros((frame['height'],frame['width']),dtype=bool)
                    small=np.asarray(Image.fromarray(union).resize((252,252),Image.Resampling.NEAREST),dtype=bool)
                    key=row['id']+'|'+prompt+'|'+str(t);masks[key]=small
                    diagnostics.append({'id':row['id'],'prompt':prompt,'mask_threshold':t,'masks_above_threshold':len(chosen),'surface_fraction':float(small.mean()),'frame_sha256':frame['sha256']})
            if index%6==0:print(json.dumps({'phase':phase,'frames':index+1,'seconds':time.monotonic()-started}),flush=True)
    np.savez_compressed(target/'masks.npz',**masks)
    results=[];filtered={}
    for setting in settings:
        rows=[];details=[]
        for row in predictions:
            frame=ann[row['id']]
            if setting['prompt']:
                key=row['id']+'|'+setting['prompt']+'|'+str(setting['mask_threshold'])
                value,detail=filter_row(row,masks[key],frame['width'],frame['height'],setting['dilation'])
                details.append({'id':row['id'],**detail})
            else:value=row
            rows.append(value)
        report={'setting':setting,'metrics':summarize(rows,threshold),'ap50':average_precision(rows),'frames':details}
        results.append(report);filtered[setting['key']]=rows
    report={'phase':phase,'protocol_sha256':sha(protocol),'fine_tune_selection_sha256':sha(FT/'selection.json'),
            'input_predictions_sha256':sha(source),'results':results,'surface_diagnostics':diagnostics,
            'mask_file_sha256':sha(target/'masks.npz'),'seconds':time.monotonic()-started,
            'gpu':torch.cuda.get_device_name(0),'prior_regression_results_seen':True,'code_revision':CODE,'weights_revision':REV}
    dump(OUT/(phase+'.json'),report)
    if phase=='development':
        best=max(results,key=lambda r:(r['metrics']['f1'],r['metrics']['precision'] or 0,r['setting']['key']=='unchanged'))
        selection={'protocol_sha256':sha(protocol),'development_sha256':sha(OUT/'development.json'),
                   'frozen_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'best':best,
                   'improves_validation_f1':best['metrics']['f1']>results[0]['metrics']['f1'],
                   'prior_regression_results_seen':True,'filter_regression_scores_seen':False}
        dump(OUT/'selection.json',selection);report['selection']=selection
    else:
        best=settings[1];dump(OUT/'regression-filtered-predictions.json',filtered[best['key']])
        public=[]
        for row in predictions:
            frame=ann[row['id']];key=row['id']+'|'+best['prompt']+'|'+str(best['mask_threshold'])
            packed=base64.b64encode(zlib.compress(np.packbits(masks[key]).tobytes())).decode()
            public.append({'id':row['id'],'width':frame['width'],'height':frame['height'],'mask_shape':[252,252],
                           'packed_bits_zlib_base64':packed,'frame_sha256':frame['sha256']})
        dump(OUT/'regression-masks.json',public)
    volume.commit();return report


@app.function(image=image,cpu=4,memory=8192,timeout=600,volumes={'/data':volume})
def review():
    import numpy as np
    import base64,zlib
    from PIL import Image,ImageDraw,ImageFont
    from scipy.ndimage import maximum_filter
    from nbhl_detection_metrics import match
    from nbhl_vlm_reconnect import sha,dump
    import math
    volume.reload();selection=json.loads((OUT/'selection.json').read_text())
    setting=selection['best']['setting'];threshold=selection['best']['metrics']['confidence_threshold']
    variant=json.loads((FT/'selection.json').read_text())['best']['variant']
    before={r['id']:r for r in json.loads((FT/variant/'regression-predictions.json').read_text())}
    after={r['id']:r for r in json.loads((OUT/'regression-filtered-predictions.json').read_text())}
    masks=dict(np.load(OUT/'regression/masks.npz'))
    ann={f['id']:f for p in Path('/work/rink-annotations').glob('*.json') for f in json.loads(p.read_text())['frames']}
    for phase in ['development','regression']:
        packed=[]
        for key,mask in dict(np.load(OUT/phase/'masks.npz')).items():
            frame_id,prompt,t=key.split('|');frame=ann[frame_id]
            packed.append({'id':frame_id,'prompt':prompt,'mask_threshold':float(t),'width':frame['width'],'height':frame['height'],
                           'mask_shape':[252,252],'packed_bits_zlib_base64':base64.b64encode(zlib.compress(np.packbits(mask).tobytes())).decode(),
                           'frame_sha256':frame['sha256']})
        dump(OUT/(phase+'-masks.json'),packed)
    ids=sorted(before);assert len(ids)==36 and set(before)==set(after)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    files=[];dest=OUT/'public';dest.mkdir(parents=True,exist_ok=True)
    for sheet in range(6):
        canvas=Image.new('RGB',(1920,2352),'white');draw=ImageDraw.Draw(canvas)
        for row,key in enumerate(ids[sheet*6:(sheet+1)*6]):
            frame=ann[key];assert sha(frame['image'])==frame['sha256']
            picture=Image.open(frame['image']).convert('RGB').resize((640,360));w,h=frame['width'],frame['height']
            mask=masks[key+'|'+setting['prompt']+'|'+str(setting['mask_threshold'])]
            fallback=mask.mean()<.05 or mask.mean()>.98
            ry=math.ceil(setting['dilation']*252);rx=math.ceil(setting['dilation']*h/w*252)
            region=maximum_filter(mask.astype(np.uint8),size=(2*ry+1,2*rx+1)).astype(bool)
            before_ranked,_=match([before[key]]);after_ranked,_=match([after[key]])
            before_ranked=[p for p in before_ranked if p['confidence']>=threshold]
            after_ranked=[p for p in after_ranked if p['confidence']>=threshold]
            kept={tuple(p['bbox']) for p in after_ranked}
            for col in range(3):
                im=picture.copy();dr=ImageDraw.Draw(im)
                if col==1:
                    overlay=Image.new('RGB',im.size,'#00b9d4')
                    alpha=Image.fromarray((region*65).astype(np.uint8)).resize(im.size,Image.Resampling.NEAREST)
                    if not fallback:im=Image.composite(overlay,im,alpha)
                    title=f'{key} | '+('No valid mask: no filter' if fallback else 'Predicted rink + selected margin')
                else:
                    predicted=before_ranked if col==0 else after_ranked
                    for p in predicted:
                        dr.rectangle([v/s*z for v,s,z in zip(p['bbox'],[w,h,w,h],[640,360,640,360])],outline='#00df55' if p['hit'] else '#ff3a35',width=2)
                    if col==2:
                        for p in before_ranked:
                            if tuple(p['bbox']) not in kept:dr.rectangle([v/s*z for v,s,z in zip(p['bbox'],[w,h,w,h],[640,360,640,360])],outline='#ffab00',width=2)
                    tp=sum(p['hit'] for p in predicted);fp=len(predicted)-tp;fn=len(frame['players'])-tp
                    title=f'{"Fine-tuned" if col==0 else "With rink filter"} | {tp} found, {fp} false, {fn} missed'
                x=col*640;y=row*392;draw.text((x+4,y+7),title,font=font,fill='black');canvas.paste(im,(x,y+32))
        path=dest/f'rink-filter-review-{sheet+1}.jpg';canvas.save(path,quality=88)
        files.append({'name':path.name,'sha256':sha(path),'bytes':path.stat().st_size})
    report={'files':files,'frames':ids,'selection':'All36 regression frames in fixed order.','legend':'Left: fine-tune. Middle: predicted rink region and margin, cyan. Right: filtered output; orange boxes were removed, including any true players removed in error. Green/red follow the same IoU0.5 matching as detection metrics.','prior_regression_results_seen':True}
    dump(OUT/'review.json',report);volume.commit();return report


@app.local_entrypoint()
def main(phase='development',task='run'):
    result=review.remote() if task=='review' else run.remote(phase)
    if task=='review':phase='review'
    p=ROOT/'notes/ball-hockey/sam3-20261024'/('rink-filter-'+phase+'.json')
    p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='surface_diagnostics'},indent=2))
