"""Pinned multimodal embedding alternative to generative reconnection judgments."""
import json
from pathlib import Path
import modal
from modal_campaign import ROOT, base_image, volume

app=modal.App('sport-scans-nbhl-vlm-embeddings')
OUT=Path('/data/ball-hockey/reliability-20261018/vlm-reconnect-20261019-v1')
CODE='393e2978d27852b0d0230d6994f37f9c15bed73c'
MODELS={'2b':('Qwen/Qwen3-VL-Embedding-2B','9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda'),
        '8b':('Qwen/Qwen3-VL-Embedding-8B','2c4565515e0f265c6511776e7193b22c0968ddc7')}
INSTRUCTION='Represent the identity of the central ball-hockey player for matching the same player across camera views. Focus on visible jersey digits, individual clothing and equipment details. Ignore background, pose and camera position.'
image=(base_image.pip_install('transformers==4.57.6','qwen-vl-utils==0.0.14','accelerate==1.11.0','av==16.0.1')
    .run_commands('git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git /opt/qwen-embedding',
                  'git -C /opt/qwen-embedding checkout '+CODE)
    .env({'PYTHONPATH':'/opt/qwen-embedding:/opt/smp:/work','HF_HOME':'/data/cache/huggingface','HF_HUB_DISABLE_XET':'1','HF_XET_CACHE':'/tmp/hf-xet'})
    .add_local_file(ROOT/'scripts/modal_campaign.py','/work/modal_campaign.py')
    .add_local_file(ROOT/'data/ball-hockey/vlm-embedding-protocol-v1.json','/work/embedding-protocol.json'))


@app.function(image=image,cpu=4,memory=8192,timeout=1800,volumes={'/data':volume})
def cache(model:str):
    from huggingface_hub import snapshot_download
    name,revision=MODELS[model]
    path=snapshot_download(name,revision=revision)
    volume.commit()
    return {'model':name,'revision':revision,'path':path}


@app.function(image=image,cpu=8,memory=49152,gpu='L40S',timeout=7200,scaledown_window=10,volumes={'/data':volume})
def embed(model:str):
    import hashlib,time,subprocess
    import numpy as np
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
    volume.reload()
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert subprocess.check_output(['git','-C','/opt/qwen-embedding','rev-parse','HEAD'],text=True).strip()==CODE
    name,revision=MODELS[model]
    path=snapshot_download(name,revision=revision,local_files_only=True)
    start=time.monotonic()
    encoder=Qwen3VLEmbedder(path,max_length=8192,min_pixels=65536,max_pixels=262144,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa')
    loading=time.monotonic()-start;reports=[]
    for phase in ['development','regression']:
        mp=OUT/phase/'evidence.json';manifest=json.loads(mp.read_text())
        folder=OUT/phase/('embedding-'+model);folder.mkdir(parents=True,exist_ok=True)
        jobs=[];tracks=[]
        for clip in manifest['clips']:
            for tid,row in clip['tracks'].items():
                key=clip['clip']+'/'+tid
                tracks.append({'key':key,'clip':clip['clip'],'track_id':tid,'best_image':row['best_image'],'image_count':len(row['images'])})
                for i,im in enumerate(row['images']):jobs.append({'key':key+'/image/'+str(i),'images':[im]})
                jobs.append({'key':key+'/joint','images':row['images']})
        partial=folder/'partial.npz'
        vectors=dict(np.load(partial)) if partial.exists() else {}
        pending=[j for j in jobs if j['key'] not in vectors]
        start=time.monotonic()
        for offset in range(0,len(pending),4):
            batch=pending[offset:offset+4];inputs=[]
            for job in batch:
                for im in job['images']:assert sha(im['path'])==im['sha256']
                inputs.append({'image':[im['path'] for im in job['images']],'instruction':INSTRUCTION})
            conversations=[encoder.format_model_input(**item) for item in inputs]
            processed=encoder._preprocess_inputs(conversations)
            # Official preprocessing catches errors by returning a NULL text.
            # Never silently accept that fallback as an image representation.
            assert 'pixel_values' in processed and len(processed['image_grid_thw'])==sum(len(j['images']) for j in batch)
            assert processed['input_ids'].shape[0]==len(batch)
            processed={k:v.to(encoder.model.device) for k,v in processed.items()}
            with torch.inference_mode():
                answer=encoder.forward(processed)
                pooled=encoder._pooling_last(answer['last_hidden_state'],answer['attention_mask']).float()
                values=F.normalize(pooled,p=2,dim=-1).cpu().numpy()
            assert np.isfinite(values).all() and len(values)==len(batch)
            for job,value in zip(batch,values):vectors[job['key']]=value
            if offset%128==0 or offset+4>=len(pending):
                np.savez_compressed(partial,**vectors);volume.commit()
                print(json.dumps({'model':model,'phase':phase,'completed':len(vectors),'total':len(jobs)}),flush=True)
        output={}
        for track in tracks:
            key=track['key'];views=np.stack([vectors[key+'/image/'+str(i)] for i in range(track['image_count'])]);mean=views.mean(axis=0);mean/=np.linalg.norm(mean)
            output[key+'/single']=views[track['best_image']];output[key+'/mean']=mean;output[key+'/joint']=vectors[key+'/joint']
        target=folder/'features.npz';np.savez_compressed(target,**output)
        report={'model':name,'revision':revision,'code_revision':CODE,'phase':phase,'tracks':tracks,'requests':len(jobs),'seconds_this_run':time.monotonic()-start,'loading_seconds':loading,
            'evidence_sha256':sha(mp),'features_sha256':sha(target),'protocol_sha256':sha('/work/embedding-protocol.json'),'instruction':INSTRUCTION,'gpu':'L40S','labels_used_for_inference':False}
        (folder/'metadata.json').write_text(json.dumps(report,indent=2)+'\n');volume.commit();reports.append({k:v for k,v in report.items() if k!='tracks'})
    return reports


@app.local_entrypoint()
def main(model:str='2b'):
    cache.remote(model)
    result=embed.remote(model)
    (ROOT/f'notes/ball-hockey/vlm-embedding-run-{model}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
