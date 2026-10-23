"""Run bounded Qwen VLM reconnection experiments on cached NBHL tracks."""
from pathlib import Path
import modal
from modal_campaign import ROOT,base_image,volume
app=modal.App('sport-scans-nbhl-vlm-reconnect')
OUT=Path('/data/ball-hockey/reliability-20261018/vlm-reconnect-20261019-v1')
prep_image=base_image
for name in ['modal_campaign.py','nbhl_vlm_reconnect.py','nbhl_tracking_fourhour.py','nbhl_tracking_features_v2.py','nbhl_tracking_global_links.py','nbhl_detection_metrics.py','nbhl_tracking_reference_audit.py']:
    prep_image=prep_image.add_local_file(ROOT/'scripts'/name,'/work/'+name)
prep_image=prep_image.add_local_file(ROOT/'data/ball-hockey/vlm-reconnect-protocol-v1.json','/work/vlm-protocol.json')
prep_image=prep_image.add_local_file(ROOT/'data/ball-hockey/tracking-four-hour-reference-corrections-v1.json','/work/tracking-four-hour-reference-corrections-v1.json')
runtime_image=(modal.Image.debian_slim(python_version='3.12').apt_install('libgl1','libglib2.0-0')
    .pip_install('vllm==0.11.1','huggingface_hub==0.36.0','pillow')
    .env({'HF_HOME':'/data/cache/huggingface','HF_XET_CACHE':'/tmp/hf-xet','HF_HUB_DISABLE_XET':'1','VLLM_WORKER_MULTIPROC_METHOD':'spawn','TOKENIZERS_PARALLELISM':'false','PYTHONPATH':'/work'}))
# The runtime imports this module too; keep the Modal image declaration available.
for name in ['modal_campaign.py','nbhl_vlm_reconnect.py']:
    runtime_image=runtime_image.add_local_file(ROOT/'scripts'/name,'/work/'+name)
runtime_image=runtime_image.add_local_file(ROOT/'data/ball-hockey/vlm-reconnect-protocol-v1.json','/work/vlm-protocol.json')

@app.function(image=prep_image,cpu=4,memory=16384,timeout=1800,volumes={'/data':volume})
def prepare(phase:str):
    from nbhl_vlm_reconnect import prepare,sha,dump
    import json
    volume.reload();protocol=json.loads(Path('/work/vlm-protocol.json').read_text());dump(OUT/'protocol.json',protocol)
    result=prepare(phase,sha('/work/vlm-protocol.json'));volume.commit();return result

@app.function(image=prep_image,cpu=1,memory=2048,timeout=180,volumes={'/data':volume})
def probe():
    import json
    from huggingface_hub import HfApi,snapshot_download
    volume.reload();rows=[]
    for model in ['Qwen/Qwen3-VL-8B-Instruct','Qwen/Qwen3-VL-235B-A22B-Instruct-FP8']:
        info=HfApi().model_info(model);row={'model':model,'revision':info.sha}
        try:
            p=Path(snapshot_download(model,revision=info.sha,local_files_only=True));row['path']=str(p);row['files']=len(list(p.glob('*')));row['weight_bytes']=sum(f.stat().st_size for f in p.glob('*.safetensors'))
        except Exception as e:row['cached']=False;row['error_type']=type(e).__name__
        rows.append(row)
    return rows

@app.function(image=prep_image,cpu=4,memory=16384,timeout=1800,volumes={'/data':volume})
def evaluate(phase:str,model_key:str):
    from nbhl_vlm_reconnect import score
    volume.reload();result=score(phase,model_key);volume.commit();return result

@app.function(image=runtime_image,cpu=16,memory=131072,gpu='H200:2',timeout=10800,max_containers=1,scaledown_window=10,volumes={'/data':volume})
def review(phase:str,model_key:str,tasks:str='number,single,multi,reverse'):
    import json,re,time
    from PIL import Image
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor
    from vllm import LLM,SamplingParams
    from nbhl_vlm_reconnect import requests,sha,dump
    volume.reload();protocol=json.loads(Path('/work/vlm-protocol.json').read_text());spec=protocol['models'][model_key]
    manifest_path=OUT/phase/'evidence.json';manifest=json.loads(manifest_path.read_text());assert manifest['protocol_sha256']==sha('/work/vlm-protocol.json')
    all_requests=requests(manifest,tasks.split(','));target=OUT/phase/model_key/'predictions.json'
    result=json.loads(target.read_text()) if target.exists() else {'model':spec,'phase':phase,'protocol_sha256':sha('/work/vlm-protocol.json'),'evidence_sha256':sha(manifest_path),'results':[]}
    assert result['protocol_sha256']==sha('/work/vlm-protocol.json') and result['evidence_sha256']==sha(manifest_path)
    done={r['id'] for r in result['results']};pending=[r for r in all_requests if r['id'] not in done]
    if not pending:return {'requests':len(result['results']),'cache_complete':True}
    start=time.monotonic();path=snapshot_download(spec['model'],revision=spec['revision'],local_files_only=True);processor=AutoProcessor.from_pretrained(path)
    llm=LLM(model=path,tensor_parallel_size=spec['tensor_parallel'],max_model_len=8192,gpu_memory_utilization=.92,max_num_seqs=8,limit_mm_per_prompt={'image':6,'video':0},enforce_eager=True,mm_processor_kwargs={'min_pixels':64*32*32,'max_pixels':256*32*32},seed=17)
    result['loading_seconds']=time.monotonic()-start
    for offset in range(0,len(pending),8):
        batch=pending[offset:offset+8];inputs=[]
        for row in batch:
            images=[]
            for im in row['images']:
                assert sha(im['path'])==im['sha256'];images.append(Image.open(im['path']).convert('RGB'))
            content=[{'type':'text','text':row['prompt']}]+[{'type':'image','image':im} for im in images]
            inputs.append({'prompt':processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True),'multi_modal_data':{'image':images}})
        start=time.monotonic();answers=llm.generate(inputs,sampling_params=SamplingParams(temperature=0,max_tokens=320),use_tqdm=False);elapsed=time.monotonic()-start
        for row,answer in zip(batch,answers):
            value=answer.outputs[0].text;parsed=None
            try:parsed=json.loads(re.search(r'\{.*\}',value,re.S).group())
            except (ValueError,AttributeError):pass
            result['results'].append({k:v for k,v in row.items() if k not in ['images','prompt']}|{'output':value,'parsed':parsed,'seconds':elapsed/len(batch),'input_tokens':len(answer.prompt_token_ids),'output_tokens':len(answer.outputs[0].token_ids),'finish_reason':answer.outputs[0].finish_reason})
        dump(target,result);volume.commit();print(json.dumps({'model':model_key,'phase':phase,'completed':len(result['results']),'total':len(all_requests),'seconds':elapsed}),flush=True)
    return {'requests':len(result['results']),'seconds':sum(r['seconds'] for r in result['results']),'prediction_sha256':sha(target)}

@app.local_entrypoint()
def main(task:str='probe',phase:str='development',model:str='235b',tasks:str='number,single,multi,reverse'):
    import json
    if task=='probe':result=probe.remote()
    elif task=='prepare':result=prepare.remote(phase)
    elif task=='review':result=review.with_options(gpu='H200' if model=='8b' else 'H200:2').remote(phase,model,tasks)
    elif task=='score':result=evaluate.remote(phase,model)
    else:raise ValueError(task)
    path=ROOT/'notes/ball-hockey'/f'vlm-reconnect-{task}-{phase}-{model}.json';path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
