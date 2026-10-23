"""Explicitly exploratory follow-up prompt, preserving the original comparison."""
import json
from pathlib import Path
import modal
from modal_campaign import ROOT,volume
from modal_nbhl_vlm_reconnect import prep_image,runtime_image

app=modal.App('sport-scans-nbhl-vlm-conservative')
BASE=Path('/data/ball-hockey/reliability-20261018/vlm-reconnect-20261019-v1')
PROTOCOL=ROOT/'data/ball-hockey/vlm-conservative-protocol-v1.json'
runtime=runtime_image.add_local_file(ROOT/'scripts/modal_nbhl_vlm_reconnect.py','/work/modal_nbhl_vlm_reconnect.py').add_local_file(PROTOCOL,'/work/conservative-protocol.json')
scoring=prep_image.add_local_file(ROOT/'scripts/modal_nbhl_vlm_reconnect.py','/work/modal_nbhl_vlm_reconnect.py').add_local_file(PROTOCOL,'/work/conservative-protocol.json')

@app.function(image=runtime,cpu=12,memory=65536,gpu='H200',timeout=3600,scaledown_window=10,volumes={'/data':volume})
def infer():
    import re,time
    from PIL import Image
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor
    from vllm import LLM,SamplingParams
    from nbhl_vlm_reconnect import requests,PAIR_PROMPT,sha,dump
    volume.reload();protocol=json.loads(Path('/work/conservative-protocol.json').read_text());spec=protocol['model']
    start=time.monotonic();path=snapshot_download(spec['model'],revision=spec['revision'],local_files_only=True)
    processor=AutoProcessor.from_pretrained(path)
    llm=LLM(model=path,tensor_parallel_size=1,max_model_len=8192,gpu_memory_utilization=.92,max_num_seqs=8,limit_mm_per_prompt={'image':6,'video':0},enforce_eager=True,mm_processor_kwargs={'min_pixels':65536,'max_pixels':262144},seed=17)
    loading=time.monotonic()-start;reports=[]
    for phase in ['development','regression']:
        manifest_path=BASE/phase/'evidence.json';manifest=json.loads(manifest_path.read_text())
        assert manifest['protocol_sha256']==sha('/work/vlm-protocol.json')
        jobs=requests(manifest,['multi','reverse'])
        for job in jobs:
            assert job['prompt'].startswith(PAIR_PROMPT)
            job['prompt']=protocol['prompt']+job['prompt'][len(PAIR_PROMPT):]
        target=BASE/phase/'8b-conservative/predictions.json';assert not target.exists(),'Do not overwrite an existing run'
        result={'model':spec,'phase':phase,'protocol_sha256':sha('/work/conservative-protocol.json'),'evidence_sha256':sha(manifest_path),'loading_seconds':loading,'results':[]}
        for offset in range(0,len(jobs),8):
            batch=jobs[offset:offset+8];inputs=[]
            for job in batch:
                images=[]
                for im in job['images']:
                    assert sha(im['path'])==im['sha256'];images.append(Image.open(im['path']).convert('RGB'))
                content=[{'type':'text','text':job['prompt']}]+[{'type':'image','image':im} for im in images]
                inputs.append({'prompt':processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True),'multi_modal_data':{'image':images}})
            start=time.monotonic();answers=llm.generate(inputs,sampling_params=SamplingParams(temperature=0,max_tokens=320),use_tqdm=False);elapsed=time.monotonic()-start
            for job,answer in zip(batch,answers):
                value=answer.outputs[0].text;parsed=None
                try:parsed=json.loads(re.search(r'\{.*\}',value,re.S).group())
                except (ValueError,AttributeError):pass
                result['results'].append({k:v for k,v in job.items() if k not in ['images','prompt']}|{'output':value,'parsed':parsed,'seconds':elapsed/len(batch),'finish_reason':answer.outputs[0].finish_reason})
            if offset%64==0 or offset+8>=len(jobs):
                dump(target,result);volume.commit();print(json.dumps({'phase':phase,'completed':len(result['results']),'total':len(jobs)}),flush=True)
        reports.append({'phase':phase,'requests':len(jobs),'seconds':sum(r['seconds'] for r in result['results']),'prediction_sha256':sha(target)})
    return reports

@app.function(image=scoring,cpu=4,memory=16384,timeout=1800,volumes={'/data':volume})
def evaluate(phase:str):
    from collections import Counter
    from nbhl_vlm_reconnect import sha,dump,datasets,apply_policy,requests,decision
    from nbhl_tracking_fourhour import tracklet_evidence,link_candidates,evaluate as count
    from nbhl_tracking_reference_audit import corrected_report
    volume.reload();protocol=json.loads(Path('/work/conservative-protocol.json').read_text())
    folder=BASE/phase/'8b-conservative';pred=json.loads((folder/'predictions.json').read_text());manifest=json.loads((BASE/phase/'evidence.json').read_text())
    assert pred['protocol_sha256']==sha('/work/conservative-protocol.json') and pred['evidence_sha256']==sha(BASE/phase/'evidence.json')
    responses={r['id']:r for r in pred['results']};assert set(responses)=={r['id'] for r in requests(manifest,['multi','reverse'])}
    corrections={r['pair_id']:r for r in json.loads(Path('/work/tracking-four-hour-reference-corrections-v1.json').read_text())['corrections']}
    rows=[]
    for meta,features,raw,_,raw_sha in datasets(phase,labels=True):
        clip=next(c for c in manifest['clips'] if c['clip']==raw['clip']);assert raw_sha==clip['raw_tracks_sha256']
        evidence=tracklet_evidence(meta,features,raw);edges=link_candidates(meta,features,evidence)
        baseline=json.loads((BASE/phase/'8b/tracks/osnet_full'/(raw['clip']+'.json')).read_text())
        for policy in protocol['policies']:
            linked=apply_policy(raw,evidence,edges,clip,responses,policy)
            if policy=='osnet_full':assert linked['frames']==baseline['frames']
            report=count(meta,linked)
            if phase=='development':report=corrected_report(report,corrections)
            rows.append(report|{'policy':policy,'links':len(linked['links'])})
            dump(folder/'tracks'/policy/(raw['clip']+'.json'),linked)
    totals=[]
    for policy in protocol['policies']:
        for scope in (['all'] if phase=='development' else ['short','stress']):
            group=[r for r in rows if r['policy']==policy and (scope=='all' or ('stress' in r['clip'])==(scope=='stress'))]
            totals.append({'policy':policy,'scope':scope,'counts':{k:sum(r['counts'][k] for r in group) for k in group[0]['counts']}})
    order=Counter(decision(responses[e['id']+'/multi'])+' / '+decision(responses[e['id']+'/reverse']) for c in manifest['clips'] for e in c['selected_edges'])
    result={'phase':phase,'totals':totals,'clips':rows,'order_consistency':dict(order),'prediction_sha256':sha(folder/'predictions.json'),'protocol_sha256':pred['protocol_sha256'],'requests':len(pred['results']),'generation_seconds':sum(r['seconds'] for r in pred['results']),'invalid_json':sum(not isinstance(r['parsed'],dict) for r in pred['results']),'limits':protocol['limits']}
    dump(folder/'evaluation.json',result);volume.commit();return {k:v for k,v in result.items() if k!='clips'}

@app.local_entrypoint()
def main():
    import datetime,hashlib
    run=infer.remote();(ROOT/'notes/ball-hockey/vlm-conservative-inference.json').write_text(json.dumps(run,indent=2)+'\n')
    development=evaluate.remote('development')
    target=ROOT/'notes/ball-hockey/vlm-conservative-development.json';target.write_text(json.dumps(development,indent=2)+'\n')
    def objective(r):
        c=r['counts'];return c['correct']-10*c['wrong'],-c['unknown_same'],r['policy']=='osnet_full'
    selected=max(development['totals'],key=objective)
    selection={'frozen_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selected':selected,'development_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'first_round_regression_seen_before_prompt_design':True,'revised_prompt_regression_scores_seen':False}
    selection_path=ROOT/'data/ball-hockey/vlm-conservative-selection-v1.json'
    with selection_path.open('x') as stream:stream.write(json.dumps(selection,indent=2)+'\n')
    regression=evaluate.remote('regression')
    (ROOT/'notes/ball-hockey/vlm-conservative-regression.json').write_text(json.dumps(regression,indent=2)+'\n')
    print(json.dumps({'selected':selected,'development':development['totals'],'regression':regression['totals']}))
