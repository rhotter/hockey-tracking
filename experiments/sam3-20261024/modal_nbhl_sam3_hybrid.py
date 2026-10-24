"""Reproducible SAM3/YOLO/appearance experiments, 2026-10-24."""
from pathlib import Path
import json
import modal
from modal_campaign import ROOT,base_image,volume
BASE=Path('/data/ball-hockey/reliability-20261018')
OLD=BASE/'sam3-20261023-v1'
OUT=BASE/'sam3-hybrid-20261024-v1'
app=modal.App('sport-scans-sam3-hybrid-20261024')
image=base_image
for name in ['modal_campaign.py','nbhl_vlm_reconnect.py','nbhl_tracking_fourhour.py','nbhl_tracking_features_v2.py','nbhl_tracking_global_links.py','nbhl_detection_metrics.py','nbhl_tracking_reference_audit.py','nbhl_sam3_hybrid.py']:
    image=image.add_local_file(ROOT/'scripts'/name,'/work/'+name)
image=image.add_local_file(ROOT/'data/ball-hockey/sam3-hybrid-protocol-20261024.json','/work/hybrid-protocol.json')
image=image.add_local_file(ROOT/'data/ball-hockey/tracking-four-hour-reference-corrections-v1.json','/work/tracking-four-hour-reference-corrections-v1.json')


@app.function(image=image,cpu=8,memory=24576,timeout=7200,scaledown_window=10,volumes={'/data':volume})
def score(phase:str='development',clips:str=''):
    import time,numpy as np
    from nbhl_vlm_reconnect import datasets,SETTING,dump,sha
    from nbhl_tracking_fourhour import evaluate,tracklet_evidence,link_candidates
    from nbhl_tracking_global_links import global_links
    from nbhl_tracking_reference_audit import corrected_report
    from nbhl_sam3_hybrid import anchor,split_appearance,endpoint_votes,consensus_edges
    volume.reload();started=time.monotonic()
    protocol=json.loads(Path('/work/hybrid-protocol.json').read_text())
    corrections={r['pair_id']:r for r in json.loads(Path('/work/tracking-four-hour-reference-corrections-v1.json').read_text())['corrections']}
    wanted=clips.split(',') if clips else protocol[phase]
    selection=None
    if phase=='regression':
        selection=json.loads((OUT/'selection.json').read_text())
        assert selection['protocol_sha256']==sha('/work/hybrid-protocol.json')
    results={};specs={};inputs=[]
    qwen_root=BASE/'vlm-reconnect-20261019-v1'/phase
    qwen_vectors=dict(np.load(qwen_root/'embedding-2b/features.npz'))
    qwen_manifest=json.loads((qwen_root/'evidence.json').read_text())
    for meta,features,base,_,base_sha in datasets(phase,labels=True):
        clip=meta['summary']['clip']
        if clip not in wanted:continue
        source=OLD/'tracking'/phase/'person'/clip/'tracks.json'
        if not source.exists():source=OUT/'native-settings'/phase/'default-offload'/clip/'tracks.json'
        assert source.exists(),f'Native SAM3 not ready: {clip}'
        native=json.loads(source.read_text())
        inputs.append({'clip':clip,'sam3_sha256':sha(source),'sam2_sha256':base_sha,'features_sha256':meta['summary']['feature_sha256']})
        def record(key,value,setting,family):
            report=evaluate(meta,value)
            if phase=='development':report=corrected_report(report,corrections)
            target=OUT/phase/key/clip
            dump(target/'tracks.json',value);dump(target/'metrics.json',report)
            results.setdefault(key,[]).append({k:v for k,v in report.items() if k!='pairs'})
            specs[key]={'family':family,'setting':setting}
        record('sam3-native',native,{},'baseline')
        record('sam2-raw',base,{},'baseline')
        old_best=BASE/'vlm-reconnect-20261019-v1/embedding-evaluation/2b'/phase/'joint'/(clip+'.json')
        if old_best.exists():record('sam2-qwen-joint',json.loads(old_best.read_text()),{},'baseline')
        for cutoff in [.3,.5]:
            anchored=anchor(meta,native,cutoff)
            if phase=='development' or f'anchor-{cutoff}' in selection['selected_keys']:
                record(f'anchor-{cutoff}',anchored,{'iou':cutoff},'anchor')
            for split in [0.,.45,.6,.75]:
                raw=split_appearance(meta,features,anchored,split)
                evidence=tracklet_evidence(meta,features,raw);edges=link_candidates(meta,features,evidence)
                for threshold in [None,.75,.85,.92]:
                    key=f'appearance-{cutoff}-{split}-{threshold}'
                    setting={'iou':cutoff,'split':split,'cosine':threshold}
                    if phase=='regression' and key not in selection['selected_keys']:continue
                    linked=global_links(raw,evidence,edges,SETTING|{'cosine':threshold}) if threshold is not None else raw
                    record(key,linked,setting,'appearance')
        evidence=tracklet_evidence(meta,features,base);edges=link_candidates(meta,features,evidence)
        votes=endpoint_votes(base,native)
        selected={(e['a'],e['b']) for c in qwen_manifest['clips'] if c['clip']==clip for e in c['selected_edges']}
        qwen_edges=[]
        for edge in edges:
            if (edge['a'],edge['b']) not in selected:continue
            cosine=float(qwen_vectors[clip+'/'+edge['a']+'/joint']@qwen_vectors[clip+'/'+edge['b']+'/joint'])
            qwen_edges.append({**edge,'cosine_mean':.5*(edge['cosine_mean']+cosine),'osnet_cosine':edge['cosine_mean'],'qwen_cosine':cosine})
        for mode,bonus in [('bonus',.025),('bonus',.05),('bonus',.10),('require',0.),('veto',0.)]:
            for family,source in [('consensus',edges),('consensus-qwen',qwen_edges)]:
                key=f'{family}-{mode}-{bonus}'
                if phase=='regression' and key not in selection['selected_keys']:continue
                adjusted=consensus_edges(source,votes,mode,bonus)
                linked=global_links(base,evidence,adjusted,SETTING|{'pooling':'consensus'})
                record(key,linked,{'mode':mode,'bonus':bonus},family)
        volume.commit()
        print(json.dumps({'phase':phase,'clip':clip,'settings':len(results),'seconds':time.monotonic()-started}),flush=True)
    assert {r['clip'] for r in inputs}==set(wanted)
    summary=[]
    for key,rows in results.items():
        counts={k:sum(r['counts'][k] for r in rows) for k in rows[0]['counts']}
        summary.append({'key':key,**specs[key],'counts':counts,'clips':rows})
    report={'phase':phase,'protocol_sha256':sha('/work/hybrid-protocol.json'),'inputs':inputs,'results':summary,'seconds':time.monotonic()-started}
    dest=OUT/(phase+('-subset' if clips else '')+'.json');dump(dest,report);volume.commit()
    return report


@app.function(image=image,cpu=1,memory=2048,timeout=180,volumes={'/data':volume})
def select():
    import datetime
    from nbhl_vlm_reconnect import dump,sha
    volume.reload();path=OUT/'development.json';report=json.loads(path.read_text())
    assert not (OUT/'selection.json').exists(),'Selection is immutable'
    assert not (OUT/'regression.json').exists()
    def objective(row):
        c=row['counts'];return c['correct']-10*c['wrong'],c['matched_observations'],-c['unknown_same']
    winners=[max([r for r in report['results'] if r['family']==family],key=objective) for family in ['anchor','appearance','consensus','consensus-qwen']]
    result={'frozen_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':report['protocol_sha256'],
            'development_sha256':sha(path),'selected_keys':[r['key'] for r in winners],'winners':winners,'new_regression_scores_seen':False}
    dump(OUT/'selection.json',result);volume.commit();return result


@app.local_entrypoint()
def main(task:str='score',phase:str='development',clips:str=''):
    if task=='score':result=score.remote(phase,clips)
    elif task=='select':result=select.remote()
    else:raise ValueError(task)
    target=ROOT/'notes/ball-hockey'/f'sam3-hybrid-{task}-{phase}{"-subset" if clips else ""}-20261024.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    if task=='score':print(json.dumps([{k:v for k,v in r.items() if k!='clips'} for r in result['results']],indent=2))
    else:print(json.dumps(result,indent=2))
