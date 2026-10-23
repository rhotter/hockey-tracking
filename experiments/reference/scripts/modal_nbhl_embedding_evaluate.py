"""Development-selected evaluation of generic multimodal player embeddings."""
import json
from pathlib import Path
import modal
from modal_campaign import ROOT, volume
from modal_nbhl_vlm_reconnect import prep_image

app=modal.App('sport-scans-nbhl-embedding-evaluation')
image=prep_image.add_local_file(ROOT/'scripts/modal_nbhl_vlm_reconnect.py','/work/modal_nbhl_vlm_reconnect.py')
image=image.add_local_file(ROOT/'data/ball-hockey/vlm-embedding-protocol-v1.json','/work/embedding-protocol.json')
image=image.add_local_file(ROOT/'data/ball-hockey/vlm-embedding-selection-amendment-v1.json','/work/embedding-selection-amendment.json')


def linked(raw,evidence,edges,selected,vectors,clip,setting):
    import numpy as np
    from nbhl_vlm_reconnect import SETTING
    from nbhl_tracking_global_links import global_links
    if setting['policy']=='baseline':return global_links(raw,evidence,edges,SETTING)
    output=[];representation=setting['representation'];threshold=setting['threshold']
    for edge in edges:
        key=(edge['a'],edge['b']);cosine=None
        if key in selected:
            cosine=float(np.dot(vectors[clip+'/'+key[0]+'/'+representation],vectors[clip+'/'+key[1]+'/'+representation]))
        if setting['policy']=='veto':
            if cosine is not None and cosine<threshold:continue
            score=edge['cosine_mean']
        else:
            if cosine is None:continue
            score=cosine if setting['policy']=='replace' else .5*(cosine+edge['cosine_mean'])
        output.append(edge|{'cosine_embedding':score})
    cutoff=SETTING['cosine'] if setting['policy']=='veto' else threshold
    return global_links(raw,evidence,output,SETTING|{'pooling':'embedding','cosine':cutoff})


def evaluate_setting(cache,setting,full=False,save=None):
    from nbhl_tracking_fourhour import evaluate
    from nbhl_vlm_reconnect import dump
    from nbhl_tracking_reference_audit import corrected_report
    reports=[]
    corrections={r['pair_id']:r for r in json.loads(Path('/work/tracking-four-hour-reference-corrections-v1.json').read_text())['corrections']}
    for row in cache:
        meta,raw,evidence,edges,selected,vectors=row
        if not full:
            ref_frames={o['frame'] for o in meta['observations']}
            raw=raw|{'frames':[r for r in raw['frames'] if r['frame'] in ref_frames]}
        result=linked(raw,evidence,edges,selected,vectors,raw['clip'],setting)
        report=evaluate(meta,result)
        if meta['summary']['source_id'] in ['plDp2kPtk_g','X83xW6iQKoc']:
            report=corrected_report(report,corrections)
        report['links']=len(result['links']);reports.append(report)
        if save:dump(save/(raw['clip']+'.json'),result)
    totals={k:sum(r['counts'][k] for r in reports) for k in reports[0]['counts']}
    return {'setting':setting,'counts':totals,'links':sum(r['links'] for r in reports),'clips':reports}


def load(phase,model):
    import numpy as np
    from nbhl_vlm_reconnect import OUT,datasets,sha
    from nbhl_tracking_fourhour import tracklet_evidence,link_candidates
    folder=OUT/phase/('embedding-'+model)
    metadata=json.loads((folder/'metadata.json').read_text())
    assert metadata['features_sha256']==sha(folder/'features.npz')
    assert metadata['evidence_sha256']==sha(OUT/phase/'evidence.json')
    assert metadata['protocol_sha256']==sha('/work/embedding-protocol.json')
    vectors=dict(np.load(folder/'features.npz'))
    manifest=json.loads((OUT/phase/'evidence.json').read_text())
    cache=[]
    for meta,features,raw,_,raw_sha in datasets(phase,labels=True):
        clip=next(c for c in manifest['clips'] if c['clip']==raw['clip'])
        assert raw_sha==clip['raw_tracks_sha256']
        evidence=tracklet_evidence(meta,features,raw);edges=link_candidates(meta,features,evidence)
        selected={(e['a'],e['b']) for e in clip['selected_edges']}
        cache.append((meta,raw,evidence,edges,selected,vectors))
    return cache,metadata


def objective(row):
    c=row['counts']
    return c['correct']-10*c['wrong'],-c['unknown_same'],row['setting']['policy']=='baseline'


@app.function(image=image,cpu=4,memory=16384,timeout=3600,volumes={'/data':volume})
def develop(model:str):
    from nbhl_vlm_reconnect import OUT,dump,sha
    volume.reload();protocol=json.loads(Path('/work/embedding-protocol.json').read_text())
    cache,metadata=load('development',model)
    baseline=evaluate_setting(cache,{'policy':'baseline'},full=True)
    rows=[baseline];chosen=[]
    folder=OUT/'embedding-evaluation'/model
    for representation in protocol['representations']:
        candidates=[]
        for policy in protocol['policies']:
            for threshold in protocol['thresholds']:
                setting={'policy':policy,'representation':representation,'threshold':threshold}
                result=evaluate_setting(cache,setting);candidates.append(result)
                rows.append({k:v for k,v in result.items() if k!='clips'})
        best=max(candidates,key=objective)
        # Full output must exactly reproduce the sparse-grid reference counts.
        verified=evaluate_setting(cache,best['setting'],full=True,save=folder/'development'/representation)
        assert verified['counts']==best['counts']
        verified['representation']=representation;verified['model']=model;chosen.append(verified)
        print(json.dumps({'model':model,'representation':representation,'setting':best['setting'],'counts':best['counts']}),flush=True)
    result={'model':model,'selected':chosen,'baseline':baseline,'all_settings':rows,'metadata':{k:v for k,v in metadata.items() if k!='tracks'},'protocol_sha256':sha('/work/embedding-protocol.json'),'selection_amendment_sha256':sha('/work/embedding-selection-amendment.json')}
    dump(folder/'development.json',result);volume.commit()
    return {k:v for k,v in result.items() if k not in ['baseline','selected']}|{'baseline':{k:v for k,v in baseline.items() if k!='clips'},'selected':[{k:v for k,v in r.items() if k!='clips'} for r in chosen]}


@app.function(image=image,cpu=1,memory=2048,timeout=180,volumes={'/data':volume})
def select():
    import datetime
    from nbhl_vlm_reconnect import OUT,dump,sha
    volume.reload();folder=OUT/'embedding-evaluation';candidates=[];sources=[]
    assert not any(folder.glob('*/regression.json'))
    for model in ['2b','8b']:
        path=folder/model/'development.json';report=json.loads(path.read_text());sources.append({'model':model,'sha256':sha(path)})
        assert report['selection_amendment_sha256']==sha('/work/embedding-selection-amendment.json')
        if not candidates:candidates.append({k:v for k,v in report['baseline'].items() if k!='clips'}|{'model':'none','representation':'baseline'})
        for row in report['selected']:candidates.append({k:v for k,v in row.items() if k!='clips'})
    best=max(candidates,key=objective)
    result={'frozen_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'best':best,'candidates':candidates,'development_reports':sources,'protocol_sha256':sha('/work/embedding-protocol.json'),'selection_amendment_sha256':sha('/work/embedding-selection-amendment.json'),'regression_scores_seen':False}
    target=folder/'selection.json';assert not target.exists(),'Selection already sealed'
    dump(target,result);volume.commit();return result


@app.function(image=image,cpu=4,memory=16384,timeout=1800,volumes={'/data':volume})
def regression(model:str):
    from nbhl_vlm_reconnect import OUT,dump,sha
    volume.reload();folder=OUT/'embedding-evaluation';selection=json.loads((folder/'selection.json').read_text())
    assert selection['protocol_sha256']==sha('/work/embedding-protocol.json')
    assert selection['selection_amendment_sha256']==sha('/work/embedding-selection-amendment.json')
    cache,metadata=load('regression',model);rows=[]
    candidates=[{'setting':{'policy':'baseline'},'representation':'baseline'}]+[r for r in selection['candidates'] if r['model']==model]
    for candidate in candidates:
        row=evaluate_setting(cache,candidate['setting'],full=True,save=folder/model/'regression'/candidate['representation'])
        row.update(model=model,representation=candidate['representation'])
        row['scopes']=[]
        for scope in ['short','stress']:
            clips=[c for c in row['clips'] if ('stress' in c['clip'])==(scope=='stress')]
            row['scopes'].append({'scope':scope,'counts':{k:sum(r['counts'][k] for r in clips) for k in clips[0]['counts']}})
        rows.append(row)
    result={'model':model,'results':rows,'selection_sha256':sha(folder/'selection.json'),'metadata':{k:v for k,v in metadata.items() if k!='tracks'}}
    dump(folder/model/'regression.json',result);volume.commit()
    return result|{'results':[{k:v for k,v in r.items() if k!='clips'} for r in rows]}


@app.local_entrypoint()
def main(task:str='develop',model:str='2b'):
    if task=='develop':result=develop.remote(model)
    elif task=='select':result=select.remote()
    elif task=='regression':result=regression.remote(model)
    else:raise ValueError(task)
    dest=ROOT/('data/ball-hockey/vlm-embedding-selection-v1.json' if task=='select' else f'notes/ball-hockey/vlm-embedding-{task}-{model}.json')
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'task':task,'model':model,'saved':str(dest)}))
