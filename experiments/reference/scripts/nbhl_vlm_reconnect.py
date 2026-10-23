"""Label-free VLM evidence and association policies for NBHL track fragments."""
import hashlib
import json
from pathlib import Path

BASE = Path('/data/ball-hockey/reliability-20261018')
OLD = BASE/'tracking-four-hour-v1'
OUT = BASE/'vlm-reconnect-20261019-v1'
SETTING = {'solver':'global_assignment','pooling':'mean','cosine':.75,'margin':0.,'gap':300,'gap_penalty':.001,'motion_base':1.,'motion_per_frame':.12,'min_support':10}
PAIR_PROMPT = '''Compare the central NBHL ball-hockey player in group A with the central player in group B. The groups are separate automatic track fragments and may show the same player or different players. Images within a group can also contain a tracking mistake. Compare visible jersey digits, gloves, helmet or hair, shoes, socks, pads and other distinguishing details. Teammates wear nearly identical uniforms; matching team colors or a similar pose alone is insufficient. Do not guess hidden digits or identify a person by name. If evidence cannot distinguish the players, return uncertain. A different viewpoint, lighting or scale alone is not proof of different identity. Return only JSON: {"decision":"same|different|uncertain","a_consistent":"yes|no|uncertain","b_consistent":"yes|no|uncertain","evidence_type":"jersey_number|distinctive_details|team_difference|uniform_only|insufficient","evidence":"one short description of directly visible matching or conflicting details"}.'''
NUMBER_PROMPT = '''These images show the central player from one automatic NBHL ball-hockey track fragment. The tracker can switch people. For each image read ONLY a complete, visibly distinguishable jersey number on that central player's shirt, sleeve or shorts. Ignore other people, words, logos and background numbers. Return null for hidden, partial, blurred or ambiguous digits; do not infer a missing digit and do not use a roster. Also assess whether the central player's identity is consistent across the images. Return only JSON: {"consistent":"yes|no|uncertain","readings":[{"image":1,"number":null,"location":"back|front|sleeve|shorts|unknown"}],"evidence":"one short description of uncertainty or visible digits"}.'''


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def datasets(phase, labels=False):
    import numpy as np
    from nbhl_tracking_features_v2 import load_features,frame_folder
    if phase=='development':
        source=load_features(BASE,OLD,'osnet_parts_generic',cuts=True)
    else:
        assert phase=='regression'
        source=[]
        for p in sorted((OLD/'final-test/features/osnet_parts_generic').glob('*/metadata.json')):
            meta=json.loads(p.read_text()); assert sha(p.with_name('features.npz'))==meta['summary']['feature_sha256']
            source.append((meta,dict(np.load(p.with_name('features.npz')))))
        assert len(source)==5
    for meta,features in source:
        clip=meta['summary']['clip']
        folder=frame_folder(BASE,meta) if phase=='development' else Path('/data/sequences')/clip/'img1'
        path=OLD/'sam2/large'/(clip+'.json') if phase=='development' else OLD/'final-test/tracks/fourhour-sam2'/(clip+'.json')
        if not labels:
            meta={k:v for k,v in meta.items() if k not in ['observations','pairs']}
        yield meta,features,json.loads(path.read_text()),folder,sha(path)


def prepare(phase,protocol_sha):
    import numpy as np
    from PIL import Image,ImageDraw
    from nbhl_tracking_fourhour import tracklet_evidence,link_candidates
    from nbhl_tracking_global_links import global_links
    manifest={'phase':phase,'protocol_sha256':protocol_sha,'selection_uses_reference_labels':False,'clips':[]}
    for meta,features,raw,frames,raw_sha in datasets(phase):
        clip=raw['clip'];dest=OUT/phase/clip;dest.mkdir(parents=True,exist_ok=True)
        evidence=tracklet_evidence(meta,features,raw);edges=link_candidates(meta,features,evidence)
        baseline=global_links(raw,evidence,edges,SETTING)
        eligible=[r for r in edges if 0<r['gap']<=300 and min(r['support_a'],r['support_b'])>=10 and r['distance_heights']<=1+r['gap']*.12 and r['cosine_mean']>=.65]
        keys={(r['a'],r['b']) for r in baseline['links']}
        for field in ['a','b']:
            for tid in {r[field] for r in eligible}:
                ranked=sorted([r for r in eligible if r[field]==tid],key=lambda r:(-r['cosine_mean'],r['gap'],r['a'],r['b']))
                keys.update((r['a'],r['b']) for r in ranked[:2])
        selected=[r for r in eligible if (r['a'],r['b']) in keys]
        assert {(r['a'],r['b']) for r in baseline['links']}<=set((r['a'],r['b']) for r in selected)
        tids=sorted({r[k] for r in selected for k in ['a','b']});tracks={};loaded={}
        for ordinal,tid in enumerate(tids):
            ev=evidence[tid];samples=ev['samples'];chosen=[]
            # Three temporal strata prevent all views collapsing to one instant.
            for group in np.array_split(sorted(samples,key=lambda r:r['frame']),3):
                if len(group):chosen.append(max(group,key=lambda r:r['quality']))
            images=[]
            for j,row in enumerate(chosen):
                f=row['frame'];path=frames/f'{f+1:06d}.jpg'
                im=Image.open(path).convert('RGB');w,h=im.size;x,y,u,v=row['bbox'];bw=u-x;bh=v-y
                bounds=[max(0,int(x-.04*bw)),max(0,int(y-.02*bh)),min(w,int(u+.04*bw+1)),min(h,int(v+.02*bh+1))]
                crop=im.crop(bounds);target=dest/'crops'/f'{ordinal:04d}-{j}.jpg';target.parent.mkdir(exist_ok=True);crop.save(target,quality=96)
                images.append({'path':str(target),'sha256':sha(target),'frame':f,'quality':row['quality'],'bbox':bounds,'source_frame_sha256':sha(path)})
            tracks[tid]={'start':ev['start'],'end':ev['end'],'count':ev['count'],'images':images,'best_image':max(range(len(images)),key=lambda n:images[n]['quality'])}
        for n,row in enumerate(selected):row['id']=f'{clip}-edge-{n:04d}'
        row={'clip':clip,'source_id':meta['summary']['source_id'],'raw_tracks_sha256':raw_sha,'feature_sha256':meta['summary']['feature_sha256'],'all_candidates':len(edges),'eligible_candidates':len(eligible),'baseline_links':len(baseline['links']),'selected_edges':selected,'tracks':tracks}
        dump(dest/'evidence.json',row);dump(dest/'baseline.json',baseline);manifest['clips'].append(row)
        print(json.dumps({'phase':phase,'clip':clip,'edges':len(selected),'tracks':len(tracks)}),flush=True)
    dump(OUT/phase/'evidence.json',manifest)
    return {'phase':phase,'clips':len(manifest['clips']),'edges':sum(len(c['selected_edges']) for c in manifest['clips']),'tracks':sum(len(c['tracks']) for c in manifest['clips']),'evidence_sha256':sha(OUT/phase/'evidence.json')}


def requests(manifest,tasks):
    output=[]
    for clip in manifest['clips']:
        for task in tasks:
            if task=='number':
                for tid,row in clip['tracks'].items():output.append({'id':clip['clip']+'/number/'+tid,'task':task,'clip':clip['clip'],'track_id':tid,'images':row['images'],'prompt':NUMBER_PROMPT})
                continue
            assert task in ['single','multi','reverse']
            for edge in clip['selected_edges']:
                a,b=[clip['tracks'][edge[k]] for k in ['a','b']]
                aa=[a['images'][a['best_image']]] if task=='single' else a['images']
                bb=[b['images'][b['best_image']]] if task=='single' else b['images']
                if task=='reverse':aa,bb=bb,aa
                prompt=PAIR_PROMPT+f' The first {len(aa)} images are group A; the following {len(bb)} images are group B.'
                output.append({'id':edge['id']+'/'+task,'task':task,'clip':clip['clip'],'edge_id':edge['id'],'images':aa+bb,'prompt':prompt})
    return output


def number_consensus(response, image_count):
    value=response.get('parsed') if response else None
    if not isinstance(value,dict) or value.get('consistent')!='yes':return None
    rows=value.get('readings')
    if not isinstance(rows,list) or len(rows)!=image_count:return None
    if {r.get('image') for r in rows if isinstance(r,dict)}!=set(range(1,image_count+1)):return None
    numbers=[]
    for row in rows:
        n=row.get('number')
        if n is None:continue
        if type(n) is not int or not 0<=n<=99:return None
        numbers.append(n)
    return numbers[0] if len(numbers)>=2 and len(set(numbers))==1 else None


def decision(response):
    value=response.get('parsed') if response else None
    if not isinstance(value,dict):return 'uncertain'
    answer=value.get('decision')
    return answer if answer in ['same','different','uncertain'] else 'uncertain'


def strict_same(response):
    value=response.get('parsed') if response else None
    return bool(isinstance(value,dict) and value.get('decision')=='same' and value.get('a_consistent')=='yes' and value.get('b_consistent')=='yes' and value.get('evidence_type') in ['jersey_number','distinctive_details'])


def apply_policy(raw,evidence,edges,clip,predictions,policy):
    from nbhl_tracking_global_links import global_links
    by_pair={(r['a'],r['b']):r for r in clip['selected_edges']}
    numbers={tid:number_consensus(predictions.get(clip['clip']+'/number/'+tid),len(row['images'])) for tid,row in clip['tracks'].items()}
    scored=[]
    for edge in edges:
        a,b=edge['a'],edge['b'];selected=by_pair.get((a,b));score=edge['cosine_mean']
        pair={task:predictions.get(selected['id']+'/'+task) if selected else None for task in ['single','multi','reverse']}
        na,nb=numbers.get(a),numbers.get(b);conflict=na is not None and nb is not None and na!=nb
        if policy=='osnet_full':pass
        elif policy=='osnet_shortlist':
            if selected is None:continue
        elif policy=='jersey_veto':
            if conflict:continue
        elif policy=='jersey_support':
            if conflict:continue
            if na is not None and na==nb:score=min(1.,score+.20)
        elif policy in ['single_veto','multi_veto']:
            if decision(pair[policy.split('_')[0]])=='different':continue
        elif policy in ['multi_strict','multi_symmetric']:
            if selected is None or not strict_same(pair['multi']):continue
            if policy=='multi_symmetric' and not strict_same(pair['reverse']):continue
            score=.80+.19*max(0.,min(1.,(score-.65)/.35))
        elif policy=='multi_hybrid':
            if decision(pair['multi'])=='different':continue
            if strict_same(pair['multi']):score=min(1.,score+.10)
        elif policy=='multi_and_jersey_veto':
            if conflict or decision(pair['multi'])=='different':continue
        else:raise ValueError(policy)
        scored.append(edge|{'cosine_vlm':score})
    return global_links(raw,evidence,scored,SETTING|{'pooling':'vlm'})


def score(phase,model_key):
    from collections import Counter
    from nbhl_tracking_fourhour import tracklet_evidence,link_candidates,evaluate
    from nbhl_tracking_reference_audit import corrected_report
    protocol=json.loads((OUT/'protocol.json').read_text())
    mp=OUT/phase/'evidence.json';manifest=json.loads(mp.read_text());pp=OUT/phase/model_key/'predictions.json';pred=json.loads(pp.read_text())
    assert pred['evidence_sha256']==sha(mp) and pred['protocol_sha256']==sha(OUT/'protocol.json')
    responses={r['id']:r for r in pred['results']};expected=requests(manifest,protocol['tasks'])
    assert set(responses)=={r['id'] for r in expected},'Inference incomplete'
    correction_path=Path('/work/tracking-four-hour-reference-corrections-v1.json')
    corrections={r['pair_id']:r for r in json.loads(correction_path.read_text())['corrections']}
    reports=[];canonical_baseline=[]
    for meta,features,raw,folder,raw_sha in datasets(phase,labels=True):
        clip=next(c for c in manifest['clips'] if c['clip']==raw['clip']);assert raw_sha==clip['raw_tracks_sha256']
        evidence=tracklet_evidence(meta,features,raw);edges=link_candidates(meta,features,evidence)
        for policy in protocol['policies']:
            linked=apply_policy(raw,evidence,edges,clip,responses,policy)
            if policy=='osnet_full' and phase=='regression':
                old=json.loads((OLD/'final-test/tracks/fourhour-sam2-parts'/(raw['clip']+'.json')).read_text())
                assert linked['frames']==old['frames'],'Generic baseline must reproduce the original output exactly'
                canonical_baseline.append(raw['clip'])
            row=evaluate(meta,linked)
            if phase=='development':row=corrected_report(row,corrections)
            row.update(policy=policy,source_id=meta['summary']['source_id'],links=len(linked['links']))
            reports.append(row);dump(OUT/phase/model_key/'tracks'/policy/(raw['clip']+'.json'),linked)
    def total(group):return {key:sum(r['counts'][key] for r in group) for key in group[0]['counts']}
    totals=[]
    for policy in protocol['policies']:
        group=[r for r in reports if r['policy']==policy]
        scopes=['all'] if phase=='development' else ['short','stress']
        for scope in scopes:
            selected=[r for r in group if scope=='all' or ('stress' in r['clip'])==(scope=='stress')]
            totals.append({'policy':policy,'scope':scope,'counts':total(selected),'fragments':sum(r['fragments'] for r in selected),'links':sum(r['links'] for r in selected)})
    diagnostic={task:dict(Counter(decision(r) if task!='number' else ('parsed' if isinstance(r.get('parsed'),dict) else 'invalid') for r in pred['results'] if r['task']==task)) for task in protocol['tasks']}
    symmetry=Counter()
    for c in manifest['clips']:
        for e in c['selected_edges']:symmetry[decision(responses[e['id']+'/multi'])+' / '+decision(responses[e['id']+'/reverse'])]+=1
    full={'phase':phase,'model_key':model_key,'model':pred['model'],'protocol_sha256':pred['protocol_sha256'],'prediction_sha256':sha(pp),'evidence_sha256':sha(mp),'totals':totals,'clips':reports,'response_counts':diagnostic,'order_consistency':dict(symmetry),'requests':len(pred['results']),'generation_seconds':sum(r['seconds'] for r in pred['results']),'loading_seconds':pred['loading_seconds'],'baseline_original_frames_reproduced':canonical_baseline,'limits':protocol['evaluation']+' VLM outputs and within-track consistency statements are unverified predictions. Fragment IDs are not roster identities.'}
    dump(OUT/phase/model_key/'evaluation.json',full)
    return {k:v for k,v in full.items() if k!='clips'}
