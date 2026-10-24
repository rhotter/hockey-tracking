"""Freeze native/adapted video settings and audit final SAM3 combinations."""
from pathlib import Path
import json
import modal
from modal_campaign import ROOT, volume
from modal_nbhl_sam3_hybrid import image as hybrid_image

BASE = Path('/data/ball-hockey/reliability-20261018')
OUT = BASE/'sam3-hybrid-20261024-v1'
OLD = BASE/'sam3-20261023-v1'
app = modal.App('sport-scans-sam3-analysis-20261024')
image = hybrid_image.add_local_file(ROOT/'scripts/modal_nbhl_sam3_hybrid.py','/work/modal_nbhl_sam3_hybrid.py')
for name in ['sam3-adapted-video-protocol-20261024.json',
             'sam3-adapted-control-protocol-20261024.json',
             'sam3-adapted-person-extension-20261024.json',
             'sam3-adapted-capacity-extension-20261024.json',
             'sam3-visual-protocol-20261024.json',
             'sam3-persistent-visual-protocol-20261024.json',
             'sam3-new-interval-audit-protocol-20261024.json',
             'sam3-new-interval-audit-reference-20261024.json',
             'tracking-four-hour-dense-audit-reference-v1.json',
             'tracking-four-hour-dense-audit-protocol-v1.json']:
    image = image.add_local_file(ROOT/'data/ball-hockey'/name, '/work/'+name)


def native_path(phase, clip, variant='default', delta=''):
    folder = OUT/'native-settings'/phase/(variant+'-offload'+('-'+delta if delta else ''))/clip
    if variant == 'default' and not delta and not (folder/'tracks.json').exists():
        folder = OLD/'tracking'/phase/'person'/clip
    return folder


def aggregate(rows):
    return {key: sum(row['counts'][key] for row in rows) for key in rows[0]['counts']}


def adapted_control_variant(variant):
    return 'default' if variant.endswith('person') else variant.replace('adapt-margin','adapt-control')


def objective(row):
    c = row['counts']
    return c['correct']-10*c['wrong'], c['matched_observations'], -c['unknown_same'], row['variant']=='default'


@app.function(image=image, cpu=4, memory=8192, timeout=600, volumes={'/data':volume})
def table(family='native', phase='development', freeze=False):
    import datetime
    from nbhl_vlm_reconnect import dump, sha
    from nbhl_tracking_reference_audit import corrected_report
    volume.reload()
    proto = json.loads(Path('/work/hybrid-protocol.json').read_text())
    variants = ['default', 'association03', 'recondition4', 'recondition0', 'alive8']
    delta = '';deltas={}
    protocol = Path('/work/hybrid-protocol.json')
    if family == 'adapted':
        selection = json.loads((BASE/'sam3-finetune-20261024-v1/selection.json').read_text())
        delta = selection['best']['variant']
        assert delta != 'original', 'No adapted detector selected'
        variants = ['adapt-margin0', 'adapt-margin1', 'adapt-margin2','adapt-person','adapt-heads-person','adapt-decoder-person']
        deltas={'adapt-heads-person':'heads-lr1e4','adapt-decoder-person':'decoder-lr1e5'}
        protocol = Path('/work/sam3-adapted-video-protocol-20261024.json')
    elif family == 'visual':
        variants = ['default', 'visual', 'visual-person', 'visual-hockey']
        protocol = Path('/work/sam3-visual-protocol-20261024.json')
    elif family == 'visual-memory':
        variants = ['default', 'visual-memory', 'visual-memory-person', 'visual-memory-hockey']
        protocol = Path('/work/sam3-persistent-visual-protocol-20261024.json')
    assert family in ['native', 'adapted', 'visual', 'visual-memory']
    dest = OUT/'video-selection'/family
    if phase == 'regression':
        chosen = json.loads((dest/'selection.json').read_text())
        assert chosen['protocol_sha256'] == sha(protocol)
        if family=='adapted':
            assert chosen['extension_protocol_sha256']==sha(Path('/work/sam3-adapted-person-extension-20261024.json'))
            assert chosen['capacity_extension_protocol_sha256']==sha(Path('/work/sam3-adapted-capacity-extension-20261024.json'))
            deltas[chosen['best']['variant']]=chosen['best']['delta']
        variants = [chosen['best']['variant']]
    results = []
    for variant in variants:
        rows = [];variant_delta=deltas.get(variant,delta)
        for clip in proto[phase]:
            folder = native_path(phase, clip, variant, variant_delta)
            raw = json.loads((folder/'metrics.json').read_text())
            if phase == 'development':
                # Earlier native files predate the existing versioned corrections;
                # newer runs already apply them. Recount both against the same truth.
                corrections = {r['pair_id']:r for r in json.loads(Path('/work/tracking-four-hour-reference-corrections-v1.json').read_text())['corrections']}
                needed = {}
                for pair in raw['pairs']:
                    if pair['id'] not in corrections:continue
                    correction = corrections[pair['id']]
                    assert pair['truth'] in [correction['previous_truth'],correction['truth']]
                    if pair['truth'] != correction['truth']:needed[pair['id']] = correction
                raw = corrected_report(raw,needed)
            rows.append({k:v for k,v in raw.items() if k!='pairs'} |
                        {'tracks_sha256':sha(folder/'tracks.json'), 'metrics_sha256':sha(folder/'metrics.json')})
        results.append({'variant':variant, 'delta':variant_delta, 'counts':aggregate(rows), 'clips':rows})
    report = {'family':family, 'phase':phase, 'results':results, 'protocol_sha256':sha(protocol)}
    if family=='adapted' and phase=='regression':
        control_variant=adapted_control_variant(variants[0])
        controls=[]
        for clip in proto[phase]:
            folder=native_path(phase,clip,control_variant)
            raw=json.loads((folder/'metrics.json').read_text())
            controls.append({k:v for k,v in raw.items() if k!='pairs'} |
                            {'tracks_sha256':sha(folder/'tracks.json'),'metrics_sha256':sha(folder/'metrics.json')})
        report['matched_original_control']={'variant':control_variant,'delta':'','counts':aggregate(controls),'clips':controls}
        report['control_protocol_sha256']=sha(Path('/work/sam3-adapted-control-protocol-20261024.json'))
    if family=='adapted':
        report['extension_protocol_sha256']=sha(Path('/work/sam3-adapted-person-extension-20261024.json'))
        report['capacity_extension_protocol_sha256']=sha(Path('/work/sam3-adapted-capacity-extension-20261024.json'))
    dump(dest/(phase+'.json'), report)
    if freeze:
        assert phase == 'development' and not (dest/'selection.json').exists()
        assert not (dest/'regression.json').exists()
        report = {'family':family, 'best':max(results,key=objective), 'candidates':results,
                  'protocol_sha256':sha(protocol), 'development_sha256':sha(dest/'development.json'),
                  'frozen_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'selection_uses_regression':False}
        if family=='adapted':
            report['extension_protocol_sha256']=sha(Path('/work/sam3-adapted-person-extension-20261024.json'))
            report['capacity_extension_protocol_sha256']=sha(Path('/work/sam3-adapted-capacity-extension-20261024.json'))
        dump(dest/'selection.json', report)
    volume.commit()
    return report


@app.function(image=image, cpu=4, memory=8192, timeout=600, volumes={'/data':volume})
def audit(include_adapted=True):
    """Existing one-second sample is a supplementary audit, never a selector."""
    from collections import Counter
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from nbhl_detection_metrics import iou
    from nbhl_vlm_reconnect import dump, sha
    volume.reload()
    reference_path = Path('/work/tracking-four-hour-dense-audit-reference-v1.json')
    reference = json.loads(reference_path.read_text())
    protocol_path = Path('/work/tracking-four-hour-dense-audit-protocol-v1.json')
    assert reference['protocol_sha256'] == sha(protocol_path)
    clips = sorted({r['clip'] for r in reference['observations']})
    for r in reference['observations']:
        assert sha(Path('/data/sequences')/r['clip']/'img1'/f'{r["frame"]+1:06d}.jpg') == r['frame_sha256']
    paths = {
        'sam3-native':{c:native_path('regression', c)/'tracks.json' for c in clips},
        'sam2-qwen-joint':{c:BASE/'vlm-reconnect-20261019-v1/embedding-evaluation/2b/regression/joint'/(c+'.json') for c in clips},
    }
    hybrid = json.loads((OUT/'selection.json').read_text())
    for key in hybrid['selected_keys']:
        paths[key] = {c:OUT/'regression'/key/c/'tracks.json' for c in clips}
    for family in ['native','visual','visual-memory']+(['adapted'] if include_adapted else []):
        chosen = json.loads((OUT/'video-selection'/family/'selection.json').read_text())['best']
        paths[family+'-'+chosen['variant']] = {c:native_path('regression', c, chosen['variant'], chosen['delta'])/'tracks.json' for c in clips}
        if family=='adapted':
            control=adapted_control_variant(chosen['variant'])
            paths['adapted-original-control']={c:native_path('regression',c,control)/'tracks.json' for c in clips}
    results = []
    inputs = {method:{clip:sha(path) for clip,path in pp.items()} for method,pp in paths.items()}
    for threshold in [.5,.3]:
        for method,pp in paths.items():
            details = []; events = []
            for clip,path in pp.items():
                tracks = json.loads(path.read_text())
                refs = [r for r in reference['observations'] if r['clip']==clip]
                assignments = {}
                for f in sorted({r['frame'] for r in refs}):
                    obs = [r for r in refs if r['frame']==f and r['state']=='visible']
                    objects = tracks['frames'][f]['objects']
                    if not obs or not objects: continue
                    overlaps = np.array([[iou(a['bbox'],b['bbox']) for b in objects] for a in obs])
                    aa,bb = linear_sum_assignment(np.where(overlaps>=threshold,1-overlaps,1e6))
                    for a,b in zip(aa,bb):
                        if overlaps[a,b]>=threshold:
                            assignments[(obs[a]['identity'],f)] = (objects[b]['track_id'],float(overlaps[a,b]))
                for identity in sorted({r['identity'] for r in refs}):
                    seq = sorted([r for r in refs if r['identity']==identity],key=lambda r:r['frame'])
                    visible = [r for r in seq if r['state']=='visible']
                    matched = [r for r in visible if (identity,r['frame']) in assignments]
                    counts = Counter()
                    for a,b in zip(seq,seq[1:]):
                        if a['state']!='visible' or b['state']!='visible': continue
                        ta = assignments.get((identity,a['frame'])); tb = assignments.get((identity,b['frame']))
                        state = (('missing_both' if not ta and not tb else 'missing_one') if not ta or not tb
                                 else ('same_track' if ta[0]==tb[0] else 'identity_change'))
                        counts[state]+=1
                        events.append({'clip':clip,'identity':identity,'frames':[a['frame'],b['frame']],
                                       'state':state,'tracks':[ta[0] if ta else None,tb[0] if tb else None]})
                    details.append({'clip':clip,'identity':identity,'visible_observations':len(visible),
                                    'matched_observations':len(matched),'sampled_transitions':dict(counts)})
            results.append({'method':method,'iou_threshold':threshold,
                            'visible_observations':sum(r['visible_observations'] for r in details),
                            'matched_observations':sum(r['matched_observations'] for r in details),
                            'sampled_transitions':dict(Counter(e['state'] for e in events)),
                            'details':details,'events':events})
    report = {'reference_sha256':sha(reference_path),'protocol_sha256':sha(protocol_path),
              'reference_counts':reference['counts'],'inputs':inputs,'results':results,
              'limits':reference['limits'],'selection_uses_this_audit':False}
    dump(OUT/'supplementary-audit.json',report); volume.commit()
    return report


@app.function(image=image, cpu=4, memory=8192, timeout=600, volumes={'/data':volume})
def fresh_audit():
    """Evaluate the frozen new temporal sample without selecting any settings."""
    from collections import Counter
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from nbhl_detection_metrics import iou
    from nbhl_vlm_reconnect import dump, sha
    volume.reload()
    rp = Path('/work/sam3-new-interval-audit-reference-20261024.json')
    pp = Path('/work/sam3-new-interval-audit-protocol-20261024.json')
    reference = json.loads(rp.read_text())
    assert reference['protocol_sha256'] == sha(pp)
    clip = reference['clip']
    cache = BASE/'tracking-four-hour-v1/long-cache-trial'/clip
    for r in reference['observations']:
        assert sha(cache/'frames'/f'{r["frame"]+1:06d}.jpg') == r['frame_sha256']
    paths = {'sam3-native':native_path('long',clip)/'tracks.json',
             'sam2-raw':cache/'sam2.json'}
    inputs = {method:sha(path) for method,path in paths.items()}
    sampled_frames=sorted({r['frame'] for r in reference['observations']})
    raw_tracks={method:json.loads(path.read_text()) for method,path in paths.items()}
    predictions={method:{'source_tracks_sha256':inputs[method],
                         'frames':[raw_tracks[method]['frames'][f] for f in sampled_frames]}
                 for method,path in paths.items()}
    dump(OUT/'fresh-audit-predictions.json',predictions)
    results = []
    for threshold in [.5,.3]:
        for method,path in paths.items():
            tracks = raw_tracks[method]
            refs = reference['observations']; assignments = {}
            for f in sorted({r['frame'] for r in refs}):
                obs = [r for r in refs if r['frame']==f and r['state']=='visible']
                objects = tracks['frames'][f]['objects']
                assert tracks['frames'][f]['frame'] == f
                if not obs or not objects:continue
                overlaps = np.array([[iou(a['bbox'],b['bbox']) for b in objects] for a in obs])
                aa,bb = linear_sum_assignment(np.where(overlaps>=threshold,1-overlaps,1e6))
                for a,b in zip(aa,bb):
                    if overlaps[a,b]>=threshold:
                        assignments[(obs[a]['identity'],f)] = (objects[b]['track_id'],float(overlaps[a,b]))
            details=[];events=[]
            for identity in sorted(reference['identities']):
                seq = sorted([r for r in refs if r['identity']==identity],key=lambda r:r['frame'])
                visible = [r for r in seq if r['state']=='visible']
                matched = [r for r in visible if (identity,r['frame']) in assignments]
                counts = Counter()
                for a,b in zip(seq,seq[1:]):
                    if a['state']!='visible' or b['state']!='visible':continue
                    ta=assignments.get((identity,a['frame']));tb=assignments.get((identity,b['frame']))
                    state=(('missing_both' if not ta and not tb else 'missing_one') if not ta or not tb
                           else ('same_track' if ta[0]==tb[0] else 'identity_change'))
                    counts[state]+=1
                    events.append({'identity':identity,'seconds':[a['second'],b['second']],
                                   'frames':[a['frame'],b['frame']],'state':state,
                                   'tracks':[ta[0] if ta else None,tb[0] if tb else None]})
                details.append({'identity':identity,'visible_observations':len(visible),
                                'matched_observations':len(matched),'sampled_transitions':dict(counts)})
            results.append({'method':method,'iou_threshold':threshold,
                            'visible_observations':sum(r['visible_observations'] for r in details),
                            'matched_observations':sum(r['matched_observations'] for r in details),
                            'sampled_transitions':dict(Counter(e['state'] for e in events)),
                            'details':details,'events':events,
                            'assignments':[{'identity':identity,'frame':frame,'track_id':value[0],'iou':value[1]}
                                           for (identity,frame),value in sorted(assignments.items())]})
    report={'reference_sha256':sha(rp),'protocol_sha256':sha(pp),'reference_frozen_at':reference['frozen_at'],
            'reference_counts':reference['counts'],'inputs':inputs,'results':results,
            'limits':reference['limits'],'selection_uses_this_audit':False}
    dump(OUT/'fresh-audit.json',report);volume.commit()
    return report


@app.local_entrypoint()
def main(task='table', family='native', phase='development', freeze:bool=False, include_adapted:bool=True):
    if task=='table':result=table.remote(family,phase,freeze)
    elif task=='fresh-audit':result=fresh_audit.remote()
    elif task=='audit':result=audit.remote(include_adapted)
    else:raise ValueError(task)
    target = ROOT/'notes/ball-hockey'/f'sam3-analysis-{task}-{family}-{phase}{"-freeze" if freeze else ""}-20261024.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
