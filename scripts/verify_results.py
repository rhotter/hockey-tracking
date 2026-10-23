"""Recompute published generative and embedding pair counts without videos/models."""
import hashlib
import json
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT/'experiments/reference'

def decode(flags):
    if isinstance(flags,str):
        assert set(flags)<={'1','0','?'}
        return [{'1':True,'0':False,'?':None}[v] for v in flags]
    return flags

def count(reference,flags,scope):
    flags=decode(flags)
    assert len(flags)==len(reference)
    selected=[(r,p) for r,p in zip(reference,flags) if scope=='all' or ('stress' in r['clip'])==(scope=='stress')]
    counts={'correct':0,'wrong':0,'unknown_same':0,'same_total':0,'different_total':0,'missing':0}
    for ref,joined in selected:
        assert joined is None or type(joined) is bool
        counts['same_total']+=ref['truth']=='same';counts['different_total']+=ref['truth']=='different';counts['missing']+=joined is None
        if joined:counts[{'same':'correct','different':'wrong','uncertain':'unknown_same'}[ref['truth']]]+=1
    return counts

def main():
    folder=PROJECT/'results'
    source=folder/'vlm-reconnect-pair-outcomes.json';outcomes=json.loads(source.read_text())
    published=json.loads((folder/'vlm-reconnect-results.json').read_text());verified=[]
    hashes={}
    for file,field in [('vlm-reconnect-code-freeze-v1.json','sha256'),('vlm-reconnect-number-amendment-v1.json','source_sha256'),('vlm-embedding-code-freeze-v2.json','sha256'),('vlm-conservative-code-freeze-v1.json','sha256')]:
        record=json.loads((ROOT/'data/ball-hockey'/file).read_text())
        for name,expected in record[field].items():
            actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            assert actual==expected,name
            hashes[name]=actual
    for report in published['selection']['development_reports']:
        assert hashlib.sha256((ROOT/report['file']).read_bytes()).hexdigest()==report['sha256']
    for row in published['observation_ceiling']['counts']:
        phase=outcomes['phases'][row['phase']]
        pairs=[(r,v) for r,v in zip(phase['references'],decode(phase['models']['8b']['osnet_full'])) if r['truth']=='same' and (row['scope']=='all' or ('stress' in r['clip'])==(row['scope']=='stress'))]
        assert row['same_pairs']==len(pairs)
        assert row['endpoints_matched']==sum(v is not None for _,v in pairs)
        assert row['unmatched_same_pairs']==sum(v is None for _,v in pairs)
        assert row['baseline_connected']==sum(v is True for _,v in pairs)
    for report in published['reports']:
        phase=outcomes['phases'][report['phase']];reference=phase['references']
        assert report['prediction_sha256']==next(r['predictions_sha256'] for r in phase['inputs'] if r['model']==report['model_key'])
        for row in report['totals']:
            flags=phase['models'][report['model_key']][row['policy']]
            counts=count(reference,flags,row['scope'])
            for key,value in counts.items():assert row['counts'][key]==value,(report['model_key'],row['policy'],row['scope'],key)
            verified.append({'model':report['model_key'],'phase':report['phase'],'policy':row['policy'],'scope':row['scope'],'counts':counts})
    embeddings=json.loads((folder/'vlm-embedding-results.json').read_text())
    embedding_verified=[]
    for report in embeddings['reports']:
        for phase_name in ['development','regression']:
            phase=outcomes['phases'][phase_name];data=report[phase_name]
            provenance=next(r for r in phase['embedding_inputs'] if r['model']==report['model'])
            assert provenance['features_sha256']==data['metadata']['features_sha256']
            if phase_name=='development':
                assert provenance['evaluation_sha256']==next(r['sha256'] for r in embeddings['selection']['development_reports'] if r['model']==report['model'])
            rows=([data['baseline']|{'representation':'baseline'}]+data['selected']) if phase_name=='development' else data['results']
            for row in rows:
                scopes=[{'scope':'all','counts':row['counts']}] if phase_name=='development' else row['scopes']
                for scope in scopes:
                    flags=phase['embedding_models'][report['model']][row['representation']]
                    counts=count(phase['references'],flags,scope['scope'])
                    for key,value in counts.items():assert scope['counts'][key]==value,(report['model'],row['representation'],scope['scope'],key)
                    embedding_verified.append({'model':report['model'],'phase':phase_name,'representation':row['representation'],'scope':scope['scope'],'counts':counts})
    conservative=json.loads((folder/'vlm-conservative-results.json').read_text());conservative_verified=[]
    for phase_name in ['development','regression']:
        phase=outcomes['phases'][phase_name];report=conservative[phase_name]
        assert report['prediction_sha256']==phase['conservative_input']['prediction_sha256']
        for row in report['totals']:
            counts=count(phase['references'],phase['conservative_policies'][row['policy']],row['scope'])
            for key,value in counts.items():assert row['counts'][key]==value,(phase_name,row['policy'],row['scope'],key)
            conservative_verified.append({'phase':phase_name,'policy':row['policy'],'scope':row['scope'],'counts':counts})
    assert conservative['selection']['development_sha256']==hashlib.sha256((ROOT/'notes/ball-hockey/vlm-conservative-development.json').read_bytes()).hexdigest()
    result={'comparisons_verified':len(verified),'embedding_comparisons_verified':len(embedding_verified),'conservative_comparisons_verified':len(conservative_verified),'frozen_source_hashes_verified':hashes,'outcomes_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'results':verified,'embedding_results':embedding_verified,'conservative_results':conservative_verified,'limit':'Pair-count recomputation from exported frozen outcomes; not a new inference run or independent ground truth.'}
    path=PROJECT/'outputs/verification.json';path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['results','embedding_results','conservative_results']}))

if __name__=='__main__':main()
