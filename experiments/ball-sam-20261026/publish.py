"""Build reproducible SAM ball comparison and single-position replay artifacts."""
from pathlib import Path
import hashlib, json
from evaluate import ROOT, HERE, read, write, derive_pilot
from fusion import derive


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positions(replay):
    fields=['frame','state','pixel','xy','reason','review_support','play_state','evidence','tracker']
    frames=[]
    for row in replay['frames']:
        item={k:row[k] for k in fields if k in row}
        if row.get('hypotheses'):
            item['carrier_track_id']=row['hypotheses'][0].get('carrier_track_id')
        frames.append(item)
    return {**{k:replay[k] for k in ['schema_version','source_id','start_s','duration_s','fps']},
            'method':'One selected SAM3/TAPNext++ or contextual ball position per frame','frames':frames}


def artifacts():
    results=HERE/'results'
    pilot=derive_pilot(results)
    decision=read(HERE/'decision.json')
    amendment=read(HERE/'exploratory-amendment.json')
    data=read(results/'all'/(decision['replay_variant']+'.json'))
    fused,full=derive(data,decision['replay_policy'])
    _,conservative=derive(data)
    _,consensus=derive(data,'sam_consensus')
    _,primary_full=derive(read(results/'all'/(pilot['selected_variant']+'.json')))
    audit=read(HERE/'output-audit.json')
    dependencies=[ROOT/'web/data/ball-sequence-20261026/context-replay.json',ROOT/'web/data/ball-sequence-20261026/replay.json',
                  ROOT/'experiments/ball-sequence-20261026/point-fusion-config.json',ROOT/'experiments/ball-sequence-20261026/output-audit.json',
                  ROOT/'experiments/ball-sequence-20261026/evaluate.py',ROOT/'hockey_tracking/ball_sequence.py']
    report={'schema_version':1,'source_id':'OLpK8enSjJs','start_s':900,'duration_s':300,'fps':15,
            'pilot':pilot,'primary_full':primary_full,'point_conservative_full':conservative,'point_consensus_full':consensus,'full':full,'decision':decision,'exploratory_amendment':amendment,'fusion_amendment':read(HERE/'fusion-amendment.json'),'output_audit':audit,
            'estimated_coverage':1-full['states'].get('unknown',0)/4500,
            'independent_human_ground_truth':False,
            'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [HERE/'protocol.json',HERE/'decision.json',HERE/'output-audit.json',HERE/'pilot-selection.json',HERE/'exploratory-amendment.json',HERE/'fusion-amendment.json',*dependencies,*sorted(results.rglob('*.json'))]},
            'code_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sorted(HERE.glob('*.py'))},
            'limits':['All references and audit decisions are approximate assistant observations on a previously examined NBHL recording.',
                      'The six-seed comparison has identical reviewed positions, with seed frames and their +-3 neighbors excluded. Configuration selection uses development windows only; validation windows are still from the same development game.',
                      'The 31-seed comparison uses a different set of windows and references. Conditional error among emitted points is not recall or full-video precision.',
                      'Context estimates are unchanged. More output is not evidence of correct invisible-ball localization. No new fine-tuning or additional language-model calls were performed.',
                      'The replay emits one selected location. Held, airborne and height-uncertain locations have no floor projection.']}
    return positions(fused),report


if __name__=='__main__':
    replay,report=artifacts();dest=ROOT/'web/data/ball-sam-20261026'
    write(dest/'positions.json',replay);write(dest/'report.json',report)
    print(json.dumps({'selected':report['pilot']['selected_variant'],'decision':report['decision'],'states':report['full']['states']}))
