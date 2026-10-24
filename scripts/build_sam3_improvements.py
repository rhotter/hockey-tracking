"""Render the SAM3 follow-up from saved, verifiable experiment reports."""
from pathlib import Path
from html import escape
import json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'web/data/reliability/sam3-20261024'
FAMILIES = ['native','visual','visual-memory','adapted']
NAMES = {
    'sam3-native':'SAM3 pretrained', 'sam2-raw':'Raw detector + SAM2',
    'sam2-qwen-joint':'Prior pipeline: SAM2 + OSNet + Qwen 2B',
    'anchor':'SAM3 + current detector boxes',
    'appearance':'SAM3 + detector + appearance linking',
    'consensus':'SAM2 + appearance + SAM3 agreement',
    'consensus-qwen':'SAM2 + appearance + Qwen + SAM3 agreement',
    'native':'SAM3 video settings', 'visual':'SAM3 initial visual prompt',
    'visual-memory':'SAM3 persistent visual prompt',
    'adapted':'Fine-tuned SAM3 video',
}


def read(name):return json.loads((DATA/name).read_text())
def pct(value):return f'{100*value:.1f}%'
def table(headers,rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def total(clips):
    return {k:sum(r['counts'][k] for r in clips) for k in clips[0]['counts']}
def counts_text(c):return f'{c["correct"]}/{c["same_total"]}'


def write_catalog(catalog):
    labels={
        'sam3-native':'SAM3 pretrained · person prompt',
        'sam3-anchor':'SAM3 + current detector boxes',
        'sam3-appearance':'SAM3 + detector + appearance links',
        'sam3-consensus':'SAM2 + appearance + SAM3 agreement',
        'sam3-consensus-qwen':'SAM2 + Qwen + SAM3 agreement',
        'sam3-native-tuned':'SAM3 · selected video settings',
        'sam3-visual-tuned':'SAM3 · selected initial visual prompt',
        'sam3-visual-memory-tuned':'SAM3 · selected persistent visual prompt',
        'sam3-adapted-tuned':'SAM3 · ball-hockey fine-tune',
        'sam3-adapted-control':'SAM3 pretrained · matched fine-tune control',
        'vlm-embedding-2b-joint':'Prior pipeline · SAM2 + OSNet + Qwen 2B',
        'sam3-long':'SAM3 · complete 20m 15s replay',
        'sam2-long-generic':'SAM2 · generic appearance · complete replay',
        'sam2-long-trained':'SAM2 · trained appearance · complete replay',
    }
    explanations={
        'sam3-native':'Pretrained SAM3 with the person prompt and original video settings.',
        'sam3-anchor':'SAM3 identities anchored to current YOLO player detections by one-to-one overlap matching.',
        'sam3-appearance':'SAM3 identities anchored to YOLO detections, with development-selected appearance checks and reconnection.',
        'sam3-consensus':'SAM2 fragments and appearance reconnection, using agreement from a separate SAM3 run.',
        'sam3-consensus-qwen':'SAM2 fragments and appearance/Qwen image embeddings, using agreement from a separate SAM3 run.',
        'sam3-native-tuned':'Pretrained SAM3 with video settings selected on nine development clips.',
        'sam3-visual-tuned':'Pretrained SAM3 with an initial player example chosen automatically from detector boxes.',
        'sam3-visual-memory-tuned':'Pretrained SAM3 with the initial player example’s visual embedding retained on later frames.',
        'sam3-adapted-tuned':'SAM3 detector layers fine-tuned on ball-hockey boxes. The detector checkpoint and video setting were selected on development clips; temporal tracking and mask weights remain pretrained.',
        'sam3-adapted-control':'Original pretrained weights with exactly the same prompt and video settings as the selected fine-tuned run.',
        'sam3-long':'Four 315-second SAM3 runs with 15-second overlaps, covering recording 29:45–50:00. Overlap agreement reconnects fragment IDs; gaps remain gaps. Frames are cached and older mask memories move to CPU without deleting history.',
        'sam2-long-generic':'The existing detector-supported SAM2 run with generic body-region appearance linking, now shown for the complete 20m 15s. Four 315-second runs share 15-second overlaps. No new inference or full-game identity reference was added.',
        'sam2-long-trained':'The existing detector-supported SAM2 run with trained body-region appearance linking, now shown for the complete 20m 15s. Four 315-second runs share 15-second overlaps. No new inference or full-game identity reference was added.',
        'vlm-embedding-2b-joint':'The prior detector-supported SAM2 pipeline, reconnecting fragments with OSNet appearance and Qwen3-VL Embedding 2B joint images. This is the unchanged prior comparison pipeline.',
    }
    specs={}
    for row in catalog:
        methods=[m['id'] for m in row['methods']]
        descriptions={}
        for m in row['methods']:
            if m['id'] not in explanations:continue
            checkpoint=(' Detector checkpoint: '+m['delta']+'.') if m.get('delta') else ''
            descriptions[m['id']]=explanations[m['id']]+' Setting: '+m['setting']+'.'+checkpoint+' Automatic fragments are not verified jersey identities. Model masks may drift or persist behind occlusion; no manual corrections or interpolated positions are added.'
        specs[row['clip']]={'frames':row['frames'],'methods':methods,
                            'methodLabels':{m:labels[m] for m in methods},
                            'methodDescriptions':descriptions}
        if row['clip']=='pittsburgh-sam3-20min':
            specs[row['clip']].update(label='Pittsburgh · full 20m 15s · 29:45–50:00',
                description='NBHL Pittsburgh Legends Tier Semis: Pittsburgh Seahorses vs. Rolling Hills Relics (09/05/2026) · Recording 29:45–50:00',
                folder='media/sam3-20261024',default='sam3-long')
    script="""(() => {
 'use strict';
 const additions=ADDITIONS;
 for(const [id,extra]of Object.entries(additions)){
  const existing=window.SportScansTrackingCatalog[id];
  if(!existing){window.SportScansTrackingCatalog[id]=extra;continue;}
  if(existing.frames!==extra.frames)throw Error('Replay frame count disagrees with catalog.');
  existing.methods=[...new Set([...existing.methods,...extra.methods])];
  Object.assign(existing.methodLabels,extra.methodLabels);
  existing.methodDescriptions={...existing.methodDescriptions,...extra.methodDescriptions};
 }
})();
""".replace('ADDITIONS',json.dumps(specs,separators=(',',':')))
    (ROOT/'web/tracking-sam3-improvements-catalog.js').write_text(script)


def main():
    fine=read('finetune/regression.json');selection=read('finetune/selection.json')
    hybrid=read('regression.json');hybrid_selection=read('selection.json')
    chosen={r['key']:r for r in hybrid_selection['winners']}
    videos={f:{n:read(f'video-{f}/{n}.json') for n in ['development','selection','regression']} for f in FAMILIES}
    runtime=read('runtime-verification.json');long=read('long-run.json');video=read('long-video.json')
    fresh=read('fresh-audit.json');audit=read('supplementary-audit.json')
    fresh_detection=read('fresh-detection/report.json')
    fresh_baseline=read('fresh-detection/baseline-control.json')
    review=read('detection-review.json');catalog=read('replay-catalog.json')
    write_catalog(catalog)
    rink_selection=read('rink-filter/selection.json')
    rink_regression=read('rink-filter/regression.json') if rink_selection['improves_validation_f1'] else None
    tracking=[]
    for r in hybrid['results']:
        if r['family']=='baseline':name=NAMES[r['key']];method=r['key']
        else:name=NAMES[r['family']];method='sam3-'+r['family']
        tracking.append({'label':name,'method':method,'key':r['key'],'counts':r['counts'],'clips':r['clips']})
    for family,report in videos.items():
        row=report['regression']['results'][0]
        if row['variant']=='default':continue
        tracking.append({'label':NAMES[family]+' · '+row['variant'],'method':'sam3-'+family+'-tuned',
                         'key':row['variant'],'counts':row['counts'],'clips':row['clips']})
    control=videos['adapted']['regression']['matched_original_control']
    tracking.append({'label':'Pretrained SAM3 · matched prompt/settings','method':'sam3-adapted-control',
                     'key':control['variant'],'counts':control['counts'],'clips':control['clips']})
    for row in tracking:
        row['short_counts']=total([r for r in row['clips'] if r['clip']!='pittsburgh-stress-0900'])
        row['stress_counts']=next(r['counts'] for r in row['clips'] if r['clip']=='pittsburgh-stress-0900')
    summary={'id':'sam3-improvements-20261024','scope':'NBHL ball hockey only','fine_tune_selection':selection['best'],
             'detection':fine['results'],'tracking':tracking,'video_selections':{f:r['selection']['best'] for f,r in videos.items()},
             'long_run':long,'long_video':video,'fresh_audit':fresh,'supplementary_audit':audit,
             'runtime_verification':runtime,'replay_catalog':catalog,
             'fresh_detection_audit':fresh_detection,
             'fresh_detection_baseline_control':fresh_baseline,
             'rink_filter':{'selection':rink_selection,'regression':rink_regression},
             'limitations':['Reused recordings and assistant-reviewed, correlated references; not an independent benchmark.',
                            'Sparse identity pairs and sampled transitions are distinct from continuous identity accuracy.',
                            'Track fragments are not validated roster identities, time on floor or metric distance.']}
    (DATA/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    before,after=fine['results'];bm=before['metrics'];am=after['metrics']
    paragraphs=[
        '<h1>SAM3 follow-up</h1><p>October 24, 2026 · NBHL ball hockey</p>',
        f'<p>The selected SAM3 fine-tune scored {pct(am["f1"])} detection F1, compared with {pct(bm["f1"])} for pretrained SAM3, on 36 reused regression frames. Adding the automatically predicted rink surface raised F1 to 90.8%, versus 89.0% for the prior YOLO detector.</p>',
        '<p>The existing SAM2 + OSNet + Qwen pipeline remains the recommended tracking baseline. Adding SAM3 agreement retained 111/129 correct connections and zero known wrong merges, while reducing unverified joins from four to three. The fine-tuned SAM3 video tracker also recovered 111 connections, but made three wrong merges. These are sparse, reused checks; the change in unverified joins is not a verified accuracy gain.</p>',
        '<p><a href="tracking-replay.html?clip=pittsburgh-sam3-20min&amp;method=sam3-long">Watch the complete 20m 15s tracking replay</a> · <a href="tracking-replay.html?clip=pittsburgh-stress-0900&amp;method=sam3-native">Compare methods on the five-minute clip</a></p>',
        '<h2>Player detection</h2><p>256 reviewed floor players and goalies across two recordings. A prediction matches at box IoU ≥ 0.5; officials, spectators and bench players count as false positives. Thresholds and the winning checkpoint were frozen using a separate validation recording.</p>',
    ]
    detection=[]
    for row,label in zip(fine['results'],['Pretrained SAM3','Fine-tuned SAM3']):
        m=row['metrics'];detection.append([label,m['tp'],m['fp'],m['fn'],pct(m['precision']),pct(m['recall']),pct(m['f1'])])
    old=json.loads((ROOT/'web/data/reliability/sam3/results.json').read_text())['detection']['fine_tuned_detector']
    tp,fp,fn=[old[k] for k in ['tp','fp','fn']]
    detection.append(['Prior fine-tuned YOLO11m',tp,fp,fn,pct(tp/(tp+fp)),pct(tp/(tp+fn)),pct(2*tp/(2*tp+fp+fn))])
    if rink_regression:
        m=rink_regression['results'][1]['metrics']
        detection.append(['Fine-tuned SAM3 + rink filter (follow-up)',m['tp'],m['fp'],m['fn'],pct(m['precision']),pct(m['recall']),pct(m['f1'])])
    paragraphs.append(table(['Detector','Found','False','Missed','Precision','Recall','F1'],detection))
    if rink_regression:
        paragraphs.append('<p>The rink filter is a post-test follow-up motivated by bench-player false positives. Its prompt, mask threshold and margin were selected on the 23 validation frames, then checked on the reused test frames. It combines the frozen fine-tuned detector with pretrained SAM3’s predicted rink surface. It does not provide calibrated rink coordinates or prove visibility.</p><details><summary>Inspect the surface-filter follow-up</summary><p>All 36 frames in fixed order. Each sheet shows the fine-tune, predicted surface region, and filtered output. Orange boxes were removed, including any true players removed in error.</p><p>')
        for i in range(1,7):paragraphs.append(f'<a href="media/sam3-20261024/rink-filter-review-{i}.jpg">Sheet {i}</a> ')
        paragraphs.append('</p><p><a href="data/reliability/sam3-20261024/rink-filter/development.json">All 19 validation settings</a> · <a href="data/reliability/sam3-20261024/rink-filter/regression.json">Follow-up results</a></p></details>')
    paragraphs.append('<h3>Fresh Pittsburgh detection audit</h3><p>Eleven fixed frames from recording 42:00–42:30 were completely annotated before inspecting these SAM3 predictions: 77 visible players and goalies. This recording was excluded from the detector training, validation and original detection regression. The recording had already been used for tracking. Existing YOLO proposals assisted the new annotations, with one manually added missing player, so its comparison is biased toward YOLO. No model, threshold or filter was selected using this audit.</p>')
    fresh_names={'original':'Pretrained SAM3','transformer-lr1e5':'Fine-tuned SAM3','fine-tuned-rink':'Fine-tuned SAM3 + rink filter','yolo-pipeline':'YOLO proposals · confidence 0.1','yolo-frozen-detector':'YOLO detector · frozen confidence 0.65'}
    fresh_rows=[]
    for r in fresh_detection['results']+[fresh_baseline]:
        m=r['metrics'];fresh_rows.append([fresh_names[r['variant']],m['tp'],m['fp'],m['fn'],pct(m['precision']),pct(m['recall']),pct(m['f1'])])
    paragraphs.append(table(['Detector','Found','False','Missed','Precision','Recall','F1'],fresh_rows))
    paragraphs.append('<p>YOLO is shown at both the tracking proposal cutoff (0.1) and its detector threshold (0.65), frozen on October 18 using the separate validation recording. The 0.65 comparison was added after the audit results to avoid comparing SAM3 only against low-confidence proposals; no threshold search was performed.</p>')
    paragraphs.append('<details><summary>Inspect all fresh audit frames</summary><p>All eleven frames, in time order. Columns: reference, pretrained SAM3, fine-tuned SAM3, fine-tune plus rink filter, and YOLO. Blue: reference; green: matched; red: unmatched prediction. Partial visible players remain in the reference.</p>')
    for r in read('fresh-detection/review.json')['files']:
        paragraphs.append(f'<img src="media/sam3-20261024/{escape(r["name"])}" loading="lazy" alt="Fresh detection audit: {escape(r["name"])}" style="display:block;max-width:100%;height:auto;margin:12px 0">')
    paragraphs.append('<p><a href="data/reliability/sam3-20261024/fresh-detection/report.json">Audit report</a></p></details>')
    paragraphs.append('<details><summary>How the fine-tuning was done</summary><p>196 complete reviewed frames from three NBHL recordings, with horizontal flips (392 training examples). Validation: 23 frames from a fourth recording. Six weight-training configurations were tried. The selected checkpoint updates the grounding transformer and scoring layers; text, the visual backbone, masks and temporal tracking weights stay frozen. The separate visual-backbone experiment also trained the final four attention blocks.</p><p>AdamW, Hungarian box assignment, L1 and generalized-IoU box losses, and focal classification loss. The selected run used learning rate 0.00001 and epoch 2. Only validation chose the epoch and confidence threshold. Gradient checks, changed-weight counts and hashes of every frozen parameter verify that training happened and frozen weights stayed fixed.</p>')
    rows=[]
    for c in selection['candidates']:
        variant=c['variant'];r=read('finetune/'+variant+'.json') if variant!='original' else None
        rows.append([variant,c['epoch'],f'{r["trainable_parameters"]:,}' if r else '0',pct(c['validation']['selected']['f1']),c['validation']['selected']['confidence_threshold']])
    paragraphs.append(table(['Training configuration','Selected epoch','Trainable parameters','Validation F1','Threshold'],rows)+'</details>')
    paragraphs.append('<details><summary>Inspect every detection test frame</summary><p>All 36 frames in fixed order. Blue: reviewed reference boxes. Green: matched predictions. Red: false positives. Left to right: reference, pretrained SAM3, fine-tuned SAM3.</p>')
    for r in review['files']:
        paragraphs.append(f'<img src="media/sam3-20261024/{escape(r["name"])}" loading="lazy" alt="Reference and model predictions: {escape(r["name"])}" style="display:block;max-width:100%;height:auto;margin:12px 0">')
    paragraphs.append('</details><h2>Player tracking</h2><p>Four 30-second regression clips and a separate five-minute clip. Correct means a reviewed same-player pair received the same track ID. Wrong means a reviewed different-player pair was joined. Unverified joins have uncertain reference identity. These sparse checks do not measure continuous full-game accuracy.</p>')
    rows=[]
    for r in tracking:
        a,b=r['short_counts'],r['stress_counts']
        rows.append([r['label'],counts_text(a),a['wrong'],a['unknown_same'],counts_text(b),b['wrong'],b['unknown_same']])
    paragraphs.append(table(['Method','30s clips: correct','Wrong','Unverified','5min: correct','Wrong','Unverified'],rows))
    paragraphs.append('<p>All settings were selected on nine development clips before their new regression scores were inspected. Previous campaigns had already used the recordings and references. A better regression result here still needs independent continuous annotation.</p><details><summary>What was combined and how settings were selected</summary><p>Detector anchoring assigns SAM3 masks to the current detector’s boxes with one-to-one overlap matching. Appearance checks split a fragment when several clear observations disagree. Consensus variants use SAM3 agreement when reconnecting SAM2 fragments, with or without Qwen image embeddings. Native experiments change association, mask reconditioning and track lifetime. Visual-prompt experiments use an automatically selected initial player box; the persistent variant keeps its encoded visual template available on later frames.</p><p>Selection maximizes correct − 10 × wrong, then observation coverage, then fewer unverified joins. Each family keeps one development winner. All tested alternatives remain in the linked evidence.</p>')
    devrows=[]
    for r in hybrid_selection['winners']:
        c=r['counts'];devrows.append([NAMES[r['family']],r['key'],counts_text(c),c['wrong'],c['unknown_same']])
    for f,r in videos.items():
        b=r['selection']['best'];c=b['counts'];setting=b['variant']+(' / '+b['delta'] if b['delta'] else '');devrows.append([NAMES[f],setting,counts_text(c),c['wrong'],c['unknown_same']])
    paragraphs.append(table(['Family','Development winner','Correct','Wrong','Unverified'],devrows)+'</details>')
    paragraphs.append('<h2>Long replay and identity audit</h2><p>The 20m 15s replay covers recording 29:45–50:00. It uses four 315-second executions with 15-second overlaps. Shared frames were checked byte-for-byte. IDs are reconciled only when simultaneous overlapping boxes agree; gaps remain gaps. The displayed labels are automatic fragments, not jersey numbers, and masks can drift or persist behind occlusion.</p><p>The method selector also offers the existing SAM2 runs with generic and trained appearance for the complete interval. Those runs are newly exported, without retraining or manual correction. Full-game identity accuracy remains unverified for all three methods.</p>')
    paragraphs.append('<p>A new 42:00–42:30 interval was labeled before looking at either tracker’s IDs. Four initially visible players were checked once per second, giving 88 visible observations; 29 out-of-view and seven uncertain observations were excluded. Both trackers had the same 135 seconds of preceding context. The recording was reused and boxes were assisted by detector proposals.</p>')
    rows=[]
    for r in fresh['results']:
        if r['iou_threshold']!=.5:continue
        c=r['sampled_transitions'];rows.append([NAMES.get(r['method'],r['method']),f'{r["matched_observations"]}/{r["visible_observations"]}',c.get('same_track',0),c.get('identity_change',0),c.get('missing_one',0)+c.get('missing_both',0)])
    paragraphs.append(table(['Method','Visible observations matched','Same ID between samples','ID changes','Missing transitions'],rows))
    paragraphs.append('<p><a href="tracking-replay.html?clip=pittsburgh-sam3-20min&amp;method=sam3-long&amp;t=735">Inspect the new audit interval</a>. This short sample does not establish accuracy for the full replay.</p><details><summary>Runtime and memory</summary><p>Frames are decoded through a four-frame cache. Old mask memories move to CPU while remaining available to the official memory reader. Control runs matched all IDs, boxes and scores across 1,200 frames. CPU memory remains substantial; this is not a low-memory or real-time system.</p>')
    rows=[]
    for r in long['inputs']:
        rows.append([r['clip'].split('-')[2],f'{r["inference_seconds"]/60:.1f}',f'{r["peak_gpu_gib"]:.1f}',f'{r["peak_process_rss_kib"]/1024**2:.1f}'])
    paragraphs.append(table(['Recording start (seconds)','Inference minutes','Peak GPU GiB','Peak process RAM GiB'],rows)+'</details>')
    paragraphs.append('<h2>Evidence</h2><p><a href="data/reliability/sam3-20261024/results.json">Results JSON</a> · <a href="https://github.com/rhotter/hockey-tracking/tree/main/experiments/sam3-20261024">Protocols, source and reproduction notes</a> · <a href="sam3.html">Previous SAM3, SAM 3.1 and body-pose experiments</a></p><p>Weights are not redistributed. These experiments do not establish reliable time on floor, distance, shots, goals, assists or possession statistics.</p>')
    html='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SAM3 follow-up · Sport Scans</title><link rel="stylesheet" href="stats.css"></head><body class="research"><header><a class="brand" href="./">Sport Scans</a><a href="tracking.html">All measured results</a></header><main>'+''.join(paragraphs)+'</main><footer>Sport Scans · Research prototype</footer></body></html>\n'
    (ROOT/'web/sam3-improvements.html').write_text(html)
    print(json.dumps({'tracking_methods':len(tracking),'detectors':len(detection),'long_frames':video['frames']}))


if __name__=='__main__':main()
