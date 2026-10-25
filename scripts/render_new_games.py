"""Render the frozen new-game comparison and extend the existing replay viewer."""
from pathlib import Path
import html
import json

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'web/data/reliability/new-games-20261025'
LABELS={'existing-sam2-osnet-qwen2b':'Existing: SAM2 + OSNet + Qwen 2B','roboflow-rfdetr-botsort':'RF-DETR + Roboflow BoT-SORT','roboflow-yolo-botsort':'YOLO + Roboflow BoT-SORT','existing-sam2-local':'SAM2 without reconnecting'}
def table(headers,rows,identifier):
    return '<div class="table-wrap"><table id="'+identifier+'"><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>'for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>'for c in row)+'</tr>'for row in rows)+'</tbody></table></div>'
def main():
    result=json.loads((DATA/'results.json').read_text());reference=json.loads((DATA/'reference.json').read_text())
    analysis=json.loads((DATA/'analysis.json').read_text())
    rows=[]
    for r in result['methods']:
        c=r['counts'];rows.append([LABELS[r['method']],f"{c['correct']} / {c['same_total']}",f"{c['wrong']} / {c['different_total']}",c['unknown_same'],f"{c['matched_observations']} / {c['reference_observations']}"])
    pieces=['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>New NBHL games · Sport Scans</title><link rel="stylesheet" href="stats.css"><link rel="stylesheet" href="roboflow.css"></head><body class="research roboflow"><header><a class="brand" href="./">Sport Scans</a><span>NBHL ball hockey</span></header><main>',
      '<p><a href="roboflow.html">← Earlier Roboflow comparison</a></p><h1>Tracking on new NBHL games</h1>',
      '<p>Two previously unused recordings. Four fixed 90-second intervals at 10:00 and 30:00, totaling six minutes and 5,400 frames. Models and settings were fixed before raw-frame review. No fine-tuning or parameter selection used these games.</p>',
      '<p><strong>SAM2 without reconnecting was safer on these checks.</strong> The appearance-based reconnection stage recovered more known connections but introduced identity mistakes. Its earlier results did not carry over cleanly to these recordings.</p>',
      table(['Method','Correct connections','Wrong pair joins','Unverified joins','Matched observations'],rows,'new-games-totals'),
      f'<p>Compared with SAM2 alone, reconnecting recovered <strong>{analysis["same_pairs_recovered"]} additional known same-player connections</strong>, adding {analysis["wrong_pair_joins_added"]} wrong pair joins and {analysis["unverified_pair_joins_added"]} unverified joins. Reconnecting left the boxes unchanged. Of the remaining missed same-player checks, {analysis["remaining_same_pairs_missing_endpoint"]} lack a matched endpoint and {analysis["remaining_same_pairs_different_ids"]} have matched boxes with different IDs.</p>',
      '<p class="note">Sparse assistant-reviewed reference, with no independent human verification. These counts describe sampled identity connections, not full-game tracking accuracy. A connection means two selected appearances receive the same track ID, through continuous tracking or reconnection. Wrong merges join known different people; unverified joins have uncertain reference identity. Multiple pair errors can come from one tracking mistake. Track IDs are fragments, not roster identities.</p>',
      '<details><summary>Short versus longer identity connections</summary>'+table(['Method','Correct ≤2s apart','Correct >2s apart','Wrong >2s apart'],[[LABELS[r['method']],f"{r['up_to_2s']['correct']} / {r['up_to_2s']['same_total']}",f"{r['over_2s']['correct']} / {r['over_2s']['same_total']}",r['over_2s']['wrong']]for r in result['methods']],'new-games-separation')+'<p>These are time separations between sampled observations, not verified occlusion durations. Most confirmed identities are supported by short-term continuity.</p></details>',
      '<h2>Watch the comparisons</h2><p>Each replay contains the full reserved interval, with automatic boxes and method switching. Reference images below use green boxes and are separate from model output.</p>']
    catalog={}
    source={s['id']:s for s in result['sources']}
    for c in result['clips']:
        name=c['clip'];title=source[c['source_id']]['title'];clock=f"{c['start_s']//60}:00";short='Pittsburgh finals'if name.startswith('pittsburgh')else'South Jersey finals'
        href=f"tracking-replay.html?clip={name}&amp;method=existing-sam2-osnet-qwen2b"
        pieces.append(f'<section><h3>{html.escape(short)} · {clock}</h3><p>{html.escape(title)}</p><p><a class="new-game-replay" href="{href}">Watch 90-second comparison</a> · <a href="https://www.youtube.com/watch?v={c["source_id"]}&amp;t={c["start_s"]}">Original recording</a></p>')
        cliprows=[]
        for r in result['per_clip']:
            if r['clip']==name:
                x=r['counts'];cliprows.append([LABELS[r['method']],f"{x['correct']} / {x['same_total']}",x['wrong'],x['unknown_same'],f"{x['matched_observations']} / {x['reference_observations']}"])
        pieces.append(table(['Method','Correct','Wrong','Unverified','Matched'],cliprows,'new-games-'+name))
        pieces.append('<details><summary>Review the reference photos</summary><p>These green boxes were drawn on raw frames before inference. Local identity labels record only what could be established visually; a question mark means unknown identity. The labels are selected observations, not exhaustive player detections.</p>')
        for f in c['reference_frames']:
            seconds=c['start_s']+f/15;stamp=f'{int(seconds)//60}:{seconds%60:05.2f}'
            pieces.append(f'<figure><img loading="lazy" src="media/new-games-20261025/{name}-reference-{f:04d}.jpg" alt="Assistant reference boxes for {html.escape(short)} at recording second {seconds:g}" style="max-width:100%;height:auto"><figcaption>Recording {stamp} · <a href="{href}&amp;t={f/15:g}">Compare automatic tracking at this moment</a></figcaption></figure>')
        pieces.append('</details></section>')
        catalog[name]={'label':f'{short} · {clock} · new test','description':title,'sourceId':c['source_id'],'frames':c['duration_s']*15,'folder':'media/new-games-20261025','methods':list(LABELS),'methodLabels':LABELS,'default':'existing-sam2-osnet-qwen2b','methodDescriptions':{m:'Frozen comparison on a previously unused NBHL recording. Automatic fragment IDs and currently observed detector boxes. No manual corrections, off-camera interpolation or verified roster identities. See the new-games report for sparse reference scores.'for m in LABELS}}
    pieces.append('<details><summary>A concrete failure: repeated scene resets</summary><p>In the first Pittsburgh interval, the scene-change heuristic resets six times around recording 10:21–10:22. Sampled raw frames show continuous close-view play with blue #96. Seven known same-person reference pairs cross this boundary, but the current linker rejects connections across scene labels. This is a specific weakness to address; it is not proof that removing the resets alone would recover all seven checks.</p><p><a href="tracking-replay.html?clip=pittsburgh-finals-0600&amp;method=existing-sam2-osnet-qwen2b&amp;t=21">Inspect the moment</a> · <a download href="data/reliability/new-games-20261025/diagnostic-scene-review.json">Post-test review notes</a></p><p>The footage, reference and settings were retained unchanged. This diagnosis was made after inference.</p></details>')
    pieces.append('<details><summary>Protocol, separation and evidence</summary><p>The two recordings are absent from the prior source registry. They were selected from the official NBHL channel before viewing the footage, using different regions; venues and teams may overlap prior recordings. All four intervals are retained regardless of activity or model outcome.</p><p>The existing pipeline uses the unchanged NBHL YOLO checkpoint, detector-supported SAM2 Large, generic OSNet body-region features and Qwen3-VL Embedding 2B joint-image blending. Roboflow BoT-SORT uses its earlier development-selected defaults, with either the same YOLO detections or the existing NBHL RF-DETR checkpoint. Both detectors remain frozen. All methods receive the same scene-reset signals.</p><p>The reference was sealed before inference. Near-in-time continuity supports some identity groups; uncertain longer-gap identity remains unknown. These labels still need independent human review. No time on floor, distance, shots or possession is validated by this test.</p>')
    pieces.append('<div class="downloads">'+''.join(f'<a download href="data/reliability/new-games-20261025/{n}.json">{label}</a>'for n,label in [('results','Results'),('analysis','Reconnection changes'),('reference','Reference labels'),('protocol','Frozen protocol'),('source-review','Source review')])+'<a href="https://github.com/rhotter/hockey-tracking/tree/main/experiments/new-games-20261025">Experiment code</a></div></details></main><footer>Sport Scans · Research prototype</footer></body></html>')
    (ROOT/'web/new-games.html').write_text('\n'.join(pieces)+'\n')
    (ROOT/'web/tracking-new-games-catalog.js').write_text('Object.assign(window.SportScansTrackingCatalog, '+json.dumps(catalog,separators=(',',':'))+');\n')

if __name__=='__main__':main()
