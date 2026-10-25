"""Render the plain research report from verified result artifacts."""
from pathlib import Path
from html import escape
import json

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'web/data/reliability/roboflow-20261025'
def read(p):return json.loads(p.read_text())
def table(headers,rows,id=''):
    return f'<div class="table-wrap"><table id="{id}"><thead><tr>'+''.join('<th>'+escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def label(method):
    if method=='existing':return 'Existing SAM2 + OSNet + Qwen'
    detector='RF-DETR' if method.startswith('rfdetr') else 'YOLO'
    tracker=next((name for key,name in [('sports_bytetrack','Sports ByteTrack'),('rf_bytetrack','Roboflow ByteTrack'),('rf_ocsort','Roboflow OC-SORT'),('rf_cbiou','Roboflow C-BIoU'),('rf_botsort','Roboflow BoT-SORT')] if key in method),method)
    return detector+' + '+tracker+(' + existing OSNet links' if method.endswith('-appearance') else '')+(' (4s buffer)' if 'long_buffer' in method else '')

def main():
    results=read(DATA/'results.json');selection=read(DATA/'selection.json');teams=read(DATA/'teams.json');torso=read(DATA/'teams-torso.json');utilities=read(DATA/'utilities.json')
    rows=[]
    for method in [results['existing_pipeline'],*results['tracking']]:
        c=method['counts'];rows.append([label(method['method']),f"{c['correct']} / {c['same_total']}",f"{c['wrong']} / {c['different_total']}",c['unknown_same'],f"{c['matched_observations']} / {c['reference_observations']}"])
    rf=selection['winners']['rfdetr_nbhl']['method'];yolo=selection['winners']['yolo_cached']['method']
    selected=next(r for r in results['tracking'] if r['method']==rf)
    hybrid=next(r for r in results['tracking'] if r['method'].endswith('-appearance'))
    teams_rows=[]
    for name,data in [('Stock full-box crop',teams),('Torso crop follow-up',torso)]:
        for r in data['sources']:
            if r['counts'] is None:continue
            c=r['counts'];teams_rows.append(['Pittsburgh' if r['source_id']=='JlBtScHdcg4' else 'Maryland women',name,
                f"{c['correct_all']} / {c['matched']}",f"{c['correct_accepted']} / {c['accepted']}",c['matched']-c['accepted'],c['unmatched']])
    capability=[
      ['Player detection','Implemented','RF-DETR uses our existing NBHL-trained checkpoint. No new weight training in this experiment.'],
      ['Local tracking','Implemented alternative','Five upstream tracker implementations tested with both detectors. Scene resets and current-detection-only output remain in our adapter.'],
      ['Team assignment','Partial','Sports TeamClassifier works well on Pittsburgh; it fails on many Maryland observations, including confident assignments.'],
      ['Rink projection','Replaceable utility','ViewTransformer matches the existing relative transform. Rink landmark detection, camera registration and physical calibration remain custom.'],
      ['Ball tracking','No measured improvement','Sports BallTracker ties the top-confidence detector control on the reviewed visible-ball frames.'],
      ['Jersey identity and reconnecting','Still custom','The hybrid keeps OSNet and our global linker. Repeated number readings still make mistakes.'],
      ['TOI, miles, shots, goals, assists, possession','Unverified','These components do not establish reliable full-game identities or game statistics.']]
    parts=['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Roboflow replacement test · Sport Scans</title><link rel="stylesheet" href="stats.css"><link rel="stylesheet" href="roboflow.css"></head><body class="research roboflow"><header><a class="brand" href="./">Sport Scans</a><a href="tracking.html">All measured results</a></header><main>',
      '<h1>Roboflow replacement test</h1><p>October 25, 2026 · NBHL ball hockey · Actual upstream Sports, Trackers and RF-DETR code, run on Modal.</p>',
      f'<p>We implemented Roboflow alternatives for detection, tracking, team clustering, ball selection and coordinate projection. The development-selected RF-DETR + Roboflow tracker recovered <strong>{selected["counts"]["correct"]}/129</strong> reviewed same-player connections with <strong>{selected["counts"]["wrong"]} known wrong merges</strong>. The existing pipeline recovered 111/129 with zero known wrong merges.</p>',
      '<p>These results support replacing individual utilities and offering a simpler tracking alternative. They do not support replacing the complete identity and statistics pipeline.</p>',
      f'<p><a id="roboflow-replay" href="tracking-replay.html?clip=pittsburgh-stress-0900&amp;method=roboflow-{rf}">Watch the complete five-minute Roboflow replay</a> · <a href="tracking-replay.html?clip=marylandwomen-0600&amp;method=roboflow-{rf}">Maryland comparison</a></p>',
      '<h2>What can be replaced?</h2>',table(['Stage','Result','What remains'],capability,'roboflow-components'),
      '<h2>Player tracking</h2><p>Four 30-second clips plus a five-minute clip: 6,300 frames from Pittsburgh and Maryland women. Correct connections join reviewed observations of the same player; wrong merges join different players. Unverified joins have uncertain reference identity. The reference is sparse and assistant-reviewed, with no independent human verification.</p>',
      table(['Method','Correct connections','Wrong merges','Unverified joins','Matched observations'],rows,'roboflow-tracking'),
      f'<p>The hybrid replaces SAM2 with {escape(label(yolo))} and then uses our unchanged generic OSNet body-region descriptors and global appearance linker. It recovered {hybrid["counts"]["correct"]}/129 connections with {hybrid["counts"]["wrong"]} wrong merges. It does not include Qwen embeddings, and is not an all-Roboflow pipeline.</p>',
      '<details><summary>Selection, implementations and timing</summary><p>Twenty configurations: two detectors × five tracker implementations × two buffer settings. Selection used six Massachusetts development clips (3,300 frames), excluding other leagues, before scoring the five regression clips. The development identity reference is small: 90 observations, 22 known same-player pairs and 46 known different-player pairs. The recordings had already been used in earlier work, so this is reused regression rather than a new untouched test.</p><p>The five implementations are the original Sports tutorial’s Supervision ByteTrack plus Roboflow Trackers’ ByteTrack, OC-SORT, C-BIoU and BoT-SORT. Native defaults use the actual 15 fps; the second profile keeps lost tracks for four seconds. BoT-SORT receives image frames for camera-motion compensation. All displayed positions are current detector boxes; the adapter emits no extrapolated off-camera positions.</p><p>RF-DETR is our previously trained medium/960 NBHL checkpoint. The detector and tracker environments are separate because current RF-DETR requires a newer Supervision release than the original tutorial’s ByteTrack. No new language-model calls or model fine-tuning were used.</p>',
      table(['Detector','Development-selected tracker','Correct','Wrong'],[[d,label(r['method']),r['counts']['correct'],r['counts']['wrong']] for d,r in selection['winners'].items()]),
      '<p>Recorded tracker timings exclude detector inference and cached appearance extraction. They are not an end-to-end speedup claim. <a href="data/reliability/roboflow-20261025/selection.json" download>All 20 development results</a> · <a href="data/reliability/roboflow-20261025/preflight.json" download>Versions and source hashes</a></p></details>',
      '<h2>Team classification</h2><p>The actual Sports TeamClassifier uses SigLIP image features, UMAP and two-cluster KMeans. Each recording supplies up to 512 unlabeled automatic player crops. Team labels are used only to align the cluster names for scoring. The evaluation footage contributes unlabeled fitting samples.</p>',
      table(['Recording','Input','Correct / matched','Correct / accepted','Abstained','Detector missed'],teams_rows,'roboflow-teams'),
      '<p>The confidence gate uses a fixed cluster-distance margin and the training-distance 95th percentile. Its errors show that cluster confidence is not a reliable guarantee. The torso crop was declared after seeing the stock classifier’s failure; it is a follow-up on reused data, not independent validation.</p>',
      '<details><summary>Compare team predictions with the frozen reference</summary><p>Fixed first frames, without cherry-picking. Left: yellow reference boxes, team and review identity. These are sparse checks; unlabeled players are not negatives. Right: stock Roboflow clusters A/B; gray “?” means the confidence gate abstained. Cluster labels are not jersey numbers.</p>'+''.join(f'<a href="media/roboflow-20261025/{c}-teams.jpg"><img src="media/roboflow-20261025/{c}-teams.jpg" loading="lazy" alt="{c}: frozen reference and Roboflow team clusters" style="width:100%;height:auto;margin:12px 0"></a>' for c in ['pittsburgh-0600','marylandwomen-0600'])+'</details>',
      '<h2>Ball, jerseys and rink coordinates</h2>',
      '<p>Sports BallTracker and the highest-confidence detection control each located 85/108 reviewed visible balls, including 11/16 validation observations. Both abstained on 23. Unknown-visibility frames were excluded from scoring; these correlated short windows do not validate shots or possession.</p>',
      '<p>Applying the tutorial-style three-consecutive-reading rule to archived VLM number readings gave 30 correct, three wrong and three abstentions on 36 readable central crops. Thirteen readings on 61 unknown crops remain unverified. This exercised a consensus rule, not a new Roboflow number detector or OCR model.</p>',
      '<p>ViewTransformer reproduced the existing four-point relative rink projection within 0.00001 relative units. This checks software compatibility only. Venue dimensions remain unverified, so miles and speed remain disabled.</p>',
      '<h2>Evidence and source</h2><p><a href="data/reliability/roboflow-20261025/results.json" download>Tracking results</a> · <a href="data/reliability/roboflow-20261025/teams.json" download>Stock teams</a> · <a href="data/reliability/roboflow-20261025/teams-torso.json" download>Torso follow-up</a> · <a href="data/reliability/roboflow-20261025/utilities.json" download>Utility results</a> · <a href="https://github.com/rhotter/hockey-tracking/tree/main/experiments/roboflow-20261025">Protocols and reproduction code</a></p>',
      '<p>Upstream: <a href="https://github.com/roboflow/sports">Roboflow Sports</a>, <a href="https://github.com/roboflow/trackers">Roboflow Trackers</a>, <a href="https://github.com/roboflow/rf-detr">RF-DETR</a>. Weights and raw recordings are not redistributed.</p></main><footer>Sport Scans · Research prototype</footer></body></html>']
    (ROOT/'web/roboflow.html').write_text(''.join(parts)+'\n')
    labels={'roboflow-'+r['method']:label(r['method']) for r in results['tracking']}
    catalog="(() => {\n 'use strict';\n const methods="+json.dumps(labels)+";\n for(const id of "+json.dumps([r['clip'] for r in selected['clips']])+"){\n const clip=window.SportScansTrackingCatalog[id];\n for(const [id,label] of Object.entries(methods)){if(!clip.methods.includes(id))clip.methods.push(id);clip.methodLabels[id]=label;clip.methodDescriptions??={};clip.methodDescriptions[id]='Roboflow experiment on NBHL ball hockey. Current detector boxes with automatic fragment IDs; no jersey identity or off-camera interpolation. See the Roboflow replacement report for measured results and limitations.';}\n }\n})();\n"
    (ROOT/'web/tracking-roboflow-catalog.js').write_text(catalog)

if __name__=='__main__':main()
