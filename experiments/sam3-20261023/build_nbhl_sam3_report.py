"""Assemble SAM3 experiment results and a plain public comparison page."""
import hashlib,json,shutil,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT.parent/'hockey-tracking'
NOTES=ROOT/'notes/ball-hockey'
DEST=PUBLIC/'web/data/reliability/sam3'
def read(name):return json.loads((NOTES/name).read_text())
def total(rows,key='counts'):
 keys=rows[0][key].keys();return {k:sum(r[key][k] for r in rows) for k in keys}
def main():
 DEST.mkdir(parents=True,exist_ok=True)
 selected=json.loads((ROOT/'data/ball-hockey/sam3-tracking-selection-20261023.json').read_text())
 prompt=selected['prompt'];tag=prompt.replace(' ','-')
 detection=read('sam3-detection-regression-hockey-player-20261023.json')['prompts']['hockey player']['at_selected_threshold']
 development={p:read('sam3-tracking-development-'+p.replace(' ','-')+'-20261023.json') for p in ['person','hockey player']}
 regression=read('sam3-tracking-regression-'+tag+'-20261023.json')
 hybrid=read('sam3-supported-development-person-20261023.json')
 hybrid_reg=NOTES/'sam3-supported-regression-person-20261023.json'
 body=read('sam3d-body-review-20261023.json')
 baseline_reports=[read('heldout-v1-'+s+'-selected_finetuned.json') for s in ['1f0Y5BIr5Ps','5Qw_g9E7EqE']]
 det_base=total(baseline_reports,'metrics')
 # Recompute the existing best pipeline on exactly these four short clips.
 outcomes=json.loads((PUBLIC/'results/vlm-reconnect-pair-outcomes.json').read_text())['phases']['regression']
 best=[]
 for clip in [r['clip'] for r in regression]:
  pairs=[(r,v) for r,v in zip(outcomes['references'],outcomes['embedding_models']['2b']['joint']) if r['clip']==clip]
  best.append({'clip':clip,'counts':{'correct':sum(r['truth']=='same' and v=='1' for r,v in pairs),'wrong':sum(r['truth']=='different' and v=='1' for r,v in pairs),'unknown_same':sum(r['truth']=='uncertain' and v=='1' for r,v in pairs),'same_total':sum(r['truth']=='same' for r,v in pairs),'different_total':sum(r['truth']=='different' for r,v in pairs),'missing':sum(v=='?'for r,v in pairs)}})
 dev_outcomes=json.loads((PUBLIC/'results/vlm-reconnect-pair-outcomes.json').read_text())['phases']['development']
 dev_clips={r['clip']for r in development['person']}
 pairs=[(r,v)for r,v in zip(dev_outcomes['references'],dev_outcomes['embedding_models']['2b']['joint']) if r['clip']in dev_clips]
 best_dev={'correct':sum(r['truth']=='same' and v=='1' for r,v in pairs),'wrong':sum(r['truth']=='different' and v=='1' for r,v in pairs),'unknown_same':sum(r['truth']=='uncertain' and v=='1' for r,v in pairs),'same_total':sum(r['truth']=='same' for r,v in pairs),'different_total':sum(r['truth']=='different' for r,v in pairs),'missing':sum(v=='?'for r,v in pairs)}
 report={'id':'nbhl-sam3-20261023-v1','scope':'Reused NBHL recordings and sparse assistant-reviewed references; not independent full-game accuracy.','training':'None; pretrained official checkpoints.','selection':selected,'detection':{'sam3':detection,'fine_tuned_detector':{k:det_base[k]for k in ['frames','targets','tp','fp','fn']}},'tracking':{'development':development,'native_regression':regression,'native_totals':total(regression),'raw_sam2_totals':total(regression,'baseline_raw_sam2'),'best_existing':best,'best_existing_totals':total(best),'hybrid_development':hybrid,'hybrid_development_totals':total(hybrid),'hybrid_regression':json.loads(hybrid_reg.read_text()) if hybrid_reg.exists() else None},'body':body,'limitations':['Native SAM3 changes both detections and IDs; it is not an association-only comparison.','SAM3 can segment officials and spectators, and may emit inferred masks behind occlusion. Sparse identity pairs do not measure every false positive or switch.','Pose emission is not accuracy or visibility; no metric distance, time on floor or roster identity is established.']}
 if hybrid_reg.exists():report['tracking']['hybrid_regression_totals']=total(report['tracking']['hybrid_regression'])
 pair_lookup={r['id']:v for r,v in zip(outcomes['references'],outcomes['embedding_models']['2b']['joint'])}
 comparisons=[];errors=[]
 for row in regression:
  pairs=json.loads((DEST/(row['clip']+'-pairs.json')).read_text())['pairs']
  comparisons.append({'clip':row['clip'],'gained_same_pairs':sum(p['truth']=='same' and p['predicted_same'] is True and pair_lookup[p['id']]!='1' for p in pairs),'lost_same_pairs':sum(p['truth']=='same' and p['predicted_same'] is not True and pair_lookup[p['id']]=='1' for p in pairs)})
  errors.extend(p for p in pairs if p['truth']=='different' and p['predicted_same'] is True)
 report['tracking']['posthoc_pair_comparison']=comparisons
 report['tracking']['wrong_joins']=errors
 report['tracking']['best_existing_development_totals']=best_dev
 if (NOTES/'sam31-tracking-regression-person-20261023.json').exists():
  report['tracking']['sam31_regression']=read('sam31-tracking-regression-person-20261023.json')
  report['tracking']['sam31_totals']=total(report['tracking']['sam31_regression'])
  report['tracking']['sam31_development']=read('sam31-tracking-development-person-20261023.json')
 if (NOTES/'sam31-cap128-tracking-development-person-20261023.json').exists():
  report['tracking']['sam31_capacity_followup']=read('sam31-cap128-tracking-development-person-20261023.json')
 if (NOTES/'sam31-geometry-tracking-development-person-20261023.json').exists():
  report['tracking']['sam31_geometry_development']=read('sam31-geometry-tracking-development-person-20261023.json')
  report['tracking']['sam31_geometry_selection']=json.loads((ROOT/'data/ball-hockey/sam31-geometry-selection-20261023.json').read_text())
 (DEST/'results.json').write_text(json.dumps(report,indent=2)+'\n')
 def pct(v):return f'{100*v:.1f}%'
 def detrow(name,m):
  tp,fp,fn=m['tp'],m['fp'],m['fn'];return f'<tr><td>{name}</td><td>{tp}</td><td>{fp}</td><td>{fn}</td><td>{pct(tp/(tp+fp))}</td><td>{pct(tp/(tp+fn))}</td><td>{pct(2*tp/(2*tp+fp+fn))}</td></tr>'
 def trackrow(name,c):return f'<tr><td>{name}</td><td>{c["correct"]} / {c["same_total"]}</td><td>{c["wrong"]} / {c["different_total"]}</td><td>{c["unknown_same"]}</td><td>{c["missing"]}</td></tr>'
 tr=report['tracking'];rows=trackrow('Raw detector-supported SAM2',tr['raw_sam2_totals'])+trackrow('Current best: SAM2 + OSNet + joint-image Qwen 2B',tr['best_existing_totals'])+trackrow('Native SAM3 · '+html.escape(prompt),tr['native_totals'])
 if hybrid_reg.exists():rows+=trackrow('Our detector + SAM3 tracker',tr['hybrid_regression_totals'])
 if 'sam31_totals' in tr:rows+=trackrow('Native SAM 3.1 Object Multiplex · person',tr['sam31_totals'])
 native=tr['native_totals'];old=tr['best_existing_totals'];delta=native['correct']-old['correct']
 conclusion=f'Native SAM3 scored {native["correct"]}/{native["same_total"]} correct same-player pairs with {native["wrong"]} wrong joins, compared with {old["correct"]}/{old["same_total"]} and {old["wrong"]} for the current best pipeline. '
 conclusion+='The existing pipeline remains the recommended research baseline; these reused sparse checks are insufficient to claim reliable full-game tracking.'
 if 'sam31_totals' in tr:
  c=tr['sam31_totals'];conclusion+=f' The newer SAM 3.1 scored {c["correct"]}/{c["same_total"]} with {c["wrong"]} wrong joins.'
 links=''.join(f'<li><a href="tracking-replay.html?clip={r["clip"]}&amp;method=sam3-native">{html.escape(r["clip"])} · SAM3 replay</a></li>'for r in regression)
 devrows=''.join(trackrow('Native SAM3 · '+p,total(r))for p,r in development.items())+trackrow('Our detector + SAM3 tracker',total(hybrid))+trackrow('Raw SAM2 on the same development clips',total(hybrid,'baseline_raw_sam2'))
 if 'sam31_development' in tr:devrows+=trackrow('Native SAM 3.1 Object Multiplex · person',total(tr['sam31_development']))
 devrows+=trackrow('Current best: SAM2 + OSNet + joint-image Qwen 2B',best_dev)
 source='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SAM3 experiments · Sport Scans</title><link rel="stylesheet" href="stats.css"></head><body class="research"><header><a class="brand" href="./">Sport Scans</a><a href="tracking.html">All measured results</a></header><main><h1>SAM3, SAM 3.1 and SAM 3D Body</h1><p>October 23, 2026 · NBHL ball hockey · Official pretrained checkpoints on Modal GPUs</p>'''
 source+=f'<p>{conclusion}</p><h2>Player tracking</h2><p>Four 30-second clips: two Pittsburgh and two Maryland women’s clips. Same frozen pair references as the current baseline. Correct means a reviewed same-player pair was connected; wrong means a reviewed different-player pair was connected. “Unverified” joins lack a definitive identity label. Missing means at least one endpoint was unmatched.</p><div class="table-wrap"><table><thead><tr><th>Pipeline</th><th>Correct</th><th>Wrong</th><th>Unverified</th><th>Missing pairs</th></tr></thead><tbody>{rows}</tbody></table></div><p>Native SAM3 uses its own masks and boxes. The hybrid uses our existing detector and detector-supported matching. These are sparse regression checks, not full-game identity accuracy.</p><ul>{links}</ul>'
 coverage=tr['raw_sam2_totals']
 source+=f'<p>SAM3 matched {native["matched_observations"]}/{native["reference_observations"]} reviewed player observations, versus {coverage["matched_observations"]}/{coverage["reference_observations"]} for the detector-supported baseline. Across the four clips it recovered eight same-player connections missed by the best pipeline, but lost eight others. The coverage gain is real on these references; the identity tradeoff prevents a default switch.</p>'
 perclip=''.join(f'<tr><td>{r["clip"]}</td><td>{b["counts"]["correct"]} / {r["counts"]["same_total"]}</td><td>{r["counts"]["correct"]} / {r["counts"]["same_total"]}</td><td>{r["counts"]["wrong"]}</td></tr>'for r,b in zip(regression,best))
 source+=f'<details><summary>Results by clip</summary><div class="table-wrap"><table><thead><tr><th>Clip</th><th>Current best correct</th><th>SAM3 correct</th><th>SAM3 wrong joins</th></tr></thead><tbody>{perclip}</tbody></table></div></details>'
 reference=json.loads((PUBLIC/'web/data/reliability/tracking-four-hour-final-reference-v1.json').read_text())
 obs={r['id']:r for r in reference['observations']};frames={(r['clip'],r['frame']):r for r in reference['frames']}
 source+='<details id="sam3-errors"><summary>Inspect SAM3’s three wrong identity joins</summary><p>Each pair below contains two different reviewed players that received the same SAM3 ID. Yellow boxes mark the reference player, not the predicted mask. Open the replay to inspect the transition.</p>'
 for pair in errors:
  replay=json.loads((PUBLIC/'web/media/tracking-fourhour'/(pair['clip']+'-sam3-native.json')).read_text())
  display_id=replay['track_id_map'][pair['track_a']]
  source+=f'<p>{pair["clip"]} · replay ID {display_id}</p><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px">'
  for side in ['a','b']:
   o=obs[pair[side]];f=frames[(o['clip'],o['frame'])];x,y,u,v=o['bbox']
   source+=f'<figure style="margin:0"><div style="position:relative"><img src="data/reliability/four-hour-reference-frames/{o["clip"]}-{o["frame"]:04d}.jpg" loading="lazy" alt="Reference player at clip second {o["frame"]/15:.2f}" style="display:block;width:100%"><svg viewBox="0 0 {f["width"]} {f["height"]}" style="position:absolute;inset:0;width:100%;height:100%" aria-hidden="true"><rect x="{x}" y="{y}" width="{u-x}" height="{v-y}" fill="none" stroke="#ffdf20" stroke-width="4"/></svg></div><figcaption><a href="tracking-replay.html?clip={o["clip"]}&amp;method=sam3-native&amp;t={o["frame"]/15}">Clip {o["frame"]/15:.2f}s · open replay</a></figcaption></figure>'
  source+='</div>'
 source+='</details>'
 if 'sam31_totals' in tr:
  source+='<h3>SAM 3.1 follow-up</h3><p>The newer Object Multiplex checkpoint was tested with the same “person” prompt and 0.5 output threshold, with 64 object slots, 16 objects per memory bucket, and compilation disabled. This follow-up was specified after the original SAM3 regression results were known; there was no new prompt sweep.</p><ul>'
  source+=''.join(f'<li><a href="tracking-replay.html?clip={r["clip"]}&amp;method=sam31-native">{html.escape(r["clip"])} · SAM 3.1 replay</a></li>'for r in tr['sam31_regression'])+'</ul>'
  source+='<p><a href="https://github.com/facebookresearch/sam3/blob/7f473cfd983d5eb43040ff608f66bff0d7bdcb53/RELEASE_SAM3p1.md">Official SAM 3.1 release notes</a>. This run measures identity behavior; it is not an optimized speed benchmark. Replay inspection found several boxes spanning much of the rink. The current box-overlap score penalizes that localization failure as well as broken identities.</p>'
  if 'sam31_capacity_followup' in tr:
   c=tr['sam31_capacity_followup'][0]['counts']
   source+=f'<p>The one-minute development clip reached the 64-object limit. A separate 128-slot follow-up scored {c["correct"]}/{c["same_total"]} correct with {c["wrong"]} wrong joins and {c["matched_observations"]}/{c["reference_observations"]} matched observations. It is not mixed into the original development total. No capacity limit was reached in the four regression clips.</p>'
 source+=f'<details><summary>Development selection and alternatives</summary><p>Two Great Britain clips, 20 and 60 seconds. Native prompt selection maximized correct − 10 × wrong, then coverage, then fewer unverified joins. Chosen prompt: <b>{html.escape(prompt)}</b>. No regression labels were used to select the prompt.</p><div class="table-wrap"><table><thead><tr><th>Variant</th><th>Correct</th><th>Wrong</th><th>Unverified</th><th>Missing pairs</th></tr></thead><tbody>{devrows}</tbody></table></div><p>The hybrid adapter uses normalized SAM3 boxes, per-object states for new arrivals, and shared image features. Earlier API integration failures produced no scored run. Byte-identical local frame staging speeds up cloud reads.</p></details>'
 if 'sam31_geometry_development' in tr:
  variants=tr['sam31_geometry_development']
  geometry_rows=''.join(trackrow(name,total([next(v for v in clip['variants'] if v['method']==name)for clip in variants]))for name in ['original-boxes','final-mask','largest-component'])
  source+='<details><summary>SAM 3.1 mask cleanup experiment</summary><p>Motivated by the oversized boxes seen in regression replays, then compared on both development clips with 64 object slots. No weights or identities changed. “Final mask” recomputes the box after the model’s overlap suppression. “Largest component” keeps only the largest 8-connected region before deriving the box. Original boxes were rescored from the same inference run for a paired comparison.</p><div class="table-wrap"><table><thead><tr><th>Box recipe</th><th>Correct</th><th>Wrong</th><th>Unverified</th><th>Missing pairs</th></tr></thead><tbody>'+geometry_rows+'</tbody></table></div><p>Final-mask boxes matched 66/79 observations versus 63/79 originally, with no extra correct connection. Largest-component boxes matched 73/79 and added four correct connections, but also one wrong join. Neither improved the predeclared correct − 10 × wrong objective, so no cleanup variant was promoted to regression testing.</p></details>'
 source+='<h2>Player detection</h2><p>36 complete reviewed frames across two other recordings, 256 visible floor players and goalies. Officials, bench players and spectators are false positives. SAM3’s “hockey player” prompt and 0.6 threshold were selected on 23 development frames before regression scoring. No manual rink filtering.</p><div class="table-wrap"><table><thead><tr><th>Detector</th><th>Found</th><th>False positives</th><th>Missed</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>'+detrow('Current fine-tuned YOLO11m',det_base)+detrow('SAM3 · hockey player',detection)+'</tbody></table></div><p>SAM3 is not a replacement for the fine-tuned detector on this test. “Person” and “ball hockey player” were also evaluated during development.</p>'
 source+=f'<h2>SAM 3D Body</h2><p>DINOv3 checkpoint: 48 existing player boxes at six fixed frames, processed in {body["seconds"]:.1f} seconds excluding model loading. It produced ankle positions for all 34 visible-foot, seven occluded-foot, and seven unclear cases. RTMW also emitted on all 48 in the earlier audit. More outputs do not establish higher accuracy.</p><p>Raw crop on the left, pose overlay on the right. Cyan: knees; red: ankles. All 48 boxes are shown, including failures. The labels describe visibility only; exact ankle coordinates and ground contact have not been annotated.</p>'
 for n in range(1,5):source+=f'<details><summary>Review sheet {n} · boxes {(n-1)*12}–{n*12-1}</summary><img src="media/sam3/body-review-{n}.jpg" alt="Raw player crops and SAM 3D Body lower-body predictions, review sheet {n}" loading="lazy" style="max-width:100%;height:auto"></details>'
 source+='<p>Pose does not identify players, establish visibility, or give calibrated rink distance. Occluded predictions must not be counted as observed feet.</p><h2>Evidence</h2><p><a href="data/reliability/sam3/results.json">Full results JSON</a> · <a href="https://github.com/rhotter/hockey-tracking/tree/main/experiments/sam3-20261023">Protocol and experiment source</a></p><p>No new fine-tuning. The recordings were already used in earlier experiments. Labels are assistant-reviewed and correlated; independent continuous annotation is still needed before TOI, distance or possession claims.</p></main><footer>Sport Scans · Research prototype</footer></body></html>'
 (PUBLIC/'web/sam3.html').write_text(source)
 print(json.dumps({'native_totals':native,'hybrid_development':tr['hybrid_development_totals'],'best_existing':old}))
if __name__=='__main__':main()
