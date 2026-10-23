"""Recompute SAM3 pair counts, selection and pose-emission counts from saved data."""
from pathlib import Path
import json,hashlib,math
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'web/data/reliability/sam3'
def read(name):return json.loads((DATA/name).read_text())
def verify_tracking(rows,totals,prefix,reference):
 summed={}
 for row in rows:
  clip=row['clip'];metrics=read(clip+('-sam31' if prefix=='sam31' else '')+'-pairs.json')
  c={'correct':0,'wrong':0,'unknown_same':0,'same_total':0,'different_total':0,'missing':0}
  for pair in metrics['pairs']:
   original=reference[pair['id']]
   assert all(pair[k]==original[k] for k in ['a','b','truth','clip'])
   pred=pair['predicted_same'];truth=pair['truth']
   assert pred is None or type(pred) is bool
   c['same_total']+=truth=='same';c['different_total']+=truth=='different';c['missing']+=pred is None
   if pred:c[{'same':'correct','different':'wrong','uncertain':'unknown_same'}[truth]]+=1
  for key,value in c.items():
   assert value==metrics['counts'][key]==row['counts'][key],(clip,key)
   summed[key]=summed.get(key,0)+value
  replay=json.loads((ROOT/'web/media/tracking-fourhour'/f'{clip}-{prefix}-native.json').read_text())
  assert replay['clip']==clip and replay['method']==prefix+'-native' and replay['fps']==15 and len(replay['frames'])==450
  assert sum(map(len,replay['frames']))==metrics['observations']
  assert len(replay['track_id_map'])==metrics['fragments']
  assert len(set(replay['track_id_map'].values()))==len(replay['track_id_map'])
  for pair in metrics['pairs']:
   for side in ['a','b']:
    if pair['track_'+side] is not None:assert pair['track_'+side] in replay['track_id_map']
  for frame in replay['frames']:
   assert len({box[0]for box in frame})==len(frame)
   assert all(len(b)==5 and all(isinstance(v,int)for v in b) and 0<=b[1]<b[3]<=10000 and 0<=b[2]<b[4]<=10000 for b in frame)
 for key,value in summed.items():assert value==totals[key]
 return summed

def main():
 report=read('results.json');tracking=report['tracking']
 ref=json.loads((ROOT/'web/data/reliability/tracking-four-hour-final-reference-v1.json').read_text())
 reference={r['id']:r for r in ref['pairs']}
 summed=verify_tracking(tracking['native_regression'],tracking['native_totals'],'sam3',reference)
 variants=1
 if 'sam31_regression' in tracking:
  verify_tracking(tracking['sam31_regression'],tracking['sam31_totals'],'sam31',reference);variants+=1
 geometry=tracking['sam31_geometry_selection']
 path=DATA/'sam31-geometry-tracking-development-person-20261023.json'
 assert hashlib.sha256(path.read_bytes()).hexdigest()==geometry['source_sha256']
 raw=json.loads(path.read_text())
 for candidate in geometry['development']:
  values=[next(v for v in r['variants'] if v['method']==candidate['method'])for r in raw]
  assert candidate['counts']=={k:sum(v['counts'][k]for v in values)for k in candidate['counts']}
  assert candidate['objective']==candidate['counts']['correct']-10*candidate['counts']['wrong']
 winner=max(geometry['development'],key=lambda r:(r['objective'],r['counts']['matched_observations'],-r['counts']['unknown_same'],-r['order']))
 assert winner['method']==geometry['selected']=='final-mask'
 assert geometry['run_regression'] is False and winner['objective']==geometry['development'][0]['objective']
 selection=report['selection'];candidates=selection['development']
 chosen=max(candidates,key=lambda r:(r['counts']['correct']-10*r['counts']['wrong'],r['counts']['matched_observations'],-r['counts']['unknown_same'],r['prompt']=='person'))
 assert chosen['prompt']==selection['prompt']=='person'
 for candidate in candidates:
  p=DATA/('sam3-tracking-development-'+candidate['prompt'].replace(' ','-')+'-20261023.json')
  assert hashlib.sha256(p.read_bytes()).hexdigest()==candidate['source_sha256']
  values=json.loads(p.read_text())
  for k,v in candidate['counts'].items():assert v==sum(r['counts'][k]for r in values)
 reference=json.loads((ROOT/'experiments/sam3-20261023/map-feet-visibility-v1.json').read_text())
 predicted=read('sam3d-body-predictions-20261023.json');groups={}
 assert len(reference['observations'])==len(predicted['observations'])==48
 for a,b in zip(reference['observations'],predicted['observations']):
  assert a['id']==b['id'] and a['frame_sha256']==b['frame_sha256']
  g=groups.setdefault(a['visibility'],{'total':0,'foot_emitted':0});g['total']+=1
  emitted=b['ankle_pixels'] is not None;g['foot_emitted']+=emitted
  if emitted:assert all(math.isfinite(v)for point in b['ankle_pixels']for v in point)
 assert groups==report['body']['groups']
 print(json.dumps({'regression_clips':4,'tracking_frames':1800*variants,'native_variants':variants,'pair_counts':summed,'pose_boxes':48,'selection_verified':True}))
if __name__=='__main__':main()
