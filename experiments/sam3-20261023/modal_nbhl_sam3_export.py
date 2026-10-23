"""Small review artifacts for the SAM3 campaign; no local bulk video transfer."""
from pathlib import Path
import modal,json
from modal_campaign import ROOT,base_image,volume
app=modal.App('sport-scans-sam3-review-export')
BASE=Path('/data/ball-hockey/reliability-20261018')
image=base_image.add_local_file(ROOT/'scripts/modal_campaign.py','/work/modal_campaign.py').add_local_file(ROOT/'data/ball-hockey/map-feet-visibility-v1.json','/work/foot-reference.json')
@app.function(image=image,cpu=4,memory=8192,timeout=600,volumes={'/data':volume})
def poses():
 from PIL import Image,ImageDraw
 import hashlib
 ref=json.loads(Path('/work/foot-reference.json').read_text())
 pred=json.loads((BASE/'sam3d-body-foot-audit-v1/predictions.json').read_text())
 out=BASE/'sam3-20261023-v1/review';out.mkdir(parents=True,exist_ok=True)
 reports=[]
 for sheet in range(4):
  canvas=Image.new('RGB',(1152,1040),'white');draw=ImageDraw.Draw(canvas)
  for n in range(12):
   idx=sheet*12+n;r=ref['observations'][idx];p=pred['observations'][idx]
   im=Image.open(BASE/'relative-map-v1/nj-0900-20s/frames'/f'{r["frame"]+1:06d}.jpg').convert('RGB')
   x,y,u,v=r['bbox'];h=v-y;box=(max(0,int(x-.12*(u-x))),max(0,int(y-.03*h)),min(im.width,int(u+.12*(u-x))),min(im.height,int(v+.15*h)))
   crop=im.crop(box);overlay=crop.copy();dr=ImageDraw.Draw(overlay)
   for j in [11,12,13,14]:
    px,py=p['keypoints_2d'][j];px-=box[0];py-=box[1];c='red' if j in [13,14] else 'cyan';dr.ellipse((px-3,py-3,px+3,py+3),outline=c,width=2)
   xx=n%3*384;yy=n//3*260
   draw.text((xx+4,yy+3),f'#{idx} frame {r["frame"]} | {r["visibility"]}',fill='black')
   for col,im in enumerate([crop,overlay]):
    im.thumbnail((186,230));canvas.paste(im,(xx+col*192+(192-im.width)//2,yy+25))
  name=f'body-review-{sheet+1}.jpg';canvas.save(out/name,quality=92)
  reports.append({'name':name,'sha256':hashlib.sha256((out/name).read_bytes()).hexdigest()})
 volume.commit();return {'sheets':reports,'keypoint_names':pred['keypoint_names'][9:17]}
@app.function(image=image,cpu=4,memory=8192,timeout=600,volumes={'/data':volume})
def tracks(version:str='sam3',recipe:str=''):
 import hashlib,math
 out=BASE/'sam3-20261023-v1';files=[]
 prefix='sam31' if version=='sam3.1' else 'sam3'
 method=prefix+'-native'
 if recipe:
  assert version=='sam3.1' and recipe in ['final-mask','largest-component']
  method='sam31-'+recipe
 for clip in ['pittsburgh-0600','pittsburgh-1800','marylandwomen-0600','marylandwomen-1800']:
  source=out/('tracking31' if version=='sam3.1' else 'tracking')/'regression/person'/clip/'tracks.json'
  if recipe:source=out/'tracking31-geometry/regression/person'/clip/recipe/'tracks.json'
  raw=json.loads(source.read_text());metrics=json.loads(source.with_name('metrics.json').read_text())
  meta=json.loads((BASE/'tracking-four-hour-v1/final-test/features/osnet_parts_generic'/clip/'metadata.json').read_text())
  height,width=meta['shape'];frames=[];ids={}
  for i,row in enumerate(raw['frames']):
   assert row['frame']==i;objects=[]
   for obj in row['objects']:
    box=obj['bbox'];assert all(math.isfinite(v) for v in box)
    assert -0.01<=box[0]<box[2]<=width+.01 and -0.01<=box[1]<box[3]<=height+.01
    objects.append([ids.setdefault(obj['track_id'],len(ids)+1),*[max(0,min(10000,round(v/scale*10000)))for v,scale in zip(box,[width,height,width,height])]])
   frames.append(objects)
  assert len(frames)==450
  value={'format':'compact_observed_boxes_v1','coordinate_scale':10000,'clip':clip,'method':method,'fps':15,'start_s':raw['frames'][0]['source_time_s'],'frames':frames,'fragment_ids_only':True,'manually_corrected':False,'source_tracks_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'model':'facebook/'+version,'prompt':'person','track_id_map':ids,'observation_definition':'A model-emitted nonempty segmentation mask, not verified visibility; may hallucinate behind occlusion. No interpolated coordinates.'}
  files.append({'path':'media/tracking-fourhour/'+clip+'-'+method+'.json','content':json.dumps(value,separators=(',',':'))+'\n'})
  suffix=('-sam31'+('-'+recipe if recipe else '')) if version=='sam3.1' else ''
  files.append({'path':'data/reliability/sam3/'+clip+suffix+'-pairs.json','content':json.dumps(metrics,indent=2)+'\n'})
 return files

@app.local_entrypoint()
def main(task:str='poses',version:str='sam3',recipe:str=''):
 if task=='poses':print(json.dumps(poses.remote(),indent=2))
 elif task=='tracks':
  root=ROOT.parent/'hockey-tracking/web';files=tracks.remote(version,recipe)
  for f in files:
   path=root/f['path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_text(f['content'])
  print(json.dumps({'exported':[f['path']for f in files]}))
 else:raise ValueError(task)
