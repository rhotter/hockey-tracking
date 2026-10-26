"""Native-resolution NBHL ball detector ablations on the existing Modal volume."""
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/work')
BASE=Path('/data/ball-hockey/reliability-20261018')
SOURCE=BASE/'overhead-20261025-v1'
OUT=BASE/'ball-rebuild-20261026-v1'
app=modal.App('sport-scans-ball-rebuild-20261026')
volume=modal.Volume.from_name('sport-scans-experiments')
@app.function(image=modal.Image.debian_slim(python_version="3.12").pip_install("pillow==11.3.0"),cpu=2,memory=4096,timeout=300,volumes={'/data':volume})
def review(targeted: bool=False):
    import json,hashlib,tarfile
    from PIL import Image,ImageDraw
    volume.reload();name='targeted' if targeted else 'review';dest=OUT/name;dest.mkdir(parents=True,exist_ok=True)
    frames=list(range(150,300,10))+list(range(2800,2950,10)) if targeted else list(range(37,4500,75));manifest=[]
    for f in frames:
        src=SOURCE/'frames'/f'{f+1:06d}.jpg';im=Image.open(src).convert('RGB')
        im.save(dest/f'frame-{f:04d}.jpg',quality=95)
        # Full-width floor view for first-pass inspection, not a ball proposal.
        crop=im.crop((0,270,1920,1080)).resize((1280,540))
        d=ImageDraw.Draw(crop);d.rectangle((0,0,260,24),fill='white')
        d.text((5,4),f'Frame {f} / native origin 0,270 / scale 2/3',fill='black')
        crop.save(dest/f'floor-{f:04d}.jpg',quality=95)
        manifest.append({'frame':f,'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
                         'split':'development' if f<1500 else 'test'})
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with tarfile.open(OUT/(name+'.tar.gz'),'w:gz')as tar:tar.add(dest,arcname=name)
    volume.commit();return {'frames':len(frames),'archive':str(OUT/(name+'.tar.gz'))}


@app.function(image=modal.Image.debian_slim(python_version="3.12").pip_install("pillow==11.3.0"),cpu=4,memory=8192,timeout=600,volumes={'/data':volume})
def audit(samples, name='final-audit',context=False):
    import json,hashlib,tarfile
    from PIL import Image,ImageDraw
    volume.reload();dest=OUT/name;dest.mkdir(parents=True,exist_ok=True)
    height=625 if context else 220;per_sheet=4 if context else 6
    for index,row in enumerate(samples):
        f=row['frame'];x,y=row['pixel'];sheet=Image.new('RGB',(720,height),'white');d=ImageDraw.Draw(sheet)
        d.text((4,4),f"{index+1} / frame {f} / {row['state']} / previous-current-next (2 frames apart)",fill='black')
        if context:
            full=Image.open(SOURCE/'frames'/f'{f+1:06d}.jpg').convert('RGB').resize((720,405));sheet.paste(full,(0,20))
            cx,cy=x*720/1920,y*405/1080+20;d.ellipse((cx-9,cy-9,cx+9,cy+9),outline='orange',width=2)
        for j,k in enumerate([f-2,f,f+2]):
            src=SOURCE/'frames'/f'{max(0,min(4499,k))+1:06d}.jpg';im=Image.open(src).convert('RGB')
            crop=im.crop((x-60,y-50,x+60,y+50)).resize((240,200));sheet.paste(crop,(j*240,425 if context else 20))
            if j==1:row['source_sha256']=hashlib.sha256(src.read_bytes()).hexdigest()
        sheet.save(dest/f'{index:02d}.jpg',quality=96)
    for start in range(0,len(samples),per_sheet):
        sheet=Image.new('RGB',(1440,height*((per_sheet+1)//2)),'white')
        for j,index in enumerate(range(start,min(start+per_sheet,len(samples)))):sheet.paste(Image.open(dest/f'{index:02d}.jpg'),(j%2*720,j//2*height))
        sheet.save(dest/f'sheet-{start//per_sheet}.jpg',quality=96)
    (dest/'manifest.json').write_text(json.dumps(samples,indent=2)+'\n')
    with tarfile.open(OUT/(name+'.tar.gz'),'w:gz')as tar:tar.add(dest,arcname=name)
    volume.commit();return {'frames':len(samples)}


@app.local_entrypoint()
def main(targeted:bool=False,final_audit:bool=False,replay_path:str='',round_name:str='final-audit',exclude_previous:bool=False,added_only:bool=False,fresh_all:bool=False,sample_count:int=36,custom_frames:str=''):
    import json
    if final_audit:
        path=ROOT/replay_path if replay_path else ROOT/'web/data/ball-rebuild-20261026/replay.json'
        data=json.loads(path.read_text())
        excluded=set()
        if exclude_previous:
            previous=json.loads((ROOT/'experiments/ball-rebuild-20261026/output-audit-v1.json').read_text())['rows']
            excluded={f for r in previous for f in range(r['frame']-3,r['frame']+4)}
        if fresh_all:
            for name in ['output-audit-v1.json','output-audit-v4.json','development-audit.json','review-labels.json','targeted-labels.json','verifier-adaptation-labels.json','output-audit-primary.json','recall-development-audit.json']:
                label_path=ROOT/'experiments/ball-rebuild-20261026'/name
                if not label_path.exists():continue
                previous=json.loads(label_path.read_text())['rows']
                excluded.update(f for r in previous for f in range(r['frame']-3,r['frame']+4))
        rows=[r for r in data['frames'] if r['frame']>=(0 if fresh_all else 1500) and r['state']!='unknown' and r['frame'] not in excluded]
        if added_only:
            baseline=json.loads((ROOT/'web/data/ball-context-20261026/replay.json').read_text())['frames']
            rows=[r for r in rows if baseline[r['frame']]['state']=='unknown']
        if custom_frames:
            chosen=[data['frames'][int(f)] for f in custom_frames.split(',')]
            if any(r['state']=='unknown'for r in chosen):raise ValueError('A custom audit frame has no proposal')
        else:
            if len(rows)<sample_count:raise ValueError(f'Only {len(rows)} eligible frames for a {sample_count}-frame audit')
            chosen=[rows[int((i+.5)*len(rows)/sample_count)] for i in range(sample_count)]
        samples=[{'frame':r['frame'],'pixel':[r['pixel'][0]*1920,r['pixel'][1]*1080],'state':r['state']} for r in chosen]
        print(audit.remote(samples,round_name,bool(fresh_all or custom_frames)))
    else:print(json.dumps(review.remote(targeted),indent=2))
