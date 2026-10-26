"""Full-frame chronological context and native crops for sequence-result audits."""
from pathlib import Path
import modal

HERE=Path(__file__).parent;ROOT=HERE.parents[1] if modal.is_local() else Path('/work')
app=modal.App('sport-scans-ball-sequence-audit-20261026')
volume=modal.Volume.from_name('sport-scans-experiments')


@app.function(image=modal.Image.debian_slim(python_version='3.12').pip_install('pillow==11.3.0'),cpu=4,memory=8192,timeout=600,volumes={'/data':volume})
def render(samples,name):
    import json,hashlib,tarfile
    from PIL import Image,ImageDraw,ImageFont
    volume.reload();base=Path('/data/ball-hockey/reliability-20261018')
    source=base/'overhead-20261025-v1/frames';out=base/'ball-sequence-20261026-v1'/name;out.mkdir(parents=True,exist_ok=True)
    font=ImageFont.load_default(size=16)
    for index,row in enumerate(samples):
        f=row['frame'];x,y=row['pixel'];sheet=Image.new('RGB',(1440,906),'white');draw=ImageDraw.Draw(sheet)
        draw.text((5,4),f"Frame {f} | {row['state']} | radius {row.get('radius_px',0):.0f} native pixels | center, then before/after context",font=font,fill='black')
        raw=source/f'{f+1:06d}.jpg';image=Image.open(raw).convert('RGB');sheet.paste(image.resize((960,540)),(0,24))
        row['source_sha256']=hashlib.sha256(raw.read_bytes()).hexdigest()
        for h in (row.get('hypotheses')or[{'pixel':[x,y],'radius_px':row.get('radius_px',12)}]):
            px,py=h['pixel'];radius=h['radius_px']/2;cx,cy=px/2,py/2+24
            draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),outline='#156a9c',width=3)
            draw.line((cx-5,cy,cx+5,cy),fill='#156a9c',width=2);draw.line((cx,cy-5,cx,cy+5),fill='#156a9c',width=2)
        for i,k in enumerate([max(0,f-15),min(4499,f+15)]):
            im=Image.open(source/f'{k+1:06d}.jpg').convert('RGB');sheet.paste(im.resize((480,270)),(960,24+i*270))
            draw.text((965,28+i*270),f'Frame {k}',font=font,fill='white',stroke_width=1,stroke_fill='black')
        for i,k in enumerate([max(0,f-8),f,min(4499,f+8)]):
            im=Image.open(source/f'{k+1:06d}.jpg').convert('RGB');crop=im.crop((x-120,y-80,x+120,y+80)).resize((480,320));sheet.paste(crop,(i*480,586))
            draw.text((i*480+5,567),f'Native crop, frame {k}',font=font,fill='black')
        sheet.save(out/f'{index:02d}.jpg',quality=96)
    (out/'manifest.json').write_text(json.dumps(samples,indent=2)+'\n')
    with tarfile.open(out.with_suffix('.tar.gz'),'w:gz')as archive:archive.add(out,arcname=name)
    volume.commit();return {'samples':len(samples),'archive':str(out.with_suffix('.tar.gz'))}


@app.local_entrypoint()
def main(replay_path:str,name:str='context-pilot-audit',frames:str=''):
    import json
    data=json.loads((ROOT/replay_path).read_text())
    wanted=[int(x)for x in frames.split(',')]if frames else[start+75 for start in [0,300,900,1800,2700,3900]]
    samples=[]
    for f in wanted:
        r=data['frames'][f]
        if not r.get('pixel'):continue
        samples.append({'frame':f,'state':r['state'],'pixel':[r['pixel'][0]*1920,r['pixel'][1]*1080],'radius_px':r['model_radius_px'],
                        'hypotheses':[{**h,'pixel':[h['pixel'][0]*1920,h['pixel'][1]*1080]}for h in r.get('hypotheses',[])]})
    if not samples:raise ValueError('No requested frames contain an estimate')
    print(render.remote(samples,name))
