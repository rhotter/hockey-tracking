"""Render raw neighboring frames for review; media stays outside Git."""
from pathlib import Path
import json
import modal

app = modal.App('sport-scans-ball-sam-audit-20261026')
volume = modal.Volume.from_name('sport-scans-experiments')


@app.function(image=modal.Image.debian_slim(python_version='3.12').pip_install('pillow==11.3.0'),
              cpu=4, memory=8192, timeout=600, volumes={'/data':volume})
def render(samples, name):
    import hashlib, tarfile
    from PIL import Image, ImageDraw, ImageFont
    volume.reload()
    base = Path('/data/ball-hockey/reliability-20261018')
    source = base/'overhead-20261025-v1/frames'
    out = base/'ball-sam-20261026-v1'/name
    out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=16)
    for index, row in enumerate(samples):
        f = row['frame']; x,y = row['pixel']
        sheet = Image.new('RGB', (1440,906), 'white'); draw = ImageDraw.Draw(sheet)
        draw.text((5,4), f"{index:02d} | Frame {f} | SAM candidate ({x:.1f}, {y:.1f}) | raw crops below", font=font, fill='black')
        raw = source/f'{f+1:06d}.jpg'
        im = Image.open(raw).convert('RGB')
        row['source_sha256'] = hashlib.sha256(raw.read_bytes()).hexdigest()
        sheet.paste(im.resize((960,540)), (0,24))
        cx,cy=x/2,y/2+24
        draw.ellipse((cx-5,cy-5,cx+5,cy+5),outline='#156a9c',width=2)
        for i,k in enumerate([max(0,f-15), min(4499,f+15)]):
            im = Image.open(source/f'{k+1:06d}.jpg').convert('RGB')
            sheet.paste(im.resize((480,270)), (960,24+i*270))
            draw.text((965,28+i*270), f'Frame {k}', font=font, fill='white',stroke_width=1,stroke_fill='black')
        for i,k in enumerate([max(0,f-8),f,min(4499,f+8)]):
            im = Image.open(source/f'{k+1:06d}.jpg').convert('RGB')
            crop = im.crop((x-120,y-80,x+120,y+80)).resize((480,320))
            sheet.paste(crop,(i*480,586))
            draw.text((i*480+5,567), f'Native crop, frame {k}',font=font,fill='black')
        sheet.save(out/f'{index:02d}.jpg',quality=96)
    (out/'manifest.json').write_text(json.dumps(samples,indent=2)+'\n')
    with tarfile.open(out.with_suffix('.tar.gz'),'w:gz') as archive:
        archive.add(out,arcname=name)
    volume.commit()
    return {'samples':len(samples),'archive':str(out.with_suffix('.tar.gz'))}


@app.local_entrypoint()
def main(selection_path:str,name:str='output-audit'):
    print(render.remote(json.loads(Path(selection_path).read_text()),name))
