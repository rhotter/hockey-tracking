"""Export review evidence without copying raw footage into the repository."""
from pathlib import Path
import modal

HERE = Path(__file__).parent
volume = modal.Volume.from_name('sport-scans-experiments')
app = modal.App('sport-scans-ball-temporal-review')
image = (modal.Image.from_registry('python:3.12-slim-bookworm')
         .apt_install('ffmpeg').pip_install('pillow==11.3.0')
         .add_local_file(HERE / 'fresh-reference.json', '/work/fresh-reference.json')
         .add_local_file(HERE / 'transfer-output-audit-selection.json', '/work/transfer-output-audit-selection.json'))


@app.function(image=image, cpu=8, memory=16384, timeout=3600, volumes={'/data': volume})
def export(task: str):
    import json, hashlib, tarfile, subprocess
    from PIL import Image, ImageDraw, ImageFont
    volume.reload()
    base = Path('/data/ball-hockey/reliability-20261018')
    out = base / 'ball-temporal-20261026-v1'
    dest = out / (task + '-review')
    dest.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=16)
    manifest = []
    if task in ('pilot','early','refinements','detector-seeds'):
        ledger = json.loads((out / 'refinement-v2/ledger.json').read_text())
        samples = []
        for request in ledger['requests'].values():
            if task == 'pilot' and request['core'][0] not in (0,900,1800,2700,3900): continue
            for sample in (request.get('answer') or {}).get('samples', []):
                if sample['status'] == 'visible':
                    if task == 'early' and sample['frame'] >= 1500: continue
                    origin = request['origins'][str(sample['frame'])]
                    xy = [origin[i] + sample['xy_1000'][i] * [256, 144][i] / 1000 for i in range(2)]
                    samples.append({**sample, 'pixel': xy})
        if task == 'detector-seeds':
            samples = json.loads((out/'tracking-seeds-dense-detector-gpu.json').read_text())['seeds']
        samples.sort(key=lambda s: s['frame'])
        for page in range(0, len(samples), 9):
            sheet = Image.new('RGB', (1536, 936), 'white')
            draw = ImageDraw.Draw(sheet)
            for k, sample in enumerate(samples[page:page+9]):
                f = sample['frame']; point = sample['pixel']
                path = base / 'overhead-20261025-v1/frames' / f'{f+1:06d}.jpg'
                raw = Image.open(path).convert('RGB')
                x, y = round(point[0]-128), round(point[1]-72)
                crop = raw.crop((x, y, x+256, y+144)).resize((512, 288))
                left, top = (k % 3)*512, (k // 3)*312
                sheet.paste(crop, (left, top+24))
                draw.text((left+4, top+3), f'Frame {f}; proposed center = crop center', font=font, fill='black')
                # No circle over the object: small ticks outside its immediate pixels.
                cx, cy = left+256, top+24+144
                draw.line((cx-22, cy, cx-12, cy), fill='yellow', width=2)
                draw.line((cx+12, cy, cx+22, cy), fill='yellow', width=2)
                manifest.append({**sample, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            sheet.save(dest / f'{task}-{page//9:02d}.jpg', quality=97)
    elif task in ('reused-reference','candidate-final-context','transfer-unresolved'):
        samples = (json.loads(Path('/work/transfer-output-audit-selection.json').read_text())['rows'] if task=='transfer-unresolved' else
                   json.loads((out/'reused-reference.json').read_text())['rows'] if task=='reused-reference'
                   else json.loads((out/'candidate-final/report.json').read_text())['audit_selection']['rows'])
        for page in range(0,len(samples),3):
            sheet=Image.new('RGB',(1536,936),'white');draw=ImageDraw.Draw(sheet)
            for k,sample in enumerate(samples[page:page+3]):
                f=sample['frame'];x,y=[round(v-96) for v in sample['pixel']];top=k*312
                for col,delta in enumerate([-3,0,3]):
                    frame=max(0,min(4499,f+delta));path=base/'overhead-20261025-v1/frames'/f'{frame+1:06d}.jpg'
                    if task=='transfer-unresolved':
                        clip=sample['clip'];folder=base/'new-games-20261025-v1/frames'/clip if clip.endswith('0600') else out/'extra-frames'/clip
                        path=folder/f'{frame+1:06d}.jpg'
                        if delta==0:assert hashlib.sha256(path.read_bytes()).hexdigest()==sample['source_sha256']
                    raw=Image.open(path).convert('RGB')
                    crop=raw.crop((x,y,x+192,y+192)).resize((288,288))
                    sheet.paste(crop,(col*512,top+24))
                    whole=raw.copy();whole.thumbnail((224,180));sheet.paste(whole,(col*512+288,top+70))
                    draw.text((col*512+4,top+3),f'Reference{f}; frame{frame}'+(' TARGET' if delta==0 else ''),font=font,fill='black')
                    cx,cy=col*512+144,top+168
                    draw.line((cx-18,cy,cx-10,cy),fill='yellow',width=2)
                    draw.line((cx+10,cy,cx+18,cy),fill='yellow',width=2)
                    if delta==0: manifest.append({**sample,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            sheet.save(dest/f'{task}-{page//3:02d}.jpg',quality=97)
    elif task.startswith('candidate-'):
        report = json.loads((out / task / 'report.json').read_text())
        samples = report['audit_selection']['rows']
        for page in range(0, len(samples), 3):
            sheet = Image.new('RGB', (1536, 936), 'white')
            draw = ImageDraw.Draw(sheet)
            for k, sample in enumerate(samples[page:page+3]):
                f = sample['frame']
                path = base / 'overhead-20261025-v1/frames' / f'{f+1:06d}.jpg'
                raw = Image.open(path).convert('RGB')
                top = k * 312
                for col, key, label in [(0, 'pixel', 'New'), (1, 'before_pixel', 'Previous')]:
                    point = sample.get(key)
                    draw.text((col*512+4, top+3), f'{label} | frame {f}', font=font, fill='black')
                    if point is None:
                        draw.text((col*512+12, top+80), 'No previous output', font=font, fill='black')
                        continue
                    x, y = round(point[0]-128), round(point[1]-72)
                    crop = raw.crop((x,y,x+256,y+144)).resize((512,288))
                    sheet.paste(crop, (col*512, top+24))
                    cx,cy = col*512+256,top+168
                    draw.line((cx-22,cy,cx-12,cy),fill='yellow',width=2)
                    draw.line((cx+12,cy,cx+22,cy),fill='yellow',width=2)
                sheet.paste(raw.resize((512,288)),(1024,top+24))
                draw.text((1028,top+3),f'Full frame {f}',font=font,fill='black')
                manifest.append({**sample,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            sheet.save(dest / f'{task}-{page//3:02d}.jpg',quality=97)
    elif task in ('extra','final-check'):
        start_s=1800 if task=='final-check' else 1200
        sample_frames=range(23,1350,90) if task=='final-check' else range(23,1350,45)
        # Fixed additional times, chosen before opening held-out model scores.
        for source, clip in [('XLyHD1ZR5o4', 'pittsburgh-finals-1200'), ('nWo-KEsSLyo', 'southjersey-finals-1200')]:
            clip=clip.replace('1200',str(start_s))
            raw = Path('/data/raw') / (source+'.mp4')
            folder = out / 'extra-frames' / clip
            folder.mkdir(parents=True, exist_ok=True)
            if len(list(folder.glob('*.jpg'))) != 1350:
                subprocess.run(['ffmpeg','-v','error','-threads','4','-ss',str(start_s),'-i',str(raw),'-t','90','-an','-vf','setpts=PTS-STARTPTS,fps=15:start_time=0','-frames:v','1350','-q:v','2','-y',str(folder/'%06d.jpg')],check=True)
            for f in sample_frames:
                path=folder/f'{f+1:06d}.jpg'; rawframe=Image.open(path).convert('RGB'); w,h=rawframe.size
                sheet=Image.new('RGB',(1920,1660),'white');draw=ImageDraw.Draw(sheet)
                sheet.paste(rawframe.resize((1920,1080)),(0,30));draw.text((8,8),f'{clip} | frame {f} | native {w} x {h} | no predictions',font=font,fill='black')
                for i,delta in enumerate([-3,3]):
                    context=Image.open(folder/f'{f+delta+1:06d}.jpg').convert('RGB');context.thumbnail((960,540));sheet.paste(context,(i*960,1120));draw.text((i*960+8,1100),f'Frame {f+delta}',font=font,fill='black')
                filename=f'{clip}-{f:04d}.jpg';sheet.save(dest/filename,quality=95)
                manifest.append({'clip':clip,'source_id':source,'start_s':start_s,'frame':f,'image':filename,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'width':w,'height':h})
    else:
        raise ValueError(task)
    (dest/'manifest.json').write_text(json.dumps({'task':task,'samples':manifest},indent=2)+'\n')
    archive=out/(task+'-review.tar.gz')
    with tarfile.open(archive,'w:gz')as tar:tar.add(dest,arcname=dest.name)
    volume.commit()
    return {'samples':len(manifest),'archive':str(archive)}


@app.local_entrypoint()
def main(task: str='pilot'):
    print(export.remote(task))
