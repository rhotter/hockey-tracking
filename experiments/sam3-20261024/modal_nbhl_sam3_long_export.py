"""Encode exact cached long-run frames and reconcile same-time SAM3 overlaps."""
from pathlib import Path
import json
import modal
from modal_campaign import ROOT,base_image,volume
BASE=Path('/data/ball-hockey/reliability-20261018');OUT=BASE/'sam3-hybrid-20261024-v1'
app=modal.App('sport-scans-sam3-long-export-20261024')
image=base_image
for name in ['modal_campaign.py','nbhl_overlap_stitch.py','nbhl_detection_metrics.py']:
    image=image.add_local_file(ROOT/'scripts'/name,'/work/'+name)


@app.function(image=image,cpu=8,memory=16384,timeout=7200,scaledown_window=10,volumes={'/data':volume})
def video():
    import hashlib,subprocess,time,concurrent.futures
    from nbhl_overlap_stitch import stitch
    volume.reload();started=time.monotonic();sources=[]
    for i,start in enumerate([1785,2085,2385,2685]):
        folder=BASE/'tracking-four-hour-v1/long-cache-trial'/f'pittsburgh-runtime-{start:04d}-315-absolute-gpu-bounds'/'frames'
        paths=sorted(folder.glob('*.jpg'));assert len(paths)==4725
        if i:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                equal=list(pool.map(lambda pair:hashlib.sha256(pair[0].read_bytes()).digest()==hashlib.sha256(pair[1].read_bytes()).digest(),zip(previous[-225:],paths[:225])))
            assert all(equal),'Every overlap JPEG must match'
        sources.extend(paths if i==3 else paths[:4500]);previous=paths
    assert len(sources)==18225
    dest=OUT/'public';dest.mkdir(parents=True,exist_ok=True);target=dest/'pittsburgh-sam3-20min.mp4'
    p=subprocess.Popen(['ffmpeg','-v','error','-y','-f','image2pipe','-framerate','15','-vcodec','mjpeg','-i','-','-an','-vf','scale=960:540','-c:v','libx264','-preset','medium','-crf','28','-pix_fmt','yuv420p','-threads','6','-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
    digest=hashlib.sha256()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for offset in range(0,len(sources),32):
                for payload in pool.map(lambda path:path.read_bytes(),sources[offset:offset+32]):
                    digest.update(hashlib.sha256(payload).digest());p.stdin.write(payload)
                if offset%960==0:print(json.dumps({'encoded_frames':offset,'total':len(sources),'seconds':time.monotonic()-started}),flush=True)
    finally:p.stdin.close()
    assert p.wait()==0
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(target)]));stream=probe['streams'][0]
    assert int(stream['nb_frames'])==18225 and abs(float(probe['format']['duration'])-1215)<.1
    report={'clip':'pittsburgh-sam3-20min','source_id':'JlBtScHdcg4','start_s':1785,'duration_s':1215,'frames':18225,'width':960,'height':540,'fps':15,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size,'input_frame_sequence_sha256':digest.hexdigest(),'overlap_frames_verified':675,'seconds':time.monotonic()-started,'source':'Exact cached inference JPEGs, each shared overlap checked byte-for-byte. Delivery resized; inference used1920x1080.'}
    (dest/'video.json').write_text(json.dumps(report,indent=2)+'\n');volume.commit();return report


@app.function(image=image,cpu=4,memory=16384,timeout=900,volumes={'/data':volume})
def tracks(variant:str='default'):
    import hashlib
    from nbhl_overlap_stitch import stitch
    volume.reload();chunks=[];inputs=[]
    for start in [1785,2085,2385,2685]:
        clip=f'pittsburgh-runtime-{start:04d}-315-absolute-gpu-bounds';folder=OUT/'native-settings/long'/variant/clip
        raw=json.loads((folder/'tracks.json').read_text());report=json.loads((folder/'completed.json').read_text())
        assert hashlib.sha256((folder/'tracks.json').read_bytes()).hexdigest()==report['tracks_sha256']
        assert len(raw['frames'])==4725
        chunks.append({'clip':clip,'start_frame':(start-1785)*15,'frames':raw['frames']});inputs.append(report)
    joined=stitch(chunks);assert len(joined['frames'])==18225
    for row in joined['frames']:row['source_time_s']+=1785
    dest=OUT/'stitched'/variant;dest.mkdir(parents=True,exist_ok=True);path=dest/'tracks.json';path.write_text(json.dumps(joined,separators=(',',':'))+'\n')
    report={'clip':'pittsburgh-sam3-20min','variant':variant,'frames':18225,'duration_s':1215,'source_id':'JlBtScHdcg4','start_s':1785,'inputs':inputs,'boundaries':joined['boundaries'],'fragments':joined['fragments'],'tracks_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'accuracy_reference':None,'limits':'Automatic same-time overlap joining. Identities across20minutes are unverified; not jersey numbers or TOI. SAM3 masks can drift or persist behind occlusion.'}
    (dest/'completed.json').write_text(json.dumps(report,indent=2)+'\n');volume.commit();return report


@app.local_entrypoint()
def main(task:str='video',variant:str='default'):
    result=video.remote() if task=='video' else tracks.remote(variant)
    (ROOT/'notes/ball-hockey'/f'sam3-long-{task}-{variant}-20261024.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
