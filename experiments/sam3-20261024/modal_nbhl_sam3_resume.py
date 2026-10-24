"""One bounded cloud coordinator; resume saved work without a local connection.

Run: modal run --detach -e main scripts/modal_nbhl_sam3_resume.py::resume
This performs only the already registered experiments. Frozen choices and
completed inference are reused. It does not publish or deploy a website.
"""
from pathlib import Path
import json
import modal
from modal_campaign import volume
import modal_nbhl_sam3_extended as extended
import modal_nbhl_sam3_analysis as analysis
import modal_nbhl_sam3_hybrid as hybrid
import modal_nbhl_sam3_long_export as long_export
import modal_nbhl_sam3_rink_filter as rink
import modal_nbhl_sam3_publish as publish

app=modal.App('sport-scans-sam3-resume-20261024')
for module in [extended,analysis,hybrid,long_export,rink,publish]:
    app.include(module.app)
OUT=Path('/data/ball-hockey/reliability-20261018/sam3-hybrid-20261024-v1')
DEV=['game2_01m','game2_06m','game2_24m','game2_33m','game2_41m',
     'gb-tracking-0600','gb-tracking-1800','gb-reid-long-0900','long-plDp2kPtk_g']
REG=['pittsburgh-0600','pittsburgh-1800','marylandwomen-0600',
     'marylandwomen-1800','pittsburgh-stress-0900']
SWEEPS={
    'native':[(v,'') for v in ['association03','recondition4','recondition0','alive8']],
    'visual':[(v,'') for v in ['visual','visual-person','visual-hockey']],
    'visual-memory':[(v,'') for v in ['visual-memory','visual-memory-person','visual-memory-hockey']],
    'adapted':[(v,'transformer-lr1e5') for v in ['adapt-margin0','adapt-margin1','adapt-margin2','adapt-person']]
              +[('adapt-heads-person','heads-lr1e4'),('adapt-decoder-person','decoder-lr1e5')],
}
coordinator_image=modal.Image.debian_slim(python_version='3.12').env({'PYTHONPATH':'/work'})
for dependency in ['modal_campaign.py','modal_nbhl_sam3_20261023.py',
                   'modal_nbhl_sam3_extended.py','modal_nbhl_sam3_analysis.py',
                   'modal_nbhl_sam3_hybrid.py','modal_nbhl_sam3_long_export.py',
                   'modal_nbhl_sam3_rink_filter.py','modal_nbhl_sam3_publish.py']:
    coordinator_image=coordinator_image.add_local_file(Path(__file__).parent/dependency,'/work/'+dependency)


@app.function(image=coordinator_image,cpu=1,memory=4096,
              timeout=21600,max_containers=1,volumes={'/data':volume})
def resume():
    from concurrent.futures import ThreadPoolExecutor,as_completed
    from datetime import datetime,timezone
    from threading import Lock
    import hashlib
    lock=Lock();events=[]
    def event(key,**values):
        with lock:
            row={'at':datetime.now(timezone.utc).isoformat(),'key':key,**values}
            events.append(row);print(json.dumps(row),flush=True)
            (OUT/'resume-events.json').write_text(json.dumps(events,indent=2)+'\n')
            volume.commit()
    def read(path):
        # The GPU functions commit their files before returning. Reload only
        # under this lock, with no local file handles kept open.
        with lock:
            volume.reload()
            return json.loads(path.read_text()) if path.exists() else None
    def infer(clip,variant='default',phase='regression',delta=''):
        fn=extended.track_long if phase=='long' or clip.endswith('stress-0900') else extended.track
        return fn.remote(clip,variant,phase,delta,True,True)
    def family(name):
        selection_path=OUT/'video-selection'/name/'selection.json'
        selected=read(selection_path)
        if selected is None:
            jobs=[(clip,variant,'development',delta,True,True)
                  for variant,delta in SWEEPS[name] for clip in DEV]
            for result in extended.track.starmap(jobs,order_outputs=False):
                event('development-complete',family=name,clip=result['clip'],variant=result['variant'])
            selected=analysis.table.remote(name,'development',True)
        best=selected['best'];event('frozen-selection',family=name,variant=best['variant'],delta=best['delta'])
        if best['variant']!='default':
            with ThreadPoolExecutor(max_workers=5) as pool:
                for result in pool.map(lambda clip:infer(clip,best['variant'],'regression',best['delta']),REG):
                    event('regression-complete',family=name,clip=result['clip'])
        if name=='adapted':
            control=analysis.adapted_control_variant(best['variant'])
            if control!='default':
                with ThreadPoolExecutor(max_workers=5) as pool:
                    list(pool.map(lambda clip:infer(clip,control),REG))
        return name
    def native_baseline():
        result=infer('pittsburgh-stress-0900')
        event('baseline-stress-complete',counts=result['counts'])
        result=hybrid.score.remote('regression','')
        event('hybrid-regression-complete')
    def long_run():
        jobs=[(f'pittsburgh-runtime-{start:04d}-315-absolute-gpu-bounds','default','long','',True,True)
              for start in [1785,2085,2385,2685]]
        for result in extended.track_long.starmap(jobs,order_outputs=False):
            event('long-window-complete',clip=result['clip'],frames=result['frames'])
        long_export.tracks.remote('default-offload')
        analysis.fresh_audit.remote();event('long-audit-complete')
    def rink_run():
        if read(OUT/'rink-filter/regression.json') is None:
            rink.run.remote('regression')
        rink.review.remote();event('rink-regression-complete')
    event('started',scope='Resume frozen campaign only; no new candidates or training.')
    try:
        with ThreadPoolExecutor(max_workers=7) as pool:
            futures=[pool.submit(family,name) for name in SWEEPS]
            futures.extend([pool.submit(native_baseline),pool.submit(long_run),pool.submit(rink_run)])
            for future in as_completed(futures):future.result()
        for name in SWEEPS:analysis.table.remote(name,'regression',False)
        analysis.audit.remote(True)
        payload=publish.replays.remote(False)
        with lock:
            dest=OUT/'public/export';dest.mkdir(exist_ok=True)
            (dest/'replays.json').write_text(json.dumps(payload,separators=(',',':'))+'\n')
            volume.commit()
        payload=publish.reports.remote(False)
        with lock:
            (dest/'reports.json').write_text(json.dumps(payload,separators=(',',':'))+'\n')
            volume.commit()
        event('complete',message='Ready for local verification and website publication.')
    except Exception as error:
        event('failed',error=repr(error));raise
    return events
