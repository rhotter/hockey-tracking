"""Advance this live campaign as complete development sweeps become available.

This is a foreground-session coordinator, not a recurring scheduled job. It
never retries a failed experiment or changes a selected setting.
"""
from pathlib import Path
import datetime,json,os,shutil,subprocess,time

ROOT=Path(__file__).resolve().parents[1]
NOTES=ROOT/'notes/ball-hockey'
LOGS=ROOT/'logs/sam3-20261024'
CLI=shutil.which('modal') or str(Path.home()/'.local/bin/modal')
DEV={'game2_01m','game2_06m','game2_24m','game2_33m','game2_41m','gb-tracking-0600','gb-tracking-1800','gb-reid-long-0900','long-plDp2kPtk_g'}
REG='pittsburgh-0600,pittsburgh-1800,marylandwomen-0600,marylandwomen-1800,pittsburgh-stress-0900'
SWEEPS={
 'native':('sam3-extended-development-association03_recondition4_recondition0_alive8-offload-20261024.json',['association03','recondition4','recondition0','alive8']),
 'visual':('sam3-extended-development-visual_visual-person_visual-hockey-offload-20261024.json',['visual','visual-person','visual-hockey']),
 'visual-memory':('sam3-extended-development-visual-memory_visual-memory-person_visual-memory-hockey-offload-20261024.json',['visual-memory','visual-memory-person','visual-memory-hockey']),
 'adapted':('sam3-extended-development-adapt-margin0_adapt-margin1_adapt-margin2-offload-transformer-lr1e5-20261024.json',['adapt-margin0','adapt-margin1','adapt-margin2']),
}
jobs={};finished=set();records=[]


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def event(**values):
    record={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),**values};records.append(record)
    print(json.dumps(record),flush=True)
    (NOTES/'sam3-20261024/followup-coordinator.json').write_text(json.dumps(records,indent=2)+'\n')


def launch(key,script,*args):
    if key in jobs or key in finished:return
    log=open(LOGS/(key+'.log'),'w')
    env={**os.environ,'MODAL_IMAGE_BUILDER_VERSION':'2025.06'}
    process=subprocess.Popen([CLI,'run','-e','main','scripts/'+script,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    jobs[key]=(process,log);event(event='launched',key=key,pid=process.pid)


def complete_sweep(name,variants):
    rows=read(NOTES/name)
    return isinstance(rows,list) and {(r['variant'],r['clip']) for r in rows}=={(v,c) for v in variants for c in DEV}


previous=read(NOTES/'sam3-20261024/followup-coordinator.json')
if previous:
    records=previous
    launched={r['key'] for r in records if r.get('event')=='launched'}
    ended={r['key'] for r in records if r.get('event')=='finished'}
    assert launched<=ended,'Inspect unfinished child jobs before resuming'
    assert all(r['returncode']==0 for r in records if r.get('event')=='finished')
    finished=ended


while True:
    for key,(process,log) in list(jobs.items()):
        code=process.poll()
        if code is None:continue
        log.close();jobs.pop(key);event(event='finished',key=key,returncode=code)
        if code:raise SystemExit(f'{key} failed; inspect its log before restarting the coordinator')
        finished.add(key)
    stress=read(NOTES/'sam3-extended-regression-default-offload-20261024.json') or []
    if read(NOTES/'sam3-20261024/rink-filter-regression.json'):
        launch('rink-filter-review','modal_nbhl_sam3_rink_filter.py','--task','review')
    if read(NOTES/'sam3-20261024/fresh-detection-run.json'):
        launch('fresh-detection-review','modal_nbhl_sam3_fresh_detection.py','--task','review')
    stress_ready=any(r['clip']=='pittsburgh-stress-0900' and r['frames']==4500 for r in stress)
    if stress_ready:launch('hybrid-full-regression','modal_nbhl_sam3_hybrid.py','--task','score','--phase','regression')
    longs=read(NOTES/'sam3-extended-long-default-offload-20261024.json') or []
    if any(r['clip']=='pittsburgh-runtime-2385-315-absolute-gpu-bounds' and r['frames']==4725 for r in longs):
        launch('fresh-audit-score','modal_nbhl_sam3_analysis.py','--task','fresh-audit')
    if len(longs)==4 and all(r['frames']==4725 for r in longs):
        launch('long-stitch','modal_nbhl_sam3_long_export.py','--task','tracks','--variant','default-offload')
    for family,(filename,variants) in SWEEPS.items():
        ready=complete_sweep(filename,variants)
        if family=='adapted':ready=ready and complete_sweep('sam3-extended-development-adapt-person-offload-transformer-lr1e5-20261024.json',['adapt-person'])
        if family=='adapted':
            for variant,delta in [('adapt-heads-person','heads-lr1e4'),('adapt-decoder-person','decoder-lr1e5')]:
                ready=ready and complete_sweep(f'sam3-extended-development-{variant}-offload-{delta}-20261024.json',[variant])
        if ready:launch('select-'+family,'modal_nbhl_sam3_analysis.py','--family',family,'--freeze')
        if 'select-'+family not in finished:continue
        selection=read(NOTES/f'sam3-analysis-table-{family}-development-freeze-20261024.json')['best']
        variant,delta=selection['variant'],selection['delta']
        key='regression-'+family
        if variant=='default':
            main_ready=stress_ready
        else:
            launch(key,'modal_nbhl_sam3_extended.py','--phase','regression','--clips',REG,'--variants',variant,
                   '--delta',delta,'--offload','--score-cache')
            main_ready=key in finished
        control_ready=True
        if family=='adapted':
            control='default' if variant.endswith('person') else variant.replace('adapt-margin','adapt-control')
            if control=='default':control_ready=stress_ready
            else:
                ck='regression-adapted-control'
                launch(ck,'modal_nbhl_sam3_extended.py','--phase','regression','--clips',REG,'--variants',control,'--offload','--score-cache')
                control_ready=ck in finished
        if main_ready and control_ready:
            launch('table-'+family,'modal_nbhl_sam3_analysis.py','--family',family,'--phase','regression')
    if 'hybrid-full-regression' in finished and all('table-'+f in finished for f in SWEEPS):
        launch('supplementary-audit','modal_nbhl_sam3_analysis.py','--task','audit')
    if {'supplementary-audit','fresh-audit-score','long-stitch','rink-filter-review','fresh-detection-review'}<=finished:
        event(event='complete',message='All selected tracking regressions and audits are ready for manual publication verification.')
        break
    time.sleep(20)
