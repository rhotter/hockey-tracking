"""Known-incomplete context responses only: medium reasoning, larger answer cap.

This separate experiment preserves the first pass and never retries ambiguous
connection failures. Its budget includes every reservation from all prior phases.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64,io,json,math,os,threading,time
import worker
from context_worker import PROMPT,SCHEMA,post

def run(protocol, config, volume):
    from PIL import Image,ImageDraw,ImageFont
    out=worker.OUT;path=out/'context-recovery-ledger.json';out.mkdir(exist_ok=True,parents=True)
    ledger=worker.read(path) if path.exists() else {'protocol_sha256':worker.sha(Path('/work/protocol.json')),
       'recovery_protocol_sha256':worker.sha(Path('/work/recovery-protocol.json')),'worker_sha256':worker.sha(Path(__file__)),
       'phase':'context-recovery','requests':{}}
    assert ledger['recovery_protocol_sha256']==worker.sha(Path('/work/recovery-protocol.json'))
    other=sum(r.get('estimated_usd',r.get('reserved_usd',0))for phase in ('scan','verify','context')for r in worker.read(out/f'{phase}-ledger.json')['requests'].values())
    assert all(r['status'] not in ('reserved',)for r in worker.read(out/'context-ledger.json')['requests'].values()),'Wait for context inference to finish'
    clips={c['id']:c for c in protocol['clips']};scan=worker.read(out/'scan-ledger.json');requests=worker.read(out/'verify-requests.json')['requests']
    prior=worker.read(out/'context-ledger.json')['requests']
    assert len(prior)==len(requests),'Wait for every context request'
    requests=[r for r in requests if prior[r['id'].replace('-verify-','-context-')]['status']=='incomplete' and prior[r['id'].replace('-verify-','-context-')].get('usage')]
    font=ImageFont.load_default(size=18);lock=threading.Lock();key=os.environ['OPENAI_API_KEY']
    def persist():
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(ledger,indent=2)+'\n');temp.replace(path);volume.commit()
    def task(request):
        rid=request['id'].replace('-verify-','-context-recovery-');c=clips[request['clip']];a,b=request['core']
        full=set(range(max(0,a-30),min(c['frames'],b+30),config['full_frame_stride']))
        full.update(range(min(c['frames']-1,b+30),min(c['frames'],b+config['after_context_seconds']*15),config['after_context_stride']))
        proposals=[e for r in scan['requests'].values() if r['clip']==c['id'] and r['status']=='completed' for e in r['answer']['events'] if a<=e['release_frame']<b]
        nets=[e['net_xy_1000']for e in proposals if e['net_xy_1000']]
        crop_frames=set(range(max(0,a-6),min(c['frames'],b+12),config['goal_crop_stride'])) if nets else set()
        sample=sorted(full|crop_frames);hashes={};bounds=None;inputs=[{'type':'input_text','text':PROMPT+f'\nCORE[{a},{b}) at15fps. Clip{c["id"]}. Team appearance: '+('green and light gray (use black for the light-gray team for compatibility).' if c['id'].startswith('pittsburgh') else str(c['team_colors']))}]
        for f in sample:
            source=worker.BASE/c['frame_folder']/f'{f+1:06d}.jpg';raw=Image.open(source).convert('RGB');hashes[str(f)]=worker.sha(source)
            images=[]
            if f in full:images.append((raw.resize((config['full_frame_width'],round(raw.height*config['full_frame_width']/raw.width))),f'FULL frame{f} | {f/15:.3f}s'+(' AFTERMATH'if f>=b+30 else'')))
            if f in crop_frames:
                x=sum(n[0]for n in nets)/len(nets)*raw.width/1000;y=sum(n[1]for n in nets)/len(nets)*raw.height/1000
                w=config['goal_crop_width'];h=round(w*9/16);left=max(0,min(raw.width-w,round(x-w/2)));top=max(0,min(raw.height-h,round(y-h/2)));bounds=[left,top,left+w,top+h]
                images.append((raw.crop(tuple(bounds)),f'CROP frame{f} | bounds {bounds}'))
            for im,title in images:
                sheet=Image.new('RGB',(im.width,im.height+26),'white');sheet.paste(im,(0,26));ImageDraw.Draw(sheet).text((4,3),title,font=font,fill='black');buf=io.BytesIO();sheet.save(buf,format='JPEG',quality=95)
                inputs.append({'type':'input_image','detail':'original','image_url':'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()})
        body={'model':config['model'],'input':[{'role':'user','content':inputs}],'reasoning':{'effort':config['reasoning']},
              'text':{'format':{'type':'json_schema','name':'contextual_nbhl_shots','strict':True,'schema':SCHEMA}},
              'max_output_tokens':config['max_output_tokens'],'store':False,'service_tier':'default'}
        count=post('responses/input_tokens',{'model':body['model'],'input':body['input']},key)['input_tokens']
        reserved=(count*config['input_usd_per_million_conservative']+config['max_output_tokens']*config['output_usd_per_million'])/1e6
        with lock:
            spent=other+sum(r.get('estimated_usd',r.get('reserved_usd',0))for r in ledger['requests'].values())
            record={**request,'id':rid,'range':[min(sample),max(sample)+1],'sample_frames':sample,'full_frames':sorted(full),'crop_frames':sorted(crop_frames),'crop_bounds':bounds,'source_sha256':hashes,'status':'reserved','reserved_usd':reserved,'input_tokens_counted':count}
            if spent+reserved>config['campaign_budget_usd']:record.update(status='budget_not_sent',reserved_usd=0);ledger['requests'][rid]=record;persist();return
            ledger['requests'][rid]=record;persist()
        start=time.monotonic()
        try:
            response=post('responses',body,key);usage=response.get('usage',{})
            output=''.join(b.get('text','')for i in response.get('output',[])if i.get('type')=='message'for b in i.get('content',[])if b.get('type')=='output_text')
            answer=json.loads(output)if response.get('status')=='completed'else None
            record.update(answer=answer,usage=usage,response_id=response.get('id'),response_status=response.get('status'),incomplete_details=response.get('incomplete_details'))
            if usage:record['estimated_usd']=(usage['input_tokens']*config['input_usd_per_million_conservative']+usage['output_tokens']*config['output_usd_per_million'])/1e6
            if answer:
                for e in answer['events']:
                    assert a<=e['release_frame']<b and e['release_frame']<=e['outcome_frame']<c['frames']
                    assert e['release_frame']in sample and e['outcome_frame']in sample
                    assert set(e['evidence_frames'])<=set(sample)
                    for k in ('shooter_xy_1000','net_xy_1000'):assert e[k]is None or all(math.isfinite(v)and 0<=v<=1000 for v in e[k])
            with lock:record.update(status='completed'if answer else'incomplete',seconds=time.monotonic()-start);persist()
        except Exception as e:
            with lock:record.update(status='failed',error_type=type(e).__name__,seconds=time.monotonic()-start);persist()
        print({'id':rid,'status':record['status'],'events':len((record.get('answer')or{}).get('events',[]))},flush=True)
    with ThreadPoolExecutor(max_workers=config['max_concurrency'])as pool:
        futures=[pool.submit(task,r)for r in requests if r['id'].replace('-verify-','-context-recovery-')not in ledger['requests']]
        for future in as_completed(futures):future.result()
    return {'phase':'context-recovery','requests':len(ledger['requests']),'completed':sum(r['status']=='completed'for r in ledger['requests'].values()),'campaign_estimated_usd':other+sum(r.get('estimated_usd',r.get('reserved_usd',0))for r in ledger['requests'].values())}
