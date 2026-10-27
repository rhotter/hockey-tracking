"""Create dense-review requests from automatic scan proposals, without labels."""
import argparse
from pathlib import Path
from detect_shots import read,sha,write,ROOT

def propose(protocol,scan,scan_sha):
    requests=[]
    for clip in protocol['clips']:
        intervals=[]
        for record in scan['requests'].values():
            if record['clip']!=clip['id']or record['status']!='completed':continue
            for event in record['answer']['events']:
                a=max(0,event['release_frame']-18)
                b=min(clip['frames'],max(event['outcome_frame']+18,event['release_frame']+30))
                intervals.append([a,b])
        merged=[]
        for a,b in sorted(intervals):
            if merged and a<=merged[-1][1]+15:merged[-1][1]=max(merged[-1][1],b)
            else:merged.append([a,b])
        for a,b in merged:
            requests.append({'id':f'{clip["id"]}-verify-{a:04d}','clip':clip['id'],'core':[a,b],
                'range':[max(0,a-30),min(clip['frames'],b+45)],'stride':3,'width':1280})
    return {'source_scan_sha256':scan_sha,
        'method':'Merge scan-proposed release/outcome intervals with temporal context; verifier receives raw frames, never scan answers. No manual labels used.',
        'requests':requests}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'experiments/sog-20261027/protocol.json')
    parser.add_argument('--scan',type=Path,default=ROOT/'experiments/sog-20261027/results/scan-ledger.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();write(args.output,propose(read(args.protocol),read(args.scan),sha(args.scan)))

if __name__=='__main__':main()
