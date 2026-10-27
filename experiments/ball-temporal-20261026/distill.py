"""Distill sequence-model rejection into small heads over frozen visual features."""
from pathlib import Path
import hashlib
import json
import time

BASE=Path('/data/ball-hockey/reliability-20261018')
OUT=BASE/'ball-temporal-20261026-v1'
SOURCE=BASE/'overhead-20261025-v1/frames'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,separators=(',',':'))+'\n')


def classification(scores,labels,threshold):
    pairs=list(zip(scores,labels));tp=sum(p>=threshold and y==1 for p,y in pairs)
    fp=sum(p>=threshold and y==0 for p,y in pairs);fn=sum(p<threshold and y==1 for p,y in pairs)
    tn=sum(p<threshold and y==0 for p,y in pairs)
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}


def train(commit):
    import cv2,numpy as np,torch
    from torch import nn
    from crop_verifier import SpatialBallVerifier,patch_inputs
    from worker import registration
    torch.set_num_threads(4);cv2.setNumThreads(1);started=time.monotonic()
    protocol=read('/work/distill-protocol.json');teacher=read('/work/distill-labels.json')['rows']
    transfer=read('/work/distill-transfer-inputs.json')['rows']
    warm=BASE/'ball-rebuild-20261026-v1/verifier-v4/best.pt'
    assert sha(warm)==protocol['warmstart_sha256']
    verifier=SpatialBallVerifier().cuda().eval()
    verifier.load_state_dict(torch.load(warm,map_location='cpu',weights_only=False)['state_dict'])
    def features(path,point):
        raw=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB);h,w=raw.shape[:2];crops=[]
        for offset in [-3,0,3]:
            neighbor=path.parent/f'{int(path.stem)+offset:06d}.jpg'
            if offset:
                im=cv2.cvtColor(cv2.imread(str(neighbor)),cv2.COLOR_BGR2RGB)
                m,_=registration(im,raw)
                im=cv2.warpAffine(im,m,(w,h),borderMode=cv2.BORDER_REFLECT_101)
            else:im=raw
            crops.append(patch_inputs(im,point))
        x=torch.from_numpy(np.stack(crops)).cuda()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            spatial=verifier.encoder(torch.cat([x[:,:3],x[:,3:]],0))
            cy,cx=spatial.shape[-2]//2,spatial.shape[-1]//2
            center=spatial[:,:,cy-2:cy+2,cx-2:cx+2].mean((-2,-1))
            local=torch.cat([center,spatial.mean((-2,-1))],1);a,b=local.chunk(2)
            f=torch.cat([a,b],1).float()
        return f.cpu().numpy()
    feature_rows=[]
    for i,row in enumerate(teacher):
        path=SOURCE/f'{row["frame"]+1:06d}.jpg';assert sha(path)==row['source_sha256']
        feature_rows.append(features(path,row['pixel']))
        if i%30==0:print({'phase':'teacher_features','sample':i,'total':len(teacher)},flush=True)
    teacher_x=torch.from_numpy(np.stack(feature_rows)).cuda()
    transfer_features=[]
    for i,row in enumerate(transfer):
        if row['pixel'] is None:transfer_features.append(None);continue
        clip=row['clip'];folder=(BASE/'new-games-20261025-v1/frames'/clip)if clip.endswith('0600')else OUT/'extra-frames'/clip
        path=folder/f'{row["frame"]+1:06d}.jpg';assert sha(path)==row['source_sha256']
        transfer_features.append(features(path,row['pixel']))
        if i%30==0:print({'phase':'transfer_features','sample':i,'total':len(transfer)},flush=True)
    split={s:[i for i,r in enumerate(teacher) if r['split']==s]for s in ['train','development','test']}
    labels=torch.tensor([r['label']for r in teacher],device='cuda',dtype=torch.float32)
    thresholds=protocol['thresholds']
    def select(scores):
        labs=[teacher[i]['label']for i in split['development']]
        trials=[{'threshold':t,**classification([scores[i]for i in split['development']],labs,t)}for t in thresholds]
        return max(trials,key=lambda r:(r['f1'],-r['fp'],r['threshold']))
    def inputs(x,variant):
        return x[:,1]if variant!='temporal-head'else torch.cat([x[:,1],x[:,0]-x[:,1],x[:,2]-x[:,1]],1)
    results={};models={};dest=OUT/'distillation';dest.mkdir(exist_ok=True)
    for variant in ['original','image-head','temporal-head']:
        torch.manual_seed(protocol['seed']);np.random.seed(protocol['seed'])
        dims=3072 if variant=='temporal-head'else 1024
        head=nn.Sequential(nn.Linear(dims,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,1)).cuda()
        with torch.no_grad():
            head[0].weight.zero_();head[0].weight[:,:1024].copy_(verifier.head[0].weight)
            head[0].bias.copy_(verifier.head[0].bias);head[3].load_state_dict(verifier.head[3].state_dict())
        X=inputs(teacher_x,variant);head.eval()
        with torch.no_grad():initial=head(X).flatten().sigmoid().cpu().tolist()
        best={'epoch':0,'selection':select(initial),'state_dict':{k:v.detach().cpu().clone()for k,v in head.state_dict().items()}}
        history=[]
        if variant!='original':
            optimizer=torch.optim.AdamW(head.parameters(),lr=protocol['learning_rate'],weight_decay=.01)
            positive=[i for i in split['train']if teacher[i]['label']==1];negative=[i for i in split['train']if teacher[i]['label']==0]
            rng=np.random.default_rng(protocol['seed'])
            for epoch in range(1,protocol['epochs']+1):
                head.train()
                for step in range(8):
                    ids=np.concatenate([rng.choice(positive,16),rng.choice(negative,16)])
                    optimizer.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(head(X[ids]).flatten(),labels[ids]);loss.backward();optimizer.step()
                head.eval()
                with torch.no_grad():scores=head(X).flatten().sigmoid().cpu().tolist()
                selection=select(scores);history.append({'epoch':epoch,**selection})
                # Keep the earlier epoch on an exact selection tie.
                rank=lambda r:(r['f1'],-r['fp'])
                if rank(selection)>rank(best['selection']):best={'epoch':epoch,'selection':selection,'state_dict':{k:v.detach().cpu().clone()for k,v in head.state_dict().items()}}
        head.load_state_dict(best['state_dict']);head.eval()
        with torch.no_grad():scores=head(X).flatten().sigmoid().cpu().tolist()
        weightpath=dest/(variant+'.pt');torch.save(best,weightpath)
        threshold=best['selection']['threshold']
        detail=[{**r,'score':s,'accepted':s>=threshold}for r,s in zip(teacher,scores)]
        transferred=[]
        for row,f in zip(transfer,transfer_features):
            if f is None:score=None
            else:
                with torch.no_grad():score=float(head(inputs(torch.from_numpy(f[None]).cuda(),variant)).sigmoid().item())
            transferred.append({**row,'verifier_score':score,'accepted':score is not None and score>=threshold})
        results[variant]={'selected_epoch':best['epoch'],'threshold':threshold,'development':best['selection'],
            'test':classification([scores[i]for i in split['test']],[teacher[i]['label']for i in split['test']],threshold),
            'weights_sha256':sha(weightpath),'rows':detail,'transfer':transferred,'history':history}
        print({'variant':variant,'selected_epoch':best['epoch'],'development':best['selection'],'test':results[variant]['test']},flush=True)
    report={'protocol':protocol,'teacher_labels_sha256':sha('/work/distill-labels.json'),
        'transfer_inputs_sha256':sha('/work/distill-transfer-inputs.json'),'variants':results,'seconds':time.monotonic()-started,
        'limits':'Classifier agreement with model-generated labels on28later same-game proposals, not independent visual truth. Transfer reuses already opened sparse game checks. No published replay changed.'}
    write(dest/'report.json',report);commit();return {v:{k:r[k]for k in ['selected_epoch','threshold','development','test']}for v,r in results.items()}
