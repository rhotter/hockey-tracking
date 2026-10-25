"""Reference-bank camera registration with floor support and temporal checks."""
import numpy as np
import cv2

def project(H,points):
    return cv2.perspectiveTransform(np.asarray(points,np.float32).reshape(1,-1,2),np.asarray(H,np.float64))[0]

def fit(source,descriptors,target,target_descriptors,mode='background'):
    if descriptors is None or target_descriptors is None or min(len(source),len(target))<12:return None
    matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=64))
    pairs=matcher.knnMatch(descriptors,target_descriptors,k=2)
    good=[a for pair in pairs if len(pair)==2 for a,b in [pair]if a.distance<.72*b.distance]
    if len(good)<12:return None
    a=source[[m.queryIdx for m in good]];b=target[[m.trainIdx for m in good]]
    H,mask=cv2.findHomography(a,b,cv2.USAC_MAGSAC,2.,maxIters=10000,confidence=.999)
    if H is None:return None
    keep=mask.ravel().astype(bool);n=int(keep.sum())
    if n<12 or n/len(good)<.35:return None
    spread=np.ptp(a[keep],axis=0)
    hull=cv2.convexHull(a[keep]);area=float(cv2.contourArea(hull))
    if spread[0]<120 or spread[1]<45 or area<3500:return None
    error=np.linalg.norm(project(H,a[keep])-b[keep],axis=1)
    if np.quantile(error,.9)>2.5:return None
    return {'matrix':H,'inliers':n,'matches':len(good),'median_error_px':float(np.median(error)*1.5),
            'p90_error_px':float(np.quantile(error,.9)*1.5),'support_area_px':area*2.25,
            'score':float(n*n/len(good)),'support':b[keep]}

def register(variant,volume):
    import time,tarfile
    from concurrent.futures import ThreadPoolExecutor
    from overhead import BASE,OUT,SOURCE,START,FPS,DURATION,read,dump,sha
    cv2.setNumThreads(1)
    calibration=read('/work/calibration-v1.json');H0=np.asarray(calibration['image_to_relative_floor'])
    bankpath=BASE/'camera-bank-v1'/SOURCE/'bank.json';bank=read(bankpath)
    assert sha(bankpath)==calibration['camera_bank_sha256']
    S=np.diag([2/3,2/3,1]);Si=np.linalg.inv(S)
    features=[]
    for row in bank['frames']:
        if row['id']not in bank['registrations']:continue
        data=np.load(bankpath.parent/(row['id']+'-features.npz'))
        points=data['points'];desc=data['descriptors'];M=np.asarray(bank['registrations'][row['id']]['anchor_to_image'])
        floor=project(H0@np.linalg.inv(M)@Si,points)
        inside=(floor[:,0]>-4.005)&(floor[:,0]<3.632)&(floor[:,1]>-.97)&(floor[:,1]<.97)
        features.append({'id':row['id'],'points':points,'descriptors':desc,'M':M,'inside':inside})
    det=read(OUT/'detections.json')['frames']if(OUT/'detections.json').exists()else None
    keys=[];started=time.monotonic();stride=5
    grid=np.array([[x,y]for x in np.linspace(-4.005,3.632,17)for y in np.linspace(-1,1,7)],np.float32)
    def fit_bank(item,points,desc,mode):
        mask=item['inside']if mode=='floor'else np.ones(len(item['points']),bool)
        match=fit(item['points'][mask],item['descriptors'][mask],points,desc,mode)
        if match is None:return None
        M=Si@match.pop('matrix')@S@item['M']
        if not np.isfinite(M).all() or abs(np.linalg.det(M))<1e-10:return None
        try:H=H0@np.linalg.inv(M);G=np.linalg.inv(H)
        except np.linalg.LinAlgError:return None
        q=project(G,grid);visible=(q[:,0]>=0)&(q[:,0]<=1920)&(q[:,1]>=200)&(q[:,1]<=1080)
        if visible.sum()<6:return None
        area=cv2.contourArea(cv2.convexHull(q[visible]))
        if area<30000:return None
        support=match.pop('support')*1.5
        floor=project(H,support);floor_n=int(((floor[:,0]>-4.005)&(floor[:,0]<3.632)&(floor[:,1]>-1)&(floor[:,1]<1)).sum())
        return match|{'bank':item['id'],'H':H.tolist(),'floor_inliers':floor_n,'visible_grid_count':int(visible.sum())}
    indices=sorted(set(range(0,DURATION*FPS,stride))|{DURATION*FPS-1})
    if variant=='quick':indices=[r['frame']for r in read('/work/floor-line-review-v1.json')['frames']]
    with ThreadPoolExecutor(max_workers=8)as pool:
        for f in indices:
            raw=cv2.imread(str(OUT/'frames'/f'{f+1:06d}.jpg'))
            gray=cv2.resize(cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY),(1280,720))
            mask=np.full(gray.shape,255,np.uint8);mask[:80]=0
            if det:
                for box in det[f]['detections']:
                    if box[4]<.1:continue
                    x,y,u,v=np.rint(np.array(box[:4])*2/3).astype(int)
                    cv2.rectangle(mask,(max(0,x-8),max(0,y-8)),(u+8,v+8),0,-1)
            kp,desc=cv2.SIFT_create(nfeatures=6000,contrastThreshold=.015).detectAndCompute(gray,mask)
            points=np.array([k.pt for k in kp],np.float32)
            modes={}
            for mode in ['background','floor']:
                candidates=list(pool.map(lambda item:fit_bank(item,points,desc,mode),features))
                candidates=sorted([c for c in candidates if c],key=lambda c:-c['score'])
                selected=None;agreement=None
                if len(candidates)>=2:
                    best=candidates[0];ref=project(np.linalg.inv(np.array(best['H'])),grid)
                    visible=(ref[:,0]>=0)&(ref[:,0]<=1920)&(ref[:,1]>=200)&(ref[:,1]<=1080)
                    errors=[]
                    for other in candidates[1:6]:
                        errors.append(float(np.quantile(np.linalg.norm(project(np.linalg.inv(np.array(other['H'])),grid[visible])-ref[visible],axis=1),.9)))
                    agreement=min(errors)
                    if agreement<=20:selected=best
                modes[mode]={'selected':selected,'agreement_p90_px':agreement,'candidates':candidates[:3]}
            keys.append({'frame':f,'methods':modes})
            if f%150==0:
                print({'stage':'register','frame':f,'accepted':{k:v['selected']is not None for k,v in modes.items()},'seconds':round(time.monotonic()-started)},flush=True)
                dump(OUT/'registration-progress.json',{'frame':f,'keyframes':len(keys),'seconds':time.monotonic()-started})
    report={'source':SOURCE,'start_s':START,'duration_s':DURATION,'fps':FPS,'stride':stride,'keyframes':keys,
            'calibration':calibration,'calibration_sha256':sha('/work/calibration-v1.json'),
            'person_masked':det is not None,'seconds':time.monotonic()-started}
    if variant=='quick':
        from evaluate_geometry import evaluate
        result=evaluate(report,read('/work/floor-line-review-v1.json'))
        dump(OUT/'registration-quick.json',report);dump(OUT/'geometry-quick.json',result)
        return {k:{a:b for a,b in v.items()if a!='frames'}for k,v in result['methods'].items()}
    dump(OUT/'registration.json',report)
    out=OUT/'registration-review';out.mkdir(exist_ok=True)
    for row in keys:
        if row['frame']%(15*FPS)!=0 and row['frame']!=DURATION*FPS-1:continue
        raw=cv2.imread(str(OUT/'frames'/f'{row["frame"]+1:06d}.jpg'))
        for method,mode in row['methods'].items():
            im=raw.copy();sel=mode['selected']
            if sel:
                G=np.linalg.inv(np.array(sel['H']))
                for x in [-4.005,-3,-2,-1,0,1,2,3,3.632]:
                    q=project(G,[[x,-1],[x,1]])
                    if np.isfinite(q).all()and abs(q).max()<100000:cv2.line(im,tuple(np.rint(q[0]).astype(int)),tuple(np.rint(q[1]).astype(int)),(0,190,255),2)
                for y in [-1,0,1]:
                    q=project(G,[[-4.005,y],[3.632,y]])
                    if np.isfinite(q).all()and abs(q).max()<100000:cv2.line(im,tuple(np.rint(q[0]).astype(int)),tuple(np.rint(q[1]).astype(int)),(0,255,0),3)
            cv2.putText(im,f'{method} frame {row["frame"]} valid={sel is not None}',(30,130),cv2.FONT_HERSHEY_SIMPLEX,1,(0,240,255),2)
            cv2.imwrite(str(out/f'{method}-{row["frame"]:04d}.jpg'),cv2.resize(im,(1280,720)))
    with tarfile.open(OUT/'registration-review.tar.gz','w:gz')as tar:
        for path in out.iterdir():tar.add(path,arcname=path.name)
        tar.add(OUT/'registration.json',arcname='registration.json')
    return {'keyframes':len(keys),'seconds':report['seconds'],'accepted':{m:sum(r['methods'][m]['selected']is not None for r in keys)for m in ['background','floor']},'archive':str(OUT/'registration-review.tar.gz')}
