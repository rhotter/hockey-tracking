"""Refine approximate blue-paint marks using raw pixels only; never read camera predictions."""
from pathlib import Path
import json
import argparse
import hashlib
import numpy as np
import cv2

def paint_line(image,points):
    points=np.asarray(points,float);coef=np.polyfit(points[:,0],points[:,1],1)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask=(hsv[:,:,0]>=100)&(hsv[:,:,0]<=130)&(hsv[:,:,1]>=45)&(hsv[:,:,2]<190)&(hsv[:,:,2]>40)
    samples=[]
    for x in range(max(0,int(points[:,0].min()-80)),min(image.shape[1],int(points[:,0].max()+80)),2):
        center=int(np.polyval(coef,x));lo=max(0,center-60);hi=min(image.shape[0],center+60)
        yy=np.flatnonzero(mask[lo:hi,x])+lo
        if not len(yy):continue
        runs=np.split(yy,np.flatnonzero(np.diff(yy)>1)+1)
        runs=[r for r in runs if 3<=len(r)<=35]
        if not runs:continue
        best=min(runs,key=lambda r:abs(np.median(r)-center))
        samples.append([x,float(np.median(best))])
    samples=np.array(samples,float)
    if len(samples)<35:raise ValueError('Insufficient paint evidence')
    rng=np.random.default_rng(17);best=None
    for _ in range(300):
        a,b=samples[rng.choice(len(samples),2,replace=False)]
        if abs(a[0]-b[0])<50:continue
        slope=(a[1]-b[1])/(a[0]-b[0]);intercept=a[1]-slope*a[0]
        if abs(slope-coef[0])>.2:continue
        residual=abs(samples[:,1]-(slope*samples[:,0]+intercept))
        keep=residual<3
        if best is None or keep.sum()>best.sum():best=keep
    if best is None or best.sum()<30:raise ValueError('No supported paint line')
    fit=np.polyfit(samples[best,0],samples[best,1],1)
    updated=[[float(x),round(float(np.polyval(fit,x)),2)]for x,y in points]
    return updated,{'pixel_columns':int(best.sum()),'x_span':float(np.ptp(samples[best,0])),
                    'median_vertical_residual_px':float(np.median(abs(samples[best,1]-np.polyval(fit,samples[best,0]))))}

def main():
    p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('images',type=Path);p.add_argument('out',type=Path);p.add_argument('review',type=Path);a=p.parse_args()
    ref=json.loads(a.reference.read_text());ref['id']='overhead-floor-line-review-v2';ref['parent_sha256']=hashlib.sha256(a.reference.read_bytes()).hexdigest()
    ref['status']='development_reference_refined_from_raw_paint_after_v1_error_review'
    ref['correction']='Original manual blue-line points were approximate. V2 snaps blue marks to a robust paint-center fit within the manually identified corridor. No calibration matrix or model output is read by this script. All refined line overlays require visual review. Other marks are unchanged.'
    a.review.mkdir(parents=True,exist_ok=True)
    for frame in ref['frames']:
        path=a.images/f'raw-{frame["frame"]:04d}.jpg';assert hashlib.sha256(path.read_bytes()).hexdigest()==frame['image_sha256']
        im=cv2.imread(str(path));overlay=im.copy()
        for group in frame['marks']:
            if 'blue'in group['line']:
                old=group['points'];group['points'],group['paint_fit']=paint_line(im,old);group['original_points']=old
                for x,y in old:cv2.circle(overlay,(round(x),round(y)),5,(0,255,255),2)
            for x,y in group['points']:cv2.circle(overlay,(round(x),round(y)),4,(0,255,0),-1)
            if 'blue'in group['line']:
                cv2.line(overlay,tuple(np.rint(group['points'][0]).astype(int)),tuple(np.rint(group['points'][-1]).astype(int)),(0,255,0),2)
        cv2.imwrite(str(a.review/f'marks-{frame["frame"]:04d}.jpg'),overlay)
    a.out.write_text(json.dumps(ref,indent=2)+'\n')

if __name__=='__main__':main()
