"""Conservative coordinate projection and path segmentation, independent of models."""
import numpy as np

BOUNDS=(-4.005,3.632,-1.,1.)

def project(H,points):
    p=np.asarray(points,dtype=float).reshape(-1,2)
    q=np.column_stack([p,np.ones(len(p))])@np.asarray(H,dtype=float).T
    with np.errstate(divide='ignore',invalid='ignore'):return q[:,:2]/q[:,2:3]

def usable(H):
    if H is None:return False
    H=np.asarray(H,float)
    if H.shape!=(3,3)or not np.isfinite(H).all():return False
    if abs(np.linalg.det(H))<1e-12:return False
    try:G=np.linalg.inv(H)
    except np.linalg.LinAlgError:return False
    # Validate the observed part of the floor. During a pan, remote off-screen
    # corners can legitimately lie behind the camera; do not reject that view.
    pixels=np.array([[x,y]for x in np.linspace(100,1820,9)for y in np.linspace(400,1000,7)])
    xy=project(H,pixels);lo,hi,far,near=BOUNDS
    inside=np.isfinite(xy).all(axis=1)&(xy[:,0]>=lo)&(xy[:,0]<=hi)&(xy[:,1]>=far)&(xy[:,1]<=near)
    if inside.sum()<3:return False
    denominator=np.column_stack([pixels[inside],np.ones(inside.sum())])@H[2]
    # Local image-to-floor orientation is positive on this camera side. Behind-
    # camera branches and reflected fits have the opposite Jacobian sign.
    return bool(np.all(np.linalg.det(H)/denominator**3>1e-8))

def floor_point(H,pixel,bounds=BOUNDS):
    if pixel is None or not usable(H):return None
    xy=project(H,[pixel])[0]
    denominator=np.asarray(H)[2]@[*pixel,1]
    if abs(denominator)<1e-9 or np.linalg.det(H)/denominator**3<=0:return None
    lo,hi,far,near=bounds
    if not np.isfinite(xy).all()or not(lo<=xy[0]<=hi and far<=xy[1]<=near):return None
    return [(float(xy[0])-lo)/(hi-lo),(float(xy[1])-far)/(near-far)]

def footpoint(pose,bbox,height=1080):
    if not pose or bbox[3]>=height-5:return None
    scores=np.asarray(pose.get('scores',[]),float)
    if scores.shape!=(2,)or not np.isfinite(scores).all()or min(scores)<.5:return None
    ankles=np.asarray(pose['ankles'],float)
    x,y,u,v=bbox;w=u-x;h=v-y
    if ankles.shape!=(2,2)or not np.isfinite(ankles).all()or w<=0 or h<=0:return None
    if np.any(ankles[:,0]<x-.1*w)or np.any(ankles[:,0]>u+.1*w):return None
    if np.any(ankles[:,1]<y+.55*h)or np.any(ankles[:,1]>v+.08*h):return None
    return ankles.mean(axis=0).tolist()

def visible_polygon(H,width=1920,height=1080,bounds=BOUNDS):
    """Clip the rink in world space against the camera frustum, including its horizon."""
    if not usable(H):return []
    G=np.linalg.inv(np.asarray(H,float));sample=project(H,[[width/2,height*.8]])[0]
    if G[2]@[*sample,1]<0:G=-G
    a,b,c=G
    lo,hi,far,near=bounds
    polygon=[np.array(p,float)for p in [[lo,far],[hi,far],[hi,near],[lo,near]]]
    for plane in [c,a,width*c-a,b,height*c-b]:
        output=[]
        for i,current in enumerate(polygon):
            previous=polygon[i-1];u=float(plane@[*previous,1]);v=float(plane@[*current,1])
            if (u>=0)!=(v>=0):output.append(previous+(current-previous)*(u/(u-v)))
            if v>=0:output.append(current)
        polygon=output
        if not polygon:break
    return [[(float(x)-lo)/(hi-lo),(float(y)-far)/(near-far)]for x,y in polygon]

def segment_path(observations,fps=15):
    """Never bridge a missing frame, fragment switch, or implausible position jump."""
    result=[];last=None;segment=-1
    for item in observations:
        row=dict(item);p=row.get('xy')
        if p is None:
            row['segment']=None;last=None;result.append(row);continue
        start=last is None or row['frame']!=last['frame']+1 or row['track_id']!=last['track_id']
        if not start:
            # Unitless guard, not a speed estimate: >=3% rink length or 10% width per frame.
            dx=np.abs(np.asarray(p)-np.asarray(last['xy']))
            start=bool(dx[0]>.03 or dx[1]>.10)
        if start:segment+=1
        row['segment']=segment;last=row;result.append(row)
    return result
