"""Apply a predicted surface mask to frozen player boxes without interpolation."""
import math
import numpy as np
from scipy.ndimage import maximum_filter


def filter_row(row,mask,width,height,dilation):
    area=float(mask.mean())
    if area<.05 or area>.98:return row,{'fallback':True,'surface_fraction':area,'removed':0}
    ry=math.ceil(dilation*mask.shape[0]);rx=math.ceil(dilation*height/width*mask.shape[1])
    region=maximum_filter(mask.astype(np.uint8),size=(2*ry+1,2*rx+1)).astype(bool)
    predictions=[]
    for p in row['predictions']:
        x,y,u,v=p['bbox'];xx=(x+u)/2;yy=v-.02*(v-y)
        ix=min(mask.shape[1]-1,max(0,int(xx/width*mask.shape[1])))
        iy=min(mask.shape[0]-1,max(0,int(yy/height*mask.shape[0])))
        if region[iy,ix]:predictions.append(p)
    return {**row,'predictions':predictions},{'fallback':False,'surface_fraction':area,
        'removed':len(row['predictions'])-len(predictions),'dilation_pixels':[rx,ry]}
