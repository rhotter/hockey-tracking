"""Bound SAM3 decoded frame storage without discarding temporal model state."""
from pathlib import Path
from collections import OrderedDict


class Frames:
    def __init__(self,folder,size,mean,std,capacity=4):
        import torch
        from sam3.model.io_utils import IMAGE_EXTS
        self.paths=sorted([p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS],key=lambda p:int(p.stem))
        assert self.paths
        self.size=size;self.capacity=capacity;self.cache=OrderedDict();self.loads=0;self.hits=0
        self.mean=torch.tensor(mean,dtype=torch.float16)[:,None,None]
        self.std=torch.tensor(std,dtype=torch.float16)[:,None,None]
        self.height=self.width=None;self[0]
    def __len__(self):return len(self.paths)
    def __getitem__(self,index):
        from sam3.model.io_utils import _load_img_as_tensor
        import operator
        index=operator.index(index)
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        if index in self.cache:
            self.hits+=1;self.cache.move_to_end(index);return self.cache[index]
        image,h,w=_load_img_as_tensor(str(self.paths[index]),self.size)
        if self.height is None:self.height,self.width=h,w
        assert (h,w)==(self.height,self.width)
        # Exactly the eager loader's fp32->fp16 assignment then fp16 normalization.
        image=image.half();image.sub_(self.mean).div_(self.std)
        if len(self.cache)>=self.capacity:self.cache.popitem(last=False)
        self.cache[index]=image;self.loads+=1;return image
    def stats(self):return {'frames':len(self),'capacity':self.capacity,'loads':self.loads,'hits':self.hits,'max_decoded_bytes':self.capacity*3*self.size*self.size*2}


def install():
    import sam3.model.sam3_video_inference as module
    original=module.load_resource_as_video_frames;instances=[]
    def load(resource_path,image_size,offload_video_to_cpu,img_mean=(.5,.5,.5),img_std=(.5,.5,.5),**kwargs):
        if not isinstance(resource_path,str) or not Path(resource_path).is_dir() or not offload_video_to_cpu:
            return original(resource_path,image_size,offload_video_to_cpu,img_mean,img_std,**kwargs)
        frames=Frames(resource_path,image_size,img_mean,img_std);instances.append(frames)
        return frames,frames.height,frames.width
    module.load_resource_as_video_frames=load
    return lambda:setattr(module,'load_resource_as_video_frames',original),instances
