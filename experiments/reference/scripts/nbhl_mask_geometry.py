"""Exact positive-mask bounds with four integers transferred from the GPU."""
_COORDINATES={}

def mask_box(logits):
    import torch
    assert logits.ndim==2 and all(n>0 for n in logits.shape)
    height,width=logits.shape;key=(str(logits.device),height,width)
    if key not in _COORDINATES:
        _COORDINATES[key]=(torch.arange(height,device=logits.device),torch.arange(width,device=logits.device))
    yy,xx=_COORDINATES[key];mask=logits>0;rows=mask.any(dim=1);cols=mask.any(dim=0)
    bounds=torch.stack([torch.where(cols,xx,width).amin(),torch.where(rows,yy,height).amin(),
                        torch.where(cols,xx,-1).amax()+1,torch.where(rows,yy,-1).amax()+1]).tolist()
    return None if bounds[2]==0 else bounds
