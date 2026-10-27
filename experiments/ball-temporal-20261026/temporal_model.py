"""Native-resolution local ball heatmaps with explicit signed motion channels.

This is our ResNet18 decoder adaptation, not an implementation or checkpoint of
TrackNetV4/V5 or TOTNet. Those papers motivate temporal/motion ablations only.
"""
import torch
from torch import nn
from heatmap_model import BallHeatmap, focal_heatmap


class TemporalBall(BallHeatmap):
    def __init__(self,variant,checkpoint=None):
        super().__init__(temporal=False,pretrained=False)
        self.variant=variant
        if checkpoint is not None:self.load_state_dict(checkpoint['state_dict'])
        if not variant.startswith('rgb-'):
            original=self.encoder.conv1
            channels=15 if variant.startswith('motion') else 9
            self.encoder.conv1=nn.Conv2d(channels,64,7,stride=2,padding=3,bias=False)
            with torch.no_grad():
                self.encoder.conv1.weight.zero_()
                if '-center' in variant:
                    # Preserve the warm-start's RGB function exactly at step zero.
                    self.encoder.conv1.weight[:,3:6].copy_(original.weight)
                else:
                    self.encoder.conv1.weight[:,:9].copy_(original.weight.repeat(1,3,1,1)/3)


def inputs(frames,variant):
    """frames: B,T,C,H,W in 0..1; center is index 1, all crops registered."""
    mean=frames.new_tensor([.485,.456,.406])[None,None,:,None,None]
    std=frames.new_tensor([.229,.224,.225])[None,None,:,None,None]
    norm=(frames-mean)/std
    if variant.startswith('rgb-'):return norm[:,1]
    rgb=norm.flatten(1,2)
    if not variant.startswith('motion'):return rgb
    # Preserve the direction of appearance/disappearance instead of abs(diff).
    past=((frames[:,1]-frames[:,0])*4).clamp(-1,1)
    future=((frames[:,2]-frames[:,1])*4).clamp(-1,1)
    return torch.cat([rgb,past,future],1)
