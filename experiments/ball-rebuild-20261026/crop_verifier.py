"""A shared ImageNet encoder sees a small ball crop and its wider context."""
import torch
from torch import nn
from torchvision.models import resnet18,ResNet18_Weights


class BallVerifier(nn.Module):
    def __init__(self,pretrained=False):
        super().__init__()
        self.encoder=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder.fc=nn.Identity();self.head=nn.Linear(1024,1)

    def forward(self,x):
        features=self.encoder(torch.cat([x[:,:3],x[:,3:]],dim=0))
        a,b=features.chunk(2);return self.head(torch.cat([a,b],dim=1)).flatten()


class SpatialBallVerifier(nn.Module):
    """Keep stride-four features and center evidence for a few-pixel ball."""
    def __init__(self,pretrained=False):
        super().__init__()
        backbone=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        backbone.conv1.stride=(1,1);backbone.maxpool=nn.Identity()
        self.encoder=nn.Sequential(*list(backbone.children())[:-3])
        self.head=nn.Sequential(nn.Linear(1024,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,1))

    def forward(self,x):
        spatial=self.encoder(torch.cat([x[:,:3],x[:,3:]],dim=0))
        cy,cx=spatial.shape[-2]//2,spatial.shape[-1]//2
        center=spatial[:,:,cy-2:cy+2,cx-2:cx+2].mean((-2,-1))
        features=torch.cat([center,spatial.mean((-2,-1))],dim=1)
        a,b=features.chunk(2);return self.head(torch.cat([a,b],dim=1)).flatten()


def patch_inputs(im,point,jitter=(0,0)):
    import cv2,numpy as np
    x,y=np.array(point)+jitter;patches=[]
    for size in [64,192]:
        # Border padding preserves the proposal's location within both views.
        padded=cv2.copyMakeBorder(im,size,size,size,size,cv2.BORDER_REFLECT_101)
        crop=cv2.getRectSubPix(padded,(size,size),(float(x+size),float(y+size)))
        patches.append(cv2.resize(crop,(128,128)))
    a=np.concatenate(patches,axis=2).transpose(2,0,1).astype(np.float32)/255
    return (a-np.array([.485,.456,.406]*2,np.float32)[:,None,None])/np.array([.229,.224,.225]*2,np.float32)[:,None,None]
