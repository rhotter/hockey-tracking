"""Small heatmap baseline with generic ImageNet initialization, no sports weights."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights


class Up(nn.Module):
    def __init__(self, incoming, skip, outgoing):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(incoming+skip,outgoing,3,padding=1,bias=False),
            nn.BatchNorm2d(outgoing),nn.ReLU(inplace=True),
            nn.Conv2d(outgoing,outgoing,3,padding=1,bias=False),nn.BatchNorm2d(outgoing),nn.ReLU(inplace=True))

    def forward(self,x,skip):
        return self.layers(torch.cat([F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip],1))


class BallHeatmap(nn.Module):
    def __init__(self, temporal=False, pretrained=False):
        super().__init__()
        self.temporal=temporal
        encoder=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        if temporal:
            original=encoder.conv1
            encoder.conv1=nn.Conv2d(9,64,7,stride=2,padding=3,bias=False)
            with torch.no_grad():encoder.conv1.weight.copy_(original.weight.repeat(1,3,1,1)/3)
        self.encoder=encoder
        self.up3=Up(512,256,256);self.up2=Up(256,128,128);self.up1=Up(128,64,64);self.up0=Up(64,64,32)
        self.head=nn.Conv2d(32,1,1)
        nn.init.constant_(self.head.bias,-2.19)

    def forward(self,x):
        e=self.encoder;s0=e.relu(e.bn1(e.conv1(x)));s1=e.layer1(e.maxpool(s0));s2=e.layer2(s1);s3=e.layer3(s2);s4=e.layer4(s3)
        return self.head(self.up0(self.up1(self.up2(self.up3(s4,s3),s2),s1),s0))


def focal_heatmap(logits,target):
    # CenterNet-style focal objective. Only reviewed current-frame targets enter.
    p=logits.float().sigmoid().clamp(1e-5,1-1e-5)
    positive=(target==1).float();negative=(target<1).float()
    pos=-(1-p).pow(2)*p.log()*positive
    neg=-p.pow(2)*(1-p).log()*(1-target).pow(4)*negative
    return (pos.sum()+neg.sum())/positive.sum().clamp_min(1)
