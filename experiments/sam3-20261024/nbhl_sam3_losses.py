"""Box-only DETR-style supervision for SAM3 adaptation; no pseudo-mask loss."""

def loss_for(out,target):
    import torch
    import torch.nn.functional as F
    from torchvision.ops import generalized_box_iou,box_convert,sigmoid_focal_loss
    from scipy.optimize import linear_sum_assignment
    boxes=out['pred_boxes'][0].float();xyxy=box_convert(boxes,'cxcywh','xyxy')
    logits=out['pred_logits'][0,:,0].float()
    # The public processor additionally multiplies by decoder presence.
    probability=(logits.sigmoid()*out['presence_logit_dec'].float().sigmoid().reshape(-1)[0]).clamp(1e-6,1-1e-6)
    joint_logits=torch.logit(probability)
    targets=box_convert(target,'xyxy','cxcywh')
    cost=5*torch.cdist(boxes.detach(),targets,p=1)-2*generalized_box_iou(xyxy.detach(),target)-probability.detach()[:,None]
    a,b=linear_sum_assignment(cost.cpu().numpy());a=torch.as_tensor(a,device=boxes.device);b=torch.as_tensor(b,device=boxes.device)
    labels=torch.zeros_like(logits);labels[a]=1.
    classification=sigmoid_focal_loss(joint_logits,labels,alpha=.25,gamma=2.,reduction='sum')/max(len(target),1)
    l1=F.l1_loss(boxes[a],targets[b],reduction='sum')/max(len(target),1)
    giou=(1-generalized_box_iou(xyxy[a],target[b]).diag()).sum()/max(len(target),1)
    return classification+5*l1+2*giou,{'classification':float(classification.detach()),'l1':float(l1.detach()),'giou':float(giou.detach())}

