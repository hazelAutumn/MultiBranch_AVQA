import torch
from torch import nn
import timm

class TechnicalVisual(nn.Module):
    """Swin features averaged across 9 spatial crops per frame."""
    def __init__(self,pretrained=True,model_name='swin_tiny_patch4_window7_224'):
        super().__init__();self.backbone=timm.create_model(model_name,pretrained=pretrained,num_classes=0)
        self.out_dim=self.backbone.num_features
    def forward(self,patches):
        # [B,I,9,3,224,224]
        b,i,n=patches.shape[:3]
        z=self.backbone(patches.reshape(b*i*n,*patches.shape[3:]))
        if z.ndim>2:z=z.mean(tuple(range(1,z.ndim-1)))
        return z.reshape(b,i,n,-1).mean(2)
