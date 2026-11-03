import torch
from torch import nn
from torchvision.models.video import r3d_18, R3D_18_Weights

class ConvergenceVisual(nn.Module):
    """Kinetics-400 R3D-18, three intermediate stages + final 512D feature."""
    def __init__(self, pretrained=True):
        super().__init__()
        net = r3d_18(weights=R3D_18_Weights.DEFAULT if pretrained else None)
        self.stem, self.l1, self.l2, self.l3, self.l4 = net.stem, net.layer1, net.layer2, net.layer3, net.layer4
    def forward(self, x):
        # x: [B,K,3,8,224,224]; return per-second feature maps
        b,k=x.shape[:2]; x=x.reshape(b*k,*x.shape[2:]); x=self.l1(self.stem(x))
        maps=[x]; x=self.l2(x); maps.append(x); x=self.l3(x); maps.append(x)
        x=self.l4(x); pooled=x.mean((2,3,4)).reshape(b,k,-1)
        return [z.reshape(b,k,*z.shape[1:]) for z in maps], pooled
