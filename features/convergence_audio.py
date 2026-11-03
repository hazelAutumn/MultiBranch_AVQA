from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

class ConvergenceAudio(nn.Module):
    """ImageNet ResNet18 on 3-channel replicated mel spectrograms."""
    def __init__(self, pretrained=True):
        super().__init__()
        net=resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem=nn.Sequential(net.conv1,net.bn1,net.relu,net.maxpool)
        self.l1,self.l2,self.l3,self.l4=net.layer1,net.layer2,net.layer3,net.layer4
    def forward(self,x):
        # [B,K,1,224,224]
        b,k=x.shape[:2];x=x.reshape(b*k,*x.shape[2:]).repeat(1,3,1,1)
        x=self.l1(self.stem(x));maps=[x]
        x=self.l2(x);maps.append(x)
        x=self.l3(x);maps.append(x)
        x=self.l4(x);pooled=x.mean((2,3)).reshape(b,k,-1)
        return [z.reshape(b,k,*z.shape[1:]) for z in maps],pooled
