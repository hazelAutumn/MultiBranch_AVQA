import torch
from torch import nn

class SoundNet(nn.Module):
    """SoundNet-style 1D CNN. Load original SoundNet weights separately if available.
    This is NOT a claim of exact equivalence to the original pretrained SoundNet.
    """
    def __init__(self,out_dim=1024):
        super().__init__()
        channels=[1,16,32,64,128,256,512,1024]
        self.net=nn.Sequential(*[layer for i in range(len(channels)-1) for layer in (
            nn.Conv1d(channels[i],channels[i+1],kernel_size=7,stride=2,padding=3),
            nn.BatchNorm1d(channels[i+1]),nn.ReLU(inplace=True))])
        self.proj=nn.Linear(1024,out_dim)
    def forward(self,wave):
        # [B,S]
        return self.proj(self.net(wave.unsqueeze(1)).mean(-1))
