import torch
from torch import nn

class CDAVF(nn.Module):
    """Equations (4)-(5): visual pooled feature * audio-derived channel weights,
    plus projected previous fusion at stages 2 and 3.
    """
    def __init__(self,channels=(64,128,256),out_dims=(128,256,512)):
        super().__init__();self.weights=nn.ModuleList();self.fcs=nn.ModuleList()
        for j,(c,d) in enumerate(zip(channels,out_dims)):
            self.weights.append(nn.Sequential(nn.Linear(c,c),nn.ReLU(),nn.Linear(c,c),nn.Sigmoid()))
            self.fcs.append(nn.Linear(c+(out_dims[j-1] if j else 0),d))
        self.out_dim=out_dims[-1]
    def forward(self,v_maps,a_maps):
        prev=None
        for j,(v,a) in enumerate(zip(v_maps,a_maps)):
            # [B,K,C,T,H,W] vs [B,K,C,H,W]
            v=v.mean(tuple(range(3,v.ndim)));a=a.mean(tuple(range(3,a.ndim)))
            fused=v*self.weights[j](a)
            if prev is not None:fused=torch.cat((prev,fused),dim=-1)
            prev=self.fcs[j](fused)
        return prev
