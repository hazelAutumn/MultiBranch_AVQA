import torch
from torch import nn

class CTFA(nn.Module):
    """Equations (20)-(26): three pair/triple sums and attention weighting."""
    def __init__(self,c_dim,t_dim,a_dim,dim=256):
        super().__init__()
        self.proj=nn.ModuleList([nn.Linear(d,dim) for d in (t_dim,c_dim,a_dim)])
        self.refine=nn.ModuleList([nn.Sequential(nn.Conv1d(1,8,3,padding=1),nn.ReLU(),nn.Conv1d(8,1,3,padding=1)) for _ in range(3)])
        self.attn=nn.ModuleList([nn.Sequential(nn.Linear(dim,dim),nn.ReLU(),nn.Linear(dim,dim),nn.Sigmoid()) for _ in range(3)])
    def forward(self,fc,ft,fa):
        t,c,a=[p(x) for p,x in zip(self.proj,(ft,fc,fa))]
        ht,hc,ha=[m(x.unsqueeze(1)).squeeze(1) for m,x in zip(self.refine,(t,c,a))]
        ft_ctf=hc+ht;fc_ctf=hc+ht+ha;fa_ctf=hc+ha
        return torch.cat((t*self.attn[0](ft_ctf),c*self.attn[1](fc_ctf),a*self.attn[2](fa_ctf)),dim=-1)
