from torch import nn
from models.branches import ConvergenceBranch,TechnicalBranch,AestheticBranch
from models.ctfa import CTFA

class MultiBranchAVQA(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__();self.convergence=ConvergenceBranch(pretrained)
        self.technical=TechnicalBranch(pretrained);self.aesthetic=AestheticBranch()
        self.ctfa=CTFA(self.convergence.out_dim,self.technical.out_dim,self.aesthetic.out_dim)
        self.head=nn.Sequential(nn.Linear(768,256),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,1))
    def forward(self,batch):
        fc=self.convergence(batch['clips'],batch['mels'])
        ft=self.technical(batch['patches'],batch['wave'])
        fa=self.aesthetic(batch['frames'],batch['wave16'])
        return self.head(self.ctfa(fc,ft,fa)).squeeze(-1)
