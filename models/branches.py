import torch
from torch import nn
from features.convergence_visual import ConvergenceVisual
from features.convergence_audio import ConvergenceAudio
from features.technical_visual import TechnicalVisual
from features.technical_audio import SoundNet
from features.aesthetic_visual import AestheticVisual
from features.aesthetic_audio import AestheticAudio
from models.cdavf import CDAVF
from models.temporal import TemporalTransformer

class ConvergenceBranch(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__();self.video=ConvergenceVisual(pretrained);self.audio=ConvergenceAudio(pretrained)
        self.cdavf=CDAVF();self.temporal=TemporalTransformer(512+512+512)
        self.out_dim=self.temporal.out_dim
    def forward(self,clips,mels):
        vm,v=self.video(clips);am,a=self.audio(mels)
        return self.temporal(torch.cat((self.cdavf(vm,am),a,v),dim=-1))

class TechnicalBranch(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__();self.video=TechnicalVisual(pretrained);self.audio=SoundNet()
        self.temporal=TemporalTransformer(self.video.out_dim)
        self.out_dim=self.temporal.out_dim+1024
    def forward(self,patches,wave):
        return torch.cat((self.audio(wave),self.temporal(self.video(patches))),dim=-1)

class AestheticBranch(nn.Module):
    def __init__(self):
        super().__init__();self.video=AestheticVisual();self.audio=AestheticAudio()
        self.out_dim=self.video.out_dim+self.audio.out_dim
    def forward(self,frames,wave16):
        return torch.cat((self.audio(wave16),self.video(frames)),dim=-1)
