import torch
from torch import nn
from transformers import CLIPVisionModel

class AestheticVisual(nn.Module):
    def __init__(self,model_id='openai/clip-vit-base-patch32',hidden=256):
        super().__init__();self.encoder=CLIPVisionModel.from_pretrained(model_id)
        self.gru=nn.GRU(self.encoder.config.hidden_size,hidden,batch_first=True)
        self.out_dim=hidden
    def forward(self,frames):
        # [B,I,3,224,224] CLIP normalized images
        b,i=frames.shape[:2];x=frames.reshape(b*i,*frames.shape[2:])
        z=self.encoder(pixel_values=x).pooler_output.reshape(b,i,-1)
        return self.gru(z)[1][-1]
