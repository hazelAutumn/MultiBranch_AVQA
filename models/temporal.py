from torch import nn

class TemporalTransformer(nn.Module):
    def __init__(self,input_dim,dim=256,layers=2,heads=4):
        super().__init__();self.proj=nn.Linear(input_dim,dim)
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(dim,heads,dim*4,batch_first=True),layers,enable_nested_tensor=False)
        self.out_dim=dim
    def forward(self,x,mask=None):
        z=self.encoder(self.proj(x),src_key_padding_mask=mask)
        if mask is None:return z.mean(1)
        valid=(~mask).unsqueeze(-1);return (z*valid).sum(1)/valid.sum(1).clamp_min(1)
