from torch import nn
from transformers import Wav2Vec2Model

class AestheticAudio(nn.Module):
    def __init__(self,model_id='facebook/wav2vec2-base',hidden=256):
        super().__init__();self.encoder=Wav2Vec2Model.from_pretrained(model_id)
        self.gru=nn.GRU(self.encoder.config.hidden_size,hidden,batch_first=True)
        self.out_dim=hidden
    def forward(self,wave,attention_mask=None):
        # 16 kHz waveform [B,S]
        z=self.encoder(input_values=wave,attention_mask=attention_mask).last_hidden_state
        return self.gru(z)[1][-1]
