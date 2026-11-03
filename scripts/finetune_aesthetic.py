"""UnB-AVC pre-finetuning: visual MOS MSE, audio MOS MSE + audio-type CE."""
import argparse,csv
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from features.aesthetic_visual import AestheticVisual
from features.aesthetic_audio import AestheticAudio
from data.preprocess import prepare_video

class FineTuneData(Dataset):
    def __init__(self,path):
        with open(path,newline='') as f:self.rows=list(csv.DictReader(f))
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i];x=prepare_video(row['path'])
        return x,torch.tensor(float(row['visual_mos'])),torch.tensor(float(row['audio_mos'])),torch.tensor(int(row['audio_class']))

def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--classes',type=int,default=3);p.add_argument('--output',default='aesthetic_finetuned.pt');args=p.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    v=AestheticVisual().to(device);a=AestheticAudio().to(device)
    vhead=nn.Linear(v.out_dim,1).to(device);aqhead=nn.Linear(a.out_dim,1).to(device);aclass=nn.Linear(a.out_dim,args.classes).to(device)
    opt=torch.optim.Adam(list(v.parameters())+list(a.parameters())+list(vhead.parameters())+list(aqhead.parameters())+list(aclass.parameters()),lr=1e-4)
    for epoch in range(args.epochs):
        total=0
        for x,qv,qa,ca in DataLoader(FineTuneData(args.csv),batch_size=1,shuffle=True):
            x={k:z.to(device) for k,z in x.items()};qv,qa,ca=qv.to(device),qa.to(device),ca.to(device)
            zv=v(x['frames']);za=a(x['wave16']);loss=nn.functional.mse_loss(vhead(zv).flatten(),qv)+nn.functional.mse_loss(aqhead(za).flatten(),qa)+nn.functional.cross_entropy(aclass(za),ca)
            opt.zero_grad();loss.backward();opt.step();total+=loss.item()
        print(epoch+1,total,flush=True)
    torch.save({'video':v.state_dict(),'audio':a.state_dict()},args.output)
if __name__=='__main__':main()
