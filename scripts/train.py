import argparse,csv,random
import torch
from torch.utils.data import Dataset,DataLoader
from data.preprocess import prepare_video
from models.avqa import MultiBranchAVQA

class AVQADataset(Dataset):
    def __init__(self,csv_path):
        with open(csv_path,newline='') as f:self.rows=list(csv.DictReader(f))
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i];batch=prepare_video(row['path']);batch['mos']=torch.tensor(float(row['mos']))
        return batch

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--csv',required=True);ap.add_argument('--epochs',type=int,default=1000)
    ap.add_argument('--batch-size',type=int,default=1);ap.add_argument('--lr',type=float,default=1e-4)
    ap.add_argument('--output',default='avqa.pt');ap.add_argument('--no-pretrained',action='store_true');args=ap.parse_args()
    torch.manual_seed(42);random.seed(42)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=MultiBranchAVQA(pretrained=not args.no_pretrained).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    # Batch size 1: variable-length clips are supported without custom padding.
    if args.batch_size!=1:raise ValueError('Use batch-size 1; variable K requires custom collate.')
    loader=DataLoader(AVQADataset(args.csv),batch_size=1,shuffle=True,num_workers=0)
    for epoch in range(args.epochs):
        model.train();total=0.
        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items()};opt.zero_grad()
            loss=torch.nn.functional.mse_loss(model(batch),batch['mos']);loss.backward();opt.step();total+=loss.item()
        print(f'epoch={epoch+1} mse={total/len(loader):.5f}',flush=True)
    torch.save(model.state_dict(),args.output)
if __name__=='__main__':main()
