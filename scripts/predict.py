import argparse,torch
from models.avqa import MultiBranchAVQA
from data.preprocess import prepare_video
p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('--checkpoint',required=True);args=p.parse_args()
device='cuda' if torch.cuda.is_available() else 'cpu'
model=MultiBranchAVQA(pretrained=False).to(device)
model.load_state_dict(torch.load(args.checkpoint,map_location=device,weights_only=True));model.eval()
x={k:v.unsqueeze(0).to(device) for k,v in prepare_video(args.video).items()}
with torch.no_grad():print('Predicted MOS:',float(model(x).item()))
