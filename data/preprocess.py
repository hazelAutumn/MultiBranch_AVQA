"""Video decoding and feature input preparation. Requires torchvision, torchaudio."""
import math
import torch
import torch.nn.functional as F
import torchaudio
from torchvision.io import read_video
from torchvision.transforms.functional import resize,normalize

CLIP_MEAN=(0.48145466,0.4578275,0.40821073)
CLIP_STD=(0.26862954,0.26130258,0.27577711)
IMAGENET_MEAN=(0.485,0.456,0.406)
IMAGENET_STD=(0.229,0.224,0.225)

def grid_patches(frame):
    # frame: [3,H,W]; nine 224x224 patches from 3x3 grid
    _,h,w=frame.shape; patches=[]
    for r in range(3):
        for c in range(3):
            y0=round(r*h/3);y1=round((r+1)*h/3)
            x0=round(c*w/3);x1=round((c+1)*w/3)
            patches.append(resize(frame[:,y0:y1,x0:x1],(224,224),antialias=True))
    return torch.stack(patches)

def prepare_video(path,seconds=8,fps_samples=8,frame_samples=8,sr=16000):
    video,audio,info=read_video(str(path),pts_unit='sec')
    if video.numel()==0:raise ValueError('No video frames: '+str(path))
    fps=float(info.get('video_fps',30));duration=min(seconds,video.shape[0]/fps)
    k=max(1,math.ceil(duration));video=video.permute(0,3,1,2).float()/255.
    def at(t):return video[min(video.shape[0]-1,int(t*fps))]
    clips=[];mels=[]
    # If source audio is absent, use zeros (explicitly missing modality).
    if audio.numel():
        if audio.ndim==2:audio=audio.mean(-1)
        original_sr=int(info.get('audio_fps',sr))
        wave=torchaudio.functional.resample(audio.float(),original_sr,sr) if original_sr!=sr else audio.float()
    else:wave=torch.zeros(max(1,int(duration*sr)))
    mel_fn=torchaudio.transforms.MelSpectrogram(sample_rate=sr,n_mels=128)
    for j in range(k):
        clip=torch.stack([resize(at(min(duration-1e-4,j+(z+0.5)/fps_samples)),(224,224),antialias=True) for z in range(fps_samples)])
        clips.append(normalize(clip,IMAGENET_MEAN,IMAGENET_STD).permute(1,0,2,3))
        chunk=wave[j*sr:(j+1)*sr];chunk=F.pad(chunk,(0,max(0,sr-chunk.numel())))
        mel=torch.log1p(mel_fn(chunk)).unsqueeze(0)
        mels.append(F.interpolate(mel.unsqueeze(0),size=(224,224),mode='bilinear',align_corners=False)[0])
    times=torch.linspace(0,max(0,duration-1e-4),frame_samples)
    raw_frames=torch.stack([resize(at(float(t)),(224,224),antialias=True) for t in times])
    frames=normalize(raw_frames,CLIP_MEAN,CLIP_STD)
    # Technical crops retain full original frame until cropping
    raw_technical=torch.stack([at(float(t)) for t in times]) if len({tuple(at(float(t)).shape) for t in times})==1 else None
    patches=torch.stack([grid_patches(f) for f in raw_technical])
    patches=normalize(patches,IMAGENET_MEAN,IMAGENET_STD)
    wave=wave[:seconds*sr];wave=F.pad(wave,(0,max(0,seconds*sr-wave.numel())))
    return dict(clips=torch.stack(clips),mels=torch.stack(mels),patches=patches,frames=frames,wave=wave,wave16=wave)
