# Multi-Branch AVQA — research implementation

PyTorch implementation based on Kwong et al., *Multi-Branch Aesthetic and Technical Perspectives With Cross Tri-Fusion Attention for No-Reference Audio-Visual Quality Assessment* (IEEE TCSVT, 2026).

## Files

Each feature extractor is separate:
- `features/convergence_visual.py` — 3D ResNet-18 (Kinetics-400)
- `features/convergence_audio.py` — 2D ResNet-18 (ImageNet) on log-mel
- `features/technical_visual.py` — Swin Transformer, nine patches/frame
- `features/technical_audio.py` — **SoundNet-style approximation** (original weights not included)
- `features/aesthetic_visual.py` — CLIP vision encoder + GRU
- `features/aesthetic_audio.py` — Wav2Vec2 + GRU

Fusion and orchestration: `models/cdavf.py`, `models/ctfa.py`, `models/temporal.py`, `models/branches.py`, `models/avqa.py`.

## Install

```bash
pip install -r requirements.txt
```

The torchvision video reader requires a working FFmpeg/PyAV installation. Hugging Face and timm checkpoints require network access for first download.

## Data CSV

For main training: `path,mos` with paths to videos and numerical MOS. For aesthetic pre-finetuning: `path,visual_mos,audio_mos,audio_class` (zero-based integer audio class).

```bash
python -m scripts.finetune_aesthetic --csv unb_train.csv
python -m scripts.train --csv train.csv --epochs 1000
python -m scripts.predict sample.mp4 --checkpoint avqa.pt
```

**Important:** Pre-finetuning checkpoints are not loaded automatically into the full model. Load them explicitly before training:

```python
weights=torch.load('aesthetic_finetuned.pt',map_location='cpu',weights_only=True)
model.aesthetic.video.load_state_dict(weights['video'])
model.aesthetic.audio.load_state_dict(weights['audio'])
```

## Fidelity and limitations

This is a **research-oriented reimplementation, not the authors' official code or a verified numerical reproduction**. Paper equations (4)-(6), (7)-(10), (11)-(19), and (20)-(27) guide the design. Some details are unspecified: precise SoundNet checkpoint, layer sizes, convolution kernel choices, temporal Transformer settings, normalization, audio resampling, and sampling schedules. Those are explicit implementation choices here. SoundNet-style audio features are randomly initialized unless you load compatible weights; for faithful reproduction replace with original pretrained SoundNet architecture and weights. CLIP and Wav2Vec2 pretraining checkpoints are defaults and may differ from the authors'.

The main script is a baseline trainer, **not** the paper's full 5-fold, 20%-validation protocol. The code expects per-video tensors, and video decoding is expensive. No MOS calibration is applied. Keep subject/reference leakage out of your train/validation/test splits. Use the pretrained encoders, aesthetic pre-finetuning, proper MOS scaling, and a held-out evaluation protocol for meaningful results.
