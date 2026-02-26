# Installation Guide

**ECCV 2026 — Hybridization vs. Co-Presence in Compositional Diffusion Models**

Python 3.11 · PyTorch 2.10 · CUDA 12.x · Stable Diffusion 3.5 Medium

---

## Prerequisites

- Conda (Miniconda or Anaconda) installed
- NVIDIA GPU with ≥ 16 GB VRAM (tested on RTX 3090 / RTX 4090 / RTX 5090)
- NVIDIA driver ≥ 525, CUDA toolkit ≥ 12.1
- HuggingFace account with access to `stabilityai/stable-diffusion-3.5-medium`

Check your CUDA version:
```bash
nvidia-smi
# Look for "CUDA Version: XX.X" in the top-right corner
```

---

## Step 1: Create the Conda Environment

```bash
conda create -n eccv2026 python=3.11 -y
conda activate eccv2026
```

---

## Step 2: Install PyTorch

Choose the command matching your CUDA version. PyTorch must be installed *before* the rest of the requirements.

**CUDA 12.1 (RTX 3090, RTX 4090, most server GPUs):**
```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 12.4:**
```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu124
```

**RTX 5090 (sm_120, requires PyTorch ≥ 2.6):**
```bash
pip install torch>=2.6.0 torchvision>=0.21.0 torchaudio>=2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
```

Verify:
```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'CUDA version: {torch.version.cuda}')
"
```

---

## Step 3: Install Remaining Dependencies

```bash
pip install -r requirements_eccv.txt
```

If you see version conflicts on torch/torchvision (pip trying to override what you just installed), add `--no-deps` for the torch-related packages:

```bash
pip install -r requirements_eccv.txt --no-deps
# Then reinstall just the non-torch packages normally if needed
```

---

## Step 4: Authenticate with HuggingFace

SD 3.5 Medium requires licence acceptance at https://huggingface.co/stabilityai/stable-diffusion-3.5-medium.

```bash
huggingface-cli login
# Paste your HuggingFace token when prompted
```

Verify model access:
```bash
python -c "
from huggingface_hub import model_info
info = model_info('stabilityai/stable-diffusion-3.5-medium')
print('Access OK:', info.id)
"
```

---

## Step 5: Verify the Full Stack

```bash
python -c "
import torch
from diffusers import SD3Transformer2DModel, AutoencoderKL
from transformers import CLIPTextModel, T5EncoderModel
import numpy, scipy, sklearn, matplotlib, plotly, PIL
import lpips, umap
print('All imports OK')
print('GPU:', torch.cuda.get_device_name(0))
"
```

---

## Step 6 (Optional): Weights & Biases

If you want experiment tracking:
```bash
wandb login
# Paste your W&B API key
```

---

## Step 7 (Optional): Plotly PNG Export

The histogram scripts export interactive HTML by default. PNG export requires Chrome/Chromium via Kaleido:

```bash
pip install kaleido
# If kaleido cannot find Chrome automatically:
plotly_get_chrome  # downloads a compatible Chrome binary
```

---

## RunPod / Remote GPU Setup

If working on a RunPod instance or any machine with a small root filesystem, redirect all caches to the network volume *before installing anything*:

```bash
export MAMBA_ROOT_PREFIX=/workspace/micromamba
export TMPDIR=/workspace/tmp
export PIP_CACHE_DIR=/workspace/pip-cache
export HF_HOME=/workspace/huggingface
mkdir -p $MAMBA_ROOT_PREFIX $TMPDIR $PIP_CACHE_DIR $HF_HOME

# Make permanent
echo "export MAMBA_ROOT_PREFIX=/workspace/micromamba" >> ~/.bashrc
echo "export TMPDIR=/workspace/tmp" >> ~/.bashrc
echo "export PIP_CACHE_DIR=/workspace/pip-cache" >> ~/.bashrc
echo "export HF_HOME=/workspace/huggingface" >> ~/.bashrc
source ~/.bashrc
```

Then create the environment with an explicit path (avoids root-filesystem quota):
```bash
conda create -p /workspace/envs/eccv2026 python=3.11 -y
conda activate /workspace/envs/eccv2026
# Then follow Steps 2–5 above
```

---

## GPU Memory Notes

SD 3.5 Medium in float16 uses approximately:

| Operation | VRAM |
|---|---|
| Transformer + text encoders (inference) | ~12 GB |
| VAE encode/decode | +2 GB |
| SuperDiff AND (two conditioning forward passes) | ~14 GB peak |
| Inverter training (batch size 16) | ~8 GB |

If you hit OOM errors:
- Add `--dtype float16` to generation scripts (default)
- Reduce `--batch-size` in training scripts
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before running

---

## Troubleshooting

**`UserWarning: NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible`**
→ Your PyTorch is too old. Use the RTX 5090 install command in Step 2.

**`OSError: stabilityai/stable-diffusion-3.5-medium is not accessible`**
→ Accept the model licence on HuggingFace and re-run `huggingface-cli login`.

**`ModuleNotFoundError: No module named 'notebooks'`**
→ Run scripts from the `eccv2026/` root directory, not from inside `scripts/`.

**`CUDA out of memory`**
→ Reduce batch size or add `--dtype float16`. Check `nvidia-smi` for any other processes consuming VRAM.

**`lpips` import error**
→ `pip install lpips` separately; it is not always pulled in correctly via `--no-deps`.
