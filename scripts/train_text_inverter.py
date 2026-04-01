"""
Train SD35TextInverter — image → text closed-loop inverter.

Reuses the (image, conditioning.pt) dataset from generate_inversion_training_data.py.
The prompt field inside conditioning.pt is used as the text target (CE loss).
The pooled_embeds field is used for the auxiliary conditioning regression loss.

Validation split is done by prompt directory (not by image) to prevent leakage:
all 8 seeds of a given prompt are either entirely in train or entirely in val.

Usage
-----
conda run -n superdiff python scripts/train_text_inverter.py \\
    [--data-dir experiments/inversion/training_data] \\
    [--ckpt-dir ckpt/text_inverter] \\
    [--epochs 60] \\
    [--batch-size 32] \\
    [--lr 2e-4] \\
    [--n-prefix 16] \\
    [--pool-loss-weight 0.1] \\
    [--val-fraction 0.1]

Phase 1 sanity check (single-concept fidelity) is run automatically at the
end of training via --eval-prompts; pass --skip-final-eval to suppress it.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.text_inverter import (
    SD35TextInverter,
    save_text_inverter,
    load_text_inverter,
    CLIP_MEAN,
    CLIP_STD,
    CLIP_SIZE,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextInversionDataset(Dataset):
    """
    Each item: (pixel_values, input_ids, attention_mask, pooled_embeds)

    pixel_values   : (3, 224, 224) float32 — CLIP-preprocessed image
    input_ids      : (max_text_len,) int64  — GPT-2 tokenised prompt (padded)
    attention_mask : (max_text_len,) int64  — 1 = real token, 0 = pad
    pooled_embeds  : (2048,) float32        — SD3.5 pooled conditioning target
    """

    def __init__(self, records: list, tokenizer, max_text_len: int = 64):
        self.records      = records
        self.tokenizer    = tokenizer
        self.max_text_len = max_text_len
        self.transform    = transforms.Compose([
            transforms.Resize(CLIP_SIZE,
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(CLIP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_path, cond_path = self.records[idx]

        # Image
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)                  # (3, 224, 224)

        # Conditioning
        cond         = torch.load(cond_path, map_location="cpu", weights_only=True)
        pooled       = cond["pooled_embeds"].squeeze(0).float()  # (2048,)
        prompt       = cond["prompt"]

        # Tokenise prompt for teacher-forcing
        # GPT-2 tokenizer: pad_token is set to eos_token in main()
        enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_text_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)          # (max_text_len,)
        attention_mask = enc["attention_mask"].squeeze(0)     # (max_text_len,)

        return pixel_values, input_ids, attention_mask, pooled


# ---------------------------------------------------------------------------
# Record building + train/val split
# ---------------------------------------------------------------------------

def build_records(data_dir: Path) -> list:
    """Return list of (img_path, cond_path) from dataset_index.json."""
    index_path = data_dir / "dataset_index.json"
    with open(index_path) as f:
        index = json.load(f)

    records = []
    for entry in index:
        prompt_dir = Path(entry["path"])
        cond_path  = prompt_dir / "conditioning.pt"
        if not cond_path.exists():
            continue
        for img_path in sorted(prompt_dir.glob("img_*.png")):
            records.append((str(img_path), str(cond_path)))
    return records


def split_records(records: list, val_fraction: float, seed: int = 42):
    """Split by prompt (cond_path), not by image, to avoid data leakage."""
    from collections import defaultdict
    by_prompt = defaultdict(list)
    for img_path, cond_path in records:
        by_prompt[cond_path].append((img_path, cond_path))

    prompts = list(by_prompt.keys())
    rng = random.Random(seed)
    rng.shuffle(prompts)

    n_val         = max(1, int(len(prompts) * val_fraction))
    val_prompts   = set(prompts[:n_val])
    train_prompts = set(prompts[n_val:])

    train_records = [r for p in train_prompts for r in by_prompt[p]]
    val_records   = [r for p in val_prompts   for r in by_prompt[p]]
    return train_records, val_records


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    # Keep CLIP eval to prevent BN drift (it's frozen anyway)
    model.clip.eval()

    total = ce_sum = pool_sum = cos_sum = 0.0

    for pixel_values, input_ids, attention_mask, pooled in loader:
        pixel_values   = pixel_values.to(device)
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pooled         = pooled.to(device)

        out = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_pooled=pooled,
        )

        optimizer.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()

        n         = pixel_values.shape[0]
        total    += out["loss"].item()      * n
        ce_sum   += out["ce_loss"].item()   * n
        pool_sum += out["pool_loss"].item() * n

        # Cosine similarity monitoring (no grad)
        with torch.no_grad():
            cls       = model._clip_cls(pixel_values)
            pred_pool = model.pool_head(cls)
            cos       = F.cosine_similarity(pred_pool, pooled.float(), dim=-1).mean().item()
        cos_sum += cos * n

    N = len(loader.dataset)
    return {"loss": total/N, "ce": ce_sum/N, "pool_mse": pool_sum/N, "pool_cos": cos_sum/N}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total = ce_sum = pool_sum = cos_sum = 0.0

    for pixel_values, input_ids, attention_mask, pooled in loader:
        pixel_values   = pixel_values.to(device)
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pooled         = pooled.to(device)

        out = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_pooled=pooled,
        )

        cls       = model._clip_cls(pixel_values)
        pred_pool = model.pool_head(cls)
        cos       = F.cosine_similarity(pred_pool, pooled.float(), dim=-1).mean().item()

        n         = pixel_values.shape[0]
        total    += out["loss"].item()      * n
        ce_sum   += out["ce_loss"].item()   * n
        pool_sum += out["pool_loss"].item() * n
        cos_sum  += cos                     * n

    N = len(loader.dataset)
    return {"loss": total/N, "ce": ce_sum/N, "pool_mse": pool_sum/N, "pool_cos": cos_sum/N}


# ---------------------------------------------------------------------------
# Phase 1 sanity eval: reconstruct single-concept prompts
# ---------------------------------------------------------------------------

_PHASE1_PROMPTS = [
    "a cat",
    "a dog",
    "a car",
    "a horse",
    "a person",
    "a red car",
    "a black cat",
    "a fluffy white dog",
]

@torch.no_grad()
def phase1_sanity_eval(model, tokenizer, device, n_samples: int = 3):
    """
    Quick closed-loop check without SD3.5.

    Encode each prompt with CLIP's text encoder, project to image-feature
    space, then decode — tests the round-trip through the mapper + GPT-2
    without needing to generate actual images.

    This is a proxy; the full closed-loop benchmark (image → p*) is in
    eval_text_inverter.py.
    """
    from transformers import CLIPModel
    clip_text = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()

    model.eval()
    print("\n── Phase 1 sanity eval (CLIP text→feature→text proxy) ──")
    print(f"  {'Prompt':40s}  {'Recovered':40s}")
    print("  " + "─" * 82)

    for prompt in _PHASE1_PROMPTS:
        # Encode prompt as if it were an image feature (proxy for a real image)
        tok = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=77).to(device)

        text_feat = clip_text.get_text_features(**tok).float()   # (1, D)
        # Project to CLIP-ViT CLS-like dimension via a tiny linear proxy:
        # This is only a proxy — real eval uses actual images (eval_text_inverter.py)
        # Here we just run a quick forward of generate() with a dummy pixel value
        # whose CLIP CLS feature is approximately the text feature.
        # Instead: generate directly from the text feature via mapper.
        prefix = model.mapper(F.normalize(text_feat, dim=-1).expand(1, -1))  # (1, n_prefix, gpt_dim)
        out_ids = model.gpt2.generate(
            inputs_embeds=prefix,
            max_new_tokens=40,
            num_beams=4,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        recovered = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        print(f"  {prompt:40s}  {recovered:40s}")

    print()
    del clip_text


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train SD35TextInverter")
    p.add_argument("--data-dir",         default="experiments/inversion/training_data")
    p.add_argument("--ckpt-dir",         default="ckpt/text_inverter")
    p.add_argument("--epochs",           type=int,   default=60)
    p.add_argument("--batch-size",       type=int,   default=32)
    p.add_argument("--lr",               type=float, default=2e-4)
    p.add_argument("--weight-decay",     type=float, default=1e-2)
    p.add_argument("--warmup-epochs",    type=int,   default=5)
    p.add_argument("--n-prefix",         type=int,   default=16,
                   help="Number of soft prefix tokens for GPT-2")
    p.add_argument("--pool-loss-weight", type=float, default=0.1,
                   help="λ_pool — weight on auxiliary pooled_embeds MSE loss")
    p.add_argument("--max-text-len",     type=int,   default=64,
                   help="Max GPT-2 token sequence length (prompt + padding)")
    p.add_argument("--val-fraction",     type=float, default=0.1)
    p.add_argument("--num-workers",      type=int,   default=4)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--clip-model-id",    default="openai/clip-vit-large-patch14")
    p.add_argument("--gpt2-model-id",    default="gpt2")
    p.add_argument("--freeze-gpt2",      action="store_true",
                   help="Freeze all GPT-2 weights; train only mapper + pool_head (~6M "
                        "params). Strongly recommended when training images < 5K.")
    p.add_argument("--skip-final-eval",  action="store_true",
                   help="Skip Phase 1 sanity eval after training")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ──────────────────────────────────────────────────────────
    print("Loading GPT-2 tokenizer ...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2_model_id)
    # GPT-2 has no pad token; set it to eos so padding is handled correctly
    tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ────────────────────────────────────────────────────────────
    print("Building dataset ...")
    records = build_records(data_dir)
    train_records, val_records = split_records(records, args.val_fraction, args.seed)
    print(f"  Train: {len(train_records)} images  |  Val: {len(val_records)} images")

    train_ds = TextInversionDataset(train_records, tokenizer, args.max_text_len)
    val_ds   = TextInversionDataset(val_records,   tokenizer, args.max_text_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ── Model ──────────────────────────────────────────────────────────────
    print(f"Building SD35TextInverter ...")
    model = SD35TextInverter(
        clip_model_id    = args.clip_model_id,
        gpt2_model_id    = args.gpt2_model_id,
        n_prefix         = args.n_prefix,
        pool_loss_weight = args.pool_loss_weight,
        freeze_gpt2      = args.freeze_gpt2,
    ).to(device)
    print(f"  Trainable: {model.num_trainable_params/1e6:.1f}M  "
          f"| Total: {model.num_total_params/1e6:.1f}M"
          + ("  [GPT-2 frozen]" if args.freeze_gpt2 else "  [GPT-2 trainable]"))

    # ── Optimiser + LR schedule ─────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                  weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return float(epoch + 1) / float(args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training ────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    log = []

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device)
        vl = validate(model, val_loader, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={tr['loss']:.4f} (ce={tr['ce']:.4f} pool={tr['pool_mse']:.4f})  "
            f"val_loss={vl['loss']:.4f} (ce={vl['ce']:.4f} cos={vl['pool_cos']:.3f})  "
            f"lr={lr_now:.2e}"
        )

        log.append({"epoch": epoch, "train": tr, "val": vl, "lr": lr_now})

        if vl["loss"] < best_val_loss:
            best_val_loss = vl["loss"]
            save_text_inverter(model, str(ckpt_dir / "best.pt"))
            print(f"  → best checkpoint (val_loss={best_val_loss:.4f})")

        if epoch % 10 == 0:
            save_text_inverter(model, str(ckpt_dir / f"epoch_{epoch:04d}.pt"))

    save_text_inverter(model, str(ckpt_dir / "final.pt"))

    with open(ckpt_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nDone. Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {ckpt_dir}")

    # ── Phase 1 sanity eval ─────────────────────────────────────────────────
    if not args.skip_final_eval:
        best_model = load_text_inverter(
            str(ckpt_dir / "best.pt"),
            clip_model_id=args.clip_model_id,
            gpt2_model_id=args.gpt2_model_id,
            device=device,
        )
        phase1_sanity_eval(best_model, tokenizer, device)


if __name__ == "__main__":
    main()
