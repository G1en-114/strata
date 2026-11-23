"""Layer parser: picture -> the painter's 6 semantic layers (background, body, skin, features, hair, accessories).

    python tools/train_layer_parser.py --out runs/layer_parser            # ~1 h on the laptop GPU
    python tools/train_layer_parser.py --out runs/layer_parser --eval-only

Why: tracing an arbitrary picture (tools/paint_image.py) needs layers. Colour clusters are not semantic: on a
light-blue-haired anime girl hair, skin and shirt fall into one cluster, and "redraw the hair" then deleted most of
the picture. The model was trained on semantic layers, so edits only work when the traced layers mean the same.
Data: the training labels we already have, both domains, the painter's split (validation images held out):
  anime  data/anime_portraits256_v2_l6view  images_XXX.npy + labels_XXX.npy (pseudo-labels, tools/anime_layers.py)
  faces  data/celebamask256                 images_XXX.npy + labels_XXX.npy (CelebAMask-HQ masks -> 6 layers)
Augmentation: flips, scale / shift, and hue rotation + saturation / brightness jitter, so hair is learnt by shape
and position, not by colour. Small U-Net (~2.6 M parameters), cross-entropy; reports per-layer IoU on validation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from lpp.data.splits import load_split_manifest  # noqa: E402

LAYERS = ["background", "body", "skin", "features", "hair", "accessories"]
SOURCES = [("anime", ROOT / "data/anime_portraits256_v2_l6view"), ("face", ROOT / "data/celebamask256")]


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, n_classes: int = 6, w: int = 24):
        super().__init__()
        ch = [w, 2 * w, 4 * w, 8 * w, 8 * w]
        self.down = nn.ModuleList([block(3, ch[0])] + [block(ch[i], ch[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([block(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(ch[0], n_classes, 1)

    def forward(self, x):                       # x [B, 3, 256, 256] in [0, 1]
        x = (x - 0.5) / 0.25
        skips = []
        for i, d in enumerate(self.down):
            x = d(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for u in self.up:
            s = skips.pop()
            x = u(torch.cat([F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1))
        return self.head(x)


def index(split_manifest: Path):
    """[(source, dir, shard, row, is_val)] for every image with labels."""
    splits = load_split_manifest(split_manifest)
    items = []
    for name, d in SOURCES:
        idx = json.loads((d / "index.json").read_text())
        for s in idx["shards"]:
            if not (d / f"labels_{s['shard']:03d}.npy").exists():
                continue
            for r, i in enumerate(s["ids"]):
                items.append((name, d, s["shard"], r, splits.get(int(i)) == "val"))
    return items


class Shards:
    def __init__(self):
        self.mm = {}

    def get(self, d: Path, shard: int, row: int):
        k = (str(d), shard)
        if k not in self.mm:
            self.mm[k] = (np.load(d / f"images_{shard:03d}.npy", mmap_mode="r"), np.load(d / f"labels_{shard:03d}.npy", mmap_mode="r"))
        img, lab = self.mm[k]
        return np.array(img[row]), np.array(lab[row])


def augment(x: torch.Tensor, y: torch.Tensor, g: torch.Generator):
    """x [B, 3, S, S] float, y [B, S, S] long, on the GPU."""
    B, _, S, _ = x.shape
    dev = x.device
    flip = torch.rand(B, device=dev, generator=g) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    y = torch.where(flip[:, None, None], y.flip(-1), y)
    # scale / shift with one affine grid (nearest for labels)
    sc = 0.8 + 0.45 * torch.rand(B, device=dev, generator=g)
    sh = (torch.rand(B, 2, device=dev, generator=g) - 0.5) * 0.3
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = theta[:, 1, 1] = 1 / sc
    theta[:, :, 2] = sh
    grid = F.affine_grid(theta, (B, 3, S, S), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    y = F.grid_sample(y[:, None].float(), grid, mode="nearest", padding_mode="border", align_corners=False)[:, 0].long()
    # hue rotation (YIQ), saturation, brightness, contrast
    a = (torch.rand(B, device=dev, generator=g) * 2 * math.pi)[:, None, None]
    yiq = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], device=dev)
    q = torch.einsum("ij,bjhw->bihw", yiq, x)
    i_, q_ = q[:, 1] * a.cos() - q[:, 2] * a.sin(), q[:, 1] * a.sin() + q[:, 2] * a.cos()
    sat = (0.6 + 0.8 * torch.rand(B, device=dev, generator=g))[:, None, None]
    q = torch.stack([q[:, 0], i_ * sat, q_ * sat], 1)
    x = torch.einsum("ij,bjhw->bihw", torch.linalg.inv(yiq), q)
    br = (0.8 + 0.4 * torch.rand(B, device=dev, generator=g))[:, None, None, None]
    ct = (0.8 + 0.4 * torch.rand(B, device=dev, generator=g))[:, None, None, None]
    m = x.mean((1, 2, 3), keepdim=True)
    x = ((x - m) * ct + m) * br
    return x.clamp(0, 1), y


def batches(items, shards: Shards, B: int, rng: np.random.Generator):
    while True:
        sel = rng.choice(len(items), size=B, replace=False)
        imgs, labs = zip(*(shards.get(items[k][1], items[k][2], items[k][3]) for k in sel))
        yield torch.from_numpy(np.stack(imgs)), torch.from_numpy(np.stack(labs))


@torch.no_grad()
def evaluate(model, items, shards, n=600):
    model.eval()
    inter = torch.zeros(len(LAYERS)); union = torch.zeros(len(LAYERS)); per = {}
    for name in ("anime", "face"):
        sub = [it for it in items if it[0] == name][: n // 2]
        I = torch.zeros(len(LAYERS)); U = torch.zeros(len(LAYERS))
        for k in range(0, len(sub), 32):
            imgs, labs = zip(*(shards.get(it[1], it[2], it[3]) for it in sub[k:k + 32]))
            x = torch.from_numpy(np.stack(imgs)).cuda().permute(0, 3, 1, 2).float() / 255
            y = torch.from_numpy(np.stack(labs)).cuda().long()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model(x).argmax(1)
            for c in range(len(LAYERS)):
                I[c] += ((p == c) & (y == c)).sum().item(); U[c] += ((p == c) | (y == c)).sum().item()
        per[name] = {LAYERS[c]: round(float(I[c] / U[c]), 3) if U[c] > 0 else None for c in range(len(LAYERS))}
        inter += I; union += U
    model.train()
    miou = float((inter / union.clamp(min=1))[union > 0].mean())
    return miou, per


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--split-manifest", type=Path, default=ROOT / "data/split_v3_faces_anime.jsonl")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    items = index(args.split_manifest)
    train = [it for it in items if not it[4]]
    val = [it for it in items if it[4]]
    print(f"{len(train)} train / {len(val)} val images "
          f"({sum(it[0] == 'anime' for it in train)} anime, {sum(it[0] == 'face' for it in train)} faces)", flush=True)
    shards = Shards()
    model = UNet().cuda()
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters", flush=True)
    ck = args.out / "layer_parser.pt"
    if args.eval_only:
        model.load_state_dict(torch.load(ck, map_location="cuda")["model"])
        print(json.dumps(evaluate(model, val, shards), indent=1))
        return
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng, g = np.random.default_rng(0), torch.Generator(device="cuda").manual_seed(0)
    it = batches(train, shards, args.batch, rng)
    # class weights: features / accessories are small
    wts = torch.tensor([0.5, 1.0, 1.0, 2.0, 1.0, 1.5], device="cuda")
    t0 = time.time()
    log = open(args.out / "log.jsonl", "a")
    for step in range(1, args.steps + 1):
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * step / args.steps)) * min(1.0, step / 300)
        for grp in opt.param_groups:
            grp["lr"] = lr
        imgs, labs = next(it)
        x = imgs.cuda(non_blocking=True).permute(0, 3, 1, 2).float() / 255
        y = labs.cuda(non_blocking=True).long()
        x, y = augment(x, y, g)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x).float(), y, weight=wts)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            rec = {"step": step, "loss": round(loss.item(), 4), "lr": round(lr, 6), "s": round(time.time() - t0)}
            if step % 2000 == 0 or step == args.steps:
                miou, per = evaluate(model, val, shards)
                rec |= {"val_miou": round(miou, 4), "val_iou": per}
                torch.save({"model": model.state_dict(), "layers": LAYERS, "step": step}, ck)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
    print(f"saved {ck}", flush=True)


if __name__ == "__main__":
    main()
