"""Assemble the Hugging Face release of the final v3 model (weights: CC BY-NC 4.0; code stays Apache-2.0).

    python tools/make_hf_release.py --out out/hf_release/strata-painter-v3
    hf auth login                                   # your own token
    hf upload <user>/strata-painter-v3 out/hf_release/strata-painter-v3 . --repo-type model

Contents:
  model.safetensors, config.json, protocol_v3.json   the PyTorch weights (lpp.model.Painter) for training / fine-tuning
  onnx-fp16/                                         the deployable export: FP16 backbone + heads, FP32 fused sampler
                                                     (decode_sample), text encoder + WordPiece tables; run with the
                                                     C++ server (runtime/, ./serve.sh) from the code repository
  samples.png, README.md (model card), LICENSE       the card holds metrics, data sources and their terms
Nothing from the training data is included.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def samples(out: Path) -> None:
    """Top rows (generated) of the final model's evaluation grids, anime then faces, 8 each."""
    rows = []
    for dom in ("anime", "face"):
        g = Image.open(ROOT / f"out/v3_train/eval/B_ft_step012000_{dom}/grid_generated_over_real.png")
        rows.append(g.crop((0, 0, 256 * 8, 256)))
    sheet = Image.new("RGB", (256 * 8, 512))
    for i, r in enumerate(rows):
        sheet.paste(r, (0, 256 * i))
    sheet.save(out / "samples.png", optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=ROOT / "runs/v3_formal/B_ft/model_step012000.pt")
    ap.add_argument("--onnx-fp16", type=Path, default=ROOT / "out/onnx/v3_final_fused_fp16")
    ap.add_argument("--onnx-text", type=Path, default=ROOT / "out/onnx/v3_final_fused",
                    help="export dir holding text_encoder.onnx + tokenizer.json (tools/export_text_encoder.py)")
    ap.add_argument("--out", type=Path, default=ROOT / "out/hf_release/strata-painter-v3")
    args = ap.parse_args()
    from safetensors.torch import save_file
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    s = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    save_file({k: v.contiguous() for k, v in s["model"].items()}, str(out / "model.safetensors"),
              metadata={"format": "pt", "step": str(s.get("step"))})
    (out / "config.json").write_text(json.dumps({"model_type": "strata-painter", "architecture": "lpp.model.Painter",
                                                 "step": s.get("step"), **s["config"]}, indent=1))
    (out / "protocol_v3.json").write_text(json.dumps(s["protocol"], indent=1, ensure_ascii=False))
    od = out / "onnx-fp16"
    shutil.copytree(args.onnx_fp16, od)
    man = json.loads((od / "manifest.json").read_text())
    tman = json.loads((args.onnx_text / "manifest.json").read_text())
    for f in ("text_encoder.onnx", "tokenizer.json"):
        shutil.copy2(args.onnx_text / f, od / f)
    man["text_encoder"] = tman["text_encoder"]
    (od / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    samples(out)
    shutil.copy2(ROOT / "docs/hf_model_card.md", out / "README.md")
    (out / "LICENSE").write_text(
        "Model weights and ONNX graphs in this repository: Creative Commons Attribution-NonCommercial 4.0 International\n"
        "(CC BY-NC 4.0), https://creativecommons.org/licenses/by-nc/4.0/legalcode\n\n"
        "The text encoder (onnx-fp16/text_encoder.onnx, tokenizer.json) is converted from OFA-Sys Chinese-CLIP\n"
        "(chinese-clip-vit-base-patch16), MIT License, Copyright (c) 2022 OFA-Sys.\n\n"
        "The code that trains and runs these weights (github.com/G1en-114/strata) is Apache-2.0.\n")
    files = sorted(p for p in out.rglob("*") if p.is_file())
    (out / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.relative_to(out)}\n" for p in files))
    total = sum(p.stat().st_size for p in files)
    print(f"{out}: {len(files)} files, {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
