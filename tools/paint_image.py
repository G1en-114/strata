"""Trace any picture into strokes: the decomposer (Gaussians, then Bezier lines) on an arbitrary image.

    python tools/paint_image.py photo.png --out out/trace/photo           # -> strokes.npy, trace.png, report
    python tools/paint_image.py photo.png --out ... --sketch-only         # only the sketch rounds (for continuation)

Used by the strata server's image endpoint (tools/trace_service.py) and by hand. The face / anime datasets come with
semantic layer labels; an arbitrary image does not, so the layers here are generic:
  background   pixels connected to the image border with a colour close to the border's (flood fill in Oklab)
  foreground   k-means colours inside the rest, biggest cluster -> the layer with the most strokes (hair, 256), then
               skin (192), features (224), body (128); tracing only needs room for strokes, not semantics
The image is padded to a square with its border colour and resized to the 256 px canvas.
Output records (kind, event, layer, attributes) are in the draft order the v3 model paints in, BOS first, EOS last,
the same layout the C++ runtime writes, so the server can animate them and continue them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lpp.codec import Codec  # noqa: E402
from lpp.color import srgb_to_oklab  # noqa: E402
from lpp.data.sequences import Example, widen  # noqa: E402
from lpp.decompose.layered import DecomposeConfig, LayeredDecomposer  # noqa: E402
from lpp.protocol import PROTOCOL_V3_PATH, load_protocol  # noqa: E402
from lpp.render.strokes import render_strokes  # noqa: E402
from lpp.sequence import build  # noqa: E402

FOREGROUND_LAYERS = [4, 2, 3, 1]          # by stroke cap: hair 256, skin 192, features 224, body 128


def load_square(path_or_img, size: int = 256) -> np.ndarray:
    im = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    a = np.asarray(im)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    fill = tuple(int(v) for v in np.median(border, 0))
    w, h = im.size
    side = max(w, h)
    sq = Image.new("RGB", (side, side), fill)
    sq.paste(im, ((side - w) // 2, (side - h) // 2))
    return np.asarray(sq.resize((size, size), Image.Resampling.LANCZOS))


def generic_labels(img: np.ndarray, k: int = 4, bg_tol: float = 0.06, seed: int = 0) -> np.ndarray:
    """[S, S, 3] uint8 -> [S, S] layer ids (see the module docstring)."""
    from scipy import ndimage
    lab = srgb_to_oklab(torch.from_numpy(img).float().div(255).reshape(-1, 3)).numpy().reshape(img.shape)
    S = img.shape[0]
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    ref = np.median(border, 0)
    close = np.linalg.norm(lab - ref, axis=-1) < bg_tol
    comp, _ = ndimage.label(close)
    edge_ids = set(np.unique(np.concatenate([comp[0], comp[-1], comp[:, 0], comp[:, -1]]))) - {0}
    bg = np.isin(comp, list(edge_ids))
    labels = np.zeros((S, S), np.int64)
    fg = ~bg
    if fg.sum() < 16:                                     # nothing but background: one big layer
        labels[:] = 4
        return labels
    x = lab[fg]
    g = np.random.default_rng(seed)
    cent = x[g.choice(len(x), size=min(k, len(x)), replace=False)]
    for _ in range(20):
        assign = np.argmin(((x[:, None] - cent[None]) ** 2).sum(-1), 1)
        cent = np.stack([x[assign == c].mean(0) if (assign == c).any() else cent[c] for c in range(len(cent))])
    order = np.argsort(-np.bincount(assign, minlength=len(cent)))
    layer_of = {int(c): FOREGROUND_LAYERS[i] for i, c in enumerate(order)}
    labels[fg] = np.array([layer_of[int(a)] for a in assign])
    return labels


def trace(img: np.ndarray, steps: int = 120, line_steps: int = 200, sketch_only: bool = False, dev: str = "cuda",
          seed: int = 0) -> dict:
    """[256, 256, 3] uint8 -> {"records": [T, 3 + A] int64 (BOS .. EOS), "psnr", "strokes", "lines", "seconds"}."""
    import decompose_lines as dl
    t0 = time.time()
    labels = generic_labels(img, seed=seed)
    proto2 = load_protocol()
    codec2 = Codec.load(proto=proto2, device=dev)
    dec = LayeredDecomposer(codec2, DecomposeConfig(steps_per_round=steps, seed=seed))
    x = torch.from_numpy(img).to(dev).permute(2, 0, 1)[None].float() / 255
    d = dec(x, torch.from_numpy(labels).to(dev)[None])[0]
    tok2 = torch.stack([d.tokens[n].long().cpu() for n in proto2.attr_names], 1)
    ex = Example(tok2, d.layer.long().cpu(), d.round.long().cpu(), image_id=0)
    proto3 = load_protocol(PROTOCOL_V3_PATH)
    codec3 = Codec.load(proto=proto3, device=dev)
    largs = argparse.Namespace(piece=20, max_curves=200, steps=line_steps, swap="global", line_qat="freeze")
    tok, lay, rnd, isl, psnr, K = dl.decompose_one(ex, img, labels, codec3, proto3, largs, dev)
    tok, lay, rnd, isl = (torch.as_tensor(np.asarray(v)) for v in (tok, lay, rnd, isl))
    seq = build(widen(tok.long(), proto3), lay.long(), rnd.long(), "draft", proto=proto3, is_line=isl.bool())
    rec = np.concatenate([np.stack([seq.kind.numpy(), seq.event.numpy(), seq.layer.numpy()], 1), seq.attrs.numpy()], 1)
    if sketch_only:                                       # everything before the first REFINE marker
        refine = {proto3.refine_event(k) for k in range(proto3.num_layers)}
        cut = next((i for i, e in enumerate(rec[:, 1]) if int(e) in refine), len(rec))
        rec = rec[:cut]
    strokes = int((rec[:, 0] == 3).sum())
    return {"records": rec.astype(np.int64), "psnr": float(psnr), "strokes": strokes, "lines": int(K),
            "seconds": time.time() - t0, "labels": labels}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=120, help="Gaussian fitting steps per round")
    ap.add_argument("--line-steps", type=int, default=200)
    ap.add_argument("--sketch-only", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    img = load_square(args.image)
    r = trace(img, args.steps, args.line_steps, args.sketch_only)
    np.save(args.out / "strokes.npy", r["records"][None])
    proto3 = load_protocol(PROTOCOL_V3_PATH)
    codec3 = Codec.load(proto=proto3, device="cuda")
    rec = torch.from_numpy(r["records"])
    st = rec[:, 0] == 3
    out = render_strokes(rec[st, 3:], rec[st, 2], rec[st, 1] == proto3.event_id("LINE"), codec3, 256)[0]
    Image.fromarray((out.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(args.out / "trace.png")
    Image.fromarray(img).save(args.out / "input.png")
    pal = np.array([[230, 230, 230], [80, 120, 200], [230, 180, 150], [220, 60, 60], [120, 80, 40], [240, 220, 60]], np.uint8)
    Image.fromarray(pal[r["labels"]]).save(args.out / "layers.png")
    rep = {k: v for k, v in r.items() if k not in ("records", "labels")}
    (args.out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
