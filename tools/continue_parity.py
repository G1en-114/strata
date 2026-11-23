"""Continuation parity: the C++ runtime's forced prefixes (lpp_paint --forced) == lpp.sample.generate(forced=...).

    python tools/continue_parity.py --ckpt runs/v3_formal/B_ft/model_step012000.pt --onnx out/onnx/v3_final_fused \
        --records out/.../records.npy --text-dir out/.../cpp_text --out out/continue_parity

From finished paintings (records.npy as lpp_paint writes it) builds the two edit prefixes the model was trained on,
for the given layers (refining needs room under the layer's stroke cap; hair is often full):
  refine   every layer in layer order (LAYER_j + its strokes, sequence order inside the layer), then REFINE_k
  redraw   the same without layer k, then REDRAW_k
then decodes the continuation with PyTorch and with the C++ runtime from the same text features and the same uniform
table, and compares the records.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from lpp.codec import Codec  # noqa: E402
from lpp.model import ModelConfig, Painter  # noqa: E402
from lpp.protocol import _parse  # noqa: E402
from lpp.sample import SampleConfig, generate  # noqa: E402
from lpp.sequence import KIND_BOS, KIND_CONTROL, KIND_PRIMITIVE, NO_ATTR, DrawSeq  # noqa: E402


def edit_prefix(rec: np.ndarray, proto, mode: str, k: int) -> np.ndarray:
    """One finished painting's records [T, 3 + A] (-2 padding) -> the edit prefix as records, BOS first."""
    A = rec.shape[1] - 3
    rows = [r for r in rec if r[0] != -2]
    strokes = [r for r in rows if r[0] == KIND_PRIMITIVE]
    out = [np.array([KIND_BOS, -1, -1] + [NO_ATTR] * A)]
    for j in sorted({int(r[2]) for r in strokes}):
        if mode == "redraw" and j == k:
            continue
        out.append(np.array([KIND_CONTROL, proto.layer_event(j), j] + [NO_ATTR] * A))
        out += [r for r in strokes if r[2] == j]
    ev = proto.refine_event(k) if mode == "refine" else proto.redraw_event(k)
    out.append(np.array([KIND_CONTROL, ev, k] + [NO_ATTR] * A))
    return np.stack(out).astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--text-dir", type=Path, required=True, help="text.npy / text_valid.npy of the same prompts")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--refine-layer", type=int, default=3, help="a layer below its stroke cap (features)")
    ap.add_argument("--redraw-layer", type=int, default=4, help="hair")
    ap.add_argument("--max-records", type=int, default=1000)
    ap.add_argument("--provider", default="cpu")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    s = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    proto = _parse(s["protocol"])
    codec = Codec.load(proto=proto, device="cpu")
    model = Painter(ModelConfig(**s["config"]), proto, codec).eval()
    model.load_state_dict(s["model"])
    rec = np.load(args.records)
    text, tv = np.load(args.text_dir / "text.npy"), np.load(args.text_dir / "text_valid.npy")
    B, A = rec.shape[0], rec.shape[2] - 3
    fields = ["event"] + list(model.attr_names)
    table = torch.rand(args.max_records, len(fields), B, generator=torch.Generator().manual_seed(0))
    np.save(args.out / "uniforms.npy", table.numpy())
    ok = True
    for mode in ("refine", "redraw"):
        k = args.refine_layer if mode == "refine" else args.redraw_layer
        pre = [edit_prefix(rec[b], proto, mode, k) for b in range(B)]
        T = max(len(p) for p in pre)
        forced = np.full((B, T, 3 + A), -2, np.int64)
        for b, p in enumerate(pre):
            forced[b, :len(p)] = p
        np.save(args.out / f"forced_{mode}.npy", forced)
        seqs = [DrawSeq(torch.tensor(p[:, 0]), torch.tensor(p[:, 1]), torch.tensor(p[:, 3:]), torch.tensor(p[:, 2])) for p in pre]
        sc = SampleConfig(temperature=1.0, top_p=0.6, cfg_scale=5.0, max_records=args.max_records)
        with torch.no_grad():
            ref = generate(model, batch=B, cfg=sc, text=torch.from_numpy(text), text_valid=torch.from_numpy(tv), codec=codec,
                           uniforms=lambda step, field: table[step, fields.index(field)], forced=seqs)
        run = args.out / f"cpp_{mode}"
        subprocess.run(["runtime/run.sh", "--onnx", str(args.onnx), "--provider", args.provider, "--text", str(args.text_dir / "text.npy"),
                        "--text-valid", str(args.text_dir / "text_valid.npy"), "--uniforms", str(args.out / "uniforms.npy"),
                        "--forced", str(args.out / f"forced_{mode}.npy"), "--max-records", str(args.max_records),
                        "--cfg-scale", "5", "--top-p", "0.6", "--require", "", "--out", str(run)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        got = np.load(run / "records.npy")
        for b in range(B):
            r = ref[b]
            a = np.concatenate([np.stack([r.kind.numpy(), r.event.numpy(), r.layer.numpy()], 1), r.attrs.numpy()], 1)
            g = got[b][got[b, :, 0] != -2]
            same = a.shape == g.shape and bool((a == g).all())
            n = min(len(a), len(g))
            first = None if same else int(np.argmax((a[:n] != g[:n]).any(1))) if (a[:n] != g[:n]).any() else n
            new = len(g) - len(pre[b])
            print(f"{mode} sample {b}: prefix {len(pre[b])} records, +{new} continued, torch {len(a)} vs C++ {len(g)}: "
                  + ("identical" if same else f"differ from record {first}"), flush=True)
            ok &= same
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
