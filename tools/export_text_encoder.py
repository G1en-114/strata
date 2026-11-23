"""Export the frozen Chinese-CLIP text tower and its tokenizer for the Python-free C++ server (strata serve).

    python tools/export_text_encoder.py --text-encoder data/models/chinese-clip-vit-base-patch16 \
        --onnx out/onnx/v3_final_fused --max-positions 32

Writes into the model's export directory:
  text_encoder.onnx   input_ids, attention_mask [B, T] int64 -> last_hidden_state [B, T, 768]
  tokenizer.json      everything a BERT WordPiece tokenizer needs, as lookup tables: the vocabulary, a per-codepoint
                      normalisation map (clean_text + lowercase + accent stripping, computed with the reference
                      normalizer itself, so the C++ side needs no Unicode library), the punctuation set, CJK ranges
and adds "text_encoder" to manifest.json. Checks: the reference tokenizer vs a pure-Python port of the C++ algorithm
on every caption of --captions (token ids identical), and ORT vs PyTorch features.
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import torch

CJK = [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0x2A700, 0x2B73F), (0x2B740, 0x2B81F),
       (0x2B820, 0x2CEAF), (0xF900, 0xFAFF), (0x2F800, 0x2FA1F)]        # BertNormalizer's _is_chinese_char


def is_cjk(cp: int) -> bool:
    return any(a <= cp <= b for a, b in CJK)


def is_punct(cp: int) -> bool:                                          # BERT's _is_punctuation
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(chr(cp)).startswith("P")


def tables(tok) -> dict:
    norm = tok.backend_tokenizer.normalizer
    mapping = {}
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF or is_cjk(cp):
            continue
        ch = chr(cp)
        out = norm.normalize_str(ch)
        if out != ch:
            mapping[cp] = out
    vocab = sorted(tok.get_vocab().items(), key=lambda kv: kv[1])
    if [i for _, i in vocab] != list(range(len(vocab))):
        raise SystemExit("vocabulary ids are not 0..V-1")
    return {"vocab": [t for t, _ in vocab], "map": {str(k): v for k, v in mapping.items()},
            "punct": [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF and is_punct(cp)],
            "cjk": CJK, "cls": tok.cls_token_id, "sep": tok.sep_token_id, "unk": tok.unk_token_id,
            "pad": tok.pad_token_id, "max_chars_per_word": 100}


def port_tokenize(text: str, t: dict, index: dict) -> list[int]:
    """The C++ algorithm (runtime/src/tokenizer.cpp) in Python: map each codepoint, space around CJK, split on
    whitespace, split punctuation off, greedy longest-match WordPiece with ##."""
    m, punct = t["map"], set(t["punct"])
    s = []
    for ch in text:
        cp = ord(ch)
        if is_cjk(cp):
            s.append(" " + ch + " ")
        else:
            s.append(m.get(str(cp), ch))
    words = []
    for w in "".join(s).split():
        cur = ""
        for ch in w:
            if ord(ch) in punct:
                if cur:
                    words.append(cur)
                words.append(ch)
                cur = ""
            else:
                cur += ch
        if cur:
            words.append(cur)
    ids = [t["cls"]]
    for w in words:
        if len(w) > t["max_chars_per_word"]:
            ids.append(t["unk"])
            continue
        start, pieces = 0, []
        while start < len(w):
            end, found = len(w), None
            while start < end:
                sub = ("##" if start else "") + w[start:end]
                if sub in index:
                    found = index[sub]
                    break
                end -= 1
            if found is None:
                pieces = [t["unk"]]
                break
            pieces.append(found)
            start = end
        ids += pieces
    return ids + [t["sep"]]


class TextTower(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask,
                      token_type_ids=torch.zeros_like(input_ids)).last_hidden_state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text-encoder", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, required=True, help="the model's export directory (manifest.json)")
    ap.add_argument("--max-positions", type=int, default=32, help="protocol max_text_positions (pooling target)")
    ap.add_argument("--captions", type=Path, default=Path("data/captions/captions_v3.jsonl"))
    args = ap.parse_args()
    from transformers import BertTokenizer, ChineseCLIPModel
    import onnxruntime as ort
    tok = BertTokenizer.from_pretrained(str(args.text_encoder))
    # eager attention: the SDPA path drops an all-ones mask while tracing, and the graph then ignores padding
    model = ChineseCLIPModel.from_pretrained(str(args.text_encoder), attn_implementation="eager").text_model.eval()
    t = tables(tok)
    (args.onnx / "tokenizer.json").write_text(json.dumps(t, ensure_ascii=False))
    index = {w: i for i, w in enumerate(t["vocab"])}
    texts = []
    for line in args.captions.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        for k in ("short", "long"):
            if r.get(k):
                texts.append(f"{r['domain']}，{r[k]}" if r.get("domain") else r[k])
    texts += ["Hello World 3D-CG ９ café", "  多个   空格\t和\n换行 ", "ＡＢＣ，全角！", "emoji 😀 符号 ©®", "Straße ÉLAN"]
    bad = [x for x in texts if port_tokenize(x, t, index) != tok(x)["input_ids"]]
    print(f"tokenizer port: {len(texts) - len(bad)}/{len(texts)} texts give identical ids"
          + (f"; first mismatch: {bad[0]!r}" if bad else ""), flush=True)
    if bad:
        raise SystemExit(1)
    path = args.onnx / "text_encoder.onnx"
    # reference BEFORE tracing: computed after torch.onnx.export it came out wrong (export leaves state behind)
    enc = tok(texts[:8], padding=True, return_tensors="pt")
    with torch.no_grad():
        ref = TextTower(model)(enc["input_ids"], enc["attention_mask"]).numpy()
    ex = tok(texts[:2] + ["短"], padding=True, return_tensors="pt")        # traced WITH padding, so masking stays
    torch.onnx.export(TextTower(model), (ex["input_ids"], ex["attention_mask"]), str(path), input_names=["input_ids", "attention_mask"],
                      output_names=["hidden"], dynamic_axes={"input_ids": {0: "B", 1: "T"},
                      "attention_mask": {0: "B", 1: "T"}, "hidden": {0: "B", 1: "T"}}, opset_version=17, dynamo=False)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()})[0]
    keep = enc["attention_mask"].numpy().astype(bool)                       # padded positions are never read
    diff = float(np.abs(got - ref)[keep].max())
    print(f"text_encoder max|ort - torch| = {diff:.2e} ({path.stat().st_size / 1e6:.0f} MB)", flush=True)
    man = json.loads((args.onnx / "manifest.json").read_text())
    man["text_encoder"] = {"file": path.name, "tokenizer": "tokenizer.json", "max_positions": args.max_positions,
                           "max_tokens": int(model.config.max_position_embeddings), "dim": int(model.config.hidden_size),
                           "pooling": "adaptive average over tokens when a caption has more than max_positions"}
    (args.onnx / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    if diff > 1e-3:
        raise SystemExit("text encoder parity failed")


if __name__ == "__main__":
    sys.exit(main())
