---
license: cc-by-nc-4.0
language:
- zh
library_name: onnxruntime
pipeline_tag: text-to-image
tags:
- autoregressive
- vector-graphics
- gaussian-splatting
- bezier
- stroke-based-rendering
- onnx
---

# Strata Painter v3

**一笔一笔作画的自回归模型**：输入中文描述，模型逐笔输出「高斯笔画 + 贝塞尔线条」，分层叠出一张 256×256 的人像或动漫头像。
代码、训练和 C++ 推理服务见 [github.com/G1en-114/strata](https://github.com/G1en-114/strata)。

Strata Painter paints a picture **stroke by stroke**. Given a Chinese caption, a transformer emits a sequence of
records: layer markers, generalized-Gaussian strokes and quadratic Bézier line strokes. A renderer composites them
layer by layer into a 256×256 image, so every output is an editable, resolution-independent stroke list rather
than a grid of pixels.

![samples](samples.png)

*Top: anime prompts; bottom: face prompts (validation captions, CFG 5, top-p 0.6, one sample each).*

## Model

| | |
|---|---|
| Parameters | 129.3 M (16-layer decoder-only backbone, d 768, GQA 12/4 heads + a 2-layer attribute decoder) |
| Output | records: `LAYER_k`, `PRIMITIVE` (Gaussian: x, y, σx, σy, θ, edge hardness, colour, α), `LINE` (Bézier: same slots + bend), `REFINE_k`, `REDRAW_k`, `EOS` |
| Layers | 6 semantic layers: background, body, skin, features, hair, accessories |
| Order | draft order: a coarse sketch of every layer first, then each layer refined |
| Conditioning | frozen Chinese-CLIP text tower (`chinese-clip-vit-base-patch16`), classifier-free guidance |
| Canvas feedback | the canvas rendered every 32 strokes is encoded and read back by cross-attention |
| Context | 1024 records (about 700 strokes per picture) |

Start each caption with the domain word, as in training: `动漫，…` (anime) or `真人照片，…` (photo of a face).
The model only knows these two domains (face and anime portraits).

## Results

Validation prompts (192 per domain), judge CLIP ViT-L/14 (a different model from the conditioning encoder):

| | faces R@5 ↑ | faces FD ↓ | anime R@5 ↑ | anime FD ↓ |
|---|---:|---:|---:|---:|
| previous model (Gaussians only) | 0.380 | 0.373 | 0.375 | 0.227 |
| **this model** | **0.406** | **0.371** | **0.417** | **0.206** |
| this model, best-of-4 | 0.422 | 0.369 | 0.500 | 0.197 |

R@5: caption retrieval of the generated image among 192; FD: Fréchet distance of CLIP features to real images.

Deployment (C++ runtime on ONNX Runtime, RTX 5070 Laptop / Ryzen 9 9955HX), records per second:

| | batch 1 | batch 32 |
|---|---:|---:|
| GPU, FP32 | 86 | 340 |
| GPU, fused sampler + FP16 (`onnx-fp16/`) | 98 | **586** |
| CPU, FP32 | 29 | 44 |

## Files

| Path | What |
|---|---|
| `model.safetensors`, `config.json`, `protocol_v3.json` | PyTorch weights for `lpp.model.Painter` (training, fine-tuning) |
| `onnx-fp16/` | deployable graphs: FP16 backbone and heads, FP32 fused sampler, text encoder, WordPiece tables |
| `SHA256SUMS` | checksums |

## Use

```bash
git clone https://github.com/G1en-114/strata && cd strata
hf download <user>/strata-painter-v3 --local-dir models/strata-painter-v3
cmake -S runtime -B runtime/build && cmake --build runtime/build --target strata
runtime/strata.sh serve --model strata-v3:fp16=models/strata-painter-v3/onnx-fp16 --web runtime/web
# http://localhost:11435 (web UI) — OpenAI-compatible: POST /v1/images/generations
curl localhost:11435/v1/images/generations -d '{"prompt": "动漫，金发碧眼的少女，微笑", "seed": 1}'
```

The server streams the painting stroke by stroke (`POST /api/generate`), batches concurrent requests, and can
continue a finished painting: redraw one layer with a new caption (e.g. other hair colour) or refine a layer.

## Training data and terms

- **Faces:** CelebAMask-HQ (30 k images, semantic masks), decomposed into stroke sequences. CelebAMask-HQ is
  released for **non-commercial research only**.
- **Anime:** huanngzh/anime_face_control_60k (about 53 k portraits), decomposed with Bézier lines. Its dataset card
  states no image licence.
- **Captions:** written by Qwen2.5-VL-7B-Instruct from the training images.

No training image or stroke sequence is distributed here. Because of the data terms, the weights are released
under **CC BY-NC 4.0** (non-commercial). The text encoder is converted from Chinese-CLIP (MIT).

## Limitations

- Only face and anime portraits; other subjects are out of distribution.
- Fine strokes are placed with some uncertainty: patchy skin on faces, stray hatching lines on anime hair.
- A picture is limited to 1024 records; a layer that reached its stroke cap can be redrawn but not refined.
- Do not use it to depict real, identifiable people.
