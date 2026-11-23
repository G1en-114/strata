// lpp_paint: text features -> layered primitive sequences -> PNG, on the exported ONNX graphs.
//
//   lpp_paint --onnx out/onnx/small_t0 --text text.npy --text-valid text_valid.npy
//             --uniforms uniforms.npy --out out/runtime_run [--provider cpu|cuda|cuda-auto|cuda-hybrid|cuda-all|cuda-tf32] [--cfg-scale 2]
//             [--top-p 0.9] [--max-records 300] [--require 0,2]
//
// uniforms.npy: float32 [max_records, 1 + A, B]; field 0 = event, 1 + i = attribute i (the same
// table tools/ort_generate.py --uniforms-out writes, so both runs consume identical randomness).
// Writes records.npy (int64 [B, Tmax, 3 + A]: kind, event, layer, attributes; -2 = padding),
// lengths.npy, sample_<b>.png and timing.json.
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>

#include "npy.h"
#include "painter.h"
#include "text_encoder.h"
#include "raster.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace lpp;

int main(int argc, char** argv) {
  std::map<std::string, std::string> a{{"--provider", "cpu"}, {"--cfg-scale", "2"}, {"--top-p", "0.9"},
                                       {"--max-records", "0"}, {"--require", "0,2"}, {"--temperature", "1"}};
  for (int i = 1; i + 1 < argc; i += 2) a[argv[i]] = argv[i + 1];
  // text: --prompts FILE (one prompt per line, tokenized and encoded here: text_encoder.onnx + tokenizer.json) or
  // precomputed --text / --text-valid (.npy, as tools/ort_generate.py --dump writes). Randomness: --uniforms .npy,
  // or --seed N (std::mt19937, float32 uniforms in [0, 1)).
  const bool prompts = a.count("--prompts") > 0;
  for (const char* k : {"--onnx", "--out"})
    if (!a.count(k)) { std::cerr << "missing " << k << "\n"; return 2; }
  if (!prompts && (!a.count("--text") || !a.count("--text-valid"))) { std::cerr << "need --prompts or --text + --text-valid\n"; return 2; }
  if (!a.count("--uniforms") && !a.count("--seed")) { std::cerr << "need --uniforms or --seed\n"; return 2; }
  try {
    NpyArray text, tv, un;
    if (prompts) {
      std::ifstream pf(a["--prompts"]);
      std::vector<std::string> lines;
      for (std::string l; std::getline(pf, l);) if (!l.empty()) lines.push_back(l);
      if (lines.empty()) throw std::runtime_error("no prompts in " + a["--prompts"]);
      TextEncoder enc(a["--onnx"]);
      const auto t0 = std::chrono::steady_clock::now();
      TextBatch tb = enc.encode(lines);
      std::printf("encoded %d prompts in %.3fs\n", tb.B, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      text = NpyArray::from_f32({tb.B, tb.Lt, tb.dim}, tb.feats);
      tv = NpyArray::from_u8({tb.B, tb.Lt}, tb.valid);
      if (a.count("--dump-text")) {
        std::filesystem::create_directories(a["--dump-text"]);
        save_npy(a["--dump-text"] + "/text.npy", "<f4", {tb.B, tb.Lt, tb.dim}, tb.feats.data(), tb.feats.size() * 4);
        save_npy(a["--dump-text"] + "/text_valid.npy", "|b1", {tb.B, tb.Lt}, tb.valid.data(), tb.valid.size());
      }
    } else {
      text = load_npy(a["--text"]), tv = load_npy(a["--text-valid"]);
    }
    // cuda-auto: placement by batch size AND model size (measured on RTX 5070 Laptop). The per-record heads
    // (decode_heads, 9 calls per record) run faster on the CPU only while they are small: small model
    // (16 MB heads) at batch 1-2 -> cuda-hybrid 266 vs cuda 174 records/s; base model (59 MB heads) at batch 1
    // -> cuda 95 vs cuda-hybrid 67 (docs/deploy_report_v3.md). From batch 8 on the GPU wins either way.
    std::string provider = a["--provider"];
    if (provider == "cuda-auto") {
      std::error_code ec;
      const auto heads = std::filesystem::file_size(std::filesystem::path(a["--onnx"]) / "decode_heads.onnx", ec);
      const bool small_heads = !ec && heads < 32u * 1024 * 1024;
      provider = (small_heads && !text.shape.empty() && text.shape[0] <= 2) ? "cuda-hybrid" : "cuda";
    }
    Painter painter(a["--onnx"], provider);
    const Protocol& P = painter.protocol();
    if (text.dtype != "<f4" || text.shape.size() != 3) throw std::runtime_error("text must be float32 [B, Lt, D]");
    const int B = static_cast<int>(text.shape[0]), Lt = static_cast<int>(text.shape[1]);
    const int fields = 1 + static_cast<int>(P.attrs.size());
    std::vector<float> seeded;
    int steps;
    if (a.count("--uniforms")) {
      un = load_npy(a["--uniforms"]);
      steps = static_cast<int>(un.shape[0]);
      if (un.shape[2] != B || un.shape[1] != fields) throw std::runtime_error("uniforms shape mismatch");
    } else {
      steps = P.max_positions;                                          // enough for any sequence
      std::mt19937 rng(static_cast<uint32_t>(std::stoul(a["--seed"])));
      std::uniform_real_distribution<float> U01(0.f, 1.f);
      seeded.resize(size_t(steps) * fields * B);
      for (auto& x : seeded) x = U01(rng);
    }
    std::vector<float> tf(text.as<float>(), text.as<float>() + text.size());
    std::vector<uint8_t> tvb(tv.data.begin(), tv.data.end());
    GenerateConfig cfg;
    cfg.sample.cfg_scale = std::stof(a["--cfg-scale"]);
    cfg.sample.top_p = std::stof(a["--top-p"]);
    cfg.sample.temperature = std::stof(a["--temperature"]);
    cfg.max_records = std::stoi(a["--max-records"]);
    std::stringstream ss(a["--require"]);
    for (std::string t; std::getline(ss, t, ',');)
      if (!t.empty()) cfg.require_before_eos.insert(std::stoi(t));
    const float* U = seeded.empty() ? un.as<float>() : seeded.data();
    Uniforms uni = [&](int step, int field, int b) {
      if (step >= steps) throw std::runtime_error("uniform table too short");
      return U[(size_t(step) * fields + field) * B + b];
    };

    auto t0 = std::chrono::steady_clock::now();
    auto seqs = painter.generate(B, cfg, tf, tvb, Lt, uni);
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::filesystem::create_directories(a["--out"]);
    const int A = static_cast<int>(P.attrs.size());
    size_t Tmax = 0, total = 0;
    for (auto& s : seqs) Tmax = std::max(Tmax, s.size()), total += s.size();
    std::vector<int64_t> rec(size_t(B) * Tmax * (3 + A), -2), lens(B);
    for (int b = 0; b < B; ++b) {
      lens[b] = static_cast<int64_t>(seqs[b].size());
      std::vector<Stroke> strokes;
      for (size_t t = 0; t < seqs[b].size(); ++t) {
        const auto& r = seqs[b][t];
        int64_t* row = &rec[(size_t(b) * Tmax + t) * (3 + A)];
        row[0] = r.kind, row[1] = r.event, row[2] = r.layer;
        std::copy(r.attrs.begin(), r.attrs.end(), row + 3);
        if (r.kind == P.kind_primitive) strokes.push_back({static_cast<int>(r.layer), r.event == P.line_ev, r.attrs.data()});
      }
      auto img = render_strokes_rgb8(P, strokes, P.canvas_size);     // per layer: Gaussians, then lines
      stbi_write_png((a["--out"] + "/sample_" + std::to_string(b) + ".png").c_str(), P.canvas_size, P.canvas_size, 3,
                     img.data(), P.canvas_size * 3);
    }
    save_npy(a["--out"] + "/records.npy", "<i8", {B, static_cast<int64_t>(Tmax), 3 + A}, rec.data(), rec.size() * 8);
    save_npy(a["--out"] + "/lengths.npy", "<i8", {B}, lens.data(), lens.size() * 8);
    std::ofstream(a["--out"] + "/timing.json") << "{\"seconds\": " << secs << ", \"records\": " << total
                                               << ", \"records_per_s\": " << total / secs << ", \"provider\": \""
                                               << provider << "\"}\n";
    std::printf("%d sequences, %zu records in %.2fs (%.0f records/s, %s)\n", B, total, secs, total / secs,
                provider.c_str());
    Profile::report();
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
