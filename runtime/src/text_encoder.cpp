#include "text_encoder.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include "json.hpp"

namespace lpp {

TextEncoder::TextEncoder(const std::string& dir) : env_(ORT_LOGGING_LEVEL_WARNING, "lpp-text") {
  std::ifstream f(dir + "/manifest.json");
  if (!f) throw std::runtime_error("cannot open " + dir + "/manifest.json");
  const auto m = nlohmann::json::parse(f);
  if (!m.contains("text_encoder")) throw std::runtime_error(dir + ": no text_encoder (run tools/export_text_encoder.py)");
  const auto& t = m["text_encoder"];
  max_positions_ = t["max_positions"];
  dim_ = t["dim"];
  tok_ = std::make_unique<Tokenizer>(dir + "/" + t["tokenizer"].get<std::string>());
  Ort::SessionOptions so;                       // CPU: 0.1 B-parameter encoder, once per request
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  so.SetIntraOpNumThreads(4);
  session_ = std::make_unique<Ort::Session>(env_, (dir + "/" + t["file"].get<std::string>()).c_str(), so);
}

TextBatch TextEncoder::encode(const std::vector<std::string>& prompts) {
  const int B = static_cast<int>(prompts.size());
  std::vector<std::vector<int64_t>> ids(B);
  size_t T = 0;
  for (int b = 0; b < B; ++b) ids[b] = tok_->encode(prompts[b]), T = std::max(T, ids[b].size());
  std::vector<int64_t> in(size_t(B) * T, tok_->pad()), mask(size_t(B) * T, 0);
  for (int b = 0; b < B; ++b)
    for (size_t t = 0; t < ids[b].size(); ++t) in[b * T + t] = ids[b][t], mask[b * T + t] = 1;
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const std::vector<int64_t> shape{B, static_cast<int64_t>(T)};
  Ort::Value vals[2] = {Ort::Value::CreateTensor<int64_t>(mem, in.data(), in.size(), shape.data(), 2),
                        Ort::Value::CreateTensor<int64_t>(mem, mask.data(), mask.size(), shape.data(), 2)};
  const char* in_names[] = {"input_ids", "attention_mask"};
  const char* out_names[] = {"hidden"};
  auto out = session_->Run(Ort::RunOptions{nullptr}, in_names, vals, 2, out_names, 1);
  const float* h = out[0].GetTensorData<float>();               // [B, T, dim]
  TextBatch tb;
  tb.B = B, tb.dim = dim_;
  int Lt = 0;
  for (int b = 0; b < B; ++b) Lt = std::max(Lt, std::min(static_cast<int>(ids[b].size()), max_positions_));
  tb.Lt = Lt;
  tb.feats.assign(size_t(B) * Lt * dim_, 0.f);
  tb.valid.assign(size_t(B) * Lt, 0);
  for (int b = 0; b < B; ++b) {
    const int n = static_cast<int>(ids[b].size());
    const float* row = h + size_t(b) * T * dim_;
    float* dst = tb.feats.data() + size_t(b) * Lt * dim_;
    if (n <= max_positions_) {
      std::copy(row, row + size_t(n) * dim_, dst);
      std::fill(tb.valid.begin() + size_t(b) * Lt, tb.valid.begin() + size_t(b) * Lt + n, 1);
      continue;
    }
    // torch adaptive_avg_pool1d bins: [floor(i n / P), ceil((i + 1) n / P))
    const int P = max_positions_;
    for (int i = 0; i < P; ++i) {
      const int s = (i * n) / P, e = ((i + 1) * n + P - 1) / P;
      for (int d = 0; d < dim_; ++d) {
        double acc = 0;
        for (int k = s; k < e; ++k) acc += row[size_t(k) * dim_ + d];
        dst[size_t(i) * dim_ + d] = static_cast<float>(acc / (e - s));
      }
      tb.valid[size_t(b) * Lt + i] = 1;
    }
  }
  return tb;
}

}  // namespace lpp
