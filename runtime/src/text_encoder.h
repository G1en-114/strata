// Prompt -> text features, as lpp.text.TextEncoder does it: tokenize, run the Chinese-CLIP text tower
// (text_encoder.onnx, tools/export_text_encoder.py), keep each prompt's real tokens, adaptive-average-pool prompts
// longer than max_positions down to max_positions. Lets the runtime take plain prompts (no Python in the loop).
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "tokenizer.h"

namespace lpp {

struct TextBatch {
  std::vector<float> feats;      // [B, Lt, dim]
  std::vector<uint8_t> valid;    // [B, Lt]
  int B = 0, Lt = 0, dim = 0;
};

class TextEncoder {
 public:
  // export_dir with manifest.json "text_encoder" (file, tokenizer, max_positions, dim)
  explicit TextEncoder(const std::string& export_dir);
  TextBatch encode(const std::vector<std::string>& prompts);
  const Tokenizer& tokenizer() const { return *tok_; }

 private:
  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Tokenizer> tok_;
  int max_positions_ = 32, dim_ = 768;
};

}  // namespace lpp
