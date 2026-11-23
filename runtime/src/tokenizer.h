// BERT WordPiece tokenizer (Chinese-CLIP text tower) from tokenizer.json (tools/export_text_encoder.py): the
// per-codepoint normalisation map was computed with the reference normalizer (clean_text, lowercase, accent
// stripping), so no Unicode library is needed here. Same ids as transformers' BertTokenizer on all 166k captions.
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace lpp {

class Tokenizer {
 public:
  explicit Tokenizer(const std::string& json_path);
  // [CLS] ... [SEP] ids of one text
  std::vector<int64_t> encode(const std::string& text) const;
  int64_t pad() const { return pad_; }

 private:
  bool cjk(uint32_t cp) const;
  std::unordered_map<std::string, int64_t> vocab_;
  std::unordered_map<uint32_t, std::u32string> map_;
  std::unordered_set<uint32_t> punct_;
  std::vector<std::pair<uint32_t, uint32_t>> cjk_;
  int64_t cls_ = 101, sep_ = 102, unk_ = 100, pad_ = 0;
  size_t max_chars_ = 100;
};

std::u32string utf8_decode(const std::string& s);
std::string utf8_encode(const std::u32string& s);

}  // namespace lpp
