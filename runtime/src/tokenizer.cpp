#include "tokenizer.h"

#include <fstream>
#include <stdexcept>

#include "json.hpp"

namespace lpp {

std::u32string utf8_decode(const std::string& s) {
  std::u32string out;
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = s[i];
    uint32_t cp;
    int n;
    if (c < 0x80) cp = c, n = 1;
    else if ((c >> 5) == 0x6) cp = c & 0x1F, n = 2;
    else if ((c >> 4) == 0xE) cp = c & 0x0F, n = 3;
    else if ((c >> 3) == 0x1E) cp = c & 0x07, n = 4;
    else { out.push_back(0xFFFD); ++i; continue; }                 // invalid lead byte
    if (i + n > s.size()) { out.push_back(0xFFFD); break; }
    for (int k = 1; k < n; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
    out.push_back(cp);
    i += n;
  }
  return out;
}

std::string utf8_encode(const std::u32string& s) {
  std::string out;
  for (uint32_t cp : s) {
    if (cp < 0x80) out += static_cast<char>(cp);
    else if (cp < 0x800) out += static_cast<char>(0xC0 | (cp >> 6)), out += static_cast<char>(0x80 | (cp & 0x3F));
    else if (cp < 0x10000) {
      out += static_cast<char>(0xE0 | (cp >> 12));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (cp >> 18));
      out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }
  return out;
}

Tokenizer::Tokenizer(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  const auto j = nlohmann::json::parse(f);
  const auto& v = j["vocab"];
  for (size_t i = 0; i < v.size(); ++i) vocab_[v[i].get<std::string>()] = static_cast<int64_t>(i);
  for (auto& [k, val] : j["map"].items()) map_[static_cast<uint32_t>(std::stoul(k))] = utf8_decode(val.get<std::string>());
  for (auto& p : j["punct"]) punct_.insert(p.get<uint32_t>());
  for (auto& r : j["cjk"]) cjk_.emplace_back(r[0].get<uint32_t>(), r[1].get<uint32_t>());
  cls_ = j["cls"], sep_ = j["sep"], unk_ = j["unk"], pad_ = j["pad"];
  max_chars_ = j["max_chars_per_word"];
}

bool Tokenizer::cjk(uint32_t cp) const {
  for (auto& [a, b] : cjk_)
    if (a <= cp && cp <= b) return true;
  return false;
}

std::vector<int64_t> Tokenizer::encode(const std::string& text) const {
  // 1. normalise per codepoint, spaces around CJK characters
  std::u32string s;
  for (uint32_t cp : utf8_decode(text)) {
    if (cjk(cp)) { s += U' '; s += cp; s += U' '; continue; }
    auto it = map_.find(cp);
    if (it != map_.end()) s += it->second;
    else s += cp;
  }
  // 2. whitespace split, punctuation split off as single-character words
  std::vector<std::u32string> words;
  std::u32string cur;
  auto flush = [&] { if (!cur.empty()) words.push_back(cur), cur.clear(); };
  for (uint32_t cp : s) {
    if (cp == U' ' || cp == U'\t' || cp == U'\n' || cp == U'\r') { flush(); continue; }
    if (punct_.count(cp)) { flush(); words.push_back(std::u32string(1, cp)); continue; }
    cur += cp;
  }
  flush();
  // 3. greedy longest-match-first WordPiece
  std::vector<int64_t> ids{cls_};
  for (const auto& w : words) {
    if (w.size() > max_chars_) { ids.push_back(unk_); continue; }
    std::vector<int64_t> pieces;
    size_t start = 0;
    bool bad = false;
    while (start < w.size()) {
      size_t end = w.size();
      int64_t found = -1;
      while (start < end) {
        std::string sub = (start ? "##" : "") + utf8_encode(w.substr(start, end - start));
        auto it = vocab_.find(sub);
        if (it != vocab_.end()) { found = it->second; break; }
        --end;
      }
      if (found < 0) { bad = true; break; }
      pieces.push_back(found);
      start = end;
    }
    if (bad) ids.push_back(unk_);
    else ids.insert(ids.end(), pieces.begin(), pieces.end());
  }
  ids.push_back(sep_);
  return ids;
}

}  // namespace lpp
