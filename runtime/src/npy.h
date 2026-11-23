// Minimal .npy reader/writer (C order, little-endian float32 / int64 / bool), enough for
// exchanging tensors with the Python tools (text features, uniforms, sampled sequences).
#pragma once
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lpp {

struct NpyArray {
  std::string dtype;            // "<f4", "<i8", "|b1"
  std::vector<int64_t> shape;
  std::vector<char> data;
  int64_t size() const {
    int64_t n = 1;
    for (auto s : shape) n *= s;
    return n;
  }
  template <class T> const T* as() const { return reinterpret_cast<const T*>(data.data()); }
  static NpyArray from_f32(std::vector<int64_t> shape, const std::vector<float>& v) {
    NpyArray a{"<f4", std::move(shape), {}};
    a.data.assign(reinterpret_cast<const char*>(v.data()), reinterpret_cast<const char*>(v.data()) + v.size() * 4);
    return a;
  }
  static NpyArray from_u8(std::vector<int64_t> shape, const std::vector<uint8_t>& v) {
    NpyArray a{"|b1", std::move(shape), {}};
    a.data.assign(v.begin(), v.end());
    return a;
  }
};

inline NpyArray load_npy(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  char magic[6];
  f.read(magic, 6);
  if (std::string(magic + 1, 5) != "NUMPY") throw std::runtime_error(path + ": not a .npy file");
  uint8_t ver[2];
  f.read(reinterpret_cast<char*>(ver), 2);
  uint32_t hlen = 0;
  if (ver[0] == 1) {
    uint16_t h16;
    f.read(reinterpret_cast<char*>(&h16), 2);
    hlen = h16;
  } else {
    f.read(reinterpret_cast<char*>(&hlen), 4);
  }
  std::string header(hlen, ' ');
  f.read(header.data(), hlen);
  NpyArray a;
  auto descr = header.find("'descr'");
  auto q1 = header.find('\'', header.find(':', descr) + 1);
  a.dtype = header.substr(q1 + 1, header.find('\'', q1 + 1) - q1 - 1);
  if (header.find("'fortran_order': True") != std::string::npos) throw std::runtime_error(path + ": fortran order");
  auto p = header.find('(', header.find("'shape'"));
  auto e = header.find(')', p);
  std::string dims = header.substr(p + 1, e - p - 1);
  size_t pos = 0;
  while (pos < dims.size()) {
    auto c = dims.find(',', pos);
    std::string tok = dims.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
    if (tok.find_first_of("0123456789") != std::string::npos) a.shape.push_back(std::stoll(tok));
    if (c == std::string::npos) break;
    pos = c + 1;
  }
  int64_t item = a.dtype == "<f4" ? 4 : a.dtype == "<i8" ? 8 : a.dtype == "|b1" ? 1 : 0;
  if (!item) throw std::runtime_error(path + ": unsupported dtype " + a.dtype);
  a.data.resize(a.size() * item);
  f.read(a.data.data(), static_cast<std::streamsize>(a.data.size()));
  if (!f) throw std::runtime_error(path + ": truncated");
  return a;
}

inline void save_npy(const std::string& path, const std::string& dtype, const std::vector<int64_t>& shape,
                     const void* data, size_t bytes) {
  std::string dims;
  for (size_t i = 0; i < shape.size(); ++i) dims += std::to_string(shape[i]) + (shape.size() == 1 || i + 1 < shape.size() ? "," : "");
  std::string header = "{'descr': '" + dtype + "', 'fortran_order': False, 'shape': (" + dims + "), }";
  size_t total = 10 + header.size() + 1;
  header.append((64 - total % 64) % 64, ' ');
  header += '\n';
  std::ofstream f(path, std::ios::binary);
  f.write("\x93NUMPY\x01\x00", 8);
  uint16_t hl = static_cast<uint16_t>(header.size());
  f.write(reinterpret_cast<char*>(&hl), 2);
  f.write(header.data(), static_cast<std::streamsize>(header.size()));
  f.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
}

}  // namespace lpp
