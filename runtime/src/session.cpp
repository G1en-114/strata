#include "session.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

#include "json.hpp"

namespace lpp {

std::map<std::string, std::pair<double, long>>& Profile::table() {
  static std::map<std::string, std::pair<double, long>> t;
  return t;
}
void Profile::add(const std::string& name, double seconds) {
  auto& e = table()[name];
  e.first += seconds;
  e.second += 1;
}
void Profile::report() {
  const char* on = std::getenv("LPP_PROFILE");
  if (!on || std::string(on) != "1") return;
  double total = 0;
  for (auto& [n, e] : table()) total += e.first;
  for (auto& [n, e] : table())
    std::fprintf(stderr, "  %-14s %8.3f s  %7ld calls  %7.3f ms/call  %5.1f%%\n", n.c_str(), e.first, e.second,
                 1e3 * e.first / e.second, 100 * e.first / total);
}

Tensor Tensor::F(std::vector<int64_t> s, float fill) {
  Tensor t;
  t.dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  t.shape = std::move(s);
  t.f.assign(static_cast<size_t>(t.numel()), fill);
  return t;
}
Tensor Tensor::I(std::vector<int64_t> s, int64_t fill) {
  Tensor t;
  t.dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  t.shape = std::move(s);
  t.i.assign(static_cast<size_t>(t.numel()), fill);
  return t;
}
Tensor Tensor::B(std::vector<int64_t> s, bool fill) {
  Tensor t;
  t.dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  t.shape = std::move(s);
  t.b.assign(static_cast<size_t>(t.numel()), fill ? 1 : 0);
  return t;
}
int64_t Tensor::numel() const {
  int64_t n = 1;
  for (auto d : shape) n *= d;
  return n;
}

Graph::Graph(Ort::Env& env, const std::string& path, const Ort::SessionOptions& so, std::vector<std::string> inputs,
             std::vector<std::string> outputs, std::string name)
    : session_(env, path.c_str(), so), in_names_(std::move(inputs)), out_names_(std::move(outputs)),
      mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)), name_(std::move(name)) {}

Ort::Value Graph::value(const Tensor& t) {
  void* data = nullptr;
  size_t bytes = 0;
  switch (t.dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: data = const_cast<float*>(t.f.data()); bytes = t.f.size() * 4; break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: data = const_cast<int64_t*>(t.i.data()); bytes = t.i.size() * 8; break;
    default: data = const_cast<uint8_t*>(t.b.data()); bytes = t.b.size(); break;
  }
  return Ort::Value::CreateTensor(mem_, data, bytes, t.shape.data(), t.shape.size(), t.dtype);
}

std::vector<Tensor> Graph::run(const std::vector<const Tensor*>& inputs) {
  if (inputs.size() != in_names_.size()) throw std::runtime_error("graph input count mismatch");
  std::vector<Ort::Value> vals;
  std::vector<const char*> in_c, out_c;
  for (size_t k = 0; k < inputs.size(); ++k) {
    vals.push_back(value(*inputs[k]));
    in_c.push_back(in_names_[k].c_str());
  }
  for (auto& n : out_names_) out_c.push_back(n.c_str());
  const auto t0 = std::chrono::steady_clock::now();
  auto outs = session_.Run(Ort::RunOptions{nullptr}, in_c.data(), vals.data(), vals.size(), out_c.data(), out_c.size());
  Profile::add(name_, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  std::vector<Tensor> res;
  for (auto& o : outs) {
    auto info = o.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {   // decode_sample's tokens
      Tensor t = Tensor::I(info.GetShape());
      const int64_t* p = o.GetTensorData<int64_t>();
      std::copy(p, p + t.i.size(), t.i.begin());
      res.push_back(std::move(t));
      continue;
    }
    Tensor t = Tensor::F(info.GetShape());
    const float* p = o.GetTensorData<float>();
    std::copy(p, p + t.f.size(), t.f.begin());
    res.push_back(std::move(t));
  }
  return res;
}

Graphs::Graphs(const std::string& dir, const std::string& provider)
    : env_(ORT_LOGGING_LEVEL_WARNING, "lpp"), provider_(provider) {
  std::ifstream f(dir + "/manifest.json");
  if (!f) throw std::runtime_error("cannot open " + dir + "/manifest.json");
  const auto m = nlohmann::json::parse(f);
  // Placement per graph. "cuda" puts every graph on the GPU; "cuda-hybrid" only the backbone
  // (chunk / chunk_cached): at batch 1 the attribute heads, record embedding and canvas encoder are
  // tiny and launch-bound, and run faster on the CPU (0.22 vs 0.40 ms per decode_heads call); the
  // backbone's output h is copied to the host anyway, so the split adds no transfer.
  //   cuda-all    : every graph on the GPU (parity reference)
  //   cuda        : backbone + attribute heads on the GPU; record/text embedding and canvas encoder on the
  //                 CPU (each costs ~19 ms per call on the CUDA EP here, < 0.5 ms on the CPU)
  //   cuda-hybrid : only the backbone on the GPU (best at batch 1-2, where the heads are launch-bound)
  //   cuda-tf32   : as cuda, with TF32 matmuls
  const bool cuda = provider == "cuda" || provider == "cuda-tf32" || provider == "cuda-hybrid" || provider == "cuda-all";
  if (!cuda && provider != "cpu")
    throw std::runtime_error("provider must be cpu, cuda, cuda-all, cuda-hybrid or cuda-tf32");
  // fused attribute sampling (export.DecodeSample): on the GPU placements by default (records/s at B = 1 / 8 / 32:
  // 86 / 295 / 340 -> 103 / 384 / 399); on the CPU it gains nothing and is slower at B = 32 (43 -> 34): the host
  // loop it replaces is cheap there, and the graph's full-vocabulary sort is slower in ORT than the C++ sampler.
  // LPP_FUSED=0 / 1 forces it off / on (A/B runs).
  const char* fu = std::getenv("LPP_FUSED");
  const bool fused = fu ? std::string(fu) != "0" : cuda && provider != "cuda-hybrid";
  auto make_options = [&](bool gpu, bool cuda_graph = false) {
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    so.SetIntraOpNumThreads(4);
    if (gpu) {
      // V2 options: ORT's CUDA EP runs FP32 matmuls in TF32 by default (10-bit mantissa), which
      // changes logits enough to fork sampled sequences within a few records. Parity runs keep
      // full FP32; "cuda-tf32" opts into the faster mode. cuDNN's default EXHAUSTIVE algorithm
      // search made every canvas_enc call take ~21 ms; HEURISTIC picks an algorithm without timing.
      const OrtApi& api = Ort::GetApi();
      OrtCUDAProviderOptionsV2* opt = nullptr;
      Ort::ThrowOnError(api.CreateCUDAProviderOptions(&opt));
      const char* keys[] = {"device_id", "use_tf32", "cudnn_conv_algo_search", "enable_cuda_graph"};
      const char* vals[] = {"0", provider == "cuda-tf32" ? "1" : "0", "HEURISTIC", cuda_graph ? "1" : "0"};
      Ort::ThrowOnError(api.UpdateCUDAProviderOptions(opt, keys, vals, 4));
      so.AppendExecutionProvider_CUDA_V2(*opt);
      api.ReleaseCUDAProviderOptions(opt);
    }
    return so;
  };
  Ort::SessionOptions so_gpu = make_options(cuda), so_cpu = make_options(false);
  // static one-step graphs: CUDA Graph capture unless LPP_CUDA_GRAPH=0 (A/B runs)
  const char* cg = std::getenv("LPP_CUDA_GRAPH");
  Ort::SessionOptions so_step = make_options(cuda, !(cg && std::string(cg) == "0"));

  for (auto& [name, g] : m["graphs"].items()) {
    if (name == "decode_sample" && !fused) continue;
    const bool backbone = name == "chunk" || name == "chunk_cached";
    const bool heads = name == "decode_heads" || name == "decode_sample";
    const bool on_gpu = cuda && (provider == "cuda-all" || backbone || (provider != "cuda-hybrid" && heads));
    const bool heads_r = name.rfind("decode_heads_r", 0) == 0;
    const bool step = name.rfind("chunk_step_r", 0) == 0 || name.rfind("chunk_inplace_r", 0) == 0 || heads_r;
    if (step && !cuda) continue;                   // the static step graphs are a CUDA-only path
    if (heads_r && provider == "cuda-hybrid") continue;   // hybrid keeps the heads on the CPU
    graphs_[name] = std::make_unique<Graph>(env_, dir + "/" + g["file"].get<std::string>(),
                                            step ? so_step : on_gpu ? so_gpu : so_cpu,
                                            g["inputs"].get<std::vector<std::string>>(),
                                            g["outputs"].get<std::vector<std::string>>(), name);
  }
}

}  // namespace lpp
