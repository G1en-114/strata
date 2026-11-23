#include "painter.h"

#include <tuple>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "grammar.h"
#include "raster.h"

#ifdef LPP_HAVE_CUDART
#include <cuda_runtime_api.h>
#endif

namespace lpp {

Painter::Painter(const std::string& dir, const std::string& provider) : proto_(Protocol::load(dir)), g_(dir, provider) {
  if (!proto_.canvas) throw std::runtime_error("runtime v0 expects a canvas-feedback model");
}

namespace {

// rows R = 2B when guided: [conditional rows ; unconditional rows] share everything but the text
template <class T>
std::vector<T> dup(const std::vector<T>& v, bool guided) {
  if (!guided) return v;
  std::vector<T> out(v);
  out.insert(out.end(), v.begin(), v.end());
  return out;
}

}  // namespace

std::vector<std::vector<Record>> Painter::generate(int B, const GenerateConfig& cfg, const std::vector<float>& text,
                                                   const std::vector<uint8_t>& text_valid, int Lt, const Uniforms& uni) {
  const Protocol& P = proto_;
  const int A = static_cast<int>(P.attrs.size()), E = static_cast<int>(P.events.size());
  const int L = P.n_layers, KV = P.n_kv_heads, HD = P.head_dim(), D = P.d_model, NL = P.num_layers();
  const bool guided = Lt > 0 && cfg.sample.cfg_scale != 1.f;
  const int R = guided ? 2 * B : B;
  const int max_records = cfg.max_records ? cfg.max_records : P.max_positions - Lt;
  // static one-step graphs (export_onnx --step-batches) are exported for a cache of max_positions slots
  const bool step_graph =
#ifdef LPP_HAVE_CUDART
      g_.provider().rfind("cuda", 0) == 0 &&
      (g_.has("chunk_inplace_r" + std::to_string(R)) || g_.has("chunk_step_r" + std::to_string(R)));
  const bool inplace = step_graph && g_.has("chunk_inplace_r" + std::to_string(R));
#else
      false;
#endif
  const int Tmax = step_graph ? P.max_positions : Lt + max_records;
  const int prim_ev = P.event_id("PRIMITIVE"), eos = P.event_id("EOS");

  // ---- KV cache owned by the runtime: static [L, R, KV, Tmax, HD]; graphs see all slots + a mask
  Tensor K = Tensor::F({L, R, KV, Tmax, HD}), V = Tensor::F({L, R, KV, Tmax, HD});
  Tensor kmask = Tensor::B({R, Tmax});
  Tensor next_pos = Tensor::I({R});
  int length = 0;

  // v1 (CUDA): the cache lives in two device buffer sets; chunk_cached reads one and writes the
  // other (ping-pong), so it never crosses PCIe. The first call reads the zeroed host tensors.
  const bool on_device = g_.provider().rfind("cuda", 0) == 0 && g_.has("chunk_cached");
  struct DevBuf {
    Ort::MemoryInfo mem{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};
    std::unique_ptr<Ort::Allocator> alloc;
    void* ptr[2][3] = {};
    std::vector<Ort::Value> val[2];
    ~DevBuf() {
      if (alloc)
        for (auto& set : ptr)
          for (void* p : set)
            if (p) alloc->Free(p);
    }
  } dev;
  int cur = -1;                                     // device set holding the cache (-1: host zeros)
  if (on_device) {
    Graph& gc = g_["chunk_cached"];
    dev.alloc = std::make_unique<Ort::Allocator>(gc.session(), dev.mem);
    const std::vector<int64_t> kv_shape{L, R, KV, Tmax, HD}, m_shape{R, Tmax};
    const size_t kv_bytes = K.f.size() * sizeof(float), m_bytes = kmask.b.size();
    for (int s = 0; s < 2; ++s) {
      dev.ptr[s][0] = dev.alloc->Alloc(kv_bytes);
      dev.ptr[s][1] = dev.alloc->Alloc(kv_bytes);
      dev.ptr[s][2] = dev.alloc->Alloc(m_bytes);
      dev.val[s].push_back(Ort::Value::CreateTensor(dev.mem, dev.ptr[s][0], kv_bytes, kv_shape.data(), 5, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
      dev.val[s].push_back(Ort::Value::CreateTensor(dev.mem, dev.ptr[s][1], kv_bytes, kv_shape.data(), 5, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
      dev.val[s].push_back(Ort::Value::CreateTensor(dev.mem, dev.ptr[s][2], m_bytes, m_shape.data(), 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL));
    }
  }
#ifdef LPP_HAVE_CUDART
  // v2 (CUDA, T = 1): the static one-step graph chunk_step_r<R>, captured as a CUDA Graph by ORT. Every
  // input lives at a fixed device address (small inputs are copied in before each run); the two cache
  // buffer sets get one captured graph each (gpu_graph_id = the set being read).
  struct StepBuf {
    Ort::Allocator* alloc = nullptr;
    std::vector<void*> ptr;
    std::vector<Ort::Value> val;                   // x, valid, write_pos, start_pos, canvas_tokens, draw, h
    ~StepBuf() { for (void* q : ptr) if (alloc && q) alloc->Free(q); }
  } sb;
  auto check = [](cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
  };
  auto chunk_step = [&](const Tensor& x, const Tensor& valid, const Tensor& tokens, const Tensor& draw) {
    Graph& gs = g_["chunk_step_r" + std::to_string(R)];
    if (sb.val.empty()) {
      sb.alloc = dev.alloc.get();
      auto add = [&](size_t bytes, std::vector<int64_t> shape, ONNXTensorElementDataType t) {
        void* q = sb.alloc->Alloc(bytes);
        sb.ptr.push_back(q);
        sb.val.push_back(Ort::Value::CreateTensor(dev.mem, q, bytes, shape.data(), shape.size(), t));
      };
      add(size_t(R) * D * 4, {R, 1, D}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      add(size_t(R), {R, 1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
      add(8, {}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
      add(size_t(R) * 8, {R}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
      add(tokens.f.size() * 4, tokens.shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      add(size_t(R), {R, 1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
      add(size_t(R) * D * 4, {R, 1, D}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    }
    const auto t0 = std::chrono::steady_clock::now();
    const int64_t wp = length;
    const auto h2d = cudaMemcpyHostToDevice;
    check(cudaMemcpy(sb.ptr[0], x.f.data(), x.f.size() * 4, h2d), "copy x");
    check(cudaMemcpy(sb.ptr[1], valid.b.data(), valid.b.size(), h2d), "copy valid");
    check(cudaMemcpy(sb.ptr[2], &wp, 8, h2d), "copy write_pos");
    check(cudaMemcpy(sb.ptr[3], next_pos.i.data(), size_t(R) * 8, h2d), "copy start_pos");
    check(cudaMemcpy(sb.ptr[4], tokens.f.data(), tokens.f.size() * 4, h2d), "copy canvas_tokens");
    check(cudaMemcpy(sb.ptr[5], draw.b.data(), draw.b.size(), h2d), "copy draw");
    Ort::IoBinding io(gs.session());
    const char* in[] = {"x", "valid", "write_pos", "start_pos", "canvas_tokens", "draw"};
    for (int k = 0; k < 6; ++k) io.BindInput(in[k], sb.val[k]);
    io.BindInput("cache_k", dev.val[cur][0]);
    io.BindInput("cache_v", dev.val[cur][1]);
    io.BindInput("cache_mask", dev.val[cur][2]);
    const int nxt = 1 - cur;
    io.BindOutput("h", sb.val[6]);
    io.BindOutput("cache_k_out", dev.val[nxt][0]);
    io.BindOutput("cache_v_out", dev.val[nxt][1]);
    io.BindOutput("cache_mask_out", dev.val[nxt][2]);
    Ort::RunOptions ro;
    ro.AddConfigEntry("gpu_graph_id", std::to_string(cur).c_str());
    gs.session().Run(ro, io);
    io.SynchronizeOutputs();                         // ORT's stream, before reading h on the host
    Tensor h = Tensor::F({R, 1, D});
    check(cudaMemcpy(h.f.data(), sb.ptr[6], h.f.size() * 4, cudaMemcpyDeviceToHost), "copy h");
    Profile::add("chunk_step", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    cur = nxt;
    for (int r = 0; r < R; ++r) next_pos.i[r] += valid.b[r];
    length += 1;
    return h;
  };
  // v3 (CUDA, T = 1): chunk_inplace_r<R> only reads the cache and returns the new position's k / v;
  // they are written into slot `length` with one strided copy each, so no step copies the whole cache.
  StepBuf ib;
  auto chunk_inplace = [&](const Tensor& x, const Tensor& valid, const Tensor& tokens, const Tensor& draw) {
    Graph& gs = g_["chunk_inplace_r" + std::to_string(R)];
    const std::vector<int64_t> kv1{L, R, KV, 1, HD};
    if (ib.val.empty()) {
      ib.alloc = dev.alloc.get();
      auto add = [&](size_t bytes, std::vector<int64_t> shape, ONNXTensorElementDataType t) {
        void* q = ib.alloc->Alloc(bytes);
        ib.ptr.push_back(q);
        ib.val.push_back(Ort::Value::CreateTensor(dev.mem, q, bytes, shape.data(), shape.size(), t));
      };
      add(size_t(R) * D * 4, {R, 1, D}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);             // 0 x
      add(size_t(R) * 8, {R}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);                        // 1 start_pos
      add(tokens.f.size() * 4, tokens.shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);         // 2 canvas_tokens
      add(size_t(R), {R, 1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);                          // 3 draw
      add(size_t(R) * D * 4, {R, 1, D}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);             // 4 h
      const size_t kvb = size_t(L) * R * KV * HD * 4;
      add(kvb, kv1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);                                  // 5 k_new
      add(kvb, kv1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);                                  // 6 v_new
    }
    const auto t0 = std::chrono::steady_clock::now();
    const auto h2d = cudaMemcpyHostToDevice;
    check(cudaMemcpy(ib.ptr[0], x.f.data(), x.f.size() * 4, h2d), "copy x");
    check(cudaMemcpy(ib.ptr[1], next_pos.i.data(), size_t(R) * 8, h2d), "copy start_pos");
    check(cudaMemcpy(ib.ptr[2], tokens.f.data(), tokens.f.size() * 4, h2d), "copy canvas_tokens");
    check(cudaMemcpy(ib.ptr[3], draw.b.data(), draw.b.size(), h2d), "copy draw");
    Ort::IoBinding io(gs.session());
    io.BindInput("x", ib.val[0]);
    io.BindInput("cache_k", dev.val[cur][0]);
    io.BindInput("cache_v", dev.val[cur][1]);
    io.BindInput("cache_mask", dev.val[cur][2]);
    io.BindInput("start_pos", ib.val[1]);
    io.BindInput("canvas_tokens", ib.val[2]);
    io.BindInput("draw", ib.val[3]);
    io.BindOutput("h", ib.val[4]);
    io.BindOutput("k_new", ib.val[5]);
    io.BindOutput("v_new", ib.val[6]);
    Ort::RunOptions ro;
    ro.AddConfigEntry("gpu_graph_id", std::to_string(cur).c_str());
    gs.session().Run(ro, io);
    io.SynchronizeOutputs();
    // write slot `length`: L*R*KV rows of HD floats, stride Tmax*HD in the cache; mask row stride Tmax
    const size_t row = size_t(HD) * 4, pitch = size_t(Tmax) * HD * 4;
    auto* kc = static_cast<char*>(dev.ptr[cur][0]) + size_t(length) * row;
    auto* vc = static_cast<char*>(dev.ptr[cur][1]) + size_t(length) * row;
    check(cudaMemcpy2D(kc, pitch, ib.ptr[5], row, row, size_t(L) * R * KV, cudaMemcpyDeviceToDevice), "write k");
    check(cudaMemcpy2D(vc, pitch, ib.ptr[6], row, row, size_t(L) * R * KV, cudaMemcpyDeviceToDevice), "write v");
    check(cudaMemcpy2D(static_cast<char*>(dev.ptr[cur][2]) + length, size_t(Tmax), valid.b.data(), 1, 1, size_t(R), h2d),
          "write mask");
    Tensor h = Tensor::F({R, 1, D});
    check(cudaMemcpy(h.f.data(), ib.ptr[4], h.f.size() * 4, cudaMemcpyDeviceToHost), "copy h");
    Profile::add("chunk_inplace", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    for (int r = 0; r < R; ++r) next_pos.i[r] += valid.b[r];
    length += 1;
    return h;
  };
#endif
  auto chunk_device = [&](const Tensor& x, const Tensor& valid, const Tensor& tokens, const Tensor& draw) {
#ifdef LPP_HAVE_CUDART
    if (step_graph && cur >= 0 && x.shape[1] == 1)
      return inplace ? chunk_inplace(x, valid, tokens, draw) : chunk_step(x, valid, tokens, draw);
#endif
    Graph& gc = g_["chunk_cached"];
    const int T = static_cast<int>(x.shape[1]);
    Tensor wp = Tensor::I({}, length);
    Ort::IoBinding io(gc.session());
    Ort::MemoryInfo cpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> host;                   // keep host-backed values alive until Run
    auto bind_host = [&](const char* name, const Tensor& t) {
      host.push_back(gc.value(t));
      io.BindInput(name, host.back());
    };
    host.reserve(9);
    bind_host("x", x);
    bind_host("valid", valid);
    if (cur < 0) {
      bind_host("cache_k", K);
      bind_host("cache_v", V);
      bind_host("cache_mask", kmask);
    } else {
      io.BindInput("cache_k", dev.val[cur][0]);
      io.BindInput("cache_v", dev.val[cur][1]);
      io.BindInput("cache_mask", dev.val[cur][2]);
    }
    bind_host("write_pos", wp);
    bind_host("start_pos", next_pos);
    bind_host("canvas_tokens", tokens);
    bind_host("draw", draw);
    const int nxt = cur < 0 ? 0 : 1 - cur;
    io.BindOutput("h", cpu);
    io.BindOutput("cache_k_out", dev.val[nxt][0]);
    io.BindOutput("cache_v_out", dev.val[nxt][1]);
    io.BindOutput("cache_mask_out", dev.val[nxt][2]);
    const auto t0 = std::chrono::steady_clock::now();
    gc.session().Run(Ort::RunOptions{nullptr}, io);
    io.SynchronizeOutputs();
    Profile::add("chunk_cached", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    cur = nxt;
    for (int r = 0; r < R; ++r)
      for (int t = 0; t < T; ++t) next_pos.i[r] += valid.b[size_t(r) * T + t];
    length += T;
    auto outs = io.GetOutputValues();
    auto shape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
    Tensor h = Tensor::F(shape);
    const float* p = outs[0].GetTensorData<float>();
    std::copy(p, p + h.f.size(), h.f.begin());
    return h;                                        // h [R, T, D]
  };

  auto chunk = [&](const Tensor& x, const Tensor& valid, const Tensor& tokens, const Tensor& draw) {
    if (on_device) return chunk_device(x, valid, tokens, draw);
    const int T = static_cast<int>(x.shape[1]);
    auto out = g_["chunk"].run({&x, &valid, &K, &V, &kmask, &next_pos, &tokens, &draw});
    const Tensor &kn = out[1], &vn = out[2];       // [L, R, KV, T, HD]
    for (int l = 0; l < L; ++l)
      for (int r = 0; r < R; ++r)
        for (int h = 0; h < KV; ++h)
          for (int t = 0; t < T; ++t) {
            const size_t src = ((((size_t(l) * R + r) * KV + h) * T + t) * HD);
            const size_t dst = ((((size_t(l) * R + r) * KV + h) * Tmax + length + t) * HD);
            std::copy_n(kn.f.begin() + src, HD, K.f.begin() + dst);
            std::copy_n(vn.f.begin() + src, HD, V.f.begin() + dst);
          }
    for (int r = 0; r < R; ++r)
      for (int t = 0; t < T; ++t) {
        kmask.b[size_t(r) * Tmax + length + t] = valid.b[size_t(r) * T + t];
        next_pos.i[r] += valid.b[size_t(r) * T + t];
      }
    length += T;
    return std::move(out[0]);                      // h [R, T, D]
  };

  // ---- canvas state and its tokens
  const int S = P.canvas_cfg_size, NT = (S / 8) * (S / 8);
  std::vector<Layered> canvas(B, Layered(NL, S));
  Tensor ctok;                                      // [B, NT, D]
  {
    Tensor cv = Tensor::F({B, 3 + NL, S, S});
    for (int b = 0; b < B; ++b) compose(P, canvas[b], cv.f.data() + size_t(b) * (3 + NL) * S * S);
    ctok = g_["canvas_enc"].run({&cv})[0];
  }
  auto canvas_rows = [&]() {
    Tensor t = Tensor::F({R, NT, D});
    for (int r = 0; r < R; ++r) std::copy_n(ctok.f.begin() + size_t(r % B) * NT * D, size_t(NT) * D, t.f.begin() + size_t(r) * NT * D);
    return t;
  };

  // ---- prefix: [text][BOS]
  Tensor h;
  {
    Tensor kind = Tensor::I({R, 1}, P.kind_bos), ev = Tensor::I({R, 1}, -1), lay = Tensor::I({R, 1}, -1),
           at = Tensor::I({R, 1, A}, P.no_attr);
    Tensor xr = g_["embed_record"].run({&kind, &ev, &lay, &at})[0];      // [R, 1, D]
    Tensor x = Tensor::F({R, Lt + 1, D}), valid = Tensor::B({R, Lt + 1}), draw = Tensor::B({R, Lt + 1});
    if (Lt) {
      Tensor tx = Tensor::F({R, Lt, P.text_dim});
      std::copy(text.begin(), text.end(), tx.f.begin());                // uncond rows stay zero
      Tensor xt = g_["embed_text"].run({&tx})[0];                       // [R, Lt, D]
      for (int r = 0; r < R; ++r) {
        std::copy_n(xt.f.begin() + size_t(r) * Lt * D, size_t(Lt) * D, x.f.begin() + size_t(r) * (Lt + 1) * D);
        for (int t = 0; t < Lt; ++t) valid.b[size_t(r) * (Lt + 1) + t] = r < B ? text_valid[size_t(r) * Lt + t] : 0;
      }
    }
    for (int r = 0; r < R; ++r) {
      std::copy_n(xr.f.begin() + size_t(r) * D, D, x.f.begin() + (size_t(r) * (Lt + 1) + Lt) * D);
      valid.b[size_t(r) * (Lt + 1) + Lt] = 1;
      draw.b[size_t(r) * (Lt + 1) + Lt] = 1;
    }
    Tensor tok = canvas_rows();
    Tensor hall = chunk(x, valid, tok, draw);
    h = Tensor::F({R, D});
    for (int r = 0; r < R; ++r) std::copy_n(hall.f.begin() + (size_t(r) * (Lt + 1) + Lt) * D, D, h.f.begin() + size_t(r) * D);
  }

  // ---- decode loop
  std::vector<Grammar> gram;
  for (int b = 0; b < B; ++b) {
    gram.emplace_back(P, cfg.require_before_eos, !cfg.forced.empty(), P.draft);   // prefixes may hold REFINE / REDRAW
  }
  std::vector<std::vector<Record>> out(B);
  for (int b = 0; b < B; ++b) out[b].push_back({P.kind_bos, -1, -1, std::vector<int64_t>(A, P.no_attr)});
  std::vector<bool> done(B, false);
  std::vector<std::vector<std::tuple<std::vector<int64_t>, int, bool>>> pending(B);   // tokens, layer, is line
  std::vector<int> since(B, 0);
  std::vector<float> lg;

  auto guided_row = [&](const Tensor& t, int b, int n) {      // [R, n] -> guided logits of sample b
    lg.resize(n);
    const float* c = t.f.data() + size_t(b) * n;
    if (guided) guide(c, t.f.data() + size_t(B + b) * n, cfg.sample.cfg_scale, lg.data(), n);
    else std::copy_n(c, n, lg.begin());
    return lg.data();
  };

  // attribute heads: in the cuda placement, the static decode_heads_r<R> graph runs as a captured CUDA
  // Graph (fixed device buffers, logits copied back); hybrid placement and other R keep the dynamic graph
#ifdef LPP_HAVE_CUDART
  const bool heads_static = on_device && g_.provider() != "cuda-hybrid" && g_.has("decode_heads_r" + std::to_string(R));
  StepBuf hb;
  std::vector<std::vector<int64_t>> head_shape;
  // need: the one output the caller reads (0 = event logits, 1 + i = attribute i); only it is copied back
  auto heads_run = [&](const Tensor& hh, const Tensor& e, const Tensor& at, int need) -> std::vector<Tensor> {
    if (!heads_static) return g_["decode_heads"].run({&hh, &e, &at});
    Graph& gh = g_["decode_heads_r" + std::to_string(R)];
    if (hb.val.empty()) {
      hb.alloc = dev.alloc.get();
      auto add = [&](size_t bytes, std::vector<int64_t> shape, ONNXTensorElementDataType t) {
        void* q = hb.alloc->Alloc(bytes);
        hb.ptr.push_back(q);
        hb.val.push_back(Ort::Value::CreateTensor(dev.mem, q, bytes, shape.data(), shape.size(), t));
      };
      add(size_t(R) * D * 4, {R, D}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      add(size_t(R) * 8, {R}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
      add(size_t(R) * A * 8, {R, A}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
      head_shape.push_back({R, E});
      for (int i = 0; i < A; ++i) head_shape.push_back({R, static_cast<int64_t>(P.vocab[i])});
      for (auto& sh : head_shape) add(size_t(sh[0]) * sh[1] * 4, sh, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    }
    const auto t0 = std::chrono::steady_clock::now();
    check(cudaMemcpy(hb.ptr[0], hh.f.data(), hh.f.size() * 4, cudaMemcpyHostToDevice), "copy h");
    check(cudaMemcpy(hb.ptr[1], e.i.data(), e.i.size() * 8, cudaMemcpyHostToDevice), "copy event");
    check(cudaMemcpy(hb.ptr[2], at.i.data(), at.i.size() * 8, cudaMemcpyHostToDevice), "copy attrs");
    Ort::IoBinding io(gh.session());
    for (int k = 0; k < 3; ++k) io.BindInput(gh.input_names()[k].c_str(), hb.val[k]);
    for (size_t k = 0; k < head_shape.size(); ++k) io.BindOutput(gh.output_names()[k].c_str(), hb.val[3 + k]);
    Ort::RunOptions ro;
    ro.AddConfigEntry("gpu_graph_id", "0");
    gh.session().Run(ro, io);
    io.SynchronizeOutputs();
    std::vector<Tensor> out(head_shape.size());
    out[need] = Tensor::F(head_shape[need]);
    check(cudaMemcpy(out[need].f.data(), hb.ptr[3 + need], out[need].f.size() * 4, cudaMemcpyDeviceToHost), "copy logits");
    Profile::add("decode_heads_r", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return out;
  };
#else
  auto heads_run = [&](const Tensor& hh, const Tensor& e, const Tensor& at, int) { return g_["decode_heads"].run({&hh, &e, &at}); };
#endif

  Tensor zeros_e = Tensor::I({R}), zeros_a = Tensor::I({R, A});
  for (int step = 1; step < max_records; ++step) {
    auto heads0 = heads_run(h, zeros_e, zeros_a, 0);
    std::vector<int64_t> ev(B);
    for (int b = 0; b < B; ++b) {
      float* l = guided_row(heads0[0], b, E);
      std::vector<int> allowed = step == max_records - 1 ? std::vector<int>{eos}
                                 : done[b] ? std::vector<int>{eos} : gram[b].allowed_events();
      std::vector<float> m(E, -std::numeric_limits<float>::infinity());
      for (int e : allowed) m[e] = l[e];
      ev[b] = sample_top_p(m.data(), E, cfg.sample, uni(step, 0, b));
    }
    auto in_prefix = [&](int b) { return !cfg.forced.empty() && step - 1 < static_cast<int>(cfg.forced[b].size()); };
    for (int b = 0; b < B; ++b)
      if (in_prefix(b)) ev[b] = cfg.forced[b][step - 1].event;
    std::vector<std::vector<int64_t>> attrs(B, std::vector<int64_t>(A, P.no_attr));
    if (std::any_of(ev.begin(), ev.end(), [&](int64_t e) { return P.is_stroke(e); })) {
      if (guided && g_.has("decode_sample")) {
        // all attributes in one call (export.DecodeSample: guidance, top-p sampling and fixed slots in the
        // graph, same uniforms): ~10 launches + device-to-host copies per record become 1 (deploy_report_v3.md)
        Tensor evb = Tensor::I({B}), u = Tensor::F({B, A});
        for (int b = 0; b < B; ++b) {
          evb.i[b] = ev[b];
          for (int i = 0; i < A; ++i) u.f[size_t(b) * A + i] = uni(step, 1 + i, b);
        }
        Tensor cs = Tensor::F({1}, cfg.sample.cfg_scale), tp = Tensor::F({1}, cfg.sample.top_p),
               tt = Tensor::F({1}, cfg.sample.temperature);
        const auto o = g_["decode_sample"].run({&h, &evb, &u, &cs, &tp, &tt});
        for (int b = 0; b < B; ++b)
          for (int i = 0; i < A; ++i) attrs[b][i] = o[0].i[size_t(b) * A + i];
      } else {
      Tensor evt = Tensor::I({R}), at = Tensor::I({R, A});
      for (int r = 0; r < R; ++r) evt.i[r] = ev[r % B];
      for (int i = 0; i < A; ++i) {
        for (int r = 0; r < R; ++r)
          for (int j = 0; j < A; ++j) at.i[size_t(r) * A + j] = std::max<int64_t>(attrs[r % B][j], 0);
        auto heads = heads_run(h, evt, at, 1 + i);
        const int n = static_cast<int>(P.vocab[i]);
        for (int b = 0; b < B; ++b) attrs[b][i] = sample_top_p(guided_row(heads[1 + i], b, n), n, cfg.sample, uni(step, 1 + i, b));
      }
      }
      for (int b = 0; b < B; ++b) {
        if (!P.is_stroke(ev[b])) std::fill(attrs[b].begin(), attrs[b].end(), P.no_attr);
        else if (ev[b] == P.line_ev && P.beta_attr >= 0) attrs[b][P.beta_attr] = 0;      // lines: beta = 0
        else if (ev[b] == prim_ev && P.bend_attr >= 0) attrs[b][P.bend_attr] = P.bend_mid;  // Gaussians: bend = 0
      }
    }
    for (int b = 0; b < B; ++b)
      if (in_prefix(b)) attrs[b] = cfg.forced[b][step - 1].attrs;
    std::vector<int64_t> layer(B), kind(B);
    std::vector<uint8_t> live(B);
    for (int b = 0; b < B; ++b) {
      live[b] = !done[b];
      kind[b] = P.is_stroke(ev[b]) ? P.kind_primitive : P.kind_control;
      if (gram[b].done()) {
        layer[b] = -1;
      } else {
        if (step == max_records - 1 && ev[b] == eos) gram[b].force_done();
        else gram[b].step(static_cast<int>(ev[b]));
        layer[b] = gram[b].done() ? -1 : gram[b].current();
      }
      if (live[b]) out[b].push_back({kind[b], ev[b], layer[b], attrs[b]});
      done[b] = done[b] || ev[b] == eos;
    }
    const bool all_done = std::all_of(done.begin(), done.end(), [](bool d) { return d; });
    if (cfg.on_progress && (all_done || step % std::max(1, cfg.progress_every) == 0)) cfg.on_progress(step, out, done);
    if (all_done) break;

    // canvas checkpoints (lpp.canvas.CanvasTracker.update)
    std::vector<int> hit;
    for (int b = 0; b < B; ++b) {
      if (!live[b]) continue;
      if (kind[b] == P.kind_primitive) {
        pending[b].push_back({attrs[b], static_cast<int>(layer[b]), ev[b] == P.line_ev});
        ++since[b];
      }
      if (kind[b] == P.kind_bos || kind[b] == P.kind_control || since[b] >= P.canvas_k) {
        hit.push_back(b);
        since[b] = 0;
      }
    }
    if (!hit.empty()) {
      Tensor cv = Tensor::F({static_cast<int64_t>(hit.size()), 3 + NL, S, S});
      for (size_t j = 0; j < hit.size(); ++j) {
        const int b = hit[j];
        auto& st = canvas[b];
        const size_t plane = size_t(S) * S;
        for (auto& [a, k, is_line] : pending[b]) {
          if (k < 0) continue;
          std::vector<int64_t> t(a);
          for (auto& v : t) v = std::max<int64_t>(v, 0);
          paint(P, P.decode_canvas(t.data(), is_line), S, st.P.data() + size_t(k) * 3 * plane, st.T.data() + size_t(k) * plane);
        }
        pending[b].clear();
        compose(P, st, cv.f.data() + j * (3 + NL) * plane);
      }
      Tensor nt = g_["canvas_enc"].run({&cv})[0];
      for (size_t j = 0; j < hit.size(); ++j)
        std::copy_n(nt.f.begin() + j * NT * D, size_t(NT) * D, ctok.f.begin() + size_t(hit[j]) * NT * D);
    }

    Tensor kd = Tensor::I({R, 1}), evt = Tensor::I({R, 1}), lay = Tensor::I({R, 1}), at = Tensor::I({R, 1, A});
    Tensor valid = Tensor::B({R, 1}), draw = Tensor::B({R, 1}, true);
    for (int r = 0; r < R; ++r) {
      const int b = r % B;
      kd.i[r] = kind[b], evt.i[r] = ev[b], lay.i[r] = layer[b], valid.b[r] = live[b];
      std::copy(attrs[b].begin(), attrs[b].end(), at.i.begin() + size_t(r) * A);
    }
    Tensor x = g_["embed_record"].run({&kd, &evt, &lay, &at})[0];
    Tensor tok = canvas_rows();
    Tensor hn = chunk(x, valid, tok, draw);
    h = Tensor::F({R, D});
    std::copy(hn.f.begin(), hn.f.end(), h.f.begin());
  }
  return out;
}

}  // namespace lpp
