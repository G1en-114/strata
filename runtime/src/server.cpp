// strata: a local painting server with an OpenAI-compatible image API (and Ollama-style endpoints).
//
//   strata serve --model strata-v3=out/onnx/v3_final_fused [--model strata-v3:fp16=out/onnx/v3_final_fused_fp16]
//                [--port 11435] [--provider cuda] [--max-batch 32] [--web runtime/web]
//   strata run --model ... "金发碧眼的少女" -o out.png [--seed 1]
//
// Endpoints
//   POST /v1/images/generations   OpenAI images API: {model, prompt, n, size, response_format: b64_json | url}
//                                 (+ seed, cfg_scale, top_p); works as an Open WebUI image-generation backend
//   GET  /v1/models               OpenAI model list
//   POST /api/generate            Ollama-style: {model, prompt, stream, options: {seed, cfg_scale, top_p}};
//                                 streamed NDJSON, one line per progress frame (PNG so far), the last has done: true
//   GET  /api/tags, /api/version  Ollama-style model list / version
//   GET  /                        the web UI (runtime/web/index.html): watch the painting stroke by stroke
// One worker thread owns the GPU. Requests waiting together with the same model and sampling settings are decoded
// as one batch (up to --max-batch images), so concurrent single-image requests share a step.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "httplib.h"
#include "json.hpp"
#include "painter.h"
#include "raster.h"
#include "text_encoder.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace lpp;
using json = nlohmann::json;

namespace {

const char* kVersion = "0.1.0";

std::string base64(const std::string& in) {
  static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve((in.size() + 2) / 3 * 4);
  size_t i = 0;
  for (; i + 2 < in.size(); i += 3) {
    const uint32_t v = (uint8_t(in[i]) << 16) | (uint8_t(in[i + 1]) << 8) | uint8_t(in[i + 2]);
    out += T[v >> 18], out += T[(v >> 12) & 63], out += T[(v >> 6) & 63], out += T[v & 63];
  }
  if (i + 1 == in.size()) {
    const uint32_t v = uint8_t(in[i]) << 16;
    out += T[v >> 18], out += T[(v >> 12) & 63], out += "==";
  } else if (i + 2 == in.size()) {
    const uint32_t v = (uint8_t(in[i]) << 16) | (uint8_t(in[i + 1]) << 8);
    out += T[v >> 18], out += T[(v >> 12) & 63], out += T[(v >> 6) & 63], out += '=';
  }
  return out;
}

std::string png(const std::vector<unsigned char>& rgb, int S) {
  std::string out;
  stbi_write_png_to_func([](void* ctx, void* data, int n) { static_cast<std::string*>(ctx)->append(static_cast<char*>(data), n); },
                         &out, S, S, 3, rgb.data(), S * 3);
  return out;
}

std::string render_png(const Protocol& P, const std::vector<Record>& recs) {
  std::vector<Stroke> strokes;
  for (const auto& r : recs)
    if (r.kind == P.kind_primitive) strokes.push_back({static_cast<int>(r.layer), r.event == P.line_ev, r.attrs.data()});
  return png(render_strokes_rgb8(P, strokes, P.canvas_size), P.canvas_size);
}

double now_s() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// ---- one request: n images of one prompt
struct Frame {                          // progress of one image
  int index = 0, step = 0, strokes = 0;
  bool done = false;
  std::string png, id;                  // id: set on the final frame (continue from it with "from")
};
// a finished painting, kept so it can be continued (refine a layer / redraw a layer)
struct Painting {
  std::string model, prompt;
  std::vector<Record> records;
};
const char* kLayerNames[] = {"background", "body", "skin", "features", "hair", "accessories"};

// the edit prefixes the model was trained on (tools/continue_parity.py, lpp.sequence.build "refine" / "redraw"):
// layers in order, each LAYER_j + its strokes in sequence order (redraw: without layer k), then REFINE_k / REDRAW_k
std::vector<Record> edit_prefix(const Protocol& P, const std::vector<Record>& recs, const std::string& mode, int k) {
  const int A = static_cast<int>(P.attrs.size());
  std::vector<Record> out;
  std::set<int> layers;
  for (auto& r : recs) if (r.kind == P.kind_primitive) layers.insert(static_cast<int>(r.layer));
  int used = 0;
  for (int j : layers) {
    if (mode == "redraw" && j == k) continue;
    out.push_back({P.kind_control, P.event_id("LAYER_" + std::to_string(j)), j, std::vector<int64_t>(A, P.no_attr)});
    for (auto& r : recs) if (r.kind == P.kind_primitive && r.layer == j) out.push_back(r);
  }
  for (auto& r : recs) used += r.kind == P.kind_primitive && r.layer == k;
  if (mode == "refine") {
    if (!layers.count(k)) throw std::runtime_error(std::string("this painting has no ") + kLayerNames[k] + " layer to refine");
    if (used >= P.layer_cap.at(k)) throw std::runtime_error(std::string("the ") + kLayerNames[k] + " layer is full (" +
                                                            std::to_string(used) + " strokes, the cap); redraw it instead");
  }
  out.push_back({P.kind_control, P.event_id((mode == "refine" ? "REFINE_" : "REDRAW_") + std::to_string(k)), k,
                 std::vector<int64_t>(A, P.no_attr)});
  return out;
}
struct Job {
  std::string model, prompt;
  int n = 1;
  float cfg_scale = 5.f, top_p = 0.6f;
  uint32_t seed = 0;
  int progress_every = 0;               // 0: no frames
  std::function<void(const Frame&)> on_frame;
  std::promise<std::vector<std::string>> result;   // final PNGs
  std::vector<std::string> ids;                    // their painting ids (set before the result)
  std::string edit_from, edit_mode;                // continue a painting: "refine" | "redraw" layer edit_layer
  int edit_layer = -1;
  std::vector<Record> forced;                      // continuation prefix: built on the worker from edit_from, or
                                                   // given (a traced sketch, /api/trace mode=sketch)
  bool continuation = false;                       // decoded with a prefix (relaxed grammar, own batches)
  double queued = now_s();
};

class Server {
 public:
  Server(std::map<std::string, std::string> models, std::string provider, int max_batch)
      : dirs_(std::move(models)), provider_(std::move(provider)), max_batch_(max_batch) {
    std::string text_dir;
    for (auto& [name, dir] : dirs_) {
      std::ifstream f(dir + "/manifest.json");
      if (!f) throw std::runtime_error("model " + name + ": no manifest.json in " + dir);
      if (text_dir.empty() && json::parse(f).contains("text_encoder")) text_dir = dir;
    }
    if (text_dir.empty()) throw std::runtime_error("no model directory has a text encoder (tools/export_text_encoder.py)");
    text_ = std::make_unique<TextEncoder>(text_dir);
    worker_ = std::thread([this] { loop(); });
  }
  ~Server() {
    { std::lock_guard<std::mutex> lk(mu_); stop_ = true; }
    cv_.notify_all();
    worker_.join();
  }
  const std::map<std::string, std::string>& models() const { return dirs_; }
  bool has(const std::string& m) const { return dirs_.count(m) > 0; }
  std::future<std::vector<std::string>> submit(std::shared_ptr<Job> j) {
    auto f = j->result.get_future();
    { std::lock_guard<std::mutex> lk(mu_); queue_.push_back(std::move(j)); }
    cv_.notify_one();
    return f;
  }
  // the protocol of a model (for rendering outside the worker): loaded once
  const Protocol& protocol(const std::string& model) {
    std::lock_guard<std::mutex> lk(protomu_);
    auto it = protos_.find(model);
    if (it == protos_.end()) it = protos_.emplace(model, std::make_unique<Protocol>(Protocol::load(dirs_.at(model)))).first;
    return *it->second;
  }
  bool recall(const std::string& id, Painting& out) {
    std::lock_guard<std::mutex> lk(pmu_);
    auto it = paintings_.find(id);
    if (it == paintings_.end()) return false;
    out = it->second;
    return true;
  }
  json stats() const {
    return {{"requests", requests_.load()}, {"images", images_.load()}, {"batches", batches_.load()},
            {"records", records_.load()}, {"busy_seconds", busy_.load()}};
  }

 private:
  Painter& painter(const std::string& model) {
    auto it = painters_.find(model);
    if (it == painters_.end()) {
      std::cerr << "loading " << model << " from " << dirs_.at(model) << " (" << provider_ << ")\n";
      it = painters_.emplace(model, std::make_unique<Painter>(dirs_.at(model), provider_)).first;
    }
    return *it->second;
  }

  // the next batch: the oldest job plus every waiting job with the same model and sampling settings
  std::vector<std::shared_ptr<Job>> take() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
    std::vector<std::shared_ptr<Job>> batch;
    if (stop_ && queue_.empty()) return batch;
    auto first = queue_.front();
    int images = 0;
    for (auto it = queue_.begin(); it != queue_.end();) {
      auto& j = *it;
      // continuations relax the grammar (REFINE / REDRAW allowed): never batched with fresh paintings
      if (j->model == first->model && j->cfg_scale == first->cfg_scale && j->top_p == first->top_p &&
          j->continuation == first->continuation &&
          images + j->n <= std::max(max_batch_, first->n)) {
        images += j->n;
        batch.push_back(j);
        it = queue_.erase(it);
      } else {
        ++it;
      }
    }
    return batch;
  }

  void loop() {
    for (;;) {
      auto batch = take();
      if (batch.empty()) return;
      try {
        run(batch);
      } catch (const std::exception& e) {
        for (auto& j : batch) j->result.set_exception(std::make_exception_ptr(std::runtime_error(e.what())));
      }
    }
  }

  void run(std::vector<std::shared_ptr<Job>>& batch) {
    const auto t0 = std::chrono::steady_clock::now();
    Painter& p = painter(batch[0]->model);
    const Protocol& P = p.protocol();
    for (auto it = batch.begin(); it != batch.end();) {       // a bad edit fails its own request only
      auto& j = *it;
      try {
        if (!j->edit_from.empty()) {
          Painting src;
          if (!recall(j->edit_from, src)) throw std::runtime_error("unknown painting id " + j->edit_from);
          j->forced = edit_prefix(P, src.records, j->edit_mode, j->edit_layer);
        }
        ++it;
      } catch (const std::exception& e) {
        j->result.set_exception(std::make_exception_ptr(std::runtime_error(e.what())));
        it = batch.erase(it);
      }
    }
    if (batch.empty()) return;
    std::vector<std::string> prompts;
    std::vector<std::pair<Job*, int>> owner;              // image b -> (job, index within the job)
    for (auto& j : batch)
      for (int i = 0; i < j->n; ++i) prompts.push_back(j->prompt), owner.emplace_back(j.get(), i);
    const int B = static_cast<int>(prompts.size());
    TextBatch tb = text_->encode(prompts);
    // per image randomness from its job's seed: the same request gives the same pictures in any batch
    const int fields = 1 + static_cast<int>(P.attrs.size()), steps = P.max_positions;
    std::vector<std::vector<float>> U(B);
    for (int b = 0; b < B; ++b) {
      std::mt19937 rng(owner[b].first->seed + 1000003u * static_cast<uint32_t>(owner[b].second));
      std::uniform_real_distribution<float> U01(0.f, 1.f);
      U[b].resize(size_t(steps) * fields);
      for (auto& x : U[b]) x = U01(rng);
    }
    Uniforms uni = [&](int step, int field, int b) { return U[b][size_t(step) * fields + field]; };
    GenerateConfig cfg;
    cfg.sample.cfg_scale = batch[0]->cfg_scale;
    cfg.sample.top_p = batch[0]->top_p;
    if (batch[0]->continuation) {
      cfg.require_before_eos.clear();
      for (int b = 0; b < B; ++b) cfg.forced.push_back(owner[b].first->forced);
    }
    int every = 0;
    for (auto& j : batch) if (j->progress_every > 0) every = every ? std::min(every, j->progress_every) : j->progress_every;
    if (every > 0) {
      cfg.progress_every = every;
      cfg.on_progress = [&](int step, const std::vector<std::vector<Record>>& out, const std::vector<bool>& done) {
        for (int b = 0; b < B; ++b) {
          Job* j = owner[b].first;
          if (!j->on_frame || j->progress_every <= 0 || done[b]) continue;
          Frame f;
          f.index = owner[b].second, f.step = step;
          for (auto& r : out[b]) f.strokes += r.kind == P.kind_primitive;
          f.png = render_png(P, out[b]);
          j->on_frame(f);
        }
      };
    }
    auto seqs = p.generate(B, cfg, tb.feats, tb.valid, tb.Lt, uni);
    std::map<Job*, std::vector<std::string>> pngs;
    long recs = 0;
    for (int b = 0; b < B; ++b) {
      recs += static_cast<long>(seqs[b].size());
      std::string img = render_png(P, seqs[b]);
      Job* j = owner[b].first;
      const std::string id = remember({j->model, j->prompt, seqs[b]});
      j->ids.push_back(id);
      if (j->on_frame && j->progress_every > 0) {
        Frame f;
        f.index = owner[b].second, f.step = static_cast<int>(seqs[b].size()), f.done = true, f.png = img, f.id = id;
        for (auto& r : seqs[b]) f.strokes += r.kind == P.kind_primitive;
        j->on_frame(f);
      }
      pngs[j].push_back(std::move(img));
    }
    for (auto& j : batch) j->result.set_value(pngs[j.get()]);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    requests_ += static_cast<long>(batch.size()), images_ += B, batches_ += 1, records_ += recs;
    busy_ = busy_.load() + secs;
    std::cerr << "batch: " << batch.size() << " requests, " << B << " images, " << recs << " records in " << secs
              << " s (" << static_cast<int>(recs / secs) << " records/s)\n";
  }


 public:
  std::string remember(Painting p) {
    std::lock_guard<std::mutex> lk(pmu_);
    const std::string id = "p" + std::to_string(++next_painting_);
    paintings_[id] = std::move(p);
    porder_.push_back(id);
    while (porder_.size() > 512) paintings_.erase(porder_.front()), porder_.pop_front();
    return id;
  }

 private:
  std::mutex pmu_, protomu_;
  std::map<std::string, std::unique_ptr<Protocol>> protos_;
  long next_painting_ = 0;
  std::map<std::string, Painting> paintings_;
  std::deque<std::string> porder_;
  std::map<std::string, std::string> dirs_;
  std::string provider_;
  int max_batch_;
  std::unique_ptr<TextEncoder> text_;
  std::map<std::string, std::unique_ptr<Painter>> painters_;   // worker thread only
  std::deque<std::shared_ptr<Job>> queue_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;
  std::thread worker_;
  std::atomic<long> requests_{0}, images_{0}, batches_{0}, records_{0};
  std::atomic<double> busy_{0};
};

// small store for response_format = url
class ImageStore {
 public:
  std::string put(std::string png) {
    std::lock_guard<std::mutex> lk(mu_);
    const std::string id = std::to_string(++next_);
    images_[id] = std::move(png);
    order_.push_back(id);
    while (order_.size() > 256) images_.erase(order_.front()), order_.pop_front();
    return id;
  }
  bool get(const std::string& id, std::string& out) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = images_.find(id);
    if (it == images_.end()) return false;
    out = it->second;
    return true;
  }

 private:
  std::mutex mu_;
  long next_ = 0;
  std::map<std::string, std::string> images_;
  std::deque<std::string> order_;
};

void cors(httplib::Response& res) {
  res.set_header("Access-Control-Allow-Origin", "*");
  res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
  res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
}

void error(httplib::Response& res, int status, const std::string& msg, const std::string& type = "invalid_request_error") {
  res.status = status;
  res.set_content(json{{"error", {{"message", msg}, {"type", type}}}}.dump(), "application/json");
}

// ---- picture -> strokes, by the local trace service (tools/trace_service.py, PyTorch decomposer)
json call_trace(int port, const std::string& image, bool sketch) {
  httplib::Client cli("127.0.0.1", port);
  cli.set_connection_timeout(5);
  cli.set_read_timeout(600);
  auto r = cli.Post(std::string("/trace?sketch=") + (sketch ? "1" : "0"), image, "application/octet-stream");
  if (!r) throw std::runtime_error("the trace service is not running on port " + std::to_string(port) +
                                   " (./serve.sh starts it: tools/trace_service.py)");
  json j = json::parse(r->body);
  if (r->status != 200) throw std::runtime_error(j.value("error", "trace service error"));
  return j;
}

std::vector<Record> records_from(const json& rows) {
  std::vector<Record> out;
  for (auto& r : rows) {
    std::vector<int64_t> v = r.get<std::vector<int64_t>>();
    out.push_back({v[0], v[1], v[2], std::vector<int64_t>(v.begin() + 3, v.end())});
  }
  return out;
}

// a streamed NDJSON response: producer(push) runs on its own thread; push(line, last)
using Push = std::function<void(std::string, bool)>;
void stream_lines(httplib::Response& res, std::function<void(Push)> producer) {
  struct Chan {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool finished = false;
  };
  auto ch = std::make_shared<Chan>();
  Push push = [ch](std::string line, bool last) {
    { std::lock_guard<std::mutex> lk(ch->mu); ch->lines.push_back(std::move(line)); ch->finished = ch->finished || last; }
    ch->cv.notify_all();
  };
  std::thread([producer, push] {
    try { producer(push); }
    catch (const std::exception& e) { push(json{{"error", e.what()}, {"done", true}}.dump() + "\n", true); }
  }).detach();
  res.set_chunked_content_provider("application/x-ndjson", [ch](size_t, httplib::DataSink& sink) {
    std::unique_lock<std::mutex> lk(ch->mu);
    ch->cv.wait(lk, [&] { return !ch->lines.empty() || ch->finished; });
    while (!ch->lines.empty()) {
      std::string l = std::move(ch->lines.front());
      ch->lines.pop_front();
      if (!sink.write(l.data(), l.size())) return false;
    }
    if (ch->finished) sink.done();
    return true;
  });
}

std::string form(const httplib::Request& req, const std::string& key, const std::string& dflt = "") {
  return req.has_file(key) ? req.get_file_value(key).content : dflt;
}

std::shared_ptr<Job> job_from(const json& body, Server& srv, std::string& err) {
  auto j = std::make_shared<Job>();
  j->model = body.value("model", srv.models().begin()->first);
  j->prompt = body.value("prompt", "");
  if (body.contains("from")) {                      // continue a finished painting
    Painting src;
    j->edit_from = body["from"].get<std::string>();
    j->continuation = true;
    if (!srv.recall(j->edit_from, src)) { err = "unknown painting id '" + j->edit_from + "' (ids live until the server restarts)"; return nullptr; }
    j->model = src.model;                           // the prefix belongs to the model that painted it
    if (j->prompt.empty()) j->prompt = src.prompt;  // a new prompt edits the layer (e.g. other hair colour)
    j->edit_mode = body.value("edit", "redraw");
    if (j->edit_mode != "refine" && j->edit_mode != "redraw") { err = "edit must be refine or redraw"; return nullptr; }
    const json& L = body.contains("layer") ? body["layer"] : json("hair");
    j->edit_layer = -1;
    if (L.is_number_integer()) j->edit_layer = L.get<int>();
    else for (int k = 0; k < 6; ++k) if (L.get<std::string>() == kLayerNames[k]) j->edit_layer = k;
    if (j->edit_layer < 0 || j->edit_layer > 5) { err = "layer must be 0..5 or one of background, body, skin, features, hair, accessories"; return nullptr; }
  }
  if (!srv.has(j->model)) { err = "unknown model '" + j->model + "'"; return nullptr; }
  if (j->prompt.empty()) { err = "prompt is required"; return nullptr; }
  const json opt = body.contains("options") ? body["options"] : body;     // Ollama nests sampling options
  j->n = body.value("n", 1);
  if (j->n < 1 || j->n > 8) { err = "n must be 1..8"; return nullptr; }
  j->cfg_scale = opt.value("cfg_scale", 5.f);
  j->top_p = opt.value("top_p", 0.6f);
  j->seed = opt.contains("seed") ? opt["seed"].get<uint32_t>() : std::random_device{}();
  return j;
}

int serve(const std::map<std::string, std::string>& a, const std::map<std::string, std::string>& models) {
  Server srv(models, a.at("--provider"), std::stoi(a.at("--max-batch")));
  ImageStore store;
  httplib::Server http;
  const std::string web = a.at("--web");
  http.Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) { cors(res); res.status = 204; });
  http.Get("/", [&](const httplib::Request&, httplib::Response& res) {
    std::ifstream f(web + "/index.html");
    if (!f) return error(res, 404, "web UI not found in " + web);
    std::stringstream ss;
    ss << f.rdbuf();
    res.set_content(ss.str(), "text/html; charset=utf-8");
  });
  http.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
    cors(res);
    json data = json::array();
    for (auto& [name, dir] : srv.models()) data.push_back({{"id", name}, {"object", "model"}, {"created", 0}, {"owned_by", "strata"}});
    res.set_content(json{{"object", "list"}, {"data", data}}.dump(), "application/json");
  });
  http.Get("/api/tags", [&](const httplib::Request&, httplib::Response& res) {
    cors(res);
    json models_j = json::array();
    for (auto& [name, dir] : srv.models()) models_j.push_back({{"name", name}, {"model", name}, {"details", {{"path", dir}}}});
    res.set_content(json{{"models", models_j}}.dump(), "application/json");
  });
  http.Get("/api/version", [&](const httplib::Request&, httplib::Response& res) {
    cors(res);
    res.set_content(json{{"version", kVersion}, {"stats", srv.stats()}}.dump(), "application/json");
  });
  http.Get(R"(/images/(\d+)\.png)", [&](const httplib::Request& req, httplib::Response& res) {
    cors(res);
    std::string img;
    if (!store.get(req.matches[1], img)) return error(res, 404, "no such image");
    res.set_content(img, "image/png");
  });
  // OpenAI images API
  http.Post("/v1/images/generations", [&](const httplib::Request& req, httplib::Response& res) {
    cors(res);
    json body;
    try { body = json::parse(req.body); } catch (...) { return error(res, 400, "body is not JSON"); }
    std::string err;
    auto j = job_from(body, srv, err);
    if (!j) return error(res, 400, err);
    const std::string fmt = body.value("response_format", "b64_json");
    if (fmt != "b64_json" && fmt != "url") return error(res, 400, "response_format must be b64_json or url");
    try {
      auto pngs = srv.submit(j).get();
      json data = json::array();
      for (auto& p : pngs) {
        if (fmt == "url") data.push_back({{"url", "http://" + req.get_header_value("Host") + "/images/" + store.put(p) + ".png"},
                                          {"revised_prompt", j->prompt}});
        else data.push_back({{"b64_json", base64(p)}, {"revised_prompt", j->prompt}});
      }
      res.set_content(json{{"created", static_cast<long>(now_s())}, {"data", data}}.dump(), "application/json");
    } catch (const std::exception& e) {
      error(res, 500, e.what(), "server_error");
    }
  });
  // Ollama-style streaming: NDJSON frames while the picture is painted
  http.Post("/api/generate", [&](const httplib::Request& req, httplib::Response& res) {
    cors(res);
    json body;
    try { body = json::parse(req.body); } catch (...) { return error(res, 400, "body is not JSON"); }
    std::string err;
    auto j = job_from(body, srv, err);
    if (!j) return error(res, 400, err);
    const bool stream = body.value("stream", true);
    const double t0 = now_s();
    if (!stream) {
      try {
        auto pngs = srv.submit(j).get();
        json imgs = json::array();
        for (auto& p : pngs) imgs.push_back(base64(p));
        res.set_content(json{{"model", j->model}, {"done", true}, {"images", imgs},
                             {"total_duration", static_cast<long>((now_s() - t0) * 1e9)}}.dump(), "application/json");
      } catch (const std::exception& e) {
        error(res, 500, e.what(), "server_error");
      }
      return;
    }
    struct Chan {
      std::mutex mu;
      std::condition_variable cv;
      std::deque<std::string> lines;
      bool finished = false;
    };
    auto ch = std::make_shared<Chan>();
    auto push = [ch](std::string line, bool last) {
      { std::lock_guard<std::mutex> lk(ch->mu); ch->lines.push_back(std::move(line)); ch->finished = ch->finished || last; }
      ch->cv.notify_all();
    };
    const std::string model = j->model;
    const int n = j->n;
    auto finished_images = std::make_shared<std::atomic<int>>(0);
    j->progress_every = body.value("progress_every", 25);
    j->on_frame = [push, model, t0](const Frame& f) {
      json line{{"model", model}, {"index", f.index}, {"step", f.step}, {"strokes", f.strokes}, {"done", false},
              {"image_done", f.done}, {"image", base64(f.png)}, {"elapsed", now_s() - t0}};
    if (!f.id.empty()) line["id"] = f.id;
    push(line.dump() + "\n", false);
    };
    auto fut = std::make_shared<std::future<std::vector<std::string>>>(srv.submit(j));
    std::thread([fut, push, model, t0, n] {
      try {
        auto pngs = fut->get();
        push(json{{"model", model}, {"done", true}, {"images", n}, {"total_duration", static_cast<long>((now_s() - t0) * 1e9)}}.dump() + "\n", true);
      } catch (const std::exception& e) {
        push(json{{"error", e.what()}, {"done", true}}.dump() + "\n", true);
      }
    }).detach();
    res.set_chunked_content_provider("application/x-ndjson", [ch](size_t, httplib::DataSink& sink) {
      std::unique_lock<std::mutex> lk(ch->mu);
      ch->cv.wait(lk, [&] { return !ch->lines.empty() || ch->finished; });
      while (!ch->lines.empty()) {
        std::string l = std::move(ch->lines.front());
        ch->lines.pop_front();
        if (!sink.write(l.data(), l.size())) return false;
      }
      if (ch->finished) sink.done();
      return true;
    });
  });
  // ---- painting from a picture: mode "trace" replays the decomposer's strokes (high fidelity, any picture);
  // mode "sketch" gives the model the traced sketch and lets it paint the rest (prompt-guided). The result gets a
  // painting id like any other, so it can be refined / redrawn afterwards.
  const int trace_port = std::stoi(a.at("--trace-port"));
  auto picture_job = [&srv, trace_port](const std::string& image, const std::string& mode, json opts, Push push,
                                        double t0, int every) -> std::vector<std::string> {
    const std::string model = opts.value("model", srv.models().begin()->first);
    if (!srv.has(model)) throw std::runtime_error("unknown model '" + model + "'");
    if (push) push(json{{"status", "tracing"}, {"done", false}, {"elapsed", now_s() - t0}}.dump() + "\n", false);
    json tr = call_trace(trace_port, image, mode == "sketch");
    auto recs = records_from(tr["records"]);
    if (push) push(json{{"status", "traced"}, {"psnr", tr["psnr"]}, {"strokes", tr["strokes"]}, {"lines", tr["lines"]},
                        {"seconds", tr["seconds"]}, {"done", false}, {"elapsed", now_s() - t0}}.dump() + "\n", false);
    const std::string prompt = opts.value("prompt", "");
    if (mode == "trace") {
      const Protocol& P = srv.protocol(model);
      std::vector<Record> cur;
      int strokes = 0;
      for (size_t t = 0; t < recs.size(); ++t) {             // replay the strokes as the painting animation
        cur.push_back(recs[t]);
        strokes += recs[t].kind == P.kind_primitive;
        if (push && recs[t].kind == P.kind_primitive && strokes % std::max(1, every) == 0)
          push(json{{"model", model}, {"index", 0}, {"step", static_cast<int>(t)}, {"strokes", strokes}, {"done", false},
                    {"image_done", false}, {"image", base64(render_png(P, cur))}, {"elapsed", now_s() - t0}}.dump() + "\n", false);
      }
      std::string img = render_png(P, recs);
      const std::string id = srv.remember({model, prompt, recs});
      if (push) push(json{{"model", model}, {"index", 0}, {"step", static_cast<int>(recs.size())}, {"strokes", strokes},
                          {"done", false}, {"image_done", true}, {"image", base64(img)}, {"id", id}, {"elapsed", now_s() - t0}}.dump() + "\n", false);
      return {img};
    }
    if (prompt.empty()) throw std::runtime_error("mode sketch needs a prompt (what the model should paint over the sketch)");
    json body = opts;
    body["model"] = model;
    std::string err;
    auto j = job_from(body, srv, err);
    if (!j) throw std::runtime_error(err);
    j->n = 1;
    j->continuation = true;
    j->forced.assign(recs.begin() + 1, recs.end());          // after BOS: the sketch, then the model goes on
    if (push) {
      j->progress_every = every;
      j->on_frame = [push, model, t0](const Frame& f) {
        json line{{"model", model}, {"index", f.index}, {"step", f.step}, {"strokes", f.strokes}, {"done", false},
                  {"image_done", f.done}, {"image", base64(f.png)}, {"elapsed", now_s() - t0}};
        if (!f.id.empty()) line["id"] = f.id;
        push(line.dump() + "\n", false);
      };
    }
    return srv.submit(j).get();
  };
  auto read_upload = [](const httplib::Request& req, std::string& image, json& opts, std::string& err) {
    if (!req.is_multipart_form_data() || !req.has_file("image")) { err = "send multipart/form-data with an 'image' file"; return false; }
    image = req.get_file_value("image").content;
    opts = json::object();
    for (const char* k : {"model", "prompt"}) if (req.has_file(k)) opts[k] = form(req, k);
    for (const char* k : {"cfg_scale", "top_p"}) if (req.has_file(k)) opts[k] = std::stof(form(req, k));
    if (req.has_file("seed")) opts["seed"] = static_cast<uint32_t>(std::stoul(form(req, "seed")));
    return true;
  };
  http.Post("/api/trace", [&](const httplib::Request& req, httplib::Response& res) {
    cors(res);
    std::string image, err;
    json opts;
    if (!read_upload(req, image, opts, err)) return error(res, 400, err);
    const std::string mode = form(req, "mode", "trace");
    if (mode != "trace" && mode != "sketch") return error(res, 400, "mode must be trace or sketch");
    const int every = std::stoi(form(req, "progress_every", "20"));
    const double t0 = now_s();
    stream_lines(res, [=](Push push) {
      picture_job(image, mode, opts, push, t0, every);
      push(json{{"done", true}, {"total_duration", static_cast<long>((now_s() - t0) * 1e9)}}.dump() + "\n", true);
    });
  });
  // OpenAI image edits: image + prompt -> picture; mode (non-standard field): trace (default) | sketch
  http.Post("/v1/images/edits", [&](const httplib::Request& req, httplib::Response& res) {
    cors(res);
    std::string image, err;
    json opts;
    if (!read_upload(req, image, opts, err)) return error(res, 400, err);
    const std::string mode = form(req, "mode", "trace");
    if (mode != "trace" && mode != "sketch") return error(res, 400, "mode must be trace or sketch");
    const std::string fmt = form(req, "response_format", "b64_json");
    try {
      auto pngs = picture_job(image, mode, opts, nullptr, now_s(), 20);
      json data = json::array();
      for (auto& pimg : pngs) {
        if (fmt == "url") data.push_back({{"url", "http://" + req.get_header_value("Host") + "/images/" + store.put(pimg) + ".png"}});
        else data.push_back({{"b64_json", base64(pimg)}});
      }
      res.set_content(json{{"created", static_cast<long>(now_s())}, {"data", data}}.dump(), "application/json");
    } catch (const std::exception& e) {
      error(res, 500, e.what(), "server_error");
    }
  });
  const int port = std::stoi(a.at("--port"));
  std::cerr << "strata " << kVersion << " listening on http://" << a.at("--host") << ":" << port << "  (models:";
  for (auto& [name, dir] : srv.models()) std::cerr << " " << name;
  std::cerr << ")\n";
  return http.listen(a.at("--host"), port) ? 0 : 1;
}

int run(const std::map<std::string, std::string>& a, const std::map<std::string, std::string>& models, const std::string& prompt) {
  Server srv(models, a.at("--provider"), 1);
  json body{{"model", a.count("--name") ? a.at("--name") : models.begin()->first}, {"prompt", prompt}};
  if (a.count("--seed")) body["seed"] = std::stoul(a.at("--seed"));
  std::string err;
  auto j = job_from(body, srv, err);
  if (!j) { std::cerr << "error: " << err << "\n"; return 2; }
  const auto t0 = std::chrono::steady_clock::now();
  auto pngs = srv.submit(j).get();
  std::ofstream(a.at("-o"), std::ios::binary) << pngs.at(0);
  std::cerr << "wrote " << a.at("-o") << " in " << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || (std::string(argv[1]) != "serve" && std::string(argv[1]) != "run" && std::string(argv[1]) != "list")) {
    std::cerr << "usage: strata serve|run|list --model NAME=DIR [--model ...] [--port 11435] [--host 127.0.0.1]\n"
                 "                [--provider cuda|cpu|cuda-auto] [--max-batch 32] [--web runtime/web]\n"
                 "       strata run --model NAME=DIR \"prompt\" -o out.png [--seed N]\n";
    return 2;
  }
  const std::string cmd = argv[1];
  std::map<std::string, std::string> a{{"--port", "11435"}, {"--host", "127.0.0.1"}, {"--provider", "cuda"},
                                       {"--max-batch", "32"}, {"--web", "runtime/web"}, {"-o", "out.png"},
                                       {"--trace-port", "11436"}};
  std::map<std::string, std::string> models;
  std::string prompt;
  for (int i = 2; i < argc; ++i) {
    std::string k = argv[i];
    if (k == "--model" && i + 1 < argc) {
      std::string v = argv[++i];
      const auto eq = v.find('=');
      if (eq == std::string::npos) { std::cerr << "--model needs NAME=DIR\n"; return 2; }
      models[v.substr(0, eq)] = v.substr(eq + 1);
    } else if (k.rfind("-", 0) == 0 && i + 1 < argc) {
      a[k] = argv[++i];
    } else {
      prompt = k;
    }
  }
  if (models.empty()) { std::cerr << "no --model NAME=DIR given\n"; return 2; }
  try {
    if (cmd == "list") {
      for (auto& [name, dir] : models) std::cout << name << "\t" << dir << "\n";
      return 0;
    }
    if (a["--provider"] == "cuda-auto") a["--provider"] = "cuda";     // batches can grow: the GPU placement
    if (cmd == "run") {
      if (prompt.empty()) { std::cerr << "run needs a prompt\n"; return 2; }
      return run(a, models, prompt);
    }
    return serve(a, models);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
