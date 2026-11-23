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
  std::string png;
};
struct Job {
  std::string model, prompt;
  int n = 1;
  float cfg_scale = 5.f, top_p = 0.6f;
  uint32_t seed = 0;
  int progress_every = 0;               // 0: no frames
  std::function<void(const Frame&)> on_frame;
  std::promise<std::vector<std::string>> result;   // final PNGs
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
      if (j->model == first->model && j->cfg_scale == first->cfg_scale && j->top_p == first->top_p &&
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
      if (j->on_frame && j->progress_every > 0) {
        Frame f;
        f.index = owner[b].second, f.step = static_cast<int>(seqs[b].size()), f.done = true, f.png = img;
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

std::shared_ptr<Job> job_from(const json& body, const Server& srv, std::string& err) {
  auto j = std::make_shared<Job>();
  j->model = body.value("model", srv.models().begin()->first);
  if (!srv.has(j->model)) { err = "unknown model '" + j->model + "'"; return nullptr; }
  j->prompt = body.value("prompt", "");
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
      push(json{{"model", model}, {"index", f.index}, {"step", f.step}, {"strokes", f.strokes}, {"done", false},
                {"image_done", f.done}, {"image", base64(f.png)}, {"elapsed", now_s() - t0}}.dump() + "\n", false);
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
                                       {"--max-batch", "32"}, {"--web", "runtime/web"}, {"-o", "out.png"}};
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
