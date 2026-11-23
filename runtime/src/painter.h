// The decode loop (tools/ort_generate.py, lpp.sample.generate): grammar-masked event, attributes
// one by one, classifier-free guidance, KV cache owned here, canvas checkpoints re-encoded.
#pragma once
#include <array>
#include <functional>
#include <set>
#include <vector>

#include "protocol.h"
#include "sampler.h"
#include "session.h"

namespace lpp {

struct Record {
  int64_t kind, event, layer;
  std::vector<int64_t> attrs;
};

struct GenerateConfig {
  SampleConfig sample;
  int max_records = 0;                 // 0: context minus text
  std::set<int> require_before_eos;    // e.g. {0, 2}: background and skin before EOS
  // streaming (strata serve): called every progress_every decode steps and once at the end, with every sample's
  // records so far and which samples have finished
  std::function<void(int step, const std::vector<std::vector<Record>>& out, const std::vector<bool>& done)> on_progress;
  int progress_every = 25;
  // continuation (lpp.sample.generate's `forced`): per sample, records after BOS taken instead of sampled, through
  // the same grammar, KV cache and canvas; decoding continues where a prefix ends. A prefix may end in a REFINE_k /
  // REDRAW_k marker (refine or repaint layer k of a finished painting). Empty: no prefixes.
  std::vector<std::vector<Record>> forced;
};

// uniform(step, field, b): field 0 = event, 1 + i = attribute i
using Uniforms = std::function<float(int step, int field, int b)>;

class Painter {
 public:
  Painter(const std::string& export_dir, const std::string& provider);
  const Protocol& protocol() const { return proto_; }
  // text [B, Lt, text_dim] (row-major), text_valid [B, Lt]; Lt may be 0
  std::vector<std::vector<Record>> generate(int B, const GenerateConfig& cfg, const std::vector<float>& text,
                                            const std::vector<uint8_t>& text_valid, int Lt, const Uniforms& u);

 private:
  Protocol proto_;
  Graphs g_;
};

}  // namespace lpp
