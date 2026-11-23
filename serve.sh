#!/bin/bash
# One command for the local Strata painting server (OpenAI-compatible image API + web UI).
#
#   ./serve.sh            start (or restart) it and open the web UI      -> http://localhost:11435
#   ./serve.sh stop       stop it
#   ./serve.sh status     is it running, which models, how many images so far
#   ./serve.sh logs       follow its log (Ctrl-C to leave)
#
# On first use it does what is missing, in order: builds the C++ runtime (runtime/build), exports the final model to
# ONNX with the fused sampler, exports the text encoder + tokenizer, makes the FP16 variant. The server runs as the
# systemd --user unit strata-serve, so it keeps running after the terminal closes. Environment knobs:
#   STRATA_PORT (11435)  STRATA_HOST (127.0.0.1; 0.0.0.0 to serve the LAN)  STRATA_PROVIDER (cuda | cpu)
#   STRATA_CKPT (runs/v3_formal/B_ft/model_step012000.pt)  STRATA_NO_BROWSER=1 (do not open the browser)
set -euo pipefail
cd "$(dirname "$0")"
PORT=${STRATA_PORT:-11435}; HOST=${STRATA_HOST:-127.0.0.1}; PROVIDER=${STRATA_PROVIDER:-cuda}
CKPT=${STRATA_CKPT:-runs/v3_formal/B_ft/model_step012000.pt}
PY="${LPP_VENV:-$HOME/AI infra/.venv}/bin/python"
TE=data/models/chinese-clip-vit-base-patch16
M=out/onnx/v3_final_fused; M16=${M}_fp16
UNIT=strata-serve
URL="http://localhost:$PORT"

say() { printf '\033[1m[strata]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[strata] %s\033[0m\n' "$*" >&2; exit 1; }

case "${1:-up}" in
  stop)
    systemctl --user stop $UNIT 2>/dev/null && say "stopped" || say "was not running"
    exit 0 ;;
  logs)
    exec journalctl --user -u $UNIT -f -o cat ;;
  status)
    if systemctl --user is-active -q $UNIT; then
      say "running at $URL"
      curl -s "$URL/v1/models" | "$PY" -c "import json,sys; print('  models:', ', '.join(m['id'] for m in json.load(sys.stdin)['data']))" || true
      curl -s "$URL/api/version" | "$PY" -c "import json,sys; s=json.load(sys.stdin)['stats']; print(f\"  {s['images']} images in {s['batches']} batches\")" || true
    else
      say "not running (./serve.sh to start)"
    fi
    exit 0 ;;
  up|start|restart) ;;
  *) die "usage: ./serve.sh [up|stop|status|logs]" ;;
esac

# 1. the C++ runtime
if [ ! -x runtime/build/strata ] || [ runtime/src/server.cpp -nt runtime/build/strata ]; then
  say "building the C++ runtime ..."
  cmake -S runtime -B runtime/build -DCMAKE_BUILD_TYPE=Release > /dev/null
  cmake --build runtime/build -j "$(nproc)" --target strata > /dev/null || die "build failed (cmake --build runtime/build --target strata)"
fi

# 2. the models (first use only; a few minutes)
[ -x "$PY" ] || die "Python environment not found at $PY (set LPP_VENV)"
if [ ! -f $M/manifest.json ] || ! grep -q decode_sample $M/manifest.json; then
  [ -f "$CKPT" ] || die "checkpoint $CKPT not found (set STRATA_CKPT)"
  say "exporting the model to ONNX (fused sampler) ..."
  "$PY" tools/export_onnx.py --ckpt "$CKPT" --out $M > out/serve_export.log 2>&1 || die "export failed: out/serve_export.log"
fi
if ! grep -q '"text_encoder"' $M/manifest.json; then
  say "exporting the text encoder and tokenizer ..."
  "$PY" tools/export_text_encoder.py --text-encoder $TE --onnx $M > out/serve_text.log 2>&1 || die "text encoder export failed: out/serve_text.log"
fi
MODELS=(--model "strata-v3=$M")
if [ "$PROVIDER" != cpu ]; then
  if [ ! -f $M16/manifest.json ]; then
    say "making the FP16 variant ..."
    "$PY" tools/quantize_onnx.py --src $M --out $M16 --mode fp16 --skip canvas_enc embed_text decode_sample \
      > out/serve_fp16.log 2>&1 || die "FP16 conversion failed: out/serve_fp16.log"
  fi
  MODELS+=(--model "strata-v3:fp16=$M16")
fi

# 3. (re)start the server as a user service
if systemctl --user is-active -q $UNIT; then
  say "restarting ..."
  systemctl --user stop $UNIT
fi
systemctl --user reset-failed $UNIT 2>/dev/null || true
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  die "port $PORT is taken by another program (STRATA_PORT=... ./serve.sh to use another)"
fi
systemd-run --user --unit $UNIT --same-dir --quiet \
  runtime/strata.sh serve "${MODELS[@]}" --port "$PORT" --host "$HOST" --provider "$PROVIDER" --web runtime/web
for i in $(seq 1 60); do
  curl -sf "$URL/v1/models" > /dev/null 2>&1 && break
  systemctl --user is-active -q $UNIT || die "the server exited: ./serve.sh logs"
  sleep 0.5
done
curl -sf "$URL/v1/models" > /dev/null || die "the server does not answer: ./serve.sh logs"

say "ready: $URL"
say "OpenAI API base: $URL/v1   (e.g. Open WebUI image engine: OpenAI, key: anything)"
say "stop with ./serve.sh stop, logs with ./serve.sh logs"
if [ -z "${STRATA_NO_BROWSER:-}" ] && command -v xdg-open > /dev/null; then xdg-open "$URL" > /dev/null 2>&1 & fi
