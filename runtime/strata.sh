#!/bin/bash
# Run strata (the server) with the CUDA libraries of the ML environment on the loader path (the Python
# onnxruntime-gpu wheel preloads them itself; a C++ binary needs them in LD_LIBRARY_PATH).
VENV="${LPP_VENV:-$HOME/AI infra/.venv}"
SP=$("$VENV/bin/python" -c "import site; print(site.getsitepackages()[0])")
LIBS=$(find "$SP/nvidia" -maxdepth 3 -type d -name lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${LIBS}${LD_LIBRARY_PATH:-}"
exec "$(dirname "$0")/build/strata" "$@"
