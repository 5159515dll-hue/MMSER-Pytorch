#!/usr/bin/env bash
set -Eeuo pipefail

# One-click environment bootstrap for a fresh Ubuntu compute server.
# Scope:
# - installs project Python dependencies that are not part of the server's prebuilt PyTorch stack
# - verifies the current mainline entrypoints
# - optionally installs archived baseline extras
#
# Assumptions:
# - python and torch are already available in the active environment
# - if you need the archived baseline, torchvision should already match the server's torch build

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_LEGACY_EXTRAS="${INSTALL_LEGACY_EXTRAS:-1}"
TRY_INSTALL_DECORD="${TRY_INSTALL_DECORD:-1}"
UPGRADE_PIP_TOOLS="${UPGRADE_PIP_TOOLS:-0}"
RUN_SMOKE_TESTS="${RUN_SMOKE_TESTS:-1}"
USE_VENV="${USE_VENV:-1}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv-server}"
BOOTSTRAPPED_VENV="${BOOTSTRAPPED_VENV:-0}"

log() {
  printf '[setup] %s\n' "$*"
}

warn() {
  printf '[setup][warn] %s\n' "$*" >&2
}

die() {
  printf '[setup][error] %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

run_python_check() {
  "$PYTHON_BIN" - "$@"
}

require_cmd "$PYTHON_BIN"

if [ "$USE_VENV" = "1" ] && [ "$BOOTSTRAPPED_VENV" != "1" ]; then
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    log "Creating isolated virtualenv at $VENV_DIR (inherits system PyTorch via --system-site-packages)"
    "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
  fi
  log "Re-entering setup inside virtualenv: $VENV_DIR"
  exec env \
    BOOTSTRAPPED_VENV=1 \
    USE_VENV=1 \
    VENV_DIR="$VENV_DIR" \
    INSTALL_LEGACY_EXTRAS="$INSTALL_LEGACY_EXTRAS" \
    TRY_INSTALL_DECORD="$TRY_INSTALL_DECORD" \
    UPGRADE_PIP_TOOLS="$UPGRADE_PIP_TOOLS" \
    RUN_SMOKE_TESTS="$RUN_SMOKE_TESTS" \
    PYTHON_BIN="$VENV_DIR/bin/python" \
    bash "$0" "$@"
fi

log "Repo root: $REPO_ROOT"
log "Python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
if [ -n "${VIRTUAL_ENV:-}" ]; then
  log "Virtualenv: $VIRTUAL_ENV"
fi

run_python_check <<'PY'
import importlib.util
import sys

missing = [name for name in ("torch",) if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "Missing preinstalled PyTorch runtime: "
        + ", ".join(missing)
        + ". This script intentionally does not install torch itself."
    )
print("torch runtime detected")
PY

run_python_check <<'PY'
import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda version:", getattr(torch.version, "cuda", None))
PY

if [ "$UPGRADE_PIP_TOOLS" = "1" ]; then
  log "Upgrading pip/setuptools/wheel"
  if ! "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel; then
    warn "pip/setuptools/wheel upgrade failed; continuing with the server's existing tooling."
  fi
fi

# These pins are chosen to stay compatible with the repo's verified local environment
# while avoiding torch/torchvision/torchaudio upgrades on the server.
CORE_PACKAGE_SPECS=(
  "numpy:numpy>=1.26,<2.0"
  "cv2:opencv-python>=4.11,<4.12"
  "openpyxl:openpyxl>=3.1,<3.2"
  "pandas:pandas>=2.0,<2.3"
  "soundfile:soundfile>=0.13,<0.14"
  "transformers:transformers>=4.57,<4.58"
  "sentencepiece:sentencepiece>=0.2,<0.3"
  "tqdm:tqdm>=4.66,<5"
  "matplotlib:matplotlib>=3.7,<3.9"
  "PIL:Pillow>=10,<11"
)

MISSING_CORE_PACKAGES=()
for entry in "${CORE_PACKAGE_SPECS[@]}"; do
  module_name="${entry%%:*}"
  pip_spec="${entry#*:}"
  if run_python_check <<PY
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("$module_name") is not None else 1)
PY
  then
    log "Dependency already available: $module_name"
  else
    MISSING_CORE_PACKAGES+=("$pip_spec")
  fi
done

if [ "${#MISSING_CORE_PACKAGES[@]}" -gt 0 ]; then
  log "Installing missing mainline + audit dependencies"
  "$PYTHON_BIN" -m pip install "${MISSING_CORE_PACKAGES[@]}"
else
  log "All mainline + audit dependencies are already available"
fi

TORCHVISION_READY=0
if run_python_check <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torchvision") is not None else 1)
PY
then
  TORCHVISION_READY=1
  log "torchvision detected in the server image"
else
  warn "torchvision not found. Current mainline can still run, but legacy/baseline_v1 extras will be skipped."
fi

if [ "$INSTALL_LEGACY_EXTRAS" = "1" ]; then
  if [ "$TORCHVISION_READY" = "1" ]; then
    if run_python_check <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("facenet_pytorch") is not None else 1)
PY
    then
      log "Archived baseline extra already available: facenet_pytorch"
    else
      log "Installing archived baseline extras"
      # Avoid pulling a different torch stack from pip.
      "$PYTHON_BIN" -m pip install --no-deps "facenet-pytorch>=2.6,<2.7"
    fi
  else
    warn "Skipped facenet-pytorch because torchvision is not available in the active torch environment."
  fi
fi

if [ "$TRY_INSTALL_DECORD" = "1" ]; then
  if run_python_check <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("decord") is not None else 1)
PY
  then
    log "Optional decord accelerator already available"
  else
    log "Trying to install optional decord accelerator"
    if ! "$PYTHON_BIN" -m pip install decord; then
      warn "decord install failed; the project will fall back to OpenCV video decoding."
    fi
  fi
fi

log "Running import checks"
run_python_check <<'PY'
import cv2
import openpyxl
import pandas
import sentencepiece
import soundfile
import transformers
import tqdm
import matplotlib

print("core_imports_ok")
print("cv2", cv2.__version__)
print("transformers", transformers.__version__)
print("pandas", pandas.__version__)
PY

if [ "$INSTALL_LEGACY_EXTRAS" = "1" ] && [ "$TORCHVISION_READY" = "1" ]; then
  if run_python_check <<'PY'
from facenet_pytorch import MTCNN
print("facenet_pytorch_ok", MTCNN.__name__)
PY
  then
    log "facenet-pytorch import check passed"
  else
    warn "facenet-pytorch import check failed; legacy baseline may need a stricter torch/torchvision pairing."
  fi
fi

if [ "$RUN_SMOKE_TESTS" = "1" ]; then
  log "Running CLI smoke tests"
  pushd "$REPO_ROOT" >/dev/null
  "$PYTHON_BIN" build_split_manifest.py --help >/dev/null
  "$PYTHON_BIN" predecode_dataset.py --help >/dev/null
  "$PYTHON_BIN" train.py --help >/dev/null
  "$PYTHON_BIN" batch_inference.py --help >/dev/null
  "$PYTHON_BIN" validate_cached_shards.py --help >/dev/null

  if [ "$INSTALL_LEGACY_EXTRAS" = "1" ] && [ "$TORCHVISION_READY" = "1" ]; then
    "$PYTHON_BIN" legacy/baseline_v1/train.py --help >/dev/null
  fi
  popd >/dev/null
fi

log "Setup finished"
log "Notes:"
log "- By default this script creates .venv-server with --system-site-packages, so server-provided torch remains visible while pip installs stay isolated."
log "- This script does not install torch/torchaudio/torchvision."
log "- Hugging Face model weights are downloaded lazily on first real train/inference run."
log "- If you only care about the current mainline, you can skip legacy extras with: INSTALL_LEGACY_EXTRAS=0 bash setup_ubuntu_server.sh"
log "- pip/setuptools/wheel upgrade is disabled by default; enable it explicitly with: UPGRADE_PIP_TOOLS=1 bash setup_ubuntu_server.sh"
