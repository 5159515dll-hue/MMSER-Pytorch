#!/usr/bin/env bash

SCRIPT_SOURCED=0
if [ "${BASH_SOURCE[0]}" != "$0" ]; then
  SCRIPT_SOURCED=1
fi

if [ "$SCRIPT_SOURCED" != "1" ]; then
  set -Eeuo pipefail
fi

# One-click environment bootstrap for Ubuntu compute servers.
# Behavior:
# - creates a local conda env under the repo
# - reuses the server's preinstalled torch stack via a .pth bridge
# - installs the repo's Python dependencies into the conda env
# - writes a success marker so the next `source setup_ubuntu_server.sh`
#   skips installs and only activates the environment

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PYTHON_BIN="${BASE_PYTHON_BIN:-python3}"
CONDA_BIN="${CONDA_BIN:-conda}"
MINICONDA_DIR="${MINICONDA_DIR:-$REPO_ROOT/.miniconda}"
MINICONDA_INSTALLER="${MINICONDA_INSTALLER:-$REPO_ROOT/.cache/Miniconda3-latest.sh}"
USE_GLOBAL_ENV="${USE_GLOBAL_ENV:-0}"
INSTALL_LEGACY_EXTRAS="${INSTALL_LEGACY_EXTRAS:-1}"
TRY_INSTALL_DECORD="${TRY_INSTALL_DECORD:-1}"
UPGRADE_PIP_TOOLS="${UPGRADE_PIP_TOOLS:-0}"
RUN_SMOKE_TESTS="${RUN_SMOKE_TESTS:-1}"
ENABLE_PYPI_FALLBACK="${ENABLE_PYPI_FALLBACK:-1}"
PYPI_FALLBACK_INDEX_URL="${PYPI_FALLBACK_INDEX_URL:-https://pypi.org/simple}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-$REPO_ROOT/.conda-server}"
SETUP_STATE_VERSION="${SETUP_STATE_VERSION:-conda-v2}"

log() {
  printf '[setup] %s\n' "$*"
}

warn() {
  printf '[setup][warn] %s\n' "$*" >&2
}

die() {
  printf '[setup][error] %s\n' "$*" >&2
  if [ "$SCRIPT_SOURCED" = "1" ]; then
    return 1
  fi
  exit 1
}

finish() {
  local code="${1:-0}"
  if [ "$SCRIPT_SOURCED" = "1" ]; then
    return "$code"
  fi
  exit "$code"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

resolve_conda_bin() {
  local requested candidate

  requested="${CONDA_BIN:-conda}"
  if command -v "$requested" >/dev/null 2>&1; then
    command -v "$requested"
    return 0
  fi

  for candidate in \
    /opt/conda/bin/conda \
    /root/miniconda3/bin/conda \
    /usr/local/miniconda3/bin/conda \
    /usr/local/anaconda3/bin/conda
  do
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

bootstrap_local_miniconda() {
  local arch installer_url installer_path

  if [ -x "$MINICONDA_DIR/bin/conda" ]; then
    printf '%s\n' "$MINICONDA_DIR/bin/conda"
    return 0
  fi

  arch="$(uname -m)"
  case "$arch" in
    x86_64|amd64)
      installer_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
      ;;
    aarch64|arm64)
      installer_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
      ;;
    *)
      die "Unsupported architecture for automatic Miniconda bootstrap: $arch"
      ;;
  esac

  mkdir -p "$(dirname "$MINICONDA_INSTALLER")"
  installer_path="$MINICONDA_INSTALLER"
  if [ ! -s "$installer_path" ]; then
    log "Downloading local Miniconda bootstrap: $installer_url"
    if command -v curl >/dev/null 2>&1; then
      curl -fsSL "$installer_url" -o "$installer_path" || die "Failed to download Miniconda installer"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "$installer_path" "$installer_url" || die "Failed to download Miniconda installer"
    else
      die "Need curl or wget to bootstrap Miniconda"
    fi
  fi

  log "Installing local Miniconda at $MINICONDA_DIR"
  bash "$installer_path" -b -p "$MINICONDA_DIR" >/dev/null || die "Miniconda bootstrap failed"
  printf '%s\n' "$MINICONDA_DIR/bin/conda"
}

pip_install() {
  if python -m pip install "$@"; then
    return 0
  fi

  if [ "$ENABLE_PYPI_FALLBACK" != "1" ]; then
    return 1
  fi

  warn "Primary pip install failed; retrying with official PyPI: $PYPI_FALLBACK_INDEX_URL"
  python -m pip install --index-url "$PYPI_FALLBACK_INDEX_URL" "$@"
}

init_conda_shell() {
  local conda_path conda_base conda_sh conda_hook

  conda_path="$(resolve_conda_bin || true)"
  [ -n "$conda_path" ] || die "Missing required command: $CONDA_BIN"
  CONDA_BIN="$conda_path"
  export CONDA_BIN

  conda_hook="$("$conda_path" shell.bash hook 2>/dev/null || true)"
  if [ -n "$conda_hook" ]; then
    eval "$conda_hook"
    return 0
  fi

  conda_base="$("$conda_path" info --base 2>/dev/null || true)"
  conda_sh="${conda_base}/etc/profile.d/conda.sh"
  [ -f "$conda_sh" ] || die "Unable to initialize conda shell integration."
  # shellcheck disable=SC1090
  source "$conda_sh"
}

ensure_conda_env() {
  if [ -d "$CONDA_ENV_PREFIX" ] && [ ! -x "$CONDA_ENV_PREFIX/bin/python" ]; then
    warn "Found incomplete conda env at $CONDA_ENV_PREFIX; recreating it"
    rm -rf "$CONDA_ENV_PREFIX"
  fi

  if [ ! -x "$CONDA_ENV_PREFIX/bin/python" ]; then
    log "Creating conda env at $CONDA_ENV_PREFIX"
    "$CONDA_BIN" create -y -p "$CONDA_ENV_PREFIX" "python=$CONDA_PYTHON_VERSION" pip
  fi
}

activate_conda_env() {
  init_conda_shell
  conda activate "$CONDA_ENV_PREFIX"
  PYTHON_BIN="$(command -v python)"
  export PYTHON_BIN
}

collect_base_site_paths() {
  "$BASE_PYTHON_BIN" - <<'PY'
import os
import site
import sys

paths = []
for getter in (getattr(site, "getsitepackages", None), getattr(site, "getusersitepackages", None)):
    if getter is None:
        continue
    try:
        value = getter()
    except Exception:
        continue
    if isinstance(value, str):
        value = [value]
    for item in value:
        if item and os.path.isdir(item):
            paths.append(os.path.realpath(item))

for item in sys.path:
    if item and any(tag in item for tag in ("site-packages", "dist-packages")) and os.path.isdir(item):
        paths.append(os.path.realpath(item))

seen = []
for item in paths:
    if item not in seen:
        seen.append(item)

print("\n".join(seen))
PY
}

write_base_site_bridge() {
  local env_purelib base_site_paths
  env_purelib="$(python - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
  base_site_paths="$(collect_base_site_paths)"
  if [ -n "$base_site_paths" ]; then
    printf '%s\n' "$base_site_paths" > "$env_purelib/_base_site_paths.pth"
  fi
}

configure_local_model_cache() {
  local cache_root hub_cache torch_cache activate_dir deactivate_dir activate_hook deactivate_hook

  cache_root="$REPO_ROOT/.hf-cache"
  hub_cache="$cache_root/hub"
  torch_cache="$REPO_ROOT/.torch-cache"
  mkdir -p "$cache_root" "$hub_cache" "$torch_cache"

  export HF_HOME="$cache_root"
  export HF_HUB_CACHE="$hub_cache"
  export TRANSFORMERS_CACHE="$hub_cache"
  export TORCH_HOME="$torch_cache"

  if [ -z "${CONDA_PREFIX:-}" ] || [ ! -d "$CONDA_PREFIX" ]; then
    return 0
  fi

  activate_dir="$CONDA_PREFIX/etc/conda/activate.d"
  deactivate_dir="$CONDA_PREFIX/etc/conda/deactivate.d"
  activate_hook="$activate_dir/mms_cache_env.sh"
  deactivate_hook="$deactivate_dir/mms_cache_env.sh"
  mkdir -p "$activate_dir" "$deactivate_dir"

  cat > "$activate_hook" <<EOF
export _MMSER_OLD_HF_HOME="\${HF_HOME-__UNSET__}"
export _MMSER_OLD_HF_HUB_CACHE="\${HF_HUB_CACHE-__UNSET__}"
export _MMSER_OLD_TRANSFORMERS_CACHE="\${TRANSFORMERS_CACHE-__UNSET__}"
export _MMSER_OLD_TORCH_HOME="\${TORCH_HOME-__UNSET__}"
export HF_HOME="$cache_root"
export HF_HUB_CACHE="$hub_cache"
export TRANSFORMERS_CACHE="$hub_cache"
export TORCH_HOME="$torch_cache"
mkdir -p "$cache_root" "$hub_cache" "$torch_cache"
EOF

  cat > "$deactivate_hook" <<'EOF'
if [ "${_MMSER_OLD_HF_HOME-__UNSET__}" = "__UNSET__" ]; then unset HF_HOME; else export HF_HOME="$_MMSER_OLD_HF_HOME"; fi
if [ "${_MMSER_OLD_HF_HUB_CACHE-__UNSET__}" = "__UNSET__" ]; then unset HF_HUB_CACHE; else export HF_HUB_CACHE="$_MMSER_OLD_HF_HUB_CACHE"; fi
if [ "${_MMSER_OLD_TRANSFORMERS_CACHE-__UNSET__}" = "__UNSET__" ]; then unset TRANSFORMERS_CACHE; else export TRANSFORMERS_CACHE="$_MMSER_OLD_TRANSFORMERS_CACHE"; fi
if [ "${_MMSER_OLD_TORCH_HOME-__UNSET__}" = "__UNSET__" ]; then unset TORCH_HOME; else export TORCH_HOME="$_MMSER_OLD_TORCH_HOME"; fi
unset _MMSER_OLD_HF_HOME _MMSER_OLD_HF_HUB_CACHE _MMSER_OLD_TRANSFORMERS_CACHE _MMSER_OLD_TORCH_HOME
EOF
}

calc_state_hash() {
  local mainline_specs legacy_specs
  mainline_specs="$(printf '%s\n' "${MAINLINE_PACKAGE_SPECS[@]}")"
  legacy_specs="$(printf '%s\n' "${LEGACY_PACKAGE_SPECS[@]}")"
  SETUP_STATE_VERSION="$SETUP_STATE_VERSION" \
  CONDA_PYTHON_VERSION="$CONDA_PYTHON_VERSION" \
  INSTALL_LEGACY_EXTRAS="$INSTALL_LEGACY_EXTRAS" \
  TRY_INSTALL_DECORD="$TRY_INSTALL_DECORD" \
  MAINLINE_SPECS="$mainline_specs" \
  LEGACY_SPECS="$legacy_specs" \
  "$BASE_PYTHON_BIN" - <<'PY'
import hashlib
import json
import os

payload = {
    "version": os.environ["SETUP_STATE_VERSION"],
    "python": os.environ["CONDA_PYTHON_VERSION"],
    "install_legacy_extras": os.environ["INSTALL_LEGACY_EXTRAS"],
    "try_install_decord": os.environ["TRY_INSTALL_DECORD"],
    "mainline_specs": os.environ["MAINLINE_SPECS"].splitlines(),
    "legacy_specs": os.environ["LEGACY_SPECS"].splitlines(),
}
print(hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest())
PY
}

verify_python_version_match() {
  local base_ver env_ver
  base_ver="$("$BASE_PYTHON_BIN" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  env_ver="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  [ "$base_ver" = "$env_ver" ] || die "Conda env Python $env_ver does not match base Python $base_ver. Set CONDA_PYTHON_VERSION=$base_ver and recreate the env."
}

run_python_check() {
  python - "$@"
}

require_cmd "$BASE_PYTHON_BIN"

if [ "$USE_GLOBAL_ENV" != "1" ]; then
  CONDA_BIN="$(resolve_conda_bin || true)"
  if [ -z "$CONDA_BIN" ]; then
    CONDA_BIN="$(bootstrap_local_miniconda || true)"
  fi
  [ -n "$CONDA_BIN" ] || die "Missing required command: conda"
  export CONDA_BIN
fi

DEFAULT_CONDA_PYTHON_VERSION="$("$BASE_PYTHON_BIN" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
CONDA_PYTHON_VERSION="${CONDA_PYTHON_VERSION:-$DEFAULT_CONDA_PYTHON_VERSION}"
if [ "$USE_GLOBAL_ENV" = "1" ]; then
  SETUP_STATE_FILE="${SETUP_STATE_FILE:-$REPO_ROOT/.mms_setup_state_global}"
else
  SETUP_STATE_FILE="${SETUP_STATE_FILE:-$CONDA_ENV_PREFIX/.mms_setup_state}"
fi

MAINLINE_PACKAGE_SPECS=(
  "numpy>=1.26,<2.0"
  "opencv-python>=4.11,<4.12"
  "openpyxl>=3.1,<3.2"
  "pandas>=2.0,<2.3"
  "soundfile>=0.13,<0.14"
  "imageio-ffmpeg>=0.5,<0.6"
  "transformers>=4.57,<4.58"
  "huggingface_hub>=0.36,<0.37"
  "sentencepiece>=0.2,<0.3"
  "tqdm>=4.66,<5"
  "matplotlib>=3.7,<3.9"
  "Pillow>=10,<11"
  "psutil>=5.9,<6"
)

LEGACY_PACKAGE_SPECS=(
  "facenet-pytorch>=2.6,<2.7"
)

log "Repo root: $REPO_ROOT"
log "Base Python: $("$BASE_PYTHON_BIN" -c 'import sys; print(sys.executable)')"

if [ "$USE_GLOBAL_ENV" = "1" ]; then
  PYTHON_BIN="$BASE_PYTHON_BIN"
  export PYTHON_BIN
  log "Using global Python environment"
else
  log "Conda env prefix: $CONDA_ENV_PREFIX"
  ensure_conda_env
  activate_conda_env
  verify_python_version_match
  write_base_site_bridge
fi

configure_local_model_cache

log "Active Python: $(python -c 'import sys; print(sys.executable)')"
if [ -n "${CONDA_PREFIX:-}" ]; then
  log "Active conda env: $CONDA_PREFIX"
fi

run_python_check <<'PY'
import importlib.util

missing = [name for name in ("torch",) if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "Missing server torch runtime after conda activation: "
        + ", ".join(missing)
        + ". This setup expects the base Python environment to already provide torch."
    )
print("torch runtime detected")
PY

run_python_check <<'PY'
import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda version:", getattr(torch.version, "cuda", None))
PY

CURRENT_STATE_HASH="$(calc_state_hash)"
if [ -f "$SETUP_STATE_FILE" ] && [ "$(cat "$SETUP_STATE_FILE")" = "$CURRENT_STATE_HASH" ]; then
  log "Environment already bootstrapped; skipping dependency installation"
  if [ "$SCRIPT_SOURCED" != "1" ]; then
    warn "Conda activation only persists when the script is sourced. Use: source setup_ubuntu_server.sh"
  fi
  finish 0
fi

if [ "$UPGRADE_PIP_TOOLS" = "1" ]; then
  log "Upgrading pip/setuptools/wheel"
  if ! pip_install --upgrade pip setuptools wheel; then
    warn "pip/setuptools/wheel upgrade failed; continuing with the current tooling."
  fi
fi

log "Installing mainline dependencies into the conda env"
pip_install "${MAINLINE_PACKAGE_SPECS[@]}"

TORCHVISION_READY=0
if run_python_check <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torchvision") is not None else 1)
PY
then
  TORCHVISION_READY=1
  log "torchvision detected through the inherited server torch stack"
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
      pip_install --no-deps "${LEGACY_PACKAGE_SPECS[@]}"
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
    if ! pip_install decord; then
      warn "decord install failed; the project will fall back to OpenCV video decoding."
    fi
  fi
fi

log "Running import checks"
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" run_python_check <<'PY'
import cv2
import huggingface_hub
import imageio_ffmpeg
import openpyxl
import pandas
import psutil
import sentencepiece
import soundfile
from hf_compat import ensure_transformers_torch_compat

ensure_transformers_torch_compat()
import transformers
import tqdm
import matplotlib

print("core_imports_ok")
print("cv2", cv2.__version__)
print("imageio_ffmpeg", imageio_ffmpeg.__version__)
print("transformers", transformers.__version__)
print("pandas", pandas.__version__)
print("psutil", psutil.__version__)
print("huggingface_hub", huggingface_hub.__version__)
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
  python build_split_manifest.py --help >/dev/null
  python prepare_dataset_media.py --help >/dev/null
  python train.py --help >/dev/null
  python batch_inference.py --help >/dev/null
  if [ "$INSTALL_LEGACY_EXTRAS" = "1" ] && [ "$TORCHVISION_READY" = "1" ]; then
    python legacy/baseline_v1/train.py --help >/dev/null
  fi
  popd >/dev/null
fi

mkdir -p "$(dirname "$SETUP_STATE_FILE")"
printf '%s\n' "$CURRENT_STATE_HASH" > "$SETUP_STATE_FILE"

log "Setup finished"
log "Notes:"
if [ "$USE_GLOBAL_ENV" = "1" ]; then
  log "- This run used the server's global Python environment directly."
else
  log "- This script creates a local conda env at $CONDA_ENV_PREFIX."
  log "- The env reuses the server's preinstalled torch/torchvision/torchaudio via a .pth bridge instead of reinstalling the torch stack."
fi
log "- Hugging Face and Transformers caches are pinned to $REPO_ROOT/.hf-cache, and torch hub assets are pinned to $REPO_ROOT/.torch-cache."
log "- The active mainline defaults to HuggingFace audio encoders (for example WavLM). The optional audio model value wav2vec2_base still requires torchaudio."
log "- Hugging Face model weights are downloaded lazily on first real train/inference run."
if [ "$USE_GLOBAL_ENV" = "1" ]; then
  log "- Re-enter the global-mode setup with: USE_GLOBAL_ENV=1 INSTALL_LEGACY_EXTRAS=0 source setup_ubuntu_server.sh"
else
  log "- If you only care about the current mainline, you can skip legacy extras with: INSTALL_LEGACY_EXTRAS=0 source setup_ubuntu_server.sh"
fi
log "- Running this script again will skip dependency installation once the setup marker matches."
log "- For persistent activation in your current shell, use: source setup_ubuntu_server.sh"

if [ "$SCRIPT_SOURCED" != "1" ]; then
  warn "Conda activation only persists when the script is sourced. Use: source setup_ubuntu_server.sh"
fi

finish 0
