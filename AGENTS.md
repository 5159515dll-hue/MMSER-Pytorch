# Repository Guidelines

## Project Structure & Module Organization
The repository root is the current mainline implementation for multimodal emotion analysis. Core files such as `train_motion_audio.py`, `predecode_motion_audio.py`, `batch_inference_motion_prosody.py`, `models.py`, `data.py`, `manifest_utils.py`, and `metrics_utils.py` live directly at the top level, while `train.py`, `batch_inference.py`, `predecode_dataset.py`, `build_split_manifest.py`, and `validate_cached_shards.py` are the stable root entrypoints. `motion_prosody/` and `experiments/motion_prosody/` now exist only as compatibility layers for older imports and commands. `legacy/baseline_v1/` contains the archived first-generation baseline with its own `src/` tree and legacy training/inference scripts. `audits/` contains dataset integrity checks. Treat `databases/` and `outputs/` as data/artifact directories, not source code.

## Build, Test, and Development Commands
Run commands from the repository root.

- `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` installs the pinned Python stack.
- `python build_split_manifest.py --data-root databases --xlsx databases/video_databases.xlsx` builds the manifest-driven train/val split for the mainline pipeline.
- `python predecode_dataset.py --data-root databases --xlsx databases/video_databases.xlsx` caches motion/audio/text tensors under `outputs/motion_prosody/`.
- `python train.py --cached-dataset <path>` trains the mainline root-level model.
- `python batch_inference.py --xlsx databases/video_databases.xlsx --checkpoint outputs/motion_prosody/checkpoints/best.pt` evaluates the mainline pipeline on spreadsheet-backed samples.
- `python validate_cached_shards.py --cached-dataset <dir>` validates cached `.pt` shards before training or inference.
- `python legacy/baseline_v1/train.py --data-root databases --output-dir outputs` runs the archived V1 baseline when a historical comparison is needed.

## Coding Style & Naming Conventions
Use 4-space indentation, `snake_case` for functions/files, `PascalCase` for classes, and `ALL_CAPS` for constants such as `EMOTIONS`. Match the existing Python style: `Path` over raw strings for filesystem paths, `parse_args()` plus `main()` for CLI scripts, and type hints on new public helpers. Keep comments short and practical; preserve Chinese dataset column names exactly where they are part of the data contract.

## Testing Guidelines
There is no checked-in `pytest` suite or coverage gate yet. Validate mainline changes with `build_split_manifest.py`, `batch_inference.py`, `audits/data_leak_check.py`, and `validate_cached_shards.py --cached-dataset <dir>`. Validate archived-baseline changes only inside `legacy/baseline_v1/`. When adding automated tests, prefer a new `tests/` package mirroring the package under test and name files `test_*.py`.

## Commit & Pull Request Guidelines
Use short imperative subjects with an optional scope, for example `train: fix AMP fallback`. PRs should state the dataset/XLSX assumptions, list the validation commands you ran, summarize metric changes, and avoid attaching large generated artifacts such as checkpoints, cached shards, or media files unless explicitly needed.

## Data & Configuration Notes
Device selection defaults vary by pipeline: the archived baseline still prefers `mps`, then `cuda`, then `cpu`, while the root-level mainline is more CUDA-oriented. If you rely on cache or decoder overrides, document environment variables such as `TORCH_HOME`, `HF_HOME`, `FORCE_FACE_CPU`, or `USE_DECORD` in the PR description.
