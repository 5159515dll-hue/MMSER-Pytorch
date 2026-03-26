# Repository Guidelines

## Project Structure & Module Organization
The repository root is the current mainline implementation for multimodal emotion analysis. The active train/infer runtime is manifest-driven `gpu_stream`, implemented through `gpu_stream_train.py`, `gpu_stream_infer.py`, `models.py`, `data.py`, `gpu_stream.py`, `manifest_utils.py`, and `metrics_utils.py`. `train.py`, `batch_inference.py`, `build_split_manifest.py`, and `prepare_dataset_media.py` are the stable root entrypoints. `train_motion_audio.py` and `batch_inference_motion_prosody.py` now exist only as thin compatibility wrappers. `motion_prosody/` is a minimal import shim kept for `legacy/baseline_v1/`; `experiments/motion_prosody/` has been removed. `legacy/baseline_v1/` contains the archived first-generation baseline with its own `src/` tree and legacy training/inference scripts. `audits/` contains dataset integrity checks. Treat `databases/` and `outputs/` as data/artifact directories, not source code.

## Build, Test, and Development Commands
Run commands from the repository root.

- `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` installs the pinned Python stack.
- `python build_split_manifest.py --dataset-kind meld --data-root <media_root> --metadata-root <metadata_root> --output <manifest.json>` builds the manifest-driven split for the mainline pipeline.
- `python prepare_dataset_media.py --dataset-kind meld --split-manifest <manifest.json> --subset all` prepares MELD audio sidecars.
- `python train.py --split-manifest <manifest.json>` trains the mainline root-level model.
- `python batch_inference.py --split-manifest <manifest.json> --subset val --checkpoint outputs/motion_prosody/checkpoints/best.pt` evaluates the mainline pipeline on a manifest subset.
- `python legacy/baseline_v1/train.py --data-root databases --output-dir outputs` runs the archived V1 baseline when a historical comparison is needed.

## Coding Style & Naming Conventions
Use 4-space indentation, `snake_case` for functions/files, `PascalCase` for classes, and `ALL_CAPS` for constants such as `EMOTIONS`. Match the existing Python style: `Path` over raw strings for filesystem paths, `parse_args()` plus `main()` for CLI scripts, and type hints on new public helpers. Keep comments short and practical; preserve Chinese dataset column names exactly where they are part of the data contract.

## Testing Guidelines
There is no checked-in `pytest` suite or coverage gate yet. Validate mainline changes with `build_split_manifest.py`, `prepare_dataset_media.py`, `train.py --help`, `batch_inference.py --help`, and targeted manifest-driven smoke runs. Validate archived-baseline changes only inside `legacy/baseline_v1/`. When adding automated tests, prefer a new `tests/` package mirroring the package under test and name files `test_*.py`.

## Commit & Pull Request Guidelines
Use short imperative subjects with an optional scope, for example `train: fix AMP fallback`. PRs should state the dataset/XLSX assumptions, list the validation commands you ran, summarize metric changes, and avoid attaching large generated artifacts such as checkpoints, cached shards, or media files unless explicitly needed.

## Data & Configuration Notes
Device selection defaults vary by pipeline: the archived baseline still prefers `mps`, then `cuda`, then `cpu`, while the root-level mainline is more CUDA-oriented. If you rely on decoder or model-download overrides, document environment variables such as `TORCH_HOME`, `HF_HOME`, `FORCE_FACE_CPU`, or `USE_DECORD` in the PR description.
