# Repository Guidelines

## Project Structure & Module Organization
`src/` holds reusable code: `src/data/` builds multimodal datasets, `src/models/` defines encoders and fusion models, and `src/utils/` contains device helpers. Root-level scripts such as `train.py`, `inference.py`, `batch_inference.py`, and `predecode_dataset.py` are the main entry points for the baseline pipeline. `experiments/motion_prosody/` is an isolated research track with its own training, predecode, and validation scripts. `audits/` contains dataset integrity checks. Treat `databases/` and `outputs/` as data/artifact directories, not source code.

## Build, Test, and Development Commands
Run commands from the repository root.

- `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` installs the pinned Python stack.
- `python train.py --data-root databases --output-dir outputs` trains the baseline model.
- `python inference.py --video-path databases/happy/1.mp4 --audio-path databases/happy/happy_audio/1.wav --checkpoint outputs/checkpoints/best.pt` runs single-sample inference.
- `python predecode_dataset.py --data-root databases --xlsx databases/video_databases.xlsx` caches decoded tensors under `outputs/predecoded/`.
- `python batch_inference.py --xlsx databases/video_databases.xlsx --checkpoint outputs/checkpoints/best.pt` evaluates a spreadsheet-backed batch.
- `python experiments/motion_prosody/train_motion_audio.py --cached-dataset <path>` trains the motion+prosody variant.

## Coding Style & Naming Conventions
Use 4-space indentation, `snake_case` for functions/files, `PascalCase` for classes, and `ALL_CAPS` for constants such as `EMOTIONS`. Match the existing Python style: `Path` over raw strings for filesystem paths, `parse_args()` plus `main()` for CLI scripts, and type hints on new public helpers. Keep comments short and practical; preserve Chinese dataset column names exactly where they are part of the data contract.

## Testing Guidelines
There is no checked-in `pytest` suite or coverage gate yet. Validate changes with runnable scripts: `batch_inference.py` for predictions, `audits/data_leak_check.py` for alignment/leakage checks, and `experiments/motion_prosody/validate_cached_shards.py --cached-dataset <dir>` for cached `.pt` shards. When adding automated tests, prefer a new `tests/` package mirroring `src/` and name files `test_*.py`.

## Commit & Pull Request Guidelines
This workspace does not include `.git`, so historic commit conventions are not inspectable here. Use short imperative subjects with an optional scope, for example `train: fix AMP fallback`. PRs should state the dataset/XLSX assumptions, list the validation commands you ran, summarize metric changes, and avoid attaching large generated artifacts such as checkpoints, cached shards, or media files unless explicitly needed.

## Data & Configuration Notes
Device selection defaults to `mps`, then `cuda`, then `cpu`; the `motion_prosody` experiment is more CUDA-oriented. If you rely on cache or decoder overrides, document environment variables such as `TORCH_HOME`, `HF_HOME`, `FORCE_FACE_CPU`, or `USE_DECORD` in the PR description.
