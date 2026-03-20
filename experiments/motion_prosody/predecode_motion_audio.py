from __future__ import annotations

import argparse
import sys
import math
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import multiprocessing as mp
from functools import partial
from typing import Any
from pathlib import Path
from typing import Dict, List

import torch


# Some torchaudio builds (notably on ARM/RPi) route torchaudio.load through torchcodec.
# If torchcodec is missing or a placeholder wheel without decoders, torchaudio.load will fail.
# In auto mode we detect this once and stop trying torchaudio for the rest of the run.
_TORCHAUDIO_DISABLED_REASON: str | None = None
_WAVLM_MODEL_CACHE: dict[str, Any] = {}


def _ensure_project_root_on_path() -> None:
    # When running as: python3 experiments/motion_prosody/predecode_motion_audio.py
    # sys.path[0] is this folder, so absolute imports like "experiments.*" won't resolve.
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


_ensure_project_root_on_path()

from experiments.motion_prosody.prosody import ProsodyConfig, extract_prosody_features
from experiments.motion_prosody.audio_aug import normalize_wav
from experiments.motion_prosody.video_motion import MotionConfig, RgbConfig, compute_face_flow_tensor, compute_face_rgb_tensor
from experiments.motion_prosody.manifest_utils import (
    detect_text_cue_flags as _detect_text_cue_flags,
    infer_speaker_id as _infer_speaker_id,
    normalize_seq as _shared_normalize_seq,
    read_xlsx_rows as _shared_read_xlsx_rows,
    resolve_label_en as _shared_resolve_label_en,
    resolve_paths_for_seq as _shared_resolve_paths_for_seq,
)


CN_TO_EN = {
    "愤怒": "angry",
    "厌恶": "disgusted",
    "恐惧": "fear",
    "高兴": "happy",
    "快乐": "happy",
    "开心": "happy",
    "中性": "neutral",
    "悲伤": "sad",
    "惊讶": "surprise",
}

EMOTIONS = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprise"]


def _atomic_torch_save(obj: Any, dst: Path) -> None:
    """Atomically save a torch object to dst.

    Prevents producing truncated/corrupt .pt shards when the process is interrupted.
    """

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd: int | None = None
    tmp_path: Path | None = None
    try:
        tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=str(dst.parent))
        tmp_path = Path(tmp_name)
        with os.fdopen(tmp_fd, "wb") as f:
            tmp_fd = None
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(dst))
        tmp_path = None
    finally:
        if tmp_fd is not None:
            try:
                os.close(tmp_fd)
            except Exception:
                pass
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _get_wavlm_model(model_name: str) -> Any:
    key = str(model_name)
    if key in _WAVLM_MODEL_CACHE:
        return _WAVLM_MODEL_CACHE[key]
    try:
        from transformers import AutoModel
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "transformers is required for WavLM embedding in predecode. Install with: pip install transformers"
        ) from e
    model = AutoModel.from_pretrained(key)
    model.eval()
    _WAVLM_MODEL_CACHE[key] = model
    return model


def _extract_wavlm_embedding(wav: torch.Tensor, model_name: str) -> torch.Tensor:
    """Extract WavLM embedding (mean+std pooling). Returns float16 tensor."""
    model = _get_wavlm_model(model_name)
    x = wav.to(dtype=torch.float32).unsqueeze(0)
    lengths = torch.tensor([x.shape[1]], dtype=torch.long)

    with torch.no_grad():
        out = model(input_values=x)
        hidden = out.last_hidden_state
        if hidden is None:
            raise RuntimeError("wavlm_output_missing")
        mean = hidden.mean(dim=1)
        std = hidden.std(dim=1, unbiased=False)
        emb = torch.cat([mean, std], dim=1).squeeze(0)
    return emb.to(torch.float16)


def _read_xlsx_rows(path: Path) -> list[dict[str, Any]]:
    return _shared_read_xlsx_rows(path)


def _normalize_seq(raw: Any) -> str:
    return _shared_normalize_seq(raw)


def parse_args():
    p = argparse.ArgumentParser(description="Predecode motion(flow) / rgb + audio + prosody into shards")
    p.add_argument("--data-root", type=Path, default=Path("databases"))
    p.add_argument("--xlsx", type=Path, required=True)
    p.add_argument("--num-frames", type=int, default=64)
    p.add_argument("--flow-size", type=int, default=112)
    p.add_argument("--rgb-size", type=int, default=224)
    p.add_argument(
        "--video-repr",
        type=str,
        default="both",
        choices=["flow", "rgb", "both"],
        help="Video representation to cache: optical flow, RGB clip, or both",
    )
    p.add_argument("--sample-rate", type=int, default=24000)
    p.add_argument(
        "--audio-repr",
        type=str,
        default="raw",
        choices=["raw", "wavlm", "both"],
        help="Audio representation to cache: raw wav, wavlm embedding, or both",
    )
    p.add_argument(
        "--audio-model",
        type=str,
        default="microsoft/wavlm-large",
        help="HF model name for WavLM embedding (used when --audio-repr includes wavlm)",
    )
    p.add_argument(
        "--audio-backend",
        type=str,
        default="auto",
        choices=["auto", "torchaudio", "soundfile"],
        help=(
            "Audio loading backend. auto=try torchaudio then fallback to soundfile; "
            "torchaudio=only torchaudio (fail if unavailable); soundfile=only soundfile"
        ),
    )
    p.add_argument("--shard-size", type=int, default=100)
    p.add_argument("--output", type=Path, required=True, help="Output directory for shards")
    p.add_argument("--max-audio-sec", type=float, default=6.0)
    p.add_argument("--num-workers", type=int, default=0, help="Use N worker processes for predecode (0=single-process)")
    p.add_argument(
        "--mp-start-method",
        type=str,
        default="spawn",
        choices=["spawn", "forkserver", "fork"],
        help="Multiprocessing start method (Linux: spawn/forkserver recommended for OpenCV/torchaudio)",
    )
    p.add_argument(
        "--pool-retry",
        type=int,
        default=100,
        help="Retry multiprocessing pool on worker crash. 0=disable retries (fallback immediately).",
    )
    p.add_argument(
        "--pool-retry-delay",
        type=float,
        default=3.0,
        help="Seconds to wait before retrying the multiprocessing pool after a crash.",
    )
    p.add_argument("--mp-chunksize", type=int, default=1, help="Chunk size for multiprocessing map")
    p.add_argument(
        "--prosody-no-pitch",
        action="store_true",
        help="Disable pitch extraction (energy-only prosody; faster, less informative)",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print per-sample video/audio/text every N samples (1=every sample, 0=disable)",
    )
    p.add_argument(
        "--print-result",
        type=str,
        default="fail",
        choices=["none", "fail", "ok", "all"],
        help="Print per-sample result status: none/fail/ok/all",
    )
    p.add_argument(
        "--result-every",
        type=int,
        default=1,
        help="Print result status every N samples (1=every sample, 0=disable)",
    )
    p.add_argument(
        "--show-first-errors",
        type=int,
        default=10,
        help="Print the first N runtime SKIP reasons immediately (helps debug why ok=0)",
    )
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    return p.parse_args()


def _load_audio_torchaudio(path: Path, sample_rate: int) -> tuple[torch.Tensor, int]:
    import torchaudio

    wav, sr = torchaudio.load(str(path))  # (C,L)
    wav = wav.mean(dim=0)  # mono
    if int(sr) != int(sample_rate):
        wav = torchaudio.functional.resample(wav, int(sr), int(sample_rate))
        sr = int(sample_rate)
    return wav, int(sr)


def _load_audio_soundfile(path: Path, sample_rate: int) -> tuple[torch.Tensor, int]:
    import numpy as np
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (L,C)
    x = x.mean(axis=1)  # mono
    if int(sr) != int(sample_rate):
        # Simple linear resample (fast, dependency-free; ok for preprocessing).
        src_len = int(x.shape[0])
        tgt_len = max(1, int(round(src_len * float(sample_rate) / float(sr))))
        if src_len > 1 and tgt_len > 1:
            t_src = np.linspace(0.0, 1.0, num=src_len, endpoint=False)
            t_tgt = np.linspace(0.0, 1.0, num=tgt_len, endpoint=False)
            x = np.interp(t_tgt, t_src, x).astype("float32", copy=False)
        sr = int(sample_rate)
    wav = torch.from_numpy(x)
    return wav, int(sr)


def load_audio(
    path: Path,
    sample_rate: int,
    max_sec: float,
    backend_mode: str = "auto",
) -> tuple[torch.Tensor, str]:
    """Load audio as mono float32 tensor at target sample_rate.

    On some Linux/RPi setups torchaudio backends can be missing; we fall back to soundfile.
    """

    wav: torch.Tensor
    backend: str

    backend_mode = (backend_mode or "auto").lower().strip()
    if backend_mode not in {"auto", "torchaudio", "soundfile"}:
        backend_mode = "auto"

    torchaudio_err: Exception | None = None

    if backend_mode in {"auto", "torchaudio"}:
        global _TORCHAUDIO_DISABLED_REASON
        if backend_mode == "auto" and _TORCHAUDIO_DISABLED_REASON:
            torchaudio_err = RuntimeError(_TORCHAUDIO_DISABLED_REASON)
        else:
            try:
                wav, _sr = _load_audio_torchaudio(path, sample_rate)
                backend = "torchaudio"
            except Exception as e:
                torchaudio_err = e
                # If this is the common TorchCodec-missing / placeholder-wheel error, disable torchaudio
                # in auto mode so we don't spam exceptions for every sample.
                msg = str(e)
                if backend_mode == "auto":
                    if (
                        "TorchCodec is required" in msg
                        or "No module named 'torchcodec.decoders'" in msg
                        or "torchcodec.decoders" in msg
                    ):
                        _TORCHAUDIO_DISABLED_REASON = (
                            "torchaudio.load disabled in auto mode: TorchCodec missing or incomplete "
                            "(e.g., placeholder wheel without torchcodec.decoders)."
                        )
                if backend_mode == "torchaudio":
                    raise

    if backend_mode in {"auto", "soundfile"}:
        try:
            wav, _sr = _load_audio_soundfile(path, sample_rate)
            backend = "soundfile"
        except Exception as sf_e:
            if torchaudio_err is None:
                raise
            msg = (
                f"torchaudio_failed({type(torchaudio_err).__name__}: {torchaudio_err}); "
                f"soundfile_failed({type(sf_e).__name__}: {sf_e})"
            )
            raise RuntimeError(msg) from sf_e

    max_len = int(sample_rate * max_sec)
    if wav.numel() > max_len:
        wav = wav[:max_len]
    return wav, backend


def _worker_process_one(
    task: Dict[str, Any],
    *,
    num_frames: int,
    flow_size: int,
    rgb_size: int,
    video_repr: str,
    sample_rate: int,
    audio_repr: str,
    audio_model: str,
    max_audio_sec: float,
    prosody_no_pitch: bool,
    audio_backend_mode: str,
) -> Dict[str, Any]:
    """Worker-side processing for a single sample.

    Returns a dict with keys:
      - ok: bool
      - sample: sample dict (if ok)
      - error: str (if not ok)
    """
    # Reduce oversubscription when multiple processes are used.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    seq = str(task["seq"])
    label = int(task["label"])
    mn = str(task.get("mn", ""))
    speaker_id = str(task.get("speaker_id", "UNKNOWN"))
    text_cue_flag = bool(task.get("text_cue_flag", False))
    intensity = task.get("intensity", None)
    video_path = Path(task["video_path"])
    audio_path = Path(task["audio_path"])

    motion_cfg = MotionConfig(num_frames=int(num_frames), flow_size=int(flow_size))
    rgb_cfg = RgbConfig(num_frames=int(num_frames), rgb_size=int(rgb_size))
    prosody_cfg = ProsodyConfig(sample_rate=int(sample_rate), use_pitch=(not prosody_no_pitch))

    flow = None
    rgb = None
    if str(video_repr) in {"flow", "both"}:
        try:
            flow_np = compute_face_flow_tensor(video_path, motion_cfg)  # float32 (3,T-1,H,W)
            flow = torch.from_numpy(flow_np).to(torch.float16)
        except Exception as e:
            return {"ok": False, "error": f"flow_error {type(e).__name__}: {e}", "seq": seq}
    if str(video_repr) in {"rgb", "both"}:
        try:
            rgb_np = compute_face_rgb_tensor(video_path, rgb_cfg)  # float32 (T,3,H,W)
            rgb = torch.from_numpy(rgb_np).to(torch.float16)
        except Exception as e:
            return {"ok": False, "error": f"rgb_error {type(e).__name__}: {e}", "seq": seq}

    try:
        wav_f32, audio_backend = load_audio(
            audio_path,
            int(sample_rate),
            float(max_audio_sec),
            backend_mode=str(audio_backend_mode),
        )
        wav_f32 = normalize_wav(wav_f32, target_rms=0.1)
        wav = wav_f32.to(torch.float16)
    except Exception as e:
        return {"ok": False, "error": f"audio_error {type(e).__name__}: {e}", "seq": seq}

    audio_emb = None
    if str(audio_repr) in {"wavlm", "both"}:
        try:
            audio_emb = _extract_wavlm_embedding(wav_f32, str(audio_model))
        except Exception as e:
            return {"ok": False, "error": f"wavlm_error {type(e).__name__}: {e}", "seq": seq}

    try:
        prosody = extract_prosody_features(wav.float(), prosody_cfg)
    except Exception as e:
        return {"ok": False, "error": f"prosody_error {type(e).__name__}: {e}", "seq": seq}

    sample = {
        "prosody": prosody,
        "label": torch.tensor(label, dtype=torch.long),
        "stem": seq,
        "mn": mn,
        "speaker_id": speaker_id,
        "text_cue_flag": text_cue_flag,
        "intensity": torch.tensor(float(intensity), dtype=torch.float32) if intensity is not None else torch.tensor(float("nan"), dtype=torch.float32),
    }
    if str(audio_repr) in {"raw", "both"}:
        sample["audio"] = wav
    if audio_emb is not None:
        sample["audio_emb"] = audio_emb
    if flow is not None:
        sample["flow"] = flow
    if rgb is not None:
        sample["rgb"] = rgb
    return {"ok": True, "sample": sample, "seq": seq, "audio_backend": audio_backend}


def main():
    args = parse_args()

    # In multi-process mode, passing many tensor storages between processes can exhaust
    # file descriptors / ancillary data and crash with errors like:
    #   RuntimeError: received 0 items of ancdata
    # Switching to file_system sharing avoids FD passing and is more robust on RPi.
    if args.num_workers and int(args.num_workers) > 0:
        try:
            import torch.multiprocessing as tmp

            tmp.set_sharing_strategy("file_system")
            print(f"torch.multiprocessing sharing_strategy={tmp.get_sharing_strategy()}", flush=True)
        except Exception as e:
            print(f"Warning: failed to set torch sharing strategy to file_system: {type(e).__name__}: {e}", flush=True)

    data_root = args.data_root.expanduser()
    xlsx = args.xlsx.expanduser()

    xlsx_rows = _read_xlsx_rows(xlsx)
    rows: list[dict[str, Any]] = []

    def _parse_intensity(v: Any) -> float | None:
        if v is None:
            return None
        if isinstance(v, bool):
            return None
        try:
            fv = float(v)
            if math.isnan(fv):
                return None
            return fv
        except Exception:
            s = str(v).strip()
            if not s:
                return None
            try:
                fv = float(s)
                if math.isnan(fv):
                    return None
                return fv
            except Exception:
                return None

    for r in xlsx_rows:
        seq = _normalize_seq(r.get("序号"))
        if not seq:
            continue
        label_raw = str(r.get("情感类别", "") or "").strip()
        label_name = _shared_resolve_label_en(label_raw)
        if label_name not in EMOTIONS:
            continue
        mn = str(r.get("蒙文", "") or "").strip()
        zh = str(r.get("中文", "") or "").strip()
        intensity = _parse_intensity(r.get("情感强度", None))
        rows.append(
            {
                "seq": seq,
                "label": EMOTIONS.index(label_name),
                "label_name": label_name,
                "mn": mn,
                "zh": zh,
                "intensity": intensity,
                "speaker_id": _infer_speaker_id(label_name),
                "text_cue_flag": any(_detect_text_cue_flags(mn, zh, label_raw).values()),
            }
        )

    motion_cfg = MotionConfig(num_frames=args.num_frames, flow_size=args.flow_size)
    rgb_cfg = RgbConfig(num_frames=args.num_frames, rgb_size=args.rgb_size)
    prosody_cfg = ProsodyConfig(sample_rate=args.sample_rate, use_pitch=(not args.prosody_no_pitch))

    out_dir = args.output.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(rows)} labeled rows from {xlsx}", flush=True)
    print(f"Data root: {data_root}", flush=True)
    print(f"Output dir: {out_dir} (shard_size={args.shard_size})", flush=True)

    cached: List[Dict] = []
    shard_id = 1
    missing: List[str] = []

    error_counts: Dict[str, int] = {}

    ok_count = 0
    skip_count = 0

    # Pre-resolve paths (fast, deterministic) and filter missing upfront.
    tasks: List[Dict[str, Any]] = []
    for item in rows:
        seq = item["seq"]
        label = int(item["label"])
        emotion = EMOTIONS[label]
        video_path, audio_path = _shared_resolve_paths_for_seq(data_root, emotion, seq)
        if video_path is None or audio_path is None or (not video_path.exists()) or (not audio_path.exists()):
            missing.append(
                f"SKIP {seq}: expected_video={str((data_root / emotion / f'{seq}.mp4'))} "
                f"expected_audio={str((data_root / emotion / f'{emotion}_audio' / f'{seq}.wav'))}"
            )
            skip_count += 1
            continue

        tasks.append(
            {
                "seq": seq,
                "label": label,
                "label_name": item.get("label_name", emotion),
                "emotion": emotion,
                "mn": item.get("mn", ""),
                "speaker_id": item.get("speaker_id", "UNKNOWN"),
                "text_cue_flag": bool(item.get("text_cue_flag", False)),
                "intensity": item.get("intensity", None),
                "video_path": str(video_path),
                "audio_path": str(audio_path),
            }
        )

    if skip_count:
        print(f"Precheck skipped {skip_count} samples due to missing files", flush=True)

    total = len(tasks)
    if total == 0:
        print("No valid samples after precheck; nothing to do.", flush=True)
        if missing:
            (out_dir / "missing.txt").write_text("\n".join(missing), encoding="utf-8")
            print(f"Wrote missing list -> {out_dir / 'missing.txt'}", flush=True)
        return

    iterator = tasks
    pbar = None
    tqdm_mod = None
    if not args.no_progress:
        try:
            from tqdm import tqdm as tqdm_mod  # type: ignore
        except Exception:
            tqdm_mod = None

    def _log(msg: str) -> None:
        if pbar is not None:
            try:
                pbar.write(msg)
                return
            except Exception:
                pass
        print(msg, flush=True)

    def _should_print_fail(idx: int) -> bool:
        if args.print_result not in {"fail", "all"}:
            return False
        if args.result_every <= 0:
            return False
        return (idx % int(args.result_every)) == 0

    def _should_print_ok(idx: int) -> bool:
        if args.print_result not in {"ok", "all"}:
            return False
        if args.result_every <= 0:
            return False
        return (idx % int(args.result_every)) == 0

    def _err_stage(err: str) -> str:
        # err examples: "flow_error X: ...", "audio_error X: ...", "prosody_error X: ..."
        head = err.split(":", 1)[0].strip()
        if head.startswith("flow_error"):
            return "flow"
        if head.startswith("audio_error"):
            return "audio"
        if head.startswith("prosody_error"):
            return "prosody"
        return head or "unknown"

    def _handle_ok(sample: Dict[str, Any]) -> None:
        nonlocal cached, shard_id, ok_count
        cached.append(sample)
        ok_count += 1
        if args.shard_size > 0 and len(cached) >= args.shard_size:
            shard_path = out_dir / f"{shard_id}.pt"
            _atomic_torch_save(
                {
                    "samples": cached,
                    "config": {
                        "num_frames": args.num_frames,
                        "flow_size": args.flow_size,
                        "rgb_size": args.rgb_size,
                        "video_repr": args.video_repr,
                        "audio_repr": args.audio_repr,
                        "audio_model": args.audio_model,
                        "sample_rate": args.sample_rate,
                        "max_audio_sec": args.max_audio_sec,
                        "emotions": EMOTIONS,
                    },
                    "missing": missing,
                },
                shard_path,
            )
            print(f"Saved shard {shard_id} ({len(cached)} samples) -> {shard_path}", flush=True)
            shard_id += 1
            cached = []

    def _format_task_line(idx: int, task: Dict[str, Any]) -> str:
        return (
            f"[#{idx}/{total}] label={task.get('label_name')}({int(task.get('label', -1))}) | "
            f"video={task.get('video_path')} | audio={task.get('audio_path')} | mn={task.get('mn','')}"
        )

    def _should_print_header_only(idx: int) -> bool:
        # If user requested per-sample result lines, we print a single combined line instead.
        if args.print_result != "none":
            return False
        if args.log_every <= 0:
            return False
        return (idx % int(args.log_every)) == 0

    if args.num_workers and args.num_workers > 0:
        # Multiprocess mode: compute-heavy parts run in worker processes.
        ctx = mp.get_context(args.mp_start_method)
        worker_fn = partial(
            _worker_process_one,
            num_frames=args.num_frames,
            flow_size=args.flow_size,
            rgb_size=args.rgb_size,
            video_repr=str(args.video_repr),
            sample_rate=args.sample_rate,
            audio_repr=str(args.audio_repr),
            audio_model=str(args.audio_model),
            max_audio_sec=args.max_audio_sec,
            prosody_no_pitch=bool(args.prosody_no_pitch),
            audio_backend_mode=str(args.audio_backend),
        )

        # Progress bar counts completed results (not task submission).
        if tqdm_mod is not None:
            pbar = tqdm_mod(total=total, desc="Predecode", unit="sample", dynamic_ncols=True)

        processed = 0
        broken_pool = False
        remaining = tasks
        max_retries = max(0, int(args.pool_retry))
        retry_delay = max(0.0, float(args.pool_retry_delay))
        for attempt in range(max_retries + 1):
            if not remaining:
                break
            try:
                with ProcessPoolExecutor(max_workers=int(args.num_workers), mp_context=ctx) as ex:
                    for i, res in enumerate(ex.map(worker_fn, remaining, chunksize=max(1, int(args.mp_chunksize))), 1):
                        processed += 1
                        task = remaining[i - 1] if i <= len(remaining) else {}
                        base_line = _format_task_line(processed, task) if task else f"[#{processed}/{total}]"
                        if _should_print_header_only(processed):
                            _log(base_line)

                        if pbar is not None:
                            seq = str(res.get("seq", "")) if isinstance(res, dict) else ""
                            pbar.set_postfix_str(f"seq={seq} ok={ok_count} skip={skip_count} shard={shard_id}")
                            pbar.update(1)

                        if not isinstance(res, dict) or not res.get("ok", False):
                            seq = str(res.get("seq", "")) if isinstance(res, dict) else ""
                            err = str(res.get("error", "unknown_error")) if isinstance(res, dict) else "unknown_error"
                            missing.append(f"SKIP {seq}: {err}")
                            skip_count += 1

                            prefix = err.split(":", 1)[0].strip() if err else "unknown_error"
                            error_counts[prefix] = error_counts.get(prefix, 0) + 1
                            if _should_print_fail(processed):
                                _log(f"{base_line} | SKIP stage={_err_stage(err)} | {err}")
                            elif args.print_result not in {"fail", "all"}:
                                if args.show_first_errors > 0 and skip_count <= int(args.show_first_errors):
                                    _log(f"{base_line} | SKIP stage={_err_stage(err)} | {err}")
                            continue

                        _handle_ok(res["sample"])
                        if _should_print_ok(processed):
                            audio_backend = str(res.get("audio_backend", ""))
                            audio_note = "soundfile(fallback)" if audio_backend == "soundfile" else (audio_backend or "unknown")
                            _log(f"{base_line} | OK audio={audio_note} ok={ok_count} skip={skip_count} shard={shard_id}")

                remaining = []
            except BrokenProcessPool as e:
                broken_pool = True
                print("ERROR: BrokenProcessPool (a worker crashed/was killed).", flush=True)
                print(f"Reason: {type(e).__name__}: {e}", flush=True)
                print(
                    "Tips: try fewer workers (e.g. --num-workers 1/2), a different start method (--mp-start-method spawn), "
                    "or run single-process (--num-workers 0). On some RPi setups, also increasing ulimit (ulimit -n 4096) helps.",
                    flush=True,
                )
                remaining = tasks[processed:]
                if attempt < max_retries:
                    print(
                        f"Retrying multiprocessing pool in {retry_delay:.1f}s... (attempt {attempt + 1}/{max_retries})",
                        flush=True,
                    )
                    try:
                        time.sleep(retry_delay)
                    except Exception:
                        pass
                    continue
                print(
                    f"Falling back to single-process for remaining samples: {total - processed} (already processed: {processed}/{total}).",
                    flush=True,
                )
                break
            except KeyboardInterrupt:
                print("Interrupted. Saving partial results...", flush=True)
                remaining = tasks[processed:]
                break

        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass

        # If the pool broke mid-run, process the remaining tasks in the main process.
        if broken_pool and processed < total:
            fb_pbar = None
            fb_iter = remaining
            if tqdm_mod is not None and not args.no_progress:
                fb_pbar = tqdm_mod(total=len(remaining), desc="Fallback", unit="sample", dynamic_ncols=True)

            for j, task in enumerate(remaining, processed + 1):
                res = worker_fn(task)
                base_line = _format_task_line(j, task)

                if fb_pbar is not None:
                    fb_pbar.set_postfix_str(f"seq={task.get('seq')} ok={ok_count} skip={skip_count} shard={shard_id}")
                    fb_pbar.update(1)

                if not isinstance(res, dict) or not res.get("ok", False):
                    seq = str(res.get("seq", "")) if isinstance(res, dict) else str(task.get("seq", ""))
                    err = str(res.get("error", "unknown_error")) if isinstance(res, dict) else "unknown_error"
                    missing.append(f"SKIP {seq}: {err}")
                    skip_count += 1
                    prefix = err.split(":", 1)[0].strip() if err else "unknown_error"
                    error_counts[prefix] = error_counts.get(prefix, 0) + 1
                    if _should_print_fail(j):
                        _log(f"{base_line} | SKIP stage={_err_stage(err)} | {err}")
                    elif args.print_result not in {"fail", "all"}:
                        if args.show_first_errors > 0 and skip_count <= int(args.show_first_errors):
                            _log(f"{base_line} | SKIP stage={_err_stage(err)} | {err}")
                    continue

                _handle_ok(res["sample"])
                if _should_print_ok(j):
                    audio_backend = str(res.get("audio_backend", ""))
                    audio_note = "soundfile(fallback)" if audio_backend == "soundfile" else (audio_backend or "unknown")
                    _log(f"{base_line} | OK audio={audio_note} ok={ok_count} skip={skip_count} shard={shard_id}")

            if fb_pbar is not None:
                try:
                    fb_pbar.close()
                except Exception:
                    pass
    else:
        # Single-process mode.
        if tqdm_mod is not None:
            pbar = tqdm_mod(iterator, total=total, desc="Predecode", unit="sample", dynamic_ncols=True)
            iterator = pbar

        for idx, task in enumerate(iterator, 1):
            if pbar is not None:
                pbar.set_postfix_str(f"seq={task.get('seq')} ok={ok_count} skip={skip_count} shard={shard_id}")

            base_line = _format_task_line(idx, task)
            if _should_print_header_only(idx):
                _log(base_line)

            video_path = Path(task["video_path"])
            audio_path = Path(task["audio_path"])
            seq = str(task["seq"])
            label = int(task["label"])

            flow = None
            rgb = None
            if str(args.video_repr) in {"flow", "both"}:
                try:
                    flow_np = compute_face_flow_tensor(video_path, motion_cfg)  # float32
                    flow = torch.from_numpy(flow_np).to(torch.float16)  # store as fp16
                except Exception as e:
                    err = f"flow_error {type(e).__name__}: {e}"
                    missing.append(f"SKIP {seq}: {err}")
                    skip_count += 1
                    error_counts[err.split(":", 1)[0]] = error_counts.get(err.split(":", 1)[0], 0) + 1
                    if _should_print_fail(idx):
                        _log(f"{base_line} | SKIP stage={_err_stage(err)} | {err}")
                    elif args.print_result not in {"fail", "all"}:
                        if args.show_first_errors > 0 and skip_count <= int(args.show_first_errors):
                            _log(f"{base_line} | SKIP stage={_err_stage(err)} | {err}")
                    continue
            if str(args.video_repr) in {"rgb", "both"}:
                try:
                    rgb_np = compute_face_rgb_tensor(video_path, rgb_cfg)  # float32
                    rgb = torch.from_numpy(rgb_np).to(torch.float16)
                except Exception as e:
                    err = f"rgb_error {type(e).__name__}: {e}"
                    missing.append(f"SKIP {seq}: {err}")
                    skip_count += 1
                    error_counts[err.split(":", 1)[0]] = error_counts.get(err.split(":", 1)[0], 0) + 1
                    if _should_print_fail(idx):
                        _log(f"{base_line} | SKIP stage={_err_stage(err)} | {err}")
                    elif args.print_result not in {"fail", "all"}:
                        if args.show_first_errors > 0 and skip_count <= int(args.show_first_errors):
                            _log(f"{base_line} | SKIP stage={_err_stage(err)} | {err}")
                    continue

            try:
                wav_f32, audio_backend = load_audio(
                    audio_path,
                    args.sample_rate,
                    args.max_audio_sec,
                    backend_mode=str(args.audio_backend),
                )
                wav_f32 = normalize_wav(wav_f32, target_rms=0.1)
                wav = wav_f32.to(torch.float16)
            except Exception as e:
                err = f"audio_error {type(e).__name__}: {e}"
                missing.append(f"SKIP {seq}: {err}")
                skip_count += 1
                error_counts[err.split(":", 1)[0]] = error_counts.get(err.split(":", 1)[0], 0) + 1
                if _should_print_fail(idx):
                    _log(f"{base_line} | SKIP stage={_err_stage(err)} | {err}")
                elif args.print_result not in {"fail", "all"}:
                    if args.show_first_errors > 0 and skip_count <= int(args.show_first_errors):
                        _log(f"{base_line} | SKIP stage={_err_stage(err)} | {err}")
                continue

            audio_emb = None
            if str(args.audio_repr) in {"wavlm", "both"}:
                try:
                    audio_emb = _extract_wavlm_embedding(wav_f32, str(args.audio_model))
                except Exception as e:
                    err = f"wavlm_error {type(e).__name__}: {e}"
                    missing.append(f"SKIP {seq}: {err}")
                    skip_count += 1
                    error_counts[err.split(":", 1)[0]] = error_counts.get(err.split(":", 1)[0], 0) + 1
                    if _should_print_fail(idx):
                        _log(f"{base_line} | SKIP stage={_err_stage(err)} | {err}")
                    elif args.print_result not in {"fail", "all"}:
                        if args.show_first_errors > 0 and skip_count <= int(args.show_first_errors):
                            _log(f"{base_line} | SKIP stage={_err_stage(err)} | {err}")
                    continue

            try:
                prosody = extract_prosody_features(wav.float(), prosody_cfg)  # float32
            except Exception as e:
                err = f"prosody_error {type(e).__name__}: {e}"
                missing.append(f"SKIP {seq}: {err}")
                skip_count += 1
                error_counts[err.split(":", 1)[0]] = error_counts.get(err.split(":", 1)[0], 0) + 1
                if _should_print_fail(idx):
                    _log(f"{base_line} | SKIP stage={_err_stage(err)} | {err}")
                elif args.print_result not in {"fail", "all"}:
                    if args.show_first_errors > 0 and skip_count <= int(args.show_first_errors):
                        _log(f"{base_line} | SKIP stage={_err_stage(err)} | {err}")
                continue

            sample = {
                "prosody": prosody,
                "label": torch.tensor(label, dtype=torch.long),
                "stem": seq,
                "mn": str(task.get("mn", "")),
                "speaker_id": str(task.get("speaker_id", "UNKNOWN")),
                "text_cue_flag": bool(task.get("text_cue_flag", False)),
            }
            if str(args.audio_repr) in {"raw", "both"}:
                sample["audio"] = wav
            if audio_emb is not None:
                sample["audio_emb"] = audio_emb
            if flow is not None:
                sample["flow"] = flow
            if rgb is not None:
                sample["rgb"] = rgb
            _handle_ok(sample)
            if _should_print_ok(idx):
                audio_note = "soundfile(fallback)" if audio_backend == "soundfile" else (audio_backend or "unknown")
                _log(f"{base_line} | OK audio={audio_note} ok={ok_count} skip={skip_count} shard={shard_id}")

        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass

    if cached:
        shard_path = out_dir / f"{shard_id}.pt"
        _atomic_torch_save(
            {
                "samples": cached,
                "config": {
                    "num_frames": args.num_frames,
                    "flow_size": args.flow_size,
                    "rgb_size": args.rgb_size,
                    "video_repr": args.video_repr,
                    "audio_repr": args.audio_repr,
                    "audio_model": args.audio_model,
                    "sample_rate": args.sample_rate,
                    "max_audio_sec": args.max_audio_sec,
                    "emotions": EMOTIONS,
                },
                "missing": missing,
            },
            shard_path,
        )
        print(f"Saved shard {shard_id} ({len(cached)} samples) -> {shard_path}", flush=True)

    if missing:
        (out_dir / "missing.txt").write_text("\n".join(missing), encoding="utf-8")
        print(f"Wrote missing list -> {out_dir / 'missing.txt'}", flush=True)

    if error_counts:
        top = sorted(error_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print("Top skip reasons:", flush=True)
        for k, v in top:
            print(f"  {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
