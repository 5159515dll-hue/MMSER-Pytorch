"""准备主线运行所需的媒体 sidecar。

当前主线在 MELD 上依赖音频 sidecar，因此训练前需要把视频里的音轨提取成
 wav 文件。这个脚本只负责这一步准备工作，不负责训练本身。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm


def _ensure_project_root_on_path() -> None:
    """在脚本直跑时把仓库根目录加入导入路径。"""

    project_root = Path(__file__).resolve().parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


_ensure_project_root_on_path()

from manifest_utils import DATASET_KINDS, load_split_manifest, resolve_dataset_kind, select_manifest_items


def parse_args() -> argparse.Namespace:
    """解析媒体准备入口参数。"""

    p = argparse.ArgumentParser(description="Prepare dataset sidecar media (currently: extract wav from video)")
    p.add_argument("--dataset-kind", type=str, default="meld", choices=list(DATASET_KINDS))
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--subset", type=str, default="all", choices=["all", "train", "val", "test"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--sample-rate", type=int, default=24000)
    p.add_argument("--num-workers", type=str, default="auto")
    p.add_argument("--ffmpeg-bin", type=str, default="ffmpeg")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def resolve_ffmpeg_bin(explicit_bin: str) -> str:
    """解析可执行的 ffmpeg 路径，优先用户显式参数，其次回退到 imageio-ffmpeg。"""

    explicit_bin = str(explicit_bin).strip() or "ffmpeg"
    resolved = shutil.which(explicit_bin)
    if resolved is not None:
        return resolved

    # Public benchmark servers often lack a system ffmpeg package. If imageio-ffmpeg
    # is installed, reuse its bundled binary instead of failing the whole pipeline.
    try:
        import imageio_ffmpeg  # type: ignore

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and Path(str(bundled)).exists():
            return str(bundled)
    except Exception:
        pass

    raise RuntimeError(
        f"ffmpeg not found: {explicit_bin}. Install ffmpeg or the Python package imageio-ffmpeg."
    )


def _extract_audio_sidecar(*, ffmpeg_bin: str, video_path: Path, audio_path: Path, sample_rate: int, overwrite: bool) -> dict[str, Any]:
    """把视频容器里的音轨提取成单声道 wav sidecar。"""

    if not video_path.exists():
        return {"ok": False, "reason": "missing_video", "video": str(video_path), "audio": str(audio_path)}
    if audio_path.exists() and not overwrite:
        return {"ok": True, "reason": "exists", "video": str(video_path), "audio": str(audio_path)}

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{audio_path.stem}.", suffix=".tmp.wav", dir=str(audio_path.parent), delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cmd = [
            ffmpeg_bin,
            "-nostdin",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(int(sample_rate)),
            "-f",
            "wav",
            str(tmp_path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            return {
                "ok": False,
                "reason": f"ffmpeg_failed({proc.returncode})",
                "video": str(video_path),
                "audio": str(audio_path),
                "stderr": proc.stderr[-500:],
            }
        tmp_path.replace(audio_path)
        return {"ok": True, "reason": "extracted", "video": str(video_path), "audio": str(audio_path)}
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def main() -> None:
    """执行 sidecar 媒体准备。"""

    args = parse_args()
    from runtime_adapt import detect_runtime, resolve_worker_count

    dataset_kind = resolve_dataset_kind(args.dataset_kind)
    if dataset_kind != "meld":
        raise RuntimeError("prepare_dataset_media.py currently only needs to run for --dataset-kind meld")
    ffmpeg_bin = resolve_ffmpeg_bin(str(args.ffmpeg_bin))

    manifest = load_split_manifest(args.split_manifest.expanduser())
    items = select_manifest_items(manifest, args.subset)
    if args.limit and int(args.limit) > 0:
        items = items[: int(args.limit)]
    tasks = []
    for item in items:
        video_path = item.get("video_path")
        audio_path = item.get("audio_path")
        if not video_path or not audio_path:
            continue
        tasks.append((Path(str(video_path)), Path(str(audio_path))))
    if not tasks:
        summary = manifest.get("summary", {})
        raise RuntimeError(
            "No usable MELD manifest items were selected for media preparation. "
            f"subset={args.subset}, manifest_usable_rows={summary.get('usable_rows')}, "
            f"manifest_total_rows={summary.get('total_rows')}. "
            "This usually means build_split_manifest.py was run with a --data-root that does not point "
            "to the MELD MP4 directory. Rebuild the manifest with the directory that actually contains "
            "`train_splits`, `dev_splits_complete`, or `output_repeated_splits_test`."
        )

    profile = detect_runtime("auto")
    num_workers = resolve_worker_count(
        args.num_workers,
        phase="predecode",
        profile=profile,
        dataset_in_memory=False,
        total_items=len(tasks),
    )
    print(
        f"Preparing audio sidecars from {len(tasks)} sample(s) "
        f"(dataset_kind={dataset_kind}, subset={args.subset}, workers={num_workers}, sample_rate={args.sample_rate}, ffmpeg={ffmpeg_bin})",
        flush=True,
    )

    ok = 0
    skipped = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as ex:
        futures = [
            ex.submit(
                _extract_audio_sidecar,
                ffmpeg_bin=ffmpeg_bin,
                video_path=video_path,
                audio_path=audio_path,
                sample_rate=int(args.sample_rate),
                overwrite=bool(args.overwrite),
            )
            for video_path, audio_path in tasks
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Prepare media", unit="file"):
            result = future.result()
            if result.get("ok"):
                if result.get("reason") == "exists":
                    skipped += 1
                else:
                    ok += 1
            else:
                failed += 1
                print(
                    f"FAILED video={result.get('video')} audio={result.get('audio')} reason={result.get('reason')}",
                    flush=True,
                )

    print(
        f"Audio sidecars ready: extracted={ok} skipped={skipped} failed={failed} total={len(tasks)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
