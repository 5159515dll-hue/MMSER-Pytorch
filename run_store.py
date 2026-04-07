from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import socket
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RUN_STORE_SCHEMA_VERSION = "run_store_v1"
PAPER_GRADE_CONTRACT_KEYS = (
    "protocol_version",
    "manifest_sha256",
    "dataset_kind",
    "task_mode",
    "speaker_id",
    "text_policy",
    "claim_scope",
    "scientific_validity",
    "ablation",
    "zero_video",
    "zero_audio",
    "zero_text",
    "use_intensity",
    "video_backbone",
    "flow_encoder_variant",
    "text_model",
    "max_text_len",
    "sample_rate",
    "max_audio_sec",
    "num_frames",
    "rgb_size",
    "label_names",
)
ATTEMPT_STATUS_ACTIVE = {"initializing", "running", "finalizing"}
ATTEMPT_STATUS_TERMINAL = {
    "completed",
    "interrupted",
    "failed_numeric",
    "failed_determinism",
    "failed_exception",
    "abandoned",
}


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _atomic_write_bytes(data: bytes, dst: Path, *, temp_dir: Path | None = None) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    temp_root = temp_dir.expanduser() if temp_dir is not None else dst.parent
    temp_root.mkdir(parents=True, exist_ok=True)
    tmp_fd: int | None = None
    tmp_path: Path | None = None
    try:
        tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=str(temp_root))
        tmp_path = Path(tmp_name)
        with os.fdopen(tmp_fd, "wb") as handle:
            tmp_fd = None
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(str(tmp_path), str(dst))
        except OSError:
            shutil.move(str(tmp_path), str(dst))
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


def atomic_write_json(payload: dict[str, Any], dst: Path, *, temp_dir: Path | None = None) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default).encode("utf-8")
    _atomic_write_bytes(data, dst, temp_dir=temp_dir)


def atomic_write_text(text: str, dst: Path, *, temp_dir: Path | None = None) -> None:
    _atomic_write_bytes(str(text).encode("utf-8"), dst, temp_dir=temp_dir)


def atomic_torch_save(torch_mod: Any, obj: Any, dst: Path, *, temp_dir: Path | None = None) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    temp_root = temp_dir.expanduser() if temp_dir is not None else dst.parent
    temp_root.mkdir(parents=True, exist_ok=True)
    tmp_fd: int | None = None
    tmp_path: Path | None = None
    try:
        tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=str(temp_root))
        tmp_path = Path(tmp_name)
        with os.fdopen(tmp_fd, "wb") as handle:
            tmp_fd = None
            torch_mod.save(obj, handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(str(tmp_path), str(dst))
        except OSError:
            shutil.move(str(tmp_path), str(dst))
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


def file_digest(path: Path, *, algo: str = "sha256", chunk_size: int = 1024 * 1024) -> dict[str, Any]:
    target = Path(path)
    h = hashlib.new(algo)
    with target.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return {
        "path": str(target),
        "algo": str(algo),
        "hexdigest": h.hexdigest(),
        "size_bytes": int(target.stat().st_size),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalize_jsonish(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_jsonish(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_normalize_jsonish(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_jsonish(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def normalize_input_cache_contract_identity(contract: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(contract, dict) or not contract:
        return None
    return dict(_normalize_jsonish(contract))


def normalize_paper_contract_subset(contract: dict[str, Any] | None) -> dict[str, Any]:
    source = contract if isinstance(contract, dict) else {}
    normalized: dict[str, Any] = {}
    for key in PAPER_GRADE_CONTRACT_KEYS:
        value = source.get(key)
        if key in {"scientific_validity", "zero_video", "zero_audio", "zero_text", "use_intensity"}:
            normalized[key] = bool(value)
        elif key in {"max_text_len", "sample_rate", "num_frames", "rgb_size"}:
            normalized[key] = int(value or 0)
        elif key == "max_audio_sec":
            normalized[key] = float(value or 0.0)
        elif key == "label_names":
            normalized[key] = [str(item) for item in (value or [])]
        elif key == "speaker_id":
            text = str(value).strip() if value is not None else ""
            normalized[key] = text or None
        else:
            normalized[key] = str(value or "")
    return normalized


def missing_paper_contract_fields(contract: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for key in PAPER_GRADE_CONTRACT_KEYS:
        value = contract.get(key)
        if key in {"scientific_validity", "zero_video", "zero_audio", "zero_text", "use_intensity"}:
            continue
        if key in {"max_text_len", "sample_rate", "num_frames", "rgb_size"}:
            if int(value or 0) <= 0:
                missing.append(key)
            continue
        if key == "max_audio_sec":
            if float(value or 0.0) <= 0.0:
                missing.append(key)
            continue
        if key == "label_names":
            if not list(value or []):
                missing.append(key)
            continue
        if key == "speaker_id":
            continue
        if not str(value or "").strip():
            missing.append(key)
    return missing


def generate_attempt_id(seed: int | None = None) -> str:
    base = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    suffix = uuid.uuid4().hex[:8]
    if seed is None:
        return f"attempt_{base}_{suffix}"
    return f"attempt_seed{int(seed)}_{base}_{suffix}"


def run_manifest_path(run_dir: Path) -> Path:
    return Path(run_dir).expanduser() / "run_manifest.json"


def is_run_dir(path: Path) -> bool:
    return run_manifest_path(path).is_file()


def load_run_manifest(run_dir: Path) -> dict[str, Any]:
    return _load_json(run_manifest_path(run_dir))


def load_attempt_manifest(attempt_dir: Path) -> dict[str, Any]:
    return _load_json(Path(attempt_dir).expanduser() / "attempt_manifest.json")


def resolve_attempt_dir(run_dir: Path, *, attempt_id: str | None = None, prefer_published: bool = True) -> Path:
    manifest = load_run_manifest(run_dir)
    selected_attempt = str(attempt_id or "").strip()
    if not selected_attempt:
        if prefer_published:
            selected_attempt = str(manifest.get("published_attempt_id", "") or "")
        if not selected_attempt:
            selected_attempt = str(manifest.get("active_attempt_id", "") or "")
    if not selected_attempt:
        raise RuntimeError(f"Run has no published or active attempt: {run_dir}")
    attempts = manifest.get("attempts", [])
    for record in attempts:
        if str(record.get("attempt_id", "")) == selected_attempt:
            relpath = str(record.get("attempt_relpath", "")).strip()
            if relpath:
                return (Path(run_dir).expanduser() / relpath).resolve()
    candidate = Path(run_dir).expanduser() / "attempts" / selected_attempt
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"Attempt not found under run dir {run_dir}: {selected_attempt}")


def resolve_best_bundle(attempt_dir: Path) -> dict[str, Any]:
    manifest = load_attempt_manifest(attempt_dir)
    relpath = str(manifest.get("best_bundle_relpath", "") or "")
    if not relpath:
        raise FileNotFoundError(f"Attempt has no best bundle: {attempt_dir}")
    bundle_dir = Path(attempt_dir).expanduser() / relpath
    bundle_manifest_path = bundle_dir / "bundle_manifest.json"
    if not bundle_manifest_path.is_file():
        raise FileNotFoundError(f"Best bundle manifest missing: {bundle_manifest_path}")
    bundle_manifest = _load_json(bundle_manifest_path)
    return {
        "bundle_dir": bundle_dir,
        "bundle_manifest": bundle_manifest,
        "checkpoint_path": bundle_dir / "checkpoint.pt",
        "inference_jsonl_path": bundle_dir / "inference_val.jsonl",
        "inference_metrics_path": bundle_dir / "inference_val.metrics.json",
    }


def resolve_published_metrics(attempt_dir: Path) -> Path:
    manifest = load_attempt_manifest(attempt_dir)
    relpath = str(manifest.get("published_metrics_relpath", "") or "")
    if not relpath:
        raise FileNotFoundError(f"Attempt has no published metrics: {attempt_dir}")
    path = Path(attempt_dir).expanduser() / relpath
    if not path.is_file():
        raise FileNotFoundError(f"Published metrics not found: {path}")
    return path


def resolve_published_inference_metrics(attempt_dir: Path, *, subset: str) -> Path:
    manifest = load_attempt_manifest(attempt_dir)
    published_metrics = manifest.get("published_inference_metrics", {})
    metrics_relpath = ""
    if isinstance(published_metrics, dict):
        metrics_relpath = str(published_metrics.get(str(subset), "") or "")
    candidates: list[Path] = []
    if metrics_relpath:
        candidates.append(Path(attempt_dir).expanduser() / metrics_relpath)
    candidates.extend(
        [
            Path(attempt_dir).expanduser() / "published" / f"inference_{subset}.metrics.json",
            Path(attempt_dir).expanduser() / "published" / f"inference_{subset}.jsonl.metrics.json",
        ]
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Published inference metrics not found for subset={subset}: {attempt_dir}")


def register_published_inference_output(attempt_dir: Path, *, subset: str, output_path: Path) -> None:
    attempt_root = Path(attempt_dir).expanduser()
    manifest_path = attempt_root / "attempt_manifest.json"
    manifest = load_attempt_manifest(attempt_root)
    published_metrics = dict(manifest.get("published_inference_metrics", {}))
    resolved_output = Path(output_path).expanduser()
    try:
        output_relpath = str(resolved_output.relative_to(attempt_root))
    except ValueError:
        output_relpath = str(resolved_output)
    published_metrics[str(subset)] = str(Path(output_relpath).with_name(Path(output_relpath).stem + ".metrics.json"))
    manifest["published_inference_outputs"] = {
        **dict(manifest.get("published_inference_outputs", {})),
        str(subset): output_relpath,
    }
    manifest["published_inference_metrics"] = published_metrics
    atomic_write_json(manifest, manifest_path)


def validate_run_dir(run_dir: Path) -> dict[str, Any]:
    run_root = Path(run_dir).expanduser()
    issues: list[str] = []
    checks: dict[str, Any] = {
        "run_manifest_exists": False,
        "published_attempt_resolved": False,
        "published_attempt_completed": False,
        "published_train_metrics_exists": False,
        "best_bundle_exists": False,
        "best_bundle_checkpoint_exists": False,
        "best_bundle_val_jsonl_exists": False,
        "best_bundle_val_metrics_exists": False,
        "published_test_metrics_exists": False,
        "paper_grade_all_eligible": False,
        "manifest_sha256_consistent": False,
        "paper_contract_consistent": False,
        "input_cache_contract_consistent": False,
        "published_bundle_checkpoint_used_for_val_test": False,
    }
    run_manifest_file = run_manifest_path(run_root)
    if not run_manifest_file.is_file():
        issues.append("run_manifest_missing")
        return {
            "run_dir": str(run_root),
            "schema_version": None,
            "published_attempt_id": None,
            "active_attempt_id": None,
            "attempt_count": 0,
            "issues": issues,
            "checks": checks,
        }
    checks["run_manifest_exists"] = True
    try:
        manifest = load_run_manifest(run_root)
    except Exception as exc:
        issues.append(f"run_manifest_invalid:{type(exc).__name__}")
        return {
            "run_dir": str(run_root),
            "schema_version": None,
            "published_attempt_id": None,
            "active_attempt_id": None,
            "attempt_count": 0,
            "issues": issues,
            "checks": checks,
        }
    published_attempt_id = str(manifest.get("published_attempt_id", "") or "")
    active_attempt_id = str(manifest.get("active_attempt_id", "") or "")
    attempts = manifest.get("attempts", [])
    attempt_map = {str(item.get("attempt_id", "")): item for item in attempts if isinstance(item, dict)}
    if not published_attempt_id:
        issues.append("published_attempt_missing")
    elif published_attempt_id not in attempt_map:
        issues.append("published_attempt_missing")
    if active_attempt_id and active_attempt_id not in attempt_map:
        issues.append("active_attempt_missing")
    published_attempt_dir = None
    if published_attempt_id in attempt_map:
        published_attempt_dir = run_root / str(attempt_map[published_attempt_id].get("attempt_relpath", ""))
        checks["published_attempt_resolved"] = bool(published_attempt_dir.is_dir())
        if not published_attempt_dir.is_dir():
            issues.append("published_attempt_dir_missing")
        attempt_manifest: dict[str, Any] | None = None
        if published_attempt_dir.is_dir():
            try:
                attempt_manifest = load_attempt_manifest(published_attempt_dir)
            except Exception as exc:
                issues.append(f"published_attempt_manifest_invalid:{type(exc).__name__}")
            if attempt_manifest is not None:
                status = str(attempt_manifest.get("status", "") or "")
                checks["published_attempt_completed"] = status == "completed"
                if status != "completed":
                    issues.append(f"published_attempt_status_invalid:{status or 'missing'}")
        try:
            bundle = resolve_best_bundle(published_attempt_dir)
            checks["best_bundle_exists"] = True
            bundle_dir = Path(bundle["bundle_dir"]).expanduser()
            checkpoint_path = Path(bundle["checkpoint_path"]).expanduser()
            val_jsonl_path = Path(bundle["inference_jsonl_path"]).expanduser()
            val_metrics_path = Path(bundle["inference_metrics_path"]).expanduser()
            checks["best_bundle_checkpoint_exists"] = checkpoint_path.is_file()
            checks["best_bundle_val_jsonl_exists"] = val_jsonl_path.is_file()
            checks["best_bundle_val_metrics_exists"] = val_metrics_path.is_file()
            if not checkpoint_path.is_file():
                issues.append("best_bundle_checkpoint_missing")
            if not val_jsonl_path.is_file():
                issues.append("best_bundle_val_jsonl_missing")
            if not val_metrics_path.is_file():
                issues.append("best_bundle_val_metrics_missing")
        except Exception as exc:
            if isinstance(exc, FileNotFoundError):
                issues.append("published_best_bundle_missing")
            else:
                issues.append(f"published_best_bundle_invalid:{type(exc).__name__}")
            bundle = None
        try:
            train_metrics_path = resolve_published_metrics(published_attempt_dir)
            checks["published_train_metrics_exists"] = True
        except Exception as exc:
            if isinstance(exc, FileNotFoundError):
                issues.append("published_train_metrics_missing")
            else:
                issues.append(f"published_train_metrics_invalid:{type(exc).__name__}")
            train_metrics_path = None
        try:
            test_metrics_path = resolve_published_inference_metrics(published_attempt_dir, subset="test")
            checks["published_test_metrics_exists"] = True
        except Exception as exc:
            if isinstance(exc, FileNotFoundError):
                issues.append("published_test_metrics_missing")
            else:
                issues.append(f"published_test_metrics_invalid:{type(exc).__name__}")
            test_metrics_path = None

        if bundle is not None and train_metrics_path is not None and test_metrics_path is not None:
            try:
                train_metrics = _load_json(Path(train_metrics_path))
                val_metrics = _load_json(Path(bundle["inference_metrics_path"]))
                test_metrics = _load_json(Path(test_metrics_path))
            except Exception as exc:
                issues.append(f"metrics_payload_invalid:{type(exc).__name__}")
            else:
                metrics_by_name = {
                    "train": train_metrics,
                    "val": val_metrics,
                    "test": test_metrics,
                }
                paper_grade_ok = True
                for name, payload in metrics_by_name.items():
                    paper_grade = payload.get("paper_grade")
                    if not isinstance(paper_grade, dict):
                        issues.append(f"{name}_paper_grade_missing")
                        paper_grade_ok = False
                        continue
                    if not bool(paper_grade.get("eligible", False)):
                        issues.append(f"{name}_paper_grade_ineligible")
                        paper_grade_ok = False
                checks["paper_grade_all_eligible"] = paper_grade_ok

                train_contract = normalize_paper_contract_subset(train_metrics.get("meta", {}).get("paper_contract", {}))
                val_contract = normalize_paper_contract_subset(val_metrics.get("paper_contract", {}))
                test_contract = normalize_paper_contract_subset(test_metrics.get("paper_contract", {}))
                contract_match = train_contract == val_contract == test_contract
                checks["paper_contract_consistent"] = contract_match
                for name, contract_payload in (("train", train_contract), ("val", val_contract), ("test", test_contract)):
                    for field in missing_paper_contract_fields(contract_payload):
                        issues.append(f"{name}_paper_contract_missing_{field}")
                if not contract_match:
                    if train_contract != val_contract:
                        issues.append("train_val_paper_contract_mismatch")
                    if train_contract != test_contract:
                        issues.append("train_test_paper_contract_mismatch")

                manifest_hashes = {
                    "train": str(train_metrics.get("meta", {}).get("manifest_sha256", "")),
                    "val": str(val_metrics.get("manifest_sha256", "")),
                    "test": str(test_metrics.get("manifest_sha256", "")),
                }
                manifest_consistent = (
                    bool(manifest_hashes["train"])
                    and manifest_hashes["train"] == manifest_hashes["val"] == manifest_hashes["test"]
                )
                checks["manifest_sha256_consistent"] = manifest_consistent
                if not manifest_consistent:
                    issues.append("manifest_sha256_mismatch")

                train_cache = normalize_input_cache_contract_identity(train_metrics.get("meta", {}).get("input_cache_contract", {}))
                val_cache = normalize_input_cache_contract_identity(val_metrics.get("input_cache_contract", {}))
                test_cache = normalize_input_cache_contract_identity(test_metrics.get("input_cache_contract", {}))
                cache_consistent = train_cache == val_cache == test_cache
                checks["input_cache_contract_consistent"] = cache_consistent
                if not cache_consistent:
                    if train_cache != val_cache:
                        issues.append("train_val_input_cache_contract_mismatch")
                    if train_cache != test_cache:
                        issues.append("train_test_input_cache_contract_mismatch")

                bundle_checkpoint = Path(bundle["checkpoint_path"]).expanduser()
                val_checkpoint = Path(str(val_metrics.get("checkpoint", ""))).expanduser()
                test_checkpoint = Path(str(test_metrics.get("checkpoint", ""))).expanduser()
                checkpoint_match = val_checkpoint == bundle_checkpoint and test_checkpoint == bundle_checkpoint
                checks["published_bundle_checkpoint_used_for_val_test"] = checkpoint_match
                if not checkpoint_match:
                    if val_checkpoint != bundle_checkpoint:
                        issues.append("val_checkpoint_not_published_bundle")
                    if test_checkpoint != bundle_checkpoint:
                        issues.append("test_checkpoint_not_published_bundle")
    return {
        "run_dir": str(run_root),
        "schema_version": str(manifest.get("schema_version", "")),
        "published_attempt_id": published_attempt_id or None,
        "active_attempt_id": active_attempt_id or None,
        "attempt_count": len(attempts),
        "issues": issues,
        "checks": checks,
    }


@dataclass
class SignalState:
    signum: int | None = None
    signal_name: str | None = None

    @property
    def requested(self) -> bool:
        return self.signum is not None


class SignalCapture:
    def __init__(self) -> None:
        self.state = SignalState()
        self._previous: dict[int, Any] = {}

    def _handler(self, signum: int, _frame: Any) -> None:
        self.state.signum = int(signum)
        try:
            self.state.signal_name = signal.Signals(signum).name
        except Exception:
            self.state.signal_name = f"SIG{int(signum)}"

    def install(self) -> "SignalCapture":
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous[int(sig)] = signal.getsignal(sig)
                signal.signal(sig, self._handler)
            except Exception:
                continue
        return self

    def restore(self) -> None:
        for sig, handler in self._previous.items():
            try:
                signal.signal(sig, handler)
            except Exception:
                continue
        self._previous.clear()


class RunAttemptStore:
    def __init__(self, run_dir: Path, attempt_dir: Path, *, run_manifest: dict[str, Any], attempt_manifest: dict[str, Any]) -> None:
        self.run_dir = Path(run_dir).expanduser()
        self.attempt_dir = Path(attempt_dir).expanduser()
        self._run_manifest = dict(run_manifest)
        self._attempt_manifest = dict(attempt_manifest)

    @property
    def attempt_id(self) -> str:
        return str(self._attempt_manifest.get("attempt_id", ""))

    @property
    def published_dir(self) -> Path:
        return self.attempt_dir / "published"

    @property
    def published_plots_dir(self) -> Path:
        return self.published_dir / "plots"

    @property
    def best_bundle_relpath(self) -> str | None:
        value = self._attempt_manifest.get("best_bundle_relpath")
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @classmethod
    def create(
        cls,
        run_dir: Path,
        *,
        seed: int | None,
        benchmark_tag: str,
        args_payload: dict[str, Any],
        run_contract: dict[str, Any],
        provenance: dict[str, Any],
        validity: dict[str, Any],
        input_cache_contract: dict[str, Any] | None,
        deterministic_policy: dict[str, Any],
    ) -> "RunAttemptStore":
        root = Path(run_dir).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        root_manifest_path = run_manifest_path(root)
        if root_manifest_path.is_file():
            run_manifest = _load_json(root_manifest_path)
        else:
            run_manifest = {
                "schema_version": RUN_STORE_SCHEMA_VERSION,
                "run_dir": str(root),
                "created_at": _utc_timestamp(),
                "updated_at": _utc_timestamp(),
                "benchmark_tag": str(benchmark_tag),
                "published_attempt_id": None,
                "active_attempt_id": None,
                "attempts": [],
            }
        previous_active = str(run_manifest.get("active_attempt_id", "") or "")
        attempts = list(run_manifest.get("attempts", []))
        if previous_active:
            for record in attempts:
                if str(record.get("attempt_id", "")) != previous_active:
                    continue
                prev_attempt_dir = root / str(record.get("attempt_relpath", ""))
                prev_manifest_path = prev_attempt_dir / "attempt_manifest.json"
                if prev_manifest_path.is_file():
                    prev_manifest = _load_json(prev_manifest_path)
                    if str(prev_manifest.get("status", "")) in ATTEMPT_STATUS_ACTIVE:
                        prev_manifest["status"] = "abandoned"
                        prev_manifest["finished_at"] = _utc_timestamp()
                        prev_manifest["updated_at"] = _utc_timestamp()
                        prev_manifest["failure"] = {
                            "type": "superseded_by_new_attempt",
                            "message": "A new attempt was started under the same run directory.",
                        }
                        atomic_write_json(prev_manifest, prev_manifest_path)
                        record["status"] = "abandoned"
                        record["finished_at"] = prev_manifest["finished_at"]
        attempt_id = generate_attempt_id(seed)
        attempt_relpath = Path("attempts") / attempt_id
        attempt_dir = root / attempt_relpath
        (attempt_dir / "state" / "epochs").mkdir(parents=True, exist_ok=True)
        (attempt_dir / "bundles").mkdir(parents=True, exist_ok=True)
        (attempt_dir / "published" / "plots").mkdir(parents=True, exist_ok=True)
        attempt_manifest = {
            "schema_version": RUN_STORE_SCHEMA_VERSION,
            "attempt_id": attempt_id,
            "attempt_relpath": str(attempt_relpath),
            "run_dir": str(root),
            "status": "initializing",
            "benchmark_tag": str(benchmark_tag),
            "seed": int(seed) if seed is not None else None,
            "created_at": _utc_timestamp(),
            "updated_at": _utc_timestamp(),
            "started_at": _utc_timestamp(),
            "finished_at": None,
            "host": socket.gethostname(),
            "pid": int(os.getpid()),
            "args": dict(args_payload),
            "run_contract": dict(run_contract),
            "provenance": dict(provenance),
            "validity": dict(validity),
            "input_cache_contract": dict(input_cache_contract or {}),
            "deterministic_policy": dict(deterministic_policy),
            "current_epoch": 0,
            "epochs_completed": [],
            "best_epoch": 0,
            "best_bundle_relpath": None,
            "published_last_checkpoint_relpath": None,
            "published_metrics_relpath": None,
            "published_results_summary_relpath": None,
            "published_inference_metrics": {},
            "run_status": "initializing",
            "stop_reason": "",
            "failure": None,
        }
        attempts.append(
            {
                "attempt_id": attempt_id,
                "attempt_relpath": str(attempt_relpath),
                "seed": int(seed) if seed is not None else None,
                "status": "initializing",
                "started_at": attempt_manifest["started_at"],
                "finished_at": None,
            }
        )
        run_manifest["attempts"] = attempts
        run_manifest["active_attempt_id"] = attempt_id
        run_manifest["updated_at"] = _utc_timestamp()
        atomic_write_json(attempt_manifest, attempt_dir / "attempt_manifest.json")
        atomic_write_json(run_manifest, root_manifest_path)
        return cls(root, attempt_dir, run_manifest=run_manifest, attempt_manifest=attempt_manifest)

    def _persist_run_manifest(self) -> None:
        self._run_manifest["updated_at"] = _utc_timestamp()
        atomic_write_json(self._run_manifest, run_manifest_path(self.run_dir))

    def _persist_attempt_manifest(self) -> None:
        self._attempt_manifest["updated_at"] = _utc_timestamp()
        atomic_write_json(self._attempt_manifest, self.attempt_dir / "attempt_manifest.json")

    def _update_attempt_record(self) -> None:
        attempts = list(self._run_manifest.get("attempts", []))
        for record in attempts:
            if str(record.get("attempt_id", "")) != self.attempt_id:
                continue
            record["status"] = str(self._attempt_manifest.get("status", ""))
            record["finished_at"] = self._attempt_manifest.get("finished_at")
            record["best_epoch"] = int(self._attempt_manifest.get("best_epoch", 0) or 0)
            record["run_status"] = str(self._attempt_manifest.get("run_status", "") or "")
            break
        self._run_manifest["attempts"] = attempts

    def mark_running(self) -> None:
        self._attempt_manifest["status"] = "running"
        self._attempt_manifest["run_status"] = "running"
        self._persist_attempt_manifest()
        self._update_attempt_record()
        self._persist_run_manifest()

    def note_signal(self, signal_name: str) -> None:
        self._attempt_manifest["signal_request"] = str(signal_name)
        self._persist_attempt_manifest()

    def write_epoch_state(
        self,
        *,
        epoch: int,
        epoch_payload: dict[str, Any],
        best_epoch: int,
        best_bundle_relpath: str | None,
    ) -> None:
        epoch_path = self.attempt_dir / "state" / "epochs" / f"epoch_{int(epoch):04d}.json"
        atomic_write_json(epoch_payload, epoch_path)
        completed = [int(x) for x in self._attempt_manifest.get("epochs_completed", [])]
        if int(epoch) not in completed:
            completed.append(int(epoch))
        completed.sort()
        self._attempt_manifest["current_epoch"] = int(epoch)
        self._attempt_manifest["epochs_completed"] = completed
        self._attempt_manifest["best_epoch"] = int(best_epoch)
        self._attempt_manifest["best_bundle_relpath"] = str(best_bundle_relpath) if best_bundle_relpath else None
        latest_payload = {
            "schema_version": RUN_STORE_SCHEMA_VERSION,
            "attempt_id": self.attempt_id,
            "current_epoch": int(epoch),
            "best_epoch": int(best_epoch),
            "best_bundle_relpath": str(best_bundle_relpath) if best_bundle_relpath else None,
            "epoch_relpath": str(epoch_path.relative_to(self.attempt_dir)),
            "updated_at": _utc_timestamp(),
            "run_status": str(self._attempt_manifest.get("run_status", "")),
        }
        atomic_write_json(latest_payload, self.attempt_dir / "state" / "latest.json")
        self._persist_attempt_manifest()
        self._update_attempt_record()
        self._persist_run_manifest()

    def publish_best_bundle(
        self,
        *,
        torch_mod: Any,
        epoch: int,
        checkpoint_payload: dict[str, Any],
        records: list[dict[str, Any]],
        metrics_summary: dict[str, Any],
        selection_meta: dict[str, Any],
    ) -> Path:
        bundle_name = f"best_epoch_{int(epoch):04d}"
        bundle_dir = self.attempt_dir / "bundles" / bundle_name
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        tmp_dir = self.attempt_dir / ".tmp" / f"{bundle_name}.{uuid.uuid4().hex[:8]}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = tmp_dir / "checkpoint.pt"
        atomic_torch_save(torch_mod, checkpoint_payload, checkpoint_path, temp_dir=tmp_dir)
        jsonl_path = tmp_dir / "inference_val.jsonl"
        lines = "\n".join(json.dumps(rec, ensure_ascii=False, default=_json_default) for rec in records)
        if lines:
            lines += "\n"
        atomic_write_text(lines, jsonl_path, temp_dir=tmp_dir)
        metrics_path = tmp_dir / "inference_val.metrics.json"
        atomic_write_json(metrics_summary, metrics_path, temp_dir=tmp_dir)
        bundle_manifest = {
            "schema_version": RUN_STORE_SCHEMA_VERSION,
            "bundle_id": bundle_name,
            "attempt_id": self.attempt_id,
            "epoch": int(epoch),
            "created_at": _utc_timestamp(),
            "checkpoint": file_digest(checkpoint_path),
            "inference_val_jsonl": file_digest(jsonl_path),
            "inference_val_metrics": file_digest(metrics_path),
            "selection_meta": dict(selection_meta),
        }
        atomic_write_json(bundle_manifest, tmp_dir / "bundle_manifest.json", temp_dir=tmp_dir)
        bundle_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(tmp_dir), str(bundle_dir))
        relpath = str(bundle_dir.relative_to(self.attempt_dir))
        self._attempt_manifest["best_epoch"] = int(epoch)
        self._attempt_manifest["best_bundle_relpath"] = relpath
        published_metrics = dict(self._attempt_manifest.get("published_inference_metrics", {}))
        published_metrics["val"] = str((bundle_dir / "inference_val.metrics.json").relative_to(self.attempt_dir))
        self._attempt_manifest["published_inference_metrics"] = published_metrics
        self._persist_attempt_manifest()
        self._update_attempt_record()
        self._persist_run_manifest()
        return bundle_dir

    def publish_last_checkpoint(self, *, torch_mod: Any, checkpoint_payload: dict[str, Any]) -> Path:
        path = self.published_dir / "last.pt"
        atomic_torch_save(torch_mod, checkpoint_payload, path)
        self._attempt_manifest["published_last_checkpoint_relpath"] = str(path.relative_to(self.attempt_dir))
        self._persist_attempt_manifest()
        return path

    def publish_reports(self, *, metrics_dir: Path) -> None:
        metrics_path = metrics_dir / "metrics.json"
        summary_path = metrics_dir / "results_summary.md"
        if metrics_path.is_file():
            self._attempt_manifest["published_metrics_relpath"] = str(metrics_path.relative_to(self.attempt_dir))
        if summary_path.is_file():
            self._attempt_manifest["published_results_summary_relpath"] = str(summary_path.relative_to(self.attempt_dir))
        self._persist_attempt_manifest()

    def record_failure(self, *, status: str, message: str, exc_type: str | None = None) -> None:
        self._attempt_manifest["failure"] = {
            "type": str(exc_type or status),
            "message": str(message),
            "recorded_at": _utc_timestamp(),
        }
        atomic_write_json(self._attempt_manifest["failure"], self.attempt_dir / "state" / "failure.json")

    def finalize(
        self,
        *,
        status: str,
        run_status: str,
        stop_reason: str,
        publish_attempt: bool,
        failure: dict[str, Any] | None = None,
    ) -> None:
        self._attempt_manifest["status"] = str(status)
        self._attempt_manifest["run_status"] = str(run_status)
        self._attempt_manifest["stop_reason"] = str(stop_reason)
        self._attempt_manifest["finished_at"] = _utc_timestamp()
        if failure is not None:
            self._attempt_manifest["failure"] = dict(failure)
            atomic_write_json(self._attempt_manifest["failure"], self.attempt_dir / "state" / "failure.json")
        self._persist_attempt_manifest()
        self._update_attempt_record()
        if str(self._run_manifest.get("active_attempt_id", "")) == self.attempt_id:
            self._run_manifest["active_attempt_id"] = None
        if publish_attempt:
            self._run_manifest["published_attempt_id"] = self.attempt_id
        self._persist_run_manifest()
