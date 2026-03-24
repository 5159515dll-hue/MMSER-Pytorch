from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


_ensure_project_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check torch/CUDA runtime status and system GPU status")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--json", action="store_true", help="Print the full result as JSON")
    p.add_argument("--verbose", action="store_true", help="Include stderr/stdout from system tools")
    return p.parse_args()


def _which(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    candidates = [
        f"/opt/hyhal/bin/{name}",
        f"/opt/rocm/bin/{name}",
        f"/usr/bin/{name}",
        f"/usr/local/bin/{name}",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


def _run_cmd(cmd: list[str], *, timeout: int = 15, verbose: bool = False) -> dict[str, Any]:
    binary = _which(cmd[0])
    if binary is None:
        return {"available": False, "cmd": cmd, "reason": "not_found"}
    real_cmd = [binary, *cmd[1:]]
    try:
        proc = subprocess.run(
            real_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        result: dict[str, Any] = {
            "available": True,
            "cmd": real_cmd,
            "returncode": int(proc.returncode),
            "ok": proc.returncode == 0,
            "stdout": (proc.stdout or "").strip(),
        }
        if verbose or proc.returncode != 0:
            result["stderr"] = (proc.stderr or "").strip()
        return result
    except Exception as e:
        return {
            "available": True,
            "cmd": real_cmd,
            "ok": False,
            "error_type": type(e).__name__,
            "error": str(e),
        }


def _parse_nvidia_smi_csv(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            continue
        rows.append(
            {
                "index": parts[0],
                "name": parts[1],
                "temperature_c": parts[2],
                "utilization_gpu_percent": parts[3],
                "memory_total_mb": parts[4],
                "memory_used_mb": parts[5],
                "memory_free_mb": parts[6],
                "driver_version": parts[7],
            }
        )
    return rows


def _collect_torch_status(device_preference: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        torch = importlib.import_module("torch")
        out["import_ok"] = True
        out["torch_version"] = str(getattr(torch, "__version__", "unknown"))
        out["cuda_is_available"] = bool(torch.cuda.is_available())
        out["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        out["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        out["torch_hip_version"] = getattr(torch.version, "hip", None)
        try:
            mps_backend = getattr(torch.backends, "mps", None)
            out["mps_is_available"] = bool(mps_backend is not None and mps_backend.is_available() and mps_backend.is_built())
        except Exception:
            out["mps_is_available"] = False

        devices: list[dict[str, Any]] = []
        if torch.cuda.is_available():
            for idx in range(int(torch.cuda.device_count())):
                try:
                    props = torch.cuda.get_device_properties(idx)
                    devices.append(
                        {
                            "index": idx,
                            "name": str(props.name),
                            "total_memory_bytes": int(props.total_memory),
                            "multi_processor_count": int(getattr(props, "multi_processor_count", 0)),
                            "major": int(getattr(props, "major", 0)),
                            "minor": int(getattr(props, "minor", 0)),
                        }
                    )
                except Exception as e:
                    devices.append(
                        {
                            "index": idx,
                            "error_type": type(e).__name__,
                            "error": str(e),
                        }
                    )
            try:
                out["current_device"] = int(torch.cuda.current_device())
            except Exception:
                out["current_device"] = None
            try:
                out["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            except Exception:
                out["bf16_supported"] = False
            try:
                current = int(out["current_device"] or 0)
                out["memory_allocated_bytes"] = int(torch.cuda.memory_allocated(current))
                out["memory_reserved_bytes"] = int(torch.cuda.memory_reserved(current))
            except Exception:
                pass
        out["devices"] = devices

        runtime_adapt = importlib.import_module("runtime_adapt")
        profile = runtime_adapt.detect_runtime(device_preference)
        selected = runtime_adapt.select_device(device_preference)
        out["selected_device"] = str(selected)
        out["runtime_profile"] = profile.to_jsonable()
        return out
    except Exception as e:
        return {
            "import_ok": False,
            "error_type": type(e).__name__,
            "error": str(e),
        }


def _collect_system_gpu_status(verbose: bool) -> dict[str, Any]:
    nvidia_query = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.total,memory.used,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        verbose=verbose,
    )
    if nvidia_query.get("ok"):
        nvidia_query["parsed"] = _parse_nvidia_smi_csv(str(nvidia_query.get("stdout", "")))

    return {
        "nvidia_smi": nvidia_query,
        "hy_smi": _run_cmd(["hy-smi", "--showuse", "--showmemuse", "--showhcuutil"], verbose=verbose),
        "rocm_smi": _run_cmd(["rocm-smi", "--showuse", "--showmemuse", "--showproductname"], verbose=verbose),
    }


def _collect_env_status() -> dict[str, Any]:
    keys = [
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HF_ENDPOINT",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
    ]
    return {key: os.environ.get(key) for key in keys}


def _print_human_report(result: dict[str, Any]) -> None:
    torch_status = result.get("torch_status", {})
    runtime_profile = torch_status.get("runtime_profile", {})
    print("Torch")
    if not torch_status.get("import_ok"):
        print(f"  import: failed ({torch_status.get('error_type')}: {torch_status.get('error')})")
    else:
        print(f"  version: {torch_status.get('torch_version')}")
        print(f"  cuda available: {torch_status.get('cuda_is_available')}")
        print(f"  selected device: {torch_status.get('selected_device')}")
        print(f"  bf16 supported: {torch_status.get('bf16_supported')}")
        if runtime_profile:
            print(
                "  runtime: cpu_count={cpu} host_cpu_count={host} gpu_name={gpu} total_vram_bytes={vram}".format(
                    cpu=runtime_profile.get("cpu_count"),
                    host=runtime_profile.get("host_cpu_count"),
                    gpu=runtime_profile.get("gpu_name"),
                    vram=runtime_profile.get("total_vram_bytes"),
                )
            )
        devices = torch_status.get("devices", [])
        for dev in devices:
            print(
                "  device[{idx}]: {name} total_memory_bytes={mem}".format(
                    idx=dev.get("index"),
                    name=dev.get("name"),
                    mem=dev.get("total_memory_bytes"),
                )
            )

    print("System GPU Tools")
    for key in ("nvidia_smi", "hy_smi", "rocm_smi"):
        tool = result.get("system_gpu_status", {}).get(key, {})
        if not tool.get("available"):
            print(f"  {key}: not found")
            continue
        if tool.get("ok"):
            print(f"  {key}: ok")
            parsed = tool.get("parsed")
            if parsed:
                for row in parsed:
                    print(
                        "    gpu[{idx}]: {name} util={util}% mem={used}/{total} MB temp={temp}C".format(
                            idx=row.get("index"),
                            name=row.get("name"),
                            util=row.get("utilization_gpu_percent"),
                            used=row.get("memory_used_mb"),
                            total=row.get("memory_total_mb"),
                            temp=row.get("temperature_c"),
                        )
                    )
        else:
            print(f"  {key}: failed")

    print("Env")
    for key, value in result.get("env", {}).items():
        if value is not None:
            print(f"  {key}={value}")


def main() -> None:
    args = parse_args()
    result = {
        "torch_status": _collect_torch_status(args.device),
        "system_gpu_status": _collect_system_gpu_status(args.verbose),
        "env": _collect_env_status(),
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    _print_human_report(result)


if __name__ == "__main__":
    main()
