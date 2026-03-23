'''
Description: 
Author: Dai Lu Lu
version: 1.0
Date: 2026-01-25 17:03:59
LastEditors: Dai Lu Lu
LastEditTime: 2026-02-03 18:00:22
'''
from __future__ import annotations

import argparse
import re
from pathlib import Path

# 匹配版本比较符（PEP 440 常见）
_VERSION_OP_RE = re.compile(r"\s*(===|==|~=|!=|<=|>=|<|>)\s*")


def strip_requirement_line(line: str) -> str:
    """
    将 requirements.txt 的一行去掉版本/环境标记，只保留名称。
    - 保留空行、注释行
    - 以 '-' 开头的 pip 选项行原样保留（如 -r, --find-links 等）
    - 支持 'name @ url' 形式，返回 name
    - 支持 'name==1.2.3; python_version>="3.10"'，返回 name
    """
    raw = line.rstrip("\n")
    s = raw.strip()

    if not s or s.startswith("#"):
        return raw  # 保留原样
    if s.startswith("-"):
        return raw  # pip 选项行不改

    # 去掉环境标记（; 后面的部分）
    s = s.split(";", 1)[0].strip()

    # 处理 direct reference: "name @ https://..."
    if "@" in s:
        left, right = s.split("@", 1)
        name = left.strip()
        if name:
            return name

    # 处理常规版本比较：name==1.2.3 / name>=1.0 等
    m = _VERSION_OP_RE.search(s)
    if m:
        return s[: m.start()].strip()

    return s  # 没版本就原样返回（例如仅写了包名）


def main() -> int:
    parser = argparse.ArgumentParser(description="去掉 requirements.txt 中的版本号，只保留包名")
    parser.add_argument(
        "-i",
        "--input",
        default="requirements.txt",
        help="输入 requirements 文件路径（默认：requirements.txt）",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="requirements_names.txt",
        help="输出文件路径（默认：requirements_names.txt）",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="直接覆盖输入文件（忽略 -o/--output）",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"未找到输入文件：{in_path}")

    out_path = in_path if args.inplace else Path(args.output)

    lines = in_path.read_text(encoding="utf-8").splitlines(True)
    out_lines = [strip_requirement_line(line) + "\n" if not line.endswith("\n") else strip_requirement_line(line) + ("\n" if strip_requirement_line(line) == line.rstrip("\n") and not line.endswith("\n") else "") for line in lines]

    # 上面那句为了稳妥保持换行，这里再规范化：按原 lines 写入时逐行写
    normalized = []
    for line in lines:
        stripped = strip_requirement_line(line)
        normalized.append(stripped + ("\n" if line.endswith("\n") else ""))

    out_path.write_text("".join(normalized), encoding="utf-8")
    print(f"已输出：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())