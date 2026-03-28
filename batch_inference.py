"""批量推理入口薄包装。

根目录 `batch_inference.py` 是主仓库对外暴露的稳定推理命令。
真正的推理实现位于 `gpu_stream_infer.py`，这里仅负责转发。
"""

def main() -> None:
    """跳转到当前主线批量推理入口。"""

    from batch_inference_motion_prosody import main as run_main

    run_main()


if __name__ == "__main__":
    main()
