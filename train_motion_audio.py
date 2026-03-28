"""训练入口兼容层。

历史上仓库曾经通过 `train_motion_audio.py` 启动训练。当前主线已经统一到
`gpu_stream_train.py`，这里保留一个最薄的兼容跳板，避免旧命令失效。
"""

def main() -> None:
    """跳转到 manifest 驱动的主线训练实现。"""

    from gpu_stream_train import main as run_main

    run_main()


if __name__ == "__main__":
    main()
