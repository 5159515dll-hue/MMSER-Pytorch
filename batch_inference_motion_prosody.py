"""批量推理入口兼容层。

这个文件保留旧入口名，但实际逻辑已经全部切换到 `gpu_stream_infer.py`。
"""

def main() -> None:
    """跳转到 manifest 驱动的主线推理实现。"""

    from gpu_stream_infer import main as run_main

    run_main()


if __name__ == "__main__":
    main()
