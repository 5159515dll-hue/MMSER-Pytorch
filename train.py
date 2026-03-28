"""训练入口薄包装。

这个文件本身不实现训练逻辑，只是把根目录 `train.py` 这个稳定命令
转发到当前主线训练入口，方便用户始终使用同一条命令。
"""

def main() -> None:
    """跳转到当前主线训练入口。"""

    from train_motion_audio import main as run_main

    run_main()


if __name__ == "__main__":
    main()
