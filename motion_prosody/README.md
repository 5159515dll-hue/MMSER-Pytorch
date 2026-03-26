<!--
 * @Description:
 * @Author: Dai Lu Lu
 * @version: 1.0
 * @Date: 2026-01-13 15:01:57
 * @LastEditors: Dai Lu Lu
 * @LastEditTime: 2026-03-26 15:12:00
-->
# Motion + Prosody（历史说明）

`motion_prosody/` 不再承载当前主线实现。

当前仓库的日常入口只有仓库根目录这 4 个脚本：
- `build_split_manifest.py`
- `prepare_dataset_media.py`
- `train.py`
- `batch_inference.py`

当前主线只支持 manifest 驱动的 `gpu_stream` 运行方式，不再支持旧的缓存式主线流程。下面这些脚本已经退役，不应再作为当前主线命令使用：
- `predecode_dataset.py`
- `build_feature_cache.py`
- `validate_cached_shards.py`
- `run_scientific_suite.py`

## 这个目录现在保留什么

- `motion_prosody/__init__.py` 和少量同名模块仅作为兼容导入层存在，主要服务于 `legacy/baseline_v1/`
- 本目录中的若干 Markdown 文档属于历史设计记录，不代表当前主线的推荐命令

## 如果你要跑当前主线

请直接查看仓库根目录文档：
- `README.md`
- `MELD_基线与上限实验运行手册.md`

当前正确流程是：
1. `python build_split_manifest.py ...`
2. `python prepare_dataset_media.py ...`
3. `python train.py --split-manifest ...`
4. `python batch_inference.py --split-manifest ...`

## 如果你要复现实验档案

历史第一代 baseline 已归档到：
- `legacy/baseline_v1/`

只有在确实需要复现旧实验结果时，才应进入该目录使用它自己的脚本和说明；不要把这些旧命令和当前主线混用。
