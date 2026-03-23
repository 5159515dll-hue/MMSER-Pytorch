# MMSER-Pytorch

当前仓库主线已经完整移动到仓库根目录。日常训练、预解码、批量推理和 split manifest 构建都直接从根目录源码与入口执行：

- `python build_split_manifest.py --data-root databases --xlsx databases/video_databases.xlsx`
- `python predecode_dataset.py --data-root databases --xlsx databases/video_databases.xlsx`
- `python train.py --cached-dataset <cached_dir_or_shard>`
- `python batch_inference.py --xlsx databases/video_databases.xlsx --checkpoint outputs/motion_prosody/checkpoints/best.pt`
- `python validate_cached_shards.py --cached-dataset <cached_dir_or_shard>`

当前根目录下的 `train_motion_audio.py`、`predecode_motion_audio.py`、`batch_inference_motion_prosody.py`、`models.py`、`data.py` 等文件就是主线实现本体。`motion_prosody/` 目录现在只保留兼容导入层，`experiments/motion_prosody/` 只保留旧命令兼容层。第一版基线已经封存在 `legacy/baseline_v1/`，仅用于历史复现、轻量对照或回归排查，不再作为默认开发主线。

更详细的主线说明暂存于 `motion_prosody/README.md`，后续会继续收敛到根目录文档。
