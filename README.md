# MMSER-Pytorch

当前仓库主线已经收敛到 manifest 驱动的 `gpu_stream` 路线。日常使用只保留这几个根目录入口：

- `python build_split_manifest.py --dataset-kind meld --data-root <media_root> --metadata-root <metadata_root> --output <manifest.json>`
- `python prepare_dataset_media.py --dataset-kind meld --split-manifest <manifest.json> --subset all`
- `python train.py --split-manifest <manifest.json> --output-dir outputs/motion_prosody`
- `python batch_inference.py --split-manifest <manifest.json> --subset val --checkpoint outputs/motion_prosody/checkpoints/best.pt`

主线训练和推理由 [train.py](/Users/dailulu/projects/MMSER-Pytorch/train.py) 和 [batch_inference.py](/Users/dailulu/projects/MMSER-Pytorch/batch_inference.py) 进入，具体实现收敛在 [gpu_stream_train.py](/Users/dailulu/projects/MMSER-Pytorch/gpu_stream_train.py) 和 [gpu_stream_infer.py](/Users/dailulu/projects/MMSER-Pytorch/gpu_stream_infer.py)。`motion_prosody/` 只保留给 `legacy/baseline_v1/` 的最小导入兼容层。

以下脚本已经退出主线支持面，只保留迁移提示：

- `predecode_dataset.py`
- `build_feature_cache.py`
- `validate_cached_shards.py`
- `run_scientific_suite.py`
