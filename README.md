# MMSER-Pytorch

当前仓库主线已经收敛到 manifest 驱动的 `gpu_stream` 路线。日常使用只保留这几个根目录入口：

- `python build_split_manifest.py --dataset-kind meld --data-root <media_root> --metadata-root <metadata_root> --output <manifest.json>`
- `python prepare_dataset_media.py --dataset-kind meld --split-manifest <manifest.json> --subset all`
- `python build_mainline_input_cache.py --split-manifest <manifest.json> --output-dir outputs/input_cache/<name>`
- `python train.py --split-manifest <manifest.json> --output-dir outputs/motion_prosody`
- `python batch_inference.py --split-manifest <manifest.json> --subset val --checkpoint outputs/motion_prosody/checkpoints/best.pt`

主线训练和推理由 [train.py](/Users/dailulu/projects/MMSER-Pytorch/train.py) 和 [batch_inference.py](/Users/dailulu/projects/MMSER-Pytorch/batch_inference.py) 进入，具体实现收敛在 [gpu_stream_train.py](/Users/dailulu/projects/MMSER-Pytorch/gpu_stream_train.py) 和 [gpu_stream_infer.py](/Users/dailulu/projects/MMSER-Pytorch/gpu_stream_infer.py)。`motion_prosody/` 只保留给 `legacy/baseline_v1/` 的最小导入兼容层。

如果你要把实验结果写进论文、答辩或正式对外表格，请先看：

- [PAPER_GRADE_PROTOCOL.md](/Users/dailulu/projects/MMSER-Pytorch/PAPER_GRADE_PROTOCOL.md)
- [MELD_基线与上限实验运行手册.md](/Users/dailulu/projects/MMSER-Pytorch/MELD_%E5%9F%BA%E7%BA%BF%E4%B8%8E%E4%B8%8A%E9%99%90%E5%AE%9E%E9%AA%8C%E8%BF%90%E8%A1%8C%E6%89%8B%E5%86%8C.md)
- [主工程认知地图.md](/Users/dailulu/projects/MMSER-Pytorch/%E4%B8%BB%E5%B7%A5%E7%A8%8B%E8%AE%A4%E7%9F%A5%E5%9C%B0%E5%9B%BE.md)

当前主线的正式论文级结果只接受：

- 同一 `manifest_sha256`
- `checkpoints/best.pt` 的验证/测试评测
- 与当前代码一致的 `paper_contract`，包括 `video_backbone` / `flow_encoder_variant`
- 如果使用缓存，则必须是 `build_mainline_input_cache.py` 生成且 `input_cache_contract` 完全兼容的主线输入缓存；视频缓存的正式表示是已经按主线规则预处理好的 RGB clip
- `paper_grade.eligible=true` 的训练、推理与多 seed 汇总产物

对于 `legacy/baseline_v1/` 或任何 `scientific_validity=false` 的结果，只能按 archival / confounded benchmark 解释，不能写成跨说话人情绪泛化证据。

以下脚本已经退出主线支持面，只保留迁移提示：

- `predecode_dataset.py`
- `build_feature_cache.py`
- `validate_cached_shards.py`
- `run_scientific_suite.py`
