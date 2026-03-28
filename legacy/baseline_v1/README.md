# Baseline V1 Archive

这里封存的是仓库第一版基线实现，包括旧的 `src/` 目录、训练脚本、单样本推理脚本和演示脚本。

- 用途：历史复现、轻量对照、排查与当前主线的行为差异。
- 状态：默认不再继续演进，不接受新功能开发。
- 入口：`python legacy/baseline_v1/train.py`、`python legacy/baseline_v1/inference.py`、`python legacy/baseline_v1/batch_inference.py`、`python legacy/baseline_v1/predecode_dataset.py`

当前主线已经切换到仓库根目录管线。

## 论文口径说明

- 这里的实现默认属于 archival baseline，不是仓库当前的论文级主线。
- 如果底层数据 manifest 被标记为 `scientific_validity=false` 或 `fatal_confound=true`，该路径产出的结果只能作为保守 benchmark 或历史对照，不能写成跨说话人情绪泛化结论。
- 正式论文级结果请以根目录 `gpu_stream` 主线和 [PAPER_GRADE_PROTOCOL.md](/Users/dailulu/projects/MMSER-Pytorch/PAPER_GRADE_PROTOCOL.md) 为准。
