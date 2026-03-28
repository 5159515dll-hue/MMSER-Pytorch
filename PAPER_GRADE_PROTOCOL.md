# 论文级可信协议

本协议定义 `MMSER-Pytorch` 中哪些结果可以作为正式论文证据，哪些结果只能作为历史归档、探索性结果或 confounded benchmark。

## 适用范围

- 当前仓库唯一的正式主线是根目录 `gpu_stream` 路线。
- `legacy/baseline_v1/` 保留为历史归档和行为对照，不默认属于正式 headline 结果路径。
- 如果 manifest 的 `validity.scientific_validity=false`，结果仍可作为保守 benchmark 报告，但不能被写成跨说话人情绪泛化证据。

## 论文级正式结果必须同时满足

- 训练、验证、测试共用同一份 `manifest_sha256`。
- 测试集评测的 checkpoint 必须是 `checkpoints/best.pt`。
- 训练与推理产物里的 `paper_grade.eligible` 必须都是 `true`。
- 训练与推理产物必须写出：
  - `paper_contract`
  - `input_cache_contract`（如果使用了 `--input-cache`）
  - `paper_grade`
  - `provenance`
  - `validity`
- `paper_contract` 必须锁住关键架构/预处理口径，包括 `video_backbone` 和当前 `flow_encoder_variant`；架构语义变化后，旧 checkpoint 不能与新主线混评。
- 如果使用 `--input-cache`，缓存契约必须与当前 `manifest_sha256`、`dataset_kind`、`sample_rate`、`max_audio_sec`、`num_frames`、`text_model`、`max_text_len` 完全兼容；否则训练/推理应直接失败，而不是静默降级。
- 允许的缓存只有当前主线 `build_mainline_input_cache.py` 生成的输入缓存；它只能缓存波形、采样帧和 token，不能缓存 prosody、flow 或 encoder embedding。
- 多 seed headline 结果必须使用固定 seed 集：`13 17 23 42 3407`。
- 同一组多 seed 正式实验必须统一缓存路径策略：要么全部不使用 `--input-cache`，要么全部使用同一套兼容的 `input_cache_contract`；不能混跑。
- 多 seed 汇总必须报告：
  - `mean ± std`
  - `95% CI`
  - 配对显著性检验
- 默认 confirmatory 比较只允许相邻实验；若使用全对比较，必须做多重比较校正，并明确标注为 exploratory。

## 直接失去论文级资格的情况

- 使用 `--allow-incompatible-checkpoint`
- 推理时更换 manifest、task、speaker、text policy、ablation 或关键预处理参数而不与 checkpoint 完全一致
- 使用旧 `build_feature_cache.py` / `validate_cached_shards.py` / `--feature-cache` / `--cached-dataset`
- 使用任何缓存过的 prosody、legacy flow、audio embedding、text embedding 作为当前主线正式结果输入
- checkpoint 非严格兼容加载
- 训练或推理出现非有限值、样本级错误或缺 seed
- 多 seed 汇总中混入不同 manifest、不同 claim scope、不同 deterministic policy 或重复/缺失 seed

## 声明边界

- `scientific_validity=true`：可以在当前 manifest 假设下作为正式 benchmark 结果表述。
- `scientific_validity=false`：只能写成 confounded benchmark、within-speaker discrimination 或 archival comparison，禁止写成“情绪泛化已被证明”。
- `paper_grade.eligible=true` 只表示协议执行合格，不自动等价于“可宣称跨说话人泛化”；最终能宣称什么，仍由 `claim_scope` 和 `scientific_validity` 决定。
