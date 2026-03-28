from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hf_compat import ensure_transformers_torch_compat

class FlowVideoEncoder(nn.Module):
    """基于光流序列的轻量 3D CNN 编码器。

        设计目的
        - 输入为光流/运动特征序列（通常由光流或运动场堆叠得到），通过 3D 卷积在时间与空间上联合建模，输出固定维度的全局视频表征。

        输入张量约定
        - `flow`: 形状为 (B, 3, T, H, W)
            - B: batch size
            - 3: 通道数（这里假设为 3；例如可表示 (u, v, magnitude) 或其它 3 通道运动特征）
            - T: 时间长度/帧数
            - H, W: 空间分辨率

        输出
        - `video_emb`: 形状为 (B, out_dim)

        备注
        - 网络结构包含下采样与 `AdaptiveAvgPool3d`，因此能适配不同的 (T, H, W) 输入尺寸。
    """

    def __init__(self, out_dim: int = 256):
        """初始化视频编码器。

        Args:
            out_dim: 输出嵌入维度。
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.proj = nn.Linear(128, out_dim)
        self.out_dim = int(out_dim)

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            flow: 光流/运动输入张量，形状 (B, 3, T, H, W)。

        Returns:
            视频嵌入，形状 (B, out_dim)。
        """
        x = self.net(flow)
        # AdaptiveAvgPool3d((1,1,1)) 后为 (B, C, 1, 1, 1)，展平成 (B, C)
        x = x.flatten(1)
        return self.proj(x)


class VideoMAEEncoder(nn.Module):
    """VideoMAE 视频编码器（Transformers 预训练模型）。

    期望输入
    - rgb: (B, T, 3, H, W)，像素值为 [0, 1] 的 float。
    """

    def __init__(self, model_name: str = "MCG-NJU/videomae-large", freeze: bool = True):
        """初始化 VideoMAE 编码器并注册 RGB 归一化缓冲区。

        Args:
            model_name: HuggingFace 上的 VideoMAE 模型名或本地目录。
            freeze: 是否冻结预训练主干。
        """
        super().__init__()
        try:
            ensure_transformers_torch_compat()
            from transformers import AutoModel
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "transformers is required for VideoMAE encoder. Install with: pip install transformers"
            ) from e

        self.model_name = str(model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.out_dim = int(getattr(self.model.config, "hidden_size", 768))
        self.expected_num_frames = int(getattr(self.model.config, "num_frames", 16))
        image_size = getattr(self.model.config, "image_size", 224)
        if isinstance(image_size, (tuple, list)):
            if len(image_size) >= 2:
                self.expected_image_size = (int(image_size[0]), int(image_size[1]))
            elif len(image_size) == 1:
                side = int(image_size[0])
                self.expected_image_size = (side, side)
            else:
                self.expected_image_size = (224, 224)
        else:
            side = int(image_size)
            self.expected_image_size = (side, side)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        # ImageNet mean/std for normalization
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

    def _adapt_clip_shape(self, rgb: torch.Tensor) -> torch.Tensor:
        """把缓存 clip 对齐到当前 VideoMAE 的期望时空尺寸。

        背景
        - 预处理阶段为了兼容 flow 分支，默认会缓存 64 帧 RGB。
        - 但 `MCG-NJU/videomae-large` 这类预训练模型通常固定期望 16 帧。
        - 如果直接把 64 帧送进去，patch/token 数会和位置编码长度不匹配，
          从而触发 `size of tensor a ... must match tensor b ...`。

        这里直接在模型前向里做一次时空重采样，避免要求用户重跑整套预处理。
        """

        x = rgb.to(dtype=torch.float32)
        target_t = max(1, int(self.expected_num_frames))
        target_h, target_w = self.expected_image_size
        need_resize = (
            int(x.shape[1]) != target_t
            or int(x.shape[3]) != int(target_h)
            or int(x.shape[4]) != int(target_w)
        )
        if not need_resize:
            return x
        # F.interpolate 处理 5D 张量时要求形状是 (B, C, T, H, W)，
        # 所以先把时间维和通道维交换，再做三线性插值。
        x_5d = x.permute(0, 2, 1, 3, 4)
        x_5d = F.interpolate(
            x_5d,
            size=(target_t, int(target_h), int(target_w)),
            mode="trilinear",
            align_corners=False,
        )
        return x_5d.permute(0, 2, 1, 3, 4).contiguous()

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """编码 RGB clip。

        这里先做 ImageNet 均值/方差归一化，再交给 VideoMAE。
        如果模型本身没有显式 `pooler_output`，就退回到 CLS token，
        再不行才做 mean pooling。
        """

        # rgb: (B, T, 3, H, W) in [0,1]
        x = self._adapt_clip_shape(rgb)
        x = (x - self._mean) / self._std
        out = self.model(pixel_values=x)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        # fallback: CLS token if available, otherwise mean pooling
        hidden = out.last_hidden_state
        if hidden is None:
            raise RuntimeError("videomae_output_missing")
        if hidden.shape[1] >= 1:
            return hidden[:, 0, :]
        return hidden.mean(dim=1)


class Wav2Vec2Encoder(nn.Module):
    """基于 torchaudio 预训练管线的 wav2vec2 编码器。"""

    def __init__(self, freeze: bool = True):
        """Wav2Vec2 音频编码器（使用 torchaudio 的预训练管线）。

        特征聚合
        - 从最后一层特征序列 `x`（形状 (B, T, D)）计算 mean 与 std，拼接得到 (B, 2D)。
        - 这样能在不引入额外注意力/池化层的情况下，获得对时序分布的粗粒度刻画。

        采样率与输入
        - torchaudio 的 `WAV2VEC2_BASE` 通常期望 16kHz 单声道波形。
        - 本模块假设输入 `wav` 已经是 float 波形（非 PCM int），且已归一化到合理范围（例如 [-1, 1]）。

        Args:
            freeze: 是否冻结 wav2vec2 参数（推理/小数据集训练时常用）。

        Raises:
            RuntimeError: torchaudio 导入失败（常见原因是 torch/torchaudio ABI 版本不匹配）。
        """
        super().__init__()
        try:
            import torchaudio
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Failed to import torchaudio. This is usually caused by a torch/torchaudio version (ABI) mismatch. "
                "Fix by reinstalling matching versions, e.g.:\n"
                "  python3 -m pip uninstall -y torchaudio torchvision torch\n"
                "  python3 -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2\n"
                f"Original error: {type(e).__name__}: {e}"
            ) from e

        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.model = bundle.get_model()
        params = getattr(bundle, "_params", None)
        embed_dim = int(getattr(params, "encoder_embed_dim", 768))
        # 采用 mean+std 池化，因此输出维度为 2*D。
        self.out_dim = 2 * embed_dim
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            wav: 波形张量，形状 (B, L)。
                - B: batch size
                - L: 采样点数（与采样率、时长相关）

        Returns:
            音频嵌入，形状 (B, 2*D)。其中 D 为 wav2vec2 编码器隐藏维度（BASE 通常为 768）。
        """
        feats, _ = self.model.extract_features(wav)
        # `extract_features` 返回多层特征列表；取最后一层作为语义更强的表征。
        x = feats[-1]  # (B, T, D)
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        return torch.cat([mean, std], dim=1)


class HFAudioEncoder(nn.Module):
    """HuggingFace 预训练音频编码器（如 WavLM-Large）。

    输入
    - wav: (B, L) float32/float16
    - lengths: (B,) 每条音频的有效长度（可选，用于掩码池化）
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-large",
        freeze: bool = True,
        revision: Optional[str] = None,
        use_safetensors: bool = True,
    ):
        """初始化 HuggingFace 音频编码器。

        Args:
            model_name: 例如 `microsoft/wavlm-large`。
            freeze: 是否冻结参数，仅训练上层融合头。
            revision: HuggingFace revision / commit hash。用于固定到一个
                已提供 `model.safetensors` 的版本。
            use_safetensors: 是否强制只使用 safetensors 权重文件。
        """
        super().__init__()
        try:
            ensure_transformers_torch_compat()
            from transformers import AutoModel
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "transformers is required for HF audio encoder. Install with: pip install transformers"
            ) from e

        self.model_name = str(model_name)
        self.revision = str(revision).strip() if revision is not None and str(revision).strip() else None
        self.use_safetensors = bool(use_safetensors)
        load_kwargs: dict[str, object] = {"use_safetensors": self.use_safetensors}
        if self.revision is not None:
            load_kwargs["revision"] = self.revision
        try:
            self.model = AutoModel.from_pretrained(self.model_name, **load_kwargs)
        except Exception as e:  # pragma: no cover
            revision_hint = self.revision if self.revision is not None else "<default>"
            raise RuntimeError(
                "Failed to load HuggingFace audio model with safetensors. "
                f"model={self.model_name!r}, revision={revision_hint!r}, "
                f"use_safetensors={self.use_safetensors}. "
                "If the default branch only provides pytorch_model.bin, pass "
                "--audio-model-revision pointing at a revision that contains model.safetensors."
            ) from e
        hidden_size = int(getattr(self.model.config, "hidden_size", 768))
        # 前向里对时序 hidden state 做了 mean+std 池化并拼接，因此最终音频维度是 2 * hidden_size。
        self.out_dim = 2 * hidden_size

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def _get_output_lengths(self, lengths: torch.Tensor) -> torch.Tensor | None:
        """把输入波形长度映射到编码器输出时间步长度。

        某些 HuggingFace 音频模型内部有卷积下采样，因此输出序列长度
        会短于输入长度。这个函数用于把 `audio_lens` 从原始采样点域
        映射到 hidden-state 的时间轴上。
        """

        fn = getattr(self.model, "_get_feat_extract_output_lengths", None)
        if fn is None:
            return None
        try:
            return fn(lengths)
        except Exception:
            return None

    def forward(self, wav: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """编码音频并做 mean+std 池化。

        当 `lengths` 可用时，会构造输出时域掩码，只对有效帧做统计；
        否则默认把整个时间轴都纳入池化。
        """

        x = wav.to(dtype=torch.float32)
        attention_mask = None
        if lengths is not None:
            max_len = int(x.shape[1])
            idx = torch.arange(max_len, device=x.device)[None, :]
            attention_mask = idx < lengths[:, None]
        out = self.model(input_values=x, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        if hidden is None:
            raise RuntimeError("audio_output_missing")

        if lengths is None:
            mean = hidden.mean(dim=1)
            std = hidden.std(dim=1, unbiased=False)
            return torch.cat([mean, std], dim=1)

        out_lens = self._get_output_lengths(lengths)
        if out_lens is None:
            mean = hidden.mean(dim=1)
            std = hidden.std(dim=1, unbiased=False)
            return torch.cat([mean, std], dim=1)

        t = hidden.shape[1]
        mask = torch.arange(t, device=hidden.device)[None, :] < out_lens[:, None]
        mask_f = mask.to(hidden.dtype)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        mean = (hidden * mask_f[:, :, None]).sum(dim=1) / denom[:, None]
        # 方差也必须只在有效帧上统计，所以这里用同一个掩码对平方误差求和。
        var = ((hidden - mean[:, None, :]) ** 2 * mask_f[:, :, None]).sum(dim=1) / denom[:, None]
        std = torch.sqrt(var.clamp_min(1e-6))
        return torch.cat([mean, std], dim=1)


class ProsodyMLP(nn.Module):
    """把低维韵律统计映射到可与其它模态融合的嵌入空间。"""

    def __init__(self, in_dim: int = 10, out_dim: int = 64):
        """韵律/统计特征的 MLP 编码器。

        典型输入
        - 这里的 `in_dim=10` 只是默认值，常用于承载诸如：基频(F0)均值/方差、能量统计、语速/停顿统计等低维韵律特征。

        Args:
            in_dim: 输入韵律特征维度。
            out_dim: 输出嵌入维度。
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 韵律特征张量，形状 (B, in_dim)。

        Returns:
            韵律嵌入，形状 (B, out_dim)。
        """
        return self.net(x)


class MbertTextEncoder(nn.Module):
    """基于 HuggingFace `AutoModel` 的文本编码器封装。"""

    def __init__(self, model_name: str = "bert-base-multilingual-cased", freeze: bool = True):
        """mBERT 文本编码器（HuggingFace Transformers）。

        编码方式
        - 使用 `AutoModel`（不包含分类头），输出 `last_hidden_state`。
        - 采用 CLS pooling：取第 0 个 token 的向量作为句级表征。

        输入约定
        - `text_inputs` 应为 HuggingFace tokenizer 的输出字典，常见键：
          - `input_ids`: (B, S)
          - `attention_mask`: (B, S)
          - 可选：`token_type_ids`: (B, S)

        Args:
            model_name: 预训练模型名或本地路径。
            freeze: 是否冻结文本编码器参数。

        Raises:
            RuntimeError: 未安装 transformers。
        """
        super().__init__()
        try:
            ensure_transformers_torch_compat()
            from transformers import AutoModel
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "transformers is required for text encoder. Install with: pip install transformers"
            ) from e

        self.model_name = str(model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.out_dim = int(getattr(self.model.config, "hidden_size", 768))

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, text_inputs: Optional[dict[str, torch.Tensor]]) -> torch.Tensor:
        """前向传播。

        Args:
            text_inputs: tokenizer 输出的张量字典。
                - 当为 None/空字典时，本函数会抛错；上层融合模型可选择用零向量替代文本模态。

        Returns:
            文本嵌入，形状 (B, hidden_size)。

        Raises:
            ValueError: 当 `text_inputs` 为空时抛出（用于提示调用方缺失文本输入）。
        """
        if not text_inputs:
            # Caller should handle zeros.
            raise ValueError("text_inputs_missing")
        out = self.model(**text_inputs)
        # CLS pooling
        return out.last_hidden_state[:, 0, :]


class FusionClassifier(nn.Module):
    """主工程的多模态融合网络。

    它负责把视频、音频、韵律、文本四路表示统一到同一个判别空间，
    并输出情绪分类结果，以及可选的连续强度回归结果。
    """

    def __init__(
        self,
        num_classes: int = 7,
        video_dim: int = 256,
        audio_dim: Optional[int] = None,
        prosody_dim: int = 64,
        text_model: str = "xlm-roberta-large",
        freeze_text: bool = True,
        hidden: int = 512,
        dropout: float = 0.1,
        audio_model: str = "microsoft/wavlm-large",
        audio_model_revision: Optional[str] = None,
        freeze_audio: bool = True,
        freeze_video: bool = False,
        freeze_flow: Optional[bool] = None,
        freeze_rgb: Optional[bool] = None,
        freeze_prosody: bool = False,
        intensity_head: bool = True,
        video_backbone: str = "dual",
        video_model: str = "MCG-NJU/videomae-large",
        fusion_mode: str = "gated_text",
        modality_dropout: float = 0.1,
        gate_temperature: float = 1.0,
        gate_scale: float = 1.0,
        delta_scale: float = 1.0,
    ):
        """多模态融合分类器（视频 + 语音 wav2vec2 + 韵律 + 文本 mBERT）。

                融合策略（默认）
                - 门控位移融合（gated_text）：将非语言模态（视频/音频/韵律）映射成“位移向量”，
                    用门控信号对文本表征进行残差式微调，然后仅使用调制后的文本向量进行分类/回归。
                - 该策略可缓解强文本主导导致的弱模态被淹没问题。
                - 可选 concat：兼容传统拼接融合。

        冻结策略
        - `freeze_audio`/`freeze_text` 常用于小数据或只训练融合头。
        - `freeze_video`/`freeze_prosody` 用于控制其它模态是否参与训练。

        Args:
            num_classes: 分类类别数。
            video_dim: 视频编码器输出维度。
            audio_dim: 音频嵌入维度（若为 None，则取 wav2vec2 的 `self.audio.out_dim`）。
            prosody_dim: 韵律编码器输出维度。
            text_model: 文本编码器预训练模型名/路径。
            freeze_text: 是否冻结文本编码器。
            hidden: 融合 MLP 隐藏层维度。
            dropout: dropout 概率。
            audio_model: 音频编码器模型名或本地目录。
            audio_model_revision: HuggingFace 音频模型 revision / commit hash。
            freeze_audio: 是否冻结音频编码器。
            freeze_video: 是否冻结视频编码器。
            freeze_prosody: 是否冻结韵律编码器。
            intensity_head: 是否启用强度回归头。
        """
        super().__init__()
        self.video_backbone = str(video_backbone)
        self.fusion_mode = str(fusion_mode)
        self.modality_dropout = float(modality_dropout)
        self.gate_temperature = float(gate_temperature)
        self.gate_scale = float(gate_scale)
        self.delta_scale = float(delta_scale)
        self._freeze_audio = bool(freeze_audio)
        self._freeze_text = bool(freeze_text)
        self._freeze_prosody = bool(freeze_prosody)
        freeze_flow = bool(freeze_video) if freeze_flow is None else bool(freeze_flow)
        freeze_rgb = bool(freeze_video) if freeze_rgb is None else bool(freeze_rgb)
        self._freeze_flow = bool(freeze_flow)
        self._freeze_rgb = bool(freeze_rgb)
        if self.video_backbone == "videomae":
            self.video = VideoMAEEncoder(model_name=video_model, freeze=freeze_rgb)
            self.video_flow = None
            self.video_rgb = None
        elif self.video_backbone == "dual":
            self.video = None
            self.video_flow = FlowVideoEncoder(out_dim=video_dim)
            self.video_rgb = VideoMAEEncoder(model_name=video_model, freeze=freeze_rgb)
        else:
            self.video = FlowVideoEncoder(out_dim=video_dim)
            self.video_flow = None
            self.video_rgb = None

        if str(audio_model) == "wav2vec2_base":
            self.audio = Wav2Vec2Encoder(freeze=freeze_audio)
        else:
            self.audio = HFAudioEncoder(
                model_name=audio_model,
                freeze=freeze_audio,
                revision=audio_model_revision,
                use_safetensors=True,
            )
        self.prosody = ProsodyMLP(in_dim=10, out_dim=prosody_dim)
        self.text = MbertTextEncoder(model_name=text_model, freeze=freeze_text)

        if self.video is not None and self.video_backbone != "videomae" and freeze_flow:
            for p in self.video.parameters():
                p.requires_grad = False
        if self.video_flow is not None and freeze_flow:
            for p in self.video_flow.parameters():
                p.requires_grad = False
        if self.video is not None and self.video_backbone == "videomae" and freeze_rgb:
            for p in self.video.parameters():
                p.requires_grad = False
        if self.video_rgb is not None and freeze_rgb:
            for p in self.video_rgb.parameters():
                p.requires_grad = False
        if freeze_prosody:
            for p in self.prosody.parameters():
                p.requires_grad = False

        if self.video_backbone == "dual":
            video_dim_final = int(getattr(self.video_flow, "out_dim", video_dim)) + int(
                getattr(self.video_rgb, "out_dim", video_dim)
            )
        else:
            video_dim_final = int(getattr(self.video, "out_dim", video_dim))
        self.video_out_dim = int(video_dim_final)
        audio_dim_final = int(audio_dim) if audio_dim is not None else int(self.audio.out_dim)
        non_text_dim = int(video_dim_final) + int(audio_dim_final) + int(prosody_dim)

        if self.fusion_mode == "gated_text":
            # Non-text -> delta(text) with gating.
            text_dim = int(self.text.out_dim)
            self.delta_proj = nn.Sequential(
                nn.Linear(non_text_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, text_dim),
            )
            self.gate_proj = nn.Sequential(
                nn.Linear(non_text_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, text_dim),
                nn.Sigmoid(),
            )
            fusion_in_dim = text_dim
        else:
            self.delta_proj = None
            self.gate_proj = None
            fusion_in_dim = non_text_dim + int(self.text.out_dim)

        self.head = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

        # Optional regression head for continuous emotion intensity.
        # Always constructed by default, but only returned/used when requested.
        self.intensity_head_enabled = bool(intensity_head)
        if self.intensity_head_enabled:
            self.reg_head = nn.Sequential(
                nn.Linear(fusion_in_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        else:
            self.reg_head = None

    def train(self, mode: bool = True) -> "FusionClassifier":
        """训练态切换时，保持被冻结编码器处于 eval 模式。"""

        super().train(mode)
        if mode:
            if self._freeze_audio:
                self.audio.eval()
            if self._freeze_text:
                self.text.eval()
            if self._freeze_prosody:
                self.prosody.eval()
            if self.video is not None and self.video_backbone == "videomae" and self._freeze_rgb:
                self.video.eval()
            if self.video is not None and self.video_backbone != "videomae" and self._freeze_flow:
                self.video.eval()
            if self.video_flow is not None and self._freeze_flow:
                self.video_flow.eval()
            if self.video_rgb is not None and self._freeze_rgb:
                self.video_rgb.eval()
        return self

    def _maybe_drop_modality(self, x: torch.Tensor) -> torch.Tensor:
        """按样本随机丢弃一个非文本模态。

        这里的 dropout 不是逐元素，而是“整条模态向量”级别：
        对同一个样本，要么完整保留该模态，要么整个置零。
        这样更接近真实缺模态/弱模态场景。
        """

        if not self.training:
            return x
        p = float(self.modality_dropout)
        if p <= 0.0:
            return x
        keep = (torch.rand((x.shape[0], 1), device=x.device) > p).to(x.dtype)
        return x * keep

    def forward(
        self,
        flow: Optional[torch.Tensor],
        wav: torch.Tensor,
        prosody: torch.Tensor,
        text_inputs: Optional[dict[str, torch.Tensor]] = None,
        return_intensity: bool = False,
        rgb: Optional[torch.Tensor] = None,
        audio_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """前向传播：融合多模态并输出分类 logits（可选强度回归）。

        Args:
            flow: 视频/光流输入，形状 (B, 3, T, H, W)。
            wav: 音频波形输入，形状 (B, L)。建议 16kHz。
            prosody: 韵律特征输入，形状 (B, 10)（默认与 `ProsodyMLP(in_dim=10)` 对齐）。
            text_inputs: tokenizer 输出字典；若为 None，则用零向量代替文本模态。
            return_intensity: 是否同时返回强度回归结果。
        """
        if self.video_backbone == "videomae":
            if rgb is None:
                raise ValueError("rgb_missing")
            if self.video is None:
                raise RuntimeError("video_encoder_missing")
            v = self.video(rgb)
        elif self.video_backbone == "dual":
            if flow is None:
                raise ValueError("flow_missing")
            if self.video_flow is None:
                raise RuntimeError("flow_encoder_missing")
            v_flow = self.video_flow(flow)
            if rgb is None:
                raise ValueError("rgb_missing")
            if self.video_rgb is None:
                raise RuntimeError("rgb_encoder_missing")
            v_rgb = self.video_rgb(rgb)
            v = torch.cat([v_flow, v_rgb], dim=1)
        else:
            if flow is None:
                raise ValueError("flow_missing")
            if self.video is None:
                raise RuntimeError("video_encoder_missing")
            v = self.video(flow)

        if audio_lens is not None:
            try:
                a = self.audio(wav, lengths=audio_lens)
            except TypeError:
                a = self.audio(wav)
        else:
            a = self.audio(wav)
        p = self.prosody(prosody)

        # Modality dropout (non-text) for robustness
        v = self._maybe_drop_modality(v)
        a = self._maybe_drop_modality(a)
        p = self._maybe_drop_modality(p)

        if text_inputs is None:
            # 允许在无文本场景下运行：使用零向量作为文本模态占位。
            t = torch.zeros(
                (v.shape[0], int(self.text.out_dim)),
                device=v.device,
                dtype=v.dtype,
            )
        else:
            # 让文本嵌入的 dtype 与其它模态对齐（例如 AMP/混合精度场景）。
            t = self.text(text_inputs).to(dtype=v.dtype)

        if self.fusion_mode == "gated_text":
            non_text = torch.cat([v, a, p], dim=1)
            if self.delta_proj is None or self.gate_proj is None:
                raise RuntimeError("gated_fusion_missing")
            delta = self.delta_proj(non_text)
            gate_logits = self.gate_proj(non_text)
            if self.gate_temperature <= 0:
                raise ValueError("gate_temperature_must_be_positive")
            # 门控位移融合的核心公式：
            #   gate = sigmoid(gate_logits / T)
            #   t'   = t + (gate_scale * gate) * (delta_scale * delta)
            #
            # 直觉上：
            # - `delta` 决定“往哪个方向修正文本向量”；
            # - `gate` 决定“修正多少”；
            # - `temperature` 控制门的平滑/尖锐程度。
            gate = torch.sigmoid(gate_logits / float(self.gate_temperature))
            t = t + (float(self.gate_scale) * gate) * (float(self.delta_scale) * delta)
            x = t
        else:
            x = torch.cat([v, a, p, t], dim=1)
        logits = self.head(x)
        if not bool(return_intensity):
            return logits

        if self.reg_head is None:
            raise RuntimeError("intensity_head_disabled")
        pred_intensity = self.reg_head(x).squeeze(1)
        return logits, pred_intensity
