from typing import Optional


# 可选类型用于类型注解
from typing import Optional
import torch.nn as nn

# 导入PyTorch相关库
import torch

# 导入神经网络模块
import torch.nn as nn

# 导入音频处理库
import torchaudio

# 导入transformers库用于文本编码
from transformers import AutoModel


# 视频编码器：用于提取视频特征
# 参数：
#   name: 视频模型名称，目前仅支持'r2plus1d_18'
#   freeze: 是否冻结参数（不参与训练）
class VideoEncoder(nn.Module):
    def __init__(self, name: str = "r2plus1d_18", freeze: bool = False):
        super().__init__()  # 初始化父类
        if name != "r2plus1d_18":  # 只支持r2plus1d_18
            raise ValueError("Only r2plus1d_18 is supported by default")
        # 导入torchvision的视频模型
        from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

        weights = R2Plus1D_18_Weights.KINETICS400_V1  # 使用Kinetics400预训练权重
        self.model = r2plus1d_18(weights=weights)  # 加载模型
        self.feature_dim = self.model.fc.in_features  # 获取全连接层输入特征维度
        self.model.fc = nn.Identity()  # 去掉原有的分类头，输出特征
        if freeze:  # 是否冻结参数
            for p in self.model.parameters():
                p.requires_grad = False  # 冻结参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向传播，输入x为视频张量，输出特征
        return self.model(x)


# 音频编码器：用于提取音频特征
# 参数：
#   name: 音频模型名称，目前仅支持'wav2vec2_base'
#   freeze: 是否冻结参数
class AudioEncoder(nn.Module):
    def __init__(self, name: str = "wav2vec2_base", freeze: bool = True):
        super().__init__()  # 初始化父类
        if name != "wav2vec2_base":  # 只支持wav2vec2_base
            raise ValueError("Only wav2vec2_base is supported by default")
        bundle = torchaudio.pipelines.WAV2VEC2_BASE  # 获取wav2vec2配置
        self.model = bundle.get_model()  # 加载模型
        # 获取特征维度
        if hasattr(bundle, "_params") and hasattr(bundle._params, "encoder_embed_dim"):
            self.feature_dim = bundle._params.encoder_embed_dim  # 直接获取
        else:
            with torch.no_grad():
                dummy = torch.zeros(1, 16000)  # 构造假数据
                feats, _ = self.model.extract_features(dummy)  # 提取特征
                self.feature_dim = feats[-1].shape[-1]  # 获取最后一层特征维度
        if freeze:  # 是否冻结参数
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # 前向传播，输入wav为(B, L)的音频张量
        features, _ = self.model.extract_features(wav)  # 提取多层特征
        feats = features[-1]  # 取最后一层特征，形状(B, T, D)
        return feats.mean(dim=1)  # 对时间维做平均，输出(B, D)


# 文本编码器：用于提取文本特征
# 参数：
#   name: 文本模型名称，默认'bert-base-multilingual-cased'
#   freeze: 是否冻结参数
class TextEncoder(nn.Module):
    def __init__(self, name: str = "bert-base-multilingual-cased", freeze: bool = True):
        super().__init__()  # 初始化父类
        self.model = AutoModel.from_pretrained(name)  # 加载BERT模型
        self.feature_dim = self.model.config.hidden_size  # 获取隐藏层维度
        if freeze:  # 是否冻结参数
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, inputs: Optional[dict]) -> torch.Tensor:
        # 前向传播，inputs为tokenizer输出的字典
        if inputs is None:
            return None  # 若无文本输入，返回None
        outputs = self.model(**inputs)  # 得到模型输出
        return outputs.last_hidden_state[:, 0, :]  # 取[CLS]位置的特征


# 融合头：将多模态特征融合后进行分类和强度回归
# 参数：
#   input_dim: 输入特征维度
#   hidden_dim: 隐藏层维度
#   num_classes: 分类数
#   dropout: dropout比例
class FusionHead(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1
    ):
        super().__init__()  # 初始化父类
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 全连接层
            nn.ReLU(inplace=True),  # 激活函数
            nn.Dropout(dropout),  # dropout
            nn.Linear(hidden_dim, num_classes),  # 输出层
        )
        # 强度回归部分
        self.intensity = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # 输出为1维
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 前向传播，输入x为融合特征
        logits = self.classifier(x)  # 分类输出
        intensity = self.intensity(x).squeeze(-1)  # 强度输出，去掉最后一维
        return logits, intensity  # 返回分类和强度


# 情感识别主模型：融合视频、音频、文本三模态特征进行情感分类和强度预测
# 参数：
#   num_classes: 情感类别数
#   video_name: 视频编码器名称
#   audio_name: 音频编码器名称
#   text_name: 文本编码器名称
#   fusion_hidden: 融合头隐藏层维度
#   dropout: dropout比例
#   freeze_video/audio/text: 是否冻结各自编码器
class EmotionModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        video_name: str = "r2plus1d_18",
        audio_name: str = "wav2vec2_base",
        text_name: str = "bert-base-multilingual-cased",
        fusion_hidden: int = 512,
        dropout: float = 0.1,
        freeze_video: bool = False,
        freeze_audio: bool = True,
        freeze_text: bool = True,
    ):
        super().__init__()  # 初始化父类
        self.video_encoder = VideoEncoder(video_name, freeze_video)  # 视频编码器
        self.audio_encoder = AudioEncoder(audio_name, freeze_audio)  # 音频编码器
        self.text_encoder = TextEncoder(text_name, freeze_text)  # 文本编码器

        # 计算融合特征维度
        fusion_dim = (
            self.video_encoder.feature_dim
            + self.audio_encoder.feature_dim
            + self.text_encoder.feature_dim
        )
        self.fusion_head = FusionHead(
            fusion_dim, fusion_hidden, num_classes, dropout
        )  # 融合头

    def forward(
        self, video: torch.Tensor, audio: torch.Tensor, text_inputs: Optional[dict]
    ):
        # 前向传播
        v = self.video_encoder(video)  # 视频特征
        a = self.audio_encoder(audio)  # 音频特征
        t = self.text_encoder(text_inputs)  # 文本特征
        if t is None:
            # 若无文本输入，补零张量
            t = torch.zeros(
                v.shape[0],
                self.text_encoder.feature_dim,
                device=v.device,
                dtype=v.dtype,
            )
        fused = torch.cat([v, a, t], dim=1)  # 拼接多模态特征
        logits, intensity = self.fusion_head(fused)  # 得到分类和强度
        return logits, intensity  # 返回结果
