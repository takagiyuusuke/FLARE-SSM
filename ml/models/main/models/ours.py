import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from s5 import S5

def compute_time_phase(timestamp, mode="cos"):
    """
    与えられた timestamp から位相値を計算する関数。
    基準日 2009-01-01 を x=0、2020-01-01 を x=2π とする1周期（4017日）でマッピングする。
    
    Args:
        timestamp: pandas.Timestamp (または datetime 互換)
        mode: "cos" または "sin"。デフォルトは "cos" で、現在は -cos を返す。
        
    Returns:
        位相値（スカラー）: 現在は -cos(phase) の値
    """
    base_date = pd.to_datetime("2009-01-01")
    period_days = 4017.0  # 1周期の日数
    delta_days = (timestamp - base_date).total_seconds() / (3600 * 24)
    phase = 2 * np.pi * (delta_days / period_days)
    
    if mode == "cos":
        return -np.cos(phase)
    elif mode == "sin":
        return np.sin(phase)
    else:
        raise ValueError("Unsupported mode. Please choose 'cos' or 'sin'.")

class TimePhaseEmbedding(nn.Module):
    """
    タイムスタンプから計算した位相値（スカラー）をそのまま返すモジュール。
    従来は全結合層によるD次元埋め込みになっていたが、今回は１次元のスカラーを返す実装に変更。
    """
    def __init__(self, D, mode="cos"):
        # 引数 D はAPI互換のため残している（今回は使用しない）
        super().__init__()
        self.mode = mode

    def forward(self, timestamps, device=None):
        """
        バッチ中の各サンプルに対して、timestamp から位相値を計算する。
        
        Args:
            timestamps: list や1次元のテンソル（または複数の pd.Timestamp）
            device: 出力テンソルのデバイス（例: "cuda:0" または "cpu"）
            
        Returns:
            位相値: Tensor of shape [batch, 1]
        """
        phase_values = []
        for ts in timestamps:
            phase = compute_time_phase(ts, mode=self.mode)
            phase_values.append(phase)
        
        # デバイスを指定してテンソルを作成
        phase_tensor = torch.tensor(phase_values, dtype=torch.float32, device=device).unsqueeze(-1)
        return phase_tensor
class Ours(nn.Module):
    def __init__(
        self,
        D=64,  # Feature dimension (scalar)
        drop_path_rate=0.3,
        layer_scale_init_value=1e-6,
        L=128,       # Sequence length
        L_SSE=3,     # number of SolarSpatialEncoder layers
        L_LT=2,      # number of LongRangeTemporalSSM layers
        dropout_rates=None  # Dropout rates for each module
    ) -> None:
        super().__init__()

        self.L = L
        self.drop_path_rate = drop_path_rate
        self.D = D
        self.layer_scale_init_value = layer_scale_init_value
        self.L_SSE = L_SSE
        self.L_LT = L_LT

        if dropout_rates is None:
            dropout_rates = {
                "sse": drop_path_rate,
                "dwcm": drop_path_rate,
                "stssm": drop_path_rate,
                "ltssm": drop_path_rate,
                "mixing_ssm": drop_path_rate,
                "head": 0.5
            }
        self.dropout_rates = dropout_rates

        self.sse = SolarSpatialEncoder(
            4,  # input_channel (k)
            D=D,
            drop_path_rate=dropout_rates["sse"],
            dwcm_dropout_rate=dropout_rates["dwcm"],
            stssm_dropout_rate=dropout_rates["stssm"],
            layer_scale_init_value=layer_scale_init_value,
            L=L,
            L_SSE=L_SSE,
        )

        self.lt_ssm = LongRangeTemporalSSM(
            D=D,
            L=L, 
            L_LT=L_LT,
            dropout_rate=dropout_rates["ltssm"]
        )

        self.mixing_ssm_block = SSMBlock(D, depth=1, dropout_rate=dropout_rates["mixing_ssm"])

        # 分類ヘッド内部でGlobal Average Pooling後にタイムスタンプ情報と結合する
        self.classification_head = ClassificationHead(D, D//2, 32, dropout_rate=dropout_rates["head"])

    def freeze_feature_extractor(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.classification_head.parameters():
            param.requires_grad = True
            
    def forward(self, x, x_history=None, timestamps=None):
        # Solar Spatial Encoder
        h_sse = self.sse(x)  # [batch, L, D]
        # Long-range Temporal SSM
        h_lt = self.lt_ssm(x_history)  # [batch, L, D]
        # 時系列方向に連結
        merged_features = torch.cat([h_sse, h_lt], dim=1)  # [batch, 2*L, D]
        mixed_features = self.mixing_ssm_block(merged_features)
        logits, hidden_features = self.classification_head(mixed_features, timestamps, return_hidden=True)
        return logits, hidden_features

class ClassificationHead(nn.Module):
    def __init__(self, d_model, hidden_dim, mlp_hidden_size, dropout_rate=0.5):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.time_phase_emb = TimePhaseEmbedding(d_model, mode="cos")
        
        # FFN を個々の層に分割する
        self.fc1 = nn.Linear(d_model + 1, mlp_hidden_size)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(mlp_hidden_size, hidden_dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_dim, 4)  # 4-class classification
    
    def forward(self, x, timestamps, return_hidden=False):
        # x: [batch, seq_len, d_model] を [batch, d_model, seq_len] に変換
        x = x.transpose(1, 2)
        pooled = self.gap(x).squeeze(-1)  # [batch, d_model]

        if timestamps is not None:
            # pooled のデバイスを取得して渡す
            phase_scalar = self.time_phase_emb(timestamps, device=pooled.device)  # [batch, 1]
        else:
            phase_scalar = torch.zeros(pooled.size(0), 1, device=pooled.device)
        
        concat_feature = torch.cat([pooled, phase_scalar], dim=-1)  # [batch, d_model+1]
        
        # FC層の逐次処理
        x1 = self.fc1(concat_feature)
        x1 = self.act1(x1)
        x1 = self.drop1(x1)
        
        hidden = self.fc2(x1)      # ここが中間層の出力（hidden_dim 次元）
        hidden = self.act2(hidden)
        hidden = self.drop2(hidden)
        
        logits = self.fc3(hidden)
        
        if return_hidden:
            return logits, hidden
        else:
            return logits


class SpatioTemporalSSM(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None, dropout_rate=0.3):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.s5_layer = SSMBlock(dim, 1, dropout_rate)

    def forward(self, x):
        B, C = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_s5 = self.s5_layer(x_flat)
        out = x_s5.transpose(-1, -2).reshape(B, C, *img_dims)
        return out

class DepthwiseChannelSelectiveModule(nn.Module):
    def __init__(self, D, dropout_rate=0.3):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.conv3d = nn.Sequential(
            nn.Conv3d(D, D, kernel_size=3, padding=1, groups=D, padding_mode="replicate"),
            nn.InstanceNorm3d(D),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        self.conv2d = nn.Sequential(
            nn.Conv3d(D, D, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=D, padding_mode="replicate"),
            nn.InstanceNorm3d(D),
            nn.ReLU(),
        )
        self.image_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)), 
            nn.Conv3d(D, D, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(D, D, kernel_size=1),
            nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            nn.Conv3d(D, D, 1),
            nn.InstanceNorm3d(D),
            nn.ReLU(),
        )

    def forward(self, h_ds):
        identity = h_ds
        feat3d = self.conv3d(h_ds)
        feat2d = self.conv2d(h_ds)
        F_fused = feat3d + feat2d
        W = self.image_attention(F_fused)
        weighted_features = F_fused * W
        refined = self.refine(weighted_features)
        h_dcs = refined + identity
        return h_dcs

class SSMLayer(nn.Module):
    def __init__(self, dim, dropout_rate=0.2, init_value=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.s5 = S5(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        residual = x
        x_norm = self.norm1(x)
        x_s5 = self.s5(x_norm)
        x = residual + self.dropout(x_s5)
        residual = x
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = residual + self.dropout(mlp_out)
        return x

class SSMBlock(nn.Module):
    def __init__(self, dim, depth=2, dropout_rate=0.4):
        super().__init__()
        self.layers = nn.ModuleList([SSMLayer(dim, dropout_rate) for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SolarSpatialEncoder(nn.Module):
    def __init__(
        self,
        in_chans=1,  # k: input channels
        D=64,        # feature dimension
        drop_path_rate=0.0,
        dwcm_dropout_rate=0.0,
        stssm_dropout_rate=0.0,
        layer_scale_init_value=1e-6,
        out_indices=[0, 1, 2],
        L=128,       # sequence length in output
        L_SSE=3,     # number of layers
    ):
        super().__init__()
        self.L = L
        self.D = D
        self.L_SSE = L_SSE
        dims = [D] * (L_SSE + 1)
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(in_chans, D, kernel_size=3, stride=(1, 4, 4), padding=(1, 1, 1)),
            nn.BatchNorm3d(D),
            nn.ReLU(),
            nn.Dropout(drop_path_rate)
        )
        self.downsample_layers.append(stem)
        for l in range(L_SSE):
            if l < 3:
                downsample_layer = nn.Sequential(
                    nn.BatchNorm3d(D),
                    nn.Conv3d(D, D, kernel_size=3, stride=(1, 2, 2), padding=(1, 1, 1)),
                    nn.ReLU(),
                    nn.Dropout(drop_path_rate)
                )
            else:
                downsample_layer = nn.Sequential(
                    nn.BatchNorm3d(D),
                    nn.Conv3d(D, D, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1)),
                    nn.ReLU(),
                    nn.Dropout(drop_path_rate)
                )
            self.downsample_layers.append(downsample_layer)
        self.dcsm_modules = nn.ModuleList()
        self.stssm_modules = nn.ModuleList()
        for l in range(L_SSE):
            dcsm = DepthwiseChannelSelectiveModule(D, dropout_rate=dwcm_dropout_rate)
            stssm = SpatioTemporalSSM(dim=D, dropout_rate=stssm_dropout_rate)
            self.dcsm_modules.append(dcsm)
            self.stssm_modules.append(stssm)
        self.out_indices = out_indices
        self.dropout = nn.Dropout(drop_path_rate)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(D * 10, D * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(D * 4),
            nn.ReLU(),
            nn.Dropout(drop_path_rate),
            nn.Conv2d(D * 4, D * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(D * 2),
            nn.ReLU(),
            nn.Dropout(drop_path_rate),
            nn.Conv2d(D * 2, L, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        h_sse = self.downsample_layers[0](x)
        for l in range(self.L_SSE):
            h_ds = self.downsample_layers[l+1](h_sse)
            h_dcs = self.dcsm_modules[l](h_ds)
            h_sse = self.stssm_modules[l](h_dcs)
        batch_size = h_sse.shape[0]
        h_sse = h_sse.reshape(batch_size, self.D * 10, h_sse.shape[3], h_sse.shape[4])
        h_sse = self.conv_layers(h_sse)
        h_sse = h_sse.reshape(batch_size, self.L, self.D)
        return h_sse

class LongRangeTemporalSSM(nn.Module):
    def __init__(self, D=64, L=128, L_LT=2, dropout_rate=0.3):
        super().__init__()
        self.D = D
        self.L = L
        self.ssm_block = SSMBlock(self.L, depth=L_LT, dropout_rate=dropout_rate)
        self.conv1d_layers = nn.Sequential(
            nn.Conv1d(self.L, self.L, kernel_size=5, stride=2, padding=2),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Conv1d(self.L, self.L, kernel_size=5, stride=2, padding=2),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.D)

    def forward(self, x):
        x = self.ssm_block(x)
        x = x.transpose(1, 2)  # [batch, D, seq_len]
        x = self.conv1d_layers(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)  # [batch, D, L]
        return x