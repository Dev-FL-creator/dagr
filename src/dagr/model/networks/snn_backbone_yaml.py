#3D聚合
import torch
import torch.nn as nn
import torch.nn.functional as F

from dagr.model.snn.snn_yaml_builder import YAMLBackbone


# 可学习的时序聚合模块
class TemporalConvAgg(nn.Module):
    """
    可学习的时序聚合模块（TCM）。

    输入:  [B, C, T, H, W]
    输出:  [B, C_out, H, W]

    结构:
      - 3D depthwise conv 建模时空局部关系
      - 3D conv 融合并调整通道
      - 可选 SE 门控（按时间维重标定）
      - 1x1x1 的 time_proj 学习每个时间切片权重
      - 对 T 求和聚合
      - 最后用 2D 1x1 proj2d 调整到目标通道
    """
    def __init__(self, in_channels, out_channels=None, k_t=3, use_se=True, depthwise=True, dropout_p=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        pad_t = k_t // 2

        groups = in_channels if depthwise else 1

        # 3D: 同时建模时间与空间（不降采样）
        self.conv3d_1 = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=(k_t, 3, 3),
            padding=(pad_t, 1, 1),
            groups=groups, bias=False
        )
        self.bn3d_1 = nn.BatchNorm3d(in_channels)

        # 仅空间融合，稳定数值/换通道
        self.conv3d_2 = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            groups=1, bias=False
        )
        self.bn3d_2 = nn.BatchNorm3d(out_channels)

        self.use_se = use_se
        if use_se:
            # 针对时间维的 SE 门控：对每个像素位置学习 T 维权重, 学习每个时间切片的重要性
            r = max(8, out_channels // 8)
            self.se_fc1 = nn.Conv3d(out_channels, r, kernel_size=1)
            self.se_fc2 = nn.Conv3d(r, out_channels, kernel_size=1)

        self.dropout = nn.Dropout3d(dropout_p) if dropout_p > 0 else nn.Identity()

        # 学习每个切片权重
        self.time_proj = nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False)

        # 最后 2D 1x1 做通道整形
        self.proj2d = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x_bcthw):
        # x: [B, C, T, H, W]
        y = F.relu(self.bn3d_1(self.conv3d_1(x_bcthw)))
        y = self.dropout(y)
        y = F.relu(self.bn3d_2(self.conv3d_2(y)))  # [B, C_out, T, H, W]

        if self.use_se:
            # 对 H,W 求均值，保留 T，得到时序门控
            s = y.mean(dim=(3, 4), keepdim=True)   # [B, C_out, T, 1, 1]
            s = F.relu(self.se_fc1(s))
            s = torch.sigmoid(self.se_fc2(s))      # [B, C_out, T, 1, 1]
            y = y * s

        y = self.time_proj(y)                      # [B, C_out, T, H, W],学习每个切片权重
        y = y.sum(dim=2)                           # 聚合 T → [B, C_out, H, W]
        y = self.proj2d(y)                         # [B, C_out, H, W]
        return y


class SNNBackboneYAMLWrapper(nn.Module):
    """
    包装 YAMLBackbone：
      - YAMLBackbone 负责事件体素化与特征提取，输出 p2,p3,p4,p5 形如 [T, B, C, H, W]
      - 本包装器在 p4/p5 上做 T 维聚合（TCM），输出标准 FPN-like 的二维特征 [B, C, H, W]
    """
    def __init__(self, args, height: int, width: int, yaml_path: str, scale: str = 's'):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        temporal_bins = getattr(args, 'snn_temporal_bins', 4)

        self.backbone = YAMLBackbone(
            yaml_path=yaml_path, scale=scale,
            in_ch=2, height=self.height, width=self.width,
            temporal_bins=temporal_bins
        )
        # 下游检测头接口
        self.out_channels = [256, 512]
        self.strides = [16, 32]
        self.num_scales = 2
        self.use_image = False
        self.is_snn = True
        self.num_classes = dict(dsec=2, ncaltech101=100).get(getattr(args, 'dataset', 'dsec'), 2)

        # TCM 聚合模块（第一次 forward 时基于实际通道创建） 
        # YAMLBackbone 返回 p4/p5 形状是 [T, B, C, H, W]，聚合到 [B, C, H, W]
        self._tcm = nn.ModuleDict()  # 键：'p4','p5'

        # 可通过 args 控制
        self.tcm_enabled = getattr(args, 'tcm_enabled', True)
        self.tcm_kt = int(getattr(args, 'tcm_k_t', 3))
        self.tcm_use_se = bool(getattr(args, 'tcm_use_se', True))
        self.tcm_depthwise = bool(getattr(args, 'tcm_depthwise', True))
        self.tcm_dropout = float(getattr(args, 'tcm_dropout', 0.0))

    def get_output_sizes(self):
        sizes = []
        for s in self.strides:
            sizes.append([max(1, self.height // s), max(1, self.width // s)])
        return [[h, w] for h, w in sizes]

    # 内部：把 [T,B,C,H,W] 聚成 [B,C,H,W]（优先用 TCM，退化为 t-mean）
    def _aggregate_time(self, x_tbchw, level_name: str):
        if x_tbchw is None:
            return None
        # x_tbchw: [T, B, C, H, W]  →  [B, C, T, H, W]
        x_bcthw = x_tbchw.permute(1, 2, 0, 3, 4).contiguous()

        # T=1 时直接挤掉时间维
        if x_bcthw.size(2) == 1:
            return x_bcthw.squeeze(2)

        if not self.tcm_enabled:
            # 兼容开关：关闭时回退到简单 t-mean
            return x_tbchw.mean(dim=0)

        key = level_name
        if key not in self._tcm:
            in_ch = x_bcthw.size(1)
            self._tcm[key] = TemporalConvAgg(
                in_channels=in_ch,
                out_channels=in_ch,          # 维持通道数不变
                k_t=self.tcm_kt,
                use_se=self.tcm_use_se,
                depthwise=self.tcm_depthwise,
                dropout_p=self.tcm_dropout
            ).to(x_bcthw.device, dtype=x_bcthw.dtype)

        return self._tcm[key](x_bcthw)

    def forward(self, data, reset: bool = True):
        # 让 backbone 知道目标分辨率
        setattr(data, 'meta_height', self.height)
        setattr(data, 'meta_width', self.width)

        # YAMLBackbone 约定输出: p2,p3,p4,p5 皆为 [T,B,C,H,W] 或 None
        p2, p3, p4, p5 = self.backbone(data)

        # 用可学习的 Temporal Conv 聚合，替换原来的 mean over T ——
        p4_bchw = self._aggregate_time(p4, 'p4')
        p5_bchw = self._aggregate_time(p5, 'p5')

        ret = []
        if p4_bchw is not None:
            ret.append(p4_bchw)
        if p5_bchw is not None:
            ret.append(p5_bchw)
        return ret

    def forward_time(self, data, reset: bool = True):
        """
        保持原语义：返回保留时间维的多尺度特征，形状为 [T,B,C,H,W]。
        """
        setattr(data, 'meta_height', self.height)
        setattr(data, 'meta_width', self.width)
        p2, p3, p4, p5 = self.backbone(data)
        ret = {}
        if p2 is not None: ret["p2"] = p2
        if p3 is not None: ret["p3"] = p3
        if p4 is not None: ret["p4"] = p4
        if p5 is not None: ret["p5"] = p5
        return ret



#简单T聚合

# import torch
# import torch.nn as nn

# from dagr.model.snn.snn_yaml_builder import YAMLBackbone


# class SNNBackboneYAMLWrapper(nn.Module):
#     def __init__(self, args, height: int, width: int, yaml_path: str, scale: str = 's'):
#         super().__init__()
#         self.height = int(height)
#         self.width = int(width)
#         temporal_bins = getattr(args, 'snn_temporal_bins', 4)
#         self.backbone = YAMLBackbone(yaml_path=yaml_path, scale=scale, in_ch=2, height=self.height, width=self.width, temporal_bins=temporal_bins)

#         self.out_channels = [256, 512]
#         self.strides = [16, 32]
#         self.num_scales = 2
#         self.use_image = False
#         self.is_snn = True
#         self.num_classes = dict(dsec=2, ncaltech101=100).get(getattr(args, 'dataset', 'dsec'), 2)

#     def get_output_sizes(self):
#         sizes = []
#         for s in self.strides:
#             sizes.append([max(1, self.height // s), max(1, self.width // s)])
#         return [[h, w] for h, w in sizes]


#     def forward(self, data, reset: bool = True):
#         # pass Data to backbone; MS_GetT_Voxel will voxelize to [T,B,2,H,W]
#         setattr(data, 'meta_height', self.height)
#         setattr(data, 'meta_width', self.width)
#         p2, p3, p4, p5 = self.backbone(data)
#         # aggregate time: mean over T -> BCHW
#         p4_bchw = p4.mean(dim=0) if p4 is not None else None
#         p5_bchw = p5.mean(dim=0) if p5 is not None else None
#         ret = []
#         if p4_bchw is not None:
#             ret.append(p4_bchw)
#         if p5_bchw is not None:
#             ret.append(p5_bchw)
#         return ret

#     def forward_time(self, data, reset: bool = True):
#         """
#         Return multi-scale features with temporal dimension preserved.
#         Output: dict with keys 'p4','p5' and values [T,B,C,H,W].
#         """
#         setattr(data, 'meta_height', self.height)
#         setattr(data, 'meta_width', self.width)
#         p2, p3, p4, p5 = self.backbone(data)
#         ret = {}
#         if p2 is not None:
#             ret["p2"] = p2
#         if p3 is not None:
#             ret["p3"] = p3
#         if p4 is not None:
#             ret["p4"] = p4
#         if p5 is not None:
#             ret["p5"] = p5
#         return ret




#复杂三分支聚合

# import torch
# import torch.nn as nn

# from dagr.model.snn.snn_yaml_builder import YAMLBackbone


# def _make_group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
#     """
#     Uses min(max_groups, num_channels) to avoid 'num_channels % num_groups != 0' errors
#     when channels are small or not divisible by 32.
#     """
#     num_groups = max(1, min(max_groups, num_channels))
#     return nn.GroupNorm(num_groups, num_channels)


# class TemporalAggHybrid(nn.Module):
#     """
#     Parallel fusion over the temporal axis (T) with stability tweaks.

#     What it does (for p shaped [T, B, C, H, W]):
#       1) mean  : temporal average (stable/steady context)
#       2) std   : temporal standard deviation (variation / motion energy)
#       3) attn  : channel-wise temporal attention (softmax across T per channel)

#     The three branches are concatenated along channel dim -> [B, 3C, H, W],
#     then projected back to [B, C, H, W] by a lightweight 1×1 Conv (+ optional GN + SiLU).
#     Two stability features are included:
#       - sqrt(T) scaling: prevents magnitude drop when slicing events into more bins.
#       - residual-to-mean: start close to your old "mean over time" baseline, then learn gains.

#     Inputs
#     ------
#     p : Tensor
#         Either [T, B, C, H, W] or [B, C, H, W] (the latter auto-expanded to T=1).

#     Output
#     ------
#     Tensor
#         [B, C, H, W] — same spatial resolution and channels as a single temporal slice.
#     """
#     def __init__(
#         self,
#         c_in: int,
#         use_gn: bool = True,
#         residual: bool = True,
#         gamma_init: float = 1.0,
#         zero_init_proj: bool = True,
#         scale_by_sqrt_t: bool = True,
#     ):
#         super().__init__()

#         # 1×1 projection: [B, 3C, H, W] -> [B, C, H, W]
#         layers = [nn.Conv2d(3 * c_in, c_in, kernel_size=1, bias=False)]
#         if use_gn:
#             layers += [_make_group_norm(c_in)]
#         layers += [nn.SiLU()]
#         self.proj = nn.Sequential(*layers)

#         # (Optional) start exactly at the "mean" behavior:
#         # zero init makes proj ≈ 0 at the beginning (so output ≈ residual branch).
#         if zero_init_proj:
#             nn.init.zeros_(self.proj[0].weight)

#         # Residual to mean (learnable gate). Keeps behavior close to baseline at init,
#         # then allows learning temporal gains on top.
#         self.residual = residual
#         self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

#         # If True, multiply input feature sequence by sqrt(T) before statistics.
#         # This keeps magnitude comparable when you move from T=1 to T>1.
#         self.scale_by_sqrt_t = scale_by_sqrt_t

#     @staticmethod
#     def _attn_pool(p: torch.Tensor) -> torch.Tensor:
#         """
#         Channel-wise temporal attention pooling:
#           g = GAP_spatial(p) -> [T, B, C]
#           w = softmax_T(g)   -> [T, B, C]
#           sum_t(p * w)       -> [B, C, H, W]
#         """
#         g = p.mean(dim=(3, 4))                               # [T, B, C]
#         w = torch.softmax(g, dim=0).unsqueeze(-1).unsqueeze(-1)  # [T, B, C, 1, 1]
#         return (p * w).sum(dim=0)                            # [B, C, H, W]

#     def forward(self, p: torch.Tensor) -> torch.Tensor:
#         # Accept both [B, C, H, W] and [T, B, C, H, W]
#         if p.dim() == 4:
#             p = p.unsqueeze(0)  # -> [1, B, C, H, W]
#         T = int(p.shape[0])

#         # Keep magnitude healthy when T>1 (mitigates "temporal mean dilution").
#         if self.scale_by_sqrt_t and T > 1:
#             p = p * (T ** 0.5)

#         # Temporal statistics
#         # mean: steady context; std: variation strength; attn: adaptive emphasis over T
#         mu  = p.mean(dim=0)                                   # [B, C, H, W]
#         std = (p.var(dim=0, unbiased=False) + 1e-6).sqrt()    # [B, C, H, W]
#         att = self._attn_pool(p)                               # [B, C, H, W]

#         # Fuse then project back to C channels
#         x = torch.cat([mu, std, att], dim=1)                  # [B, 3C, H, W]
#         out = self.proj(x)                                    # [B, C, H, W]

#         # Residual to mean (optional). At init, output ≈ mean if proj is zero-initialized.
#         return out + (self.gamma * mu if self.residual else 0.0)


# class SNNBackboneYAMLWrapper(nn.Module):
#     """
#       - passes spatial metadata to YAMLBackbone (so it can voxelize events),
#       - receives multi-scale temporal features shaped roughly [T, B, C, H/s, W/s],
#       - aggregates the temporal axis with TemporalAggHybrid,
#       - returns standard FPN-like feature maps as [B, C_l, H/s_l, W/s_l] for the head.
#     Notes:
#     * YAMLBackbone is expected to voxelize events into [T, B, 2, H, W], then extract features.
#     * We keep output channels/strides consistent with your downstream head (e.g., FCOS/YOLOX).
#     * TemporalAggHybrid keeps time information (steady + variation + adaptive emphasis)
#       while remaining lightweight and stable for small batch training.
#     """
#     def __init__(self, args, height: int, width: int, yaml_path: str, scale: str = 's'):
#         super().__init__()
#         self.height = int(height)
#         self.width  = int(width)

#         temporal_bins = int(getattr(args, 'snn_temporal_bins', 4))

#         # Backbone: will voxelize to [T, B, 2, H, W] internally and produce multi-scale features
#         self.backbone = YAMLBackbone(
#             yaml_path=yaml_path, scale=scale,
#             in_ch=2, height=self.height, width=self.width,
#             temporal_bins=temporal_bins
#         )

#         # Metadata for the detection head
#         self.out_channels = [256, 512]  # p4, p5 channels
#         self.strides      = [16, 32]
#         self.num_scales   = 2
#         self.use_image    = False
#         self.is_snn       = True
#         self.num_classes  = dict(dsec=2, ncaltech101=100).get(getattr(args, 'dataset', 'dsec'), 2)

#         # Temporal aggregation per scale (keeps interface unchanged: 256/512 channels out)
#         self.agg_p4 = TemporalAggHybrid(
#             c_in=256, use_gn=True,
#             residual=True, gamma_init=1.0,
#             zero_init_proj=True, scale_by_sqrt_t=True
#         )
#         self.agg_p5 = TemporalAggHybrid(
#             c_in=512, use_gn=True,
#             residual=True, gamma_init=1.0,
#             zero_init_proj=True, scale_by_sqrt_t=True
#         )

#     def get_output_sizes(self):
#         """
#         returns [[H/stride, W/stride], ...] for each scale.
#         """
#         sizes = []
#         for s in self.strides:
#             sizes.append([max(1, self.height // s), max(1, self.width // s)])
#         return [[h, w] for h, w in sizes]

#     def forward(self, data, reset: bool = True):
#         # Provide spatial metadata so voxelizer can size the grids
#         setattr(data, 'meta_height', self.height)
#         setattr(data, 'meta_width',  self.width)

#         # Backbone is expected to output: p3, p4, p5 each shaped ~ [T, B, C, H/s, W/s]
#         p3, p4, p5 = self.backbone(data)  # p3 is unused here

#         # Aggregate over T with hybrid fusion (steady + variation + attention) + stability tricks
#         p4_bchw = self.agg_p4(p4)  # [B, 256, H/16, W/16]
#         p5_bchw = self.agg_p5(p5)  # [B, 512, H/32, W/32]

#         return [p4_bchw, p5_bchw]

