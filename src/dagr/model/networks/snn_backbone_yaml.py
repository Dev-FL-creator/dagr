# 参考SODFORMER中TDTE设计的时序Transformer聚合
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from dagr.model.snn.snn_yaml_builder import YAMLBackbone


class TemporalAttention(nn.Module):
    def __init__(self, channels, num_heads=4, dropout=0.0):
        super().__init__()
        assert channels % num_heads == 0
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        # QKV投影层
        self.q_proj = nn.Conv3d(channels, channels, 1, bias=False)
        self.k_proj = nn.Conv3d(channels, channels, 1, bias=False) 
        self.v_proj = nn.Conv3d(channels, channels, 1, bias=False)
        self.output_proj = nn.Conv3d(channels, channels, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = (self.head_dim ** -0.5)
        
        # 小权重初始化，避免训练不稳定
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        # 数值检查，避免崩溃
        if torch.isnan(x).any() or torch.isinf(x).any():
            return x
        if H * W < 4 or T == 1:
            return x
        
        try:
            # 生成查询、键、值
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            
            if torch.isnan(q).any() or torch.isnan(k).any() or torch.isnan(v).any():
                return x
            
            # 重塑为多头格式: [B, heads, head_dim, T, H, W]
            q = q.contiguous().view(B, self.num_heads, self.head_dim, T, H, W)
            k = k.contiguous().view(B, self.num_heads, self.head_dim, T, H, W) 
            v = v.contiguous().view(B, self.num_heads, self.head_dim, T, H, W)
            
            output = self._safe_attention(q, k, v, B, T, H, W)
            
            if output is None:
                return x
            
            output = output.contiguous().view(B, C, T, H, W)
            output = self.output_proj(output)
            
            return output
            
        except Exception:
            return x
    
    def _safe_attention(self, q, k, v, B, T, H, W):
        try:
            spatial_size = H * W
            
            # 维度重排: [B, heads, H, W, T, head_dim]
            q = q.permute(0, 1, 4, 5, 3, 2).contiguous()
            k = k.permute(0, 1, 4, 5, 3, 2).contiguous()
            v = v.permute(0, 1, 4, 5, 3, 2).contiguous()
            
            batch_size = B * self.num_heads * spatial_size
            
            if batch_size <= 0 or batch_size > 1e8:
                return None
            
            # 展平为批量矩阵乘法格式: [batch_size, T, head_dim]
            # 每个空间位置独立处理时间序列
            q = q.view(batch_size, T, self.head_dim)
            k = k.view(batch_size, T, self.head_dim)
            v = v.view(batch_size, T, self.head_dim)
            
            assert q.shape == k.shape == v.shape
            
            if torch.isnan(q).any() or torch.isnan(k).any() or torch.isnan(v).any():
                return None
                
            # 计算注意力分数矩阵 [batch_size, T, T]
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_scores = torch.clamp(attn_scores, min=-10.0, max=10.0)  # 数值稳定性
            
            if torch.isnan(attn_scores).any() or torch.isinf(attn_scores).any():
                return None
            
            # softmax得到注意力权重，应用到值上
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            
            out = torch.matmul(attn_probs, v)
            
            # 重塑回原始空间格式
            out = out.view(B, self.num_heads, spatial_size, T, self.head_dim)
            out = out.view(B, self.num_heads, H, W, T, self.head_dim)
            out = out.permute(0, 1, 5, 4, 2, 3).contiguous()
            
            return out
            
        except Exception:
            return None


class TemporalTransformerEncoder(nn.Module):
    """时序Transformer编码器，类似SODFormer的TDTE"""
    def __init__(self, channels, num_heads=4, num_layers=2, dropout=0.0):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        
        self.attention_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm1_layers = nn.ModuleList()
        self.norm2_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            # 时序注意力层
            self.attention_layers.append(
                TemporalAttention(channels, num_heads, dropout)
            )
            
            # 前馈网络: C -> 4C -> C
            ffn = nn.Sequential(
                nn.Conv3d(channels, channels * 4, 1),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv3d(channels * 4, channels, 1),
                nn.Dropout(dropout)
            )
            self.ffn_layers.append(ffn)
            
            # 组归一化
            num_groups = min(32, channels)
            self.norm1_layers.append(nn.GroupNorm(num_groups, channels))
            self.norm2_layers.append(nn.GroupNorm(num_groups, channels))
        
        # 时间维度池化
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.final_proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        """
        输入: [B, C, T, H, W]
        输出: [B, C, H, W]
        """
        B, C, T, H, W = x.shape
        
        # 多层处理
        for i in range(self.num_layers):
            try:
                # 子层1: 层归一化 -> 时序注意力 -> 残差连接
                residual = x
                x_norm = self.norm1_layers[i](x)
                x_attn = self.attention_layers[i](x_norm)
                x = residual + x_attn * 0.1
                
                # 子层2: 层归一化 -> FFN -> 残差连接
                residual = x
                x_norm = self.norm2_layers[i](x)
                x_ffn = self.ffn_layers[i](x_norm)
                x = residual + x_ffn * 0.1
                
            except Exception:
                continue
        
        # 时间维度聚合: [B, C, T, H, W] -> [B, C, H, W]
        x = self.temporal_pool(x).squeeze(2)
        x = self.final_proj(x)
        
        return x


class SNNBackboneYAMLWrapper(nn.Module):
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
        
        self.out_channels = [256, 512]
        self.strides = [16, 32]
        self.num_scales = 2
        self.use_image = False
        self.is_snn = True
        self.num_classes = dict(dsec=2, ncaltech101=100).get(getattr(args, 'dataset', 'dsec'), 2)

        self._temporal_modules = nn.ModuleDict()
        
        # 参数配置
        self.num_heads = 4
        self.num_layers = 1
        self.dropout = 0.0

    def get_output_sizes(self):
        sizes = []
        for s in self.strides:
            sizes.append([max(1, self.height // s), max(1, self.width // s)])
        return sizes

    def _aggregate_temporal_features(self, x_tbchw, level_name: str):
        """时序特征聚合"""
        if x_tbchw is None:
            return None
            
        # 维度转换: [T, B, C, H, W] -> [B, C, T, H, W]
        x_bcthw = x_tbchw.permute(1, 2, 0, 3, 4).contiguous()

        if x_bcthw.size(2) == 1:
            return x_bcthw.squeeze(2)
        
        key = level_name
        if key not in self._temporal_modules:
            in_ch = x_bcthw.size(1)
            try:
                self._temporal_modules[key] = TemporalTransformerEncoder(
                    channels=in_ch,
                    num_heads=self.num_heads,
                    num_layers=self.num_layers,
                    dropout=self.dropout
                ).to(x_bcthw.device, dtype=x_bcthw.dtype)
            except Exception:
                return x_tbchw.mean(dim=0)

        try:
            return self._temporal_modules[key](x_bcthw)
        except Exception:
            return x_tbchw.mean(dim=0)

    def forward(self, data, reset: bool = True):
        setattr(data, 'meta_height', self.height)
        setattr(data, 'meta_width', self.width)

        p2, p3, p4, p5 = self.backbone(data)

        # 时序聚合
        p4_bchw = self._aggregate_temporal_features(p4, 'p4')
        p5_bchw = self._aggregate_temporal_features(p5, 'p5')

        ret = []
        if p4_bchw is not None:
            ret.append(p4_bchw)
        if p5_bchw is not None:
            ret.append(p5_bchw)
        return ret

    def forward_time(self, data, reset: bool = True):
        """保留时间维度的前向传播，用于双分支融合"""
        setattr(data, 'meta_height', self.height)
        setattr(data, 'meta_width', self.width)
        p2, p3, p4, p5 = self.backbone(data)
        ret = {}
        if p2 is not None: ret["p2"] = p2
        if p3 is not None: ret["p3"] = p3
        if p4 is not None: ret["p4"] = p4
        if p5 is not None: ret["p5"] = p5
        return ret
    



# #3D聚合
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from dagr.model.snn.snn_yaml_builder import YAMLBackbone


# # 可学习的时序聚合模块
# class TemporalConvAgg(nn.Module):
#     """
#     可学习的时序聚合模块（TCM）。

#     输入:  [B, C, T, H, W]
#     输出:  [B, C_out, H, W]

#     结构:
#       - 3D depthwise conv 建模时空局部关系
#       - 3D conv 融合并调整通道
#       - 可选 SE 门控（按时间维重标定）
#       - 1x1x1 的 time_proj 学习每个时间切片权重
#       - 对 T 求和聚合
#       - 最后用 2D 1x1 proj2d 调整到目标通道
#     """
#     def __init__(self, in_channels, out_channels=None, k_t=3, use_se=True, depthwise=True, dropout_p=0.0):
#         super().__init__()
#         out_channels = out_channels or in_channels
#         pad_t = k_t // 2

#         groups = in_channels if depthwise else 1

#         # 3D: 同时建模时间与空间（不降采样）
#         self.conv3d_1 = nn.Conv3d(
#             in_channels, in_channels,
#             kernel_size=(k_t, 3, 3),
#             padding=(pad_t, 1, 1),
#             groups=groups, bias=False
#         )
#         self.bn3d_1 = nn.BatchNorm3d(in_channels)

#         # 仅空间融合，稳定数值/换通道
#         self.conv3d_2 = nn.Conv3d(
#             in_channels, out_channels,
#             kernel_size=(1, 3, 3),
#             padding=(0, 1, 1),
#             groups=1, bias=False
#         )
#         self.bn3d_2 = nn.BatchNorm3d(out_channels)

#         self.use_se = use_se
#         if use_se:
#             # 针对时间维的 SE 门控：对每个像素位置学习 T 维权重, 学习每个时间切片的重要性
#             r = max(8, out_channels // 8)
#             self.se_fc1 = nn.Conv3d(out_channels, r, kernel_size=1)
#             self.se_fc2 = nn.Conv3d(r, out_channels, kernel_size=1)

#         self.dropout = nn.Dropout3d(dropout_p) if dropout_p > 0 else nn.Identity()

#         # 学习每个切片权重
#         self.time_proj = nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False)

#         # 最后 2D 1x1 做通道整形
#         self.proj2d = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

#     def forward(self, x_bcthw):
#         # x: [B, C, T, H, W]
#         y = F.relu(self.bn3d_1(self.conv3d_1(x_bcthw)))
#         y = self.dropout(y)
#         y = F.relu(self.bn3d_2(self.conv3d_2(y)))  # [B, C_out, T, H, W]

#         if self.use_se:
#             # 对 H,W 求均值，保留 T，得到时序门控
#             s = y.mean(dim=(3, 4), keepdim=True)   # 对空间维度求平均，保留时间维度T
#             s = F.relu(self.se_fc1(s))             # MLP层学习每个通道随时间的变化
#             s = torch.sigmoid(self.se_fc2(s))      # 学习到的时序权重
#             y = y * s                              # 乘回原特征 调整各通道在当前T切片的权重

#         y = self.time_proj(y)                      # [B, C_out, T, H, W],学习每个切片权重
#         y = y.sum(dim=2)                           # 聚合 T → [B, C_out, H, W]
#         y = self.proj2d(y)                         # [B, C_out, H, W]
#         return y


# class SNNBackboneYAMLWrapper(nn.Module):
#     """
#     包装 YAMLBackbone：
#       - YAMLBackbone 负责事件体素化与特征提取，输出 p2,p3,p4,p5 形如 [T, B, C, H, W]
#       - 本包装器在 p4/p5 上做 T 维聚合（TCM），输出标准 FPN-like 的二维特征 [B, C, H, W]
#     """
#     def __init__(self, args, height: int, width: int, yaml_path: str, scale: str = 's'):
#         super().__init__()
#         self.height = int(height)
#         self.width = int(width)
#         temporal_bins = getattr(args, 'snn_temporal_bins', 4)

#         self.backbone = YAMLBackbone(
#             yaml_path=yaml_path, scale=scale,
#             in_ch=2, height=self.height, width=self.width,
#             temporal_bins=temporal_bins
#         )
#         # 下游检测头接口
#         self.out_channels = [256, 512]
#         self.strides = [16, 32]
#         self.num_scales = 2
#         self.use_image = False
#         self.is_snn = True
#         self.num_classes = dict(dsec=2, ncaltech101=100).get(getattr(args, 'dataset', 'dsec'), 2)

#         # TCM 聚合模块（第一次 forward 时基于实际通道创建） 
#         # YAMLBackbone 返回 p4/p5 形状是 [T, B, C, H, W]，聚合到 [B, C, H, W]
#         self._tcm = nn.ModuleDict()  # 键：'p4','p5'

#         # 可通过 args 控制
#         self.tcm_enabled = getattr(args, 'tcm_enabled', True)
#         self.tcm_kt = int(getattr(args, 'tcm_k_t', 3))
#         self.tcm_use_se = bool(getattr(args, 'tcm_use_se', True))
#         self.tcm_depthwise = bool(getattr(args, 'tcm_depthwise', True))
#         self.tcm_dropout = float(getattr(args, 'tcm_dropout', 0.0))

#     def get_output_sizes(self):
#         sizes = []
#         for s in self.strides:
#             sizes.append([max(1, self.height // s), max(1, self.width // s)])
#         return [[h, w] for h, w in sizes]

#     # 内部：把 [T,B,C,H,W] 聚成 [B,C,H,W]（优先用 TCM，退化为 t-mean）
#     def _aggregate_time(self, x_tbchw, level_name: str):
#         if x_tbchw is None:
#             return None
#         # x_tbchw: [T, B, C, H, W]  →  [B, C, T, H, W]
#         x_bcthw = x_tbchw.permute(1, 2, 0, 3, 4).contiguous()

#         # T=1 时直接挤掉时间维
#         if x_bcthw.size(2) == 1:
#             return x_bcthw.squeeze(2)

#         if not self.tcm_enabled:
#             # 兼容开关：关闭时回退到简单 t-mean
#             return x_tbchw.mean(dim=0)

#         key = level_name
#         if key not in self._tcm:
#             in_ch = x_bcthw.size(1)
#             self._tcm[key] = TemporalConvAgg(
#                 in_channels=in_ch,
#                 out_channels=in_ch,          # 维持通道数不变
#                 k_t=self.tcm_kt,
#                 use_se=self.tcm_use_se,
#                 depthwise=self.tcm_depthwise,
#                 dropout_p=self.tcm_dropout
#             ).to(x_bcthw.device, dtype=x_bcthw.dtype)

#         return self._tcm[key](x_bcthw)

#     # 标准 forward: 聚合时间维，返回二维特征 用于单snn分支
#     def forward(self, data, reset: bool = True):
#         # 让 backbone 知道目标分辨率
#         setattr(data, 'meta_height', self.height)
#         setattr(data, 'meta_width', self.width)

#         # YAMLBackbone 约定输出: p2,p3,p4,p5 皆为 [T,B,C,H,W] 或 None
#         p2, p3, p4, p5 = self.backbone(data)

#         # 用可学习的 Temporal Conv 聚合，替换原来的 mean over T ——
#         p4_bchw = self._aggregate_time(p4, 'p4')
#         p5_bchw = self._aggregate_time(p5, 'p5')

#         ret = []
#         if p4_bchw is not None:
#             ret.append(p4_bchw)
#         if p5_bchw is not None:
#             ret.append(p5_bchw)
#         return ret

#     # 直接返回未经聚合的时序特征，做双分支融合的时候才用于融合前，融合中才简单mean T聚合
#     def forward_time(self, data, reset: bool = True):
#         """
#         保持原语义：返回保留时间维的多尺度特征，形状为 [T,B,C,H,W]。
#         """
#         setattr(data, 'meta_height', self.height)
#         setattr(data, 'meta_width', self.width)
#         p2, p3, p4, p5 = self.backbone(data)
#         ret = {}
#         if p2 is not None: ret["p2"] = p2
#         if p3 is not None: ret["p3"] = p3
#         if p4 is not None: ret["p4"] = p4
#         if p5 is not None: ret["p5"] = p5
#         return ret



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

