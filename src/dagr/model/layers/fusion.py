import torch
import torch.nn as nn

from dagr.model.layers.spike_cross_attention import CrossAttention

# TAFR (AdaIN) 实现
class AdaIN_block(nn.Module):
    def __init__(self):
        super(AdaIN_block, self).__init__()

    def calc_mean_std(self, feat, eps=1e-5):
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, rgb, evt): # (content, style)
        assert (rgb.size()[:2] == evt.size()[:2])
        size = rgb.size()
        style_mean, style_std = self.calc_mean_std(evt)
        content_mean, content_std = self.calc_mean_std(rgb)
        normalized_feat = (rgb - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

class SpikeCAFR(nn.Module):
    """
    CAFR-like bidirectional fusion with spike-driven cross-attention.

    Inputs
      - rgb: [B,C,H,W]
      - evt: [T,B,C,H,W]

    Output
      - fused rgb-like feature: [B,C_out,H,W]
    """

    def __init__(self, rgb_in_channels: int, evt_in_channels: int, out_channels: int, num_heads: int = 8):
        super().__init__()
        self.rgb_in_channels = rgb_in_channels
        self.evt_in_channels = evt_in_channels
        self.out_channels = out_channels

        self.conv_rgb_in = nn.Conv2d(rgb_in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_rgb_in = nn.BatchNorm2d(out_channels)

        self.conv_evt_in = nn.Conv2d(evt_in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_evt_in = nn.BatchNorm2d(out_channels)

        self.rgb_centric_attn = CrossAttention(embed_dims=out_channels, num_heads=num_heads)
        self.evt_centric_attn = CrossAttention(embed_dims=out_channels, num_heads=num_heads)

        # 初始化 TAFR (AdaIN) 模块 
        self.tafr_rgb_refine = AdaIN_block() # content=rgb, style=evt_attn_out
        self.tafr_evt_refine = AdaIN_block() # content=evt, style=rgb_attn_out

        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)

    @staticmethod
    def _bchw_to_tbnc(x: torch.Tensor) -> torch.Tensor:
        # [B,C,H,W] -> [1,B,N,C]
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1).unsqueeze(0).contiguous()
        return x

    @staticmethod
    def _tbnc_to_bchw(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # [T,B,N,C] -> mean over T -> [B,N,C] -> [B,C,H,W]
        x = x.mean(dim=0)  # [B,N,C]
        B, N, C = x.shape
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return x

    @staticmethod
    def _tbchw_to_tbnc(x: torch.Tensor) -> torch.Tensor:
        # [T,B,C,H,W] -> [T,B,N,C]
        T, B, C, H, W = x.shape
        x = x.view(T, B, C, H * W).permute(0, 1, 3, 2).contiguous()
        return x

    def forward(self, rgb: torch.Tensor, evt: torch.Tensor) -> torch.Tensor:
        if evt.shape[3:] != rgb.shape[2:]:  # evt是[T,B,C,H,W], rgb是[B,C,H,W]
            import torch.nn.functional as F
            T, B = evt.shape[:2]
            evt_resized = F.interpolate(
                evt.flatten(0,1),  # [T*B,C,H,W]
                size=rgb.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            evt = evt_resized.view(T, B, evt_resized.shape[1], *rgb.shape[2:])

        # 预投射
        rgb = self.bn_rgb_in(self.conv_rgb_in(rgb))
        T, B, C_evt, H, W = evt.shape
        evt = self.bn_evt_in(self.conv_evt_in(evt.flatten(0, 1))).view(T, B, self.out_channels, H, W)

        # element-wise interaction (REFusion-like)
        evt_mean = evt.mean(dim=0)
        mul = rgb * evt_mean
        rgb_enh = rgb + mul
        evt_enh = evt + mul.unsqueeze(0)

        # spike-driven bidirectional cross-attention
        q_rgb = self._bchw_to_tbnc(rgb_enh)   # [1,B,N,C]
        kv_evt = self._tbchw_to_tbnc(evt_enh) # [T,B,N,C]
        out_rgb = self.rgb_centric_attn(q_rgb, kv_evt, kv_evt)  # [T,B,N,C] (T aligned inside)
        out_rgb = self._tbnc_to_bchw(out_rgb, H, W)

        q_evt = kv_evt                              # [T,B,N,C]
        kv_rgb = self._bchw_to_tbnc(rgb_enh)        # [1,B,N,C]
        out_evt = self.evt_centric_attn(q_evt, kv_rgb, kv_rgb)  # [T,B,N,C] (T aligned inside)
        out_evt = self._tbnc_to_bchw(out_evt, H, W)

        # 4.TAFR (Post-Attention Refinement)
        # 遵循 FRN 逻辑: z = adain(content=x[0], style=W_y)
        
        # 细化 BCI 的 RGB 特征 (content=rgb_enh)
        # 使用 Event-centric attention 的输出作为 style (style=out_evt)
        rgb_refined = self.tafr_rgb_refine(rgb_enh, out_evt)

        # 细化 BCI 的 Event 特征 (content=evt_enh.mean(0))
        # 使用 RGB-centric attention 的输出作为 style (style=out_rgb)
        evt_refined = self.tafr_evt_refine(evt_enh.mean(0), out_rgb)

        # TAFR后融合 
        # fused = out_rgb + out_evt
        fused = rgb_refined + evt_refined
        
        fused = self.bn_out(self.conv_out(fused))
        return fused + rgb


#带有wf we可学习权重，最后Concat而不是相加

# import torch
# import torch.nn as nn
# from dagr.model.layers.spike_cross_attention import CrossAttention

# # 添加可学习线性投影的AdaIN模块
# class AdaIN_block(nn.Module):
#     def __init__(self, channel_dim):
#         super(AdaIN_block, self).__init__()
#         # 添加可学习的线性投影W
#         self.linear_projection = nn.Conv2d(channel_dim, channel_dim, kernel_size=1, bias=False)
        
#     def calc_mean_std(self, feat, eps=1e-5):
#         size = feat.size()
#         assert (len(size) == 4)
#         N, C = size[:2]
#         feat_var = feat.view(N, C, -1).var(dim=2) + eps
#         feat_std = feat_var.sqrt().view(N, C, 1, 1)
#         feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
#         return feat_mean, feat_std

#     def forward(self, content, style, apply_projection=True):
#         # 应用线性投影W
#         if apply_projection:
#             content_w = self.linear_projection(content)
#         else:
#             content_w = content
            
#         assert (content.size()[:2] == style.size()[:2])
#         size = content.size()
        
#         # 计算统计量
#         style_mean, style_std = self.calc_mean_std(style)
#         content_w_mean, content_w_std = self.calc_mean_std(content_w)
        
#         # AdaIN操作
#         normalized_feat = (content_w - content_w_mean.expand(size)) / content_w_std.expand(size)
#         return normalized_feat * style_std.expand(size) + style_mean.expand(size)

# class SpikeCAFR(nn.Module):
#     def __init__(self, rgb_in_channels: int, evt_in_channels: int, out_channels: int, num_heads: int = 8):
#         super().__init__()
#         self.rgb_in_channels = rgb_in_channels
#         self.evt_in_channels = evt_in_channels
#         self.out_channels = out_channels

#         self.conv_rgb_in = nn.Conv2d(rgb_in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn_rgb_in = nn.BatchNorm2d(out_channels)

#         self.conv_evt_in = nn.Conv2d(evt_in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn_evt_in = nn.BatchNorm2d(out_channels)

#         self.rgb_centric_attn = CrossAttention(embed_dims=out_channels, num_heads=num_heads)
#         self.evt_centric_attn = CrossAttention(embed_dims=out_channels, num_heads=num_heads)

#         # 初始化TAFR模块，传入通道数以创建可学习的线性投影
#         self.tafr_rgb_refine = AdaIN_block(out_channels)
#         self.tafr_evt_refine = AdaIN_block(out_channels)

#         # 添加用于融合的层
#         self.fusion = nn.Sequential(
#             nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels)
#         )

#     @staticmethod
#     def _bchw_to_tbnc(x: torch.Tensor) -> torch.Tensor:
#         B, C, H, W = x.shape
#         x = x.view(B, C, H * W).permute(0, 2, 1).unsqueeze(0).contiguous()
#         return x

#     @staticmethod
#     def _tbnc_to_bchw(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
#         x = x.mean(dim=0)  # [B,N,C]
#         B, N, C = x.shape
#         x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
#         return x

#     @staticmethod
#     def _tbchw_to_tbnc(x: torch.Tensor) -> torch.Tensor:
#         T, B, C, H, W = x.shape
#         x = x.view(T, B, C, H * W).permute(0, 1, 3, 2).contiguous()
#         return x

#     def forward(self, rgb: torch.Tensor, evt: torch.Tensor) -> torch.Tensor:
#         # 预投射
#         rgb = self.bn_rgb_in(self.conv_rgb_in(rgb))
#         T, B, C_evt, H, W = evt.shape
#         evt = self.bn_evt_in(self.conv_evt_in(evt.flatten(0, 1))).view(T, B, self.out_channels, H, W)

#         # element-wise interaction
#         evt_mean = evt.mean(dim=0)  # [B,C,H,W]
#         mul = rgb * evt_mean
#         rgb_enh = rgb + mul
#         evt_enh = evt + mul.unsqueeze(0)

#         # spike-driven bidirectional cross-attention
#         q_rgb = self._bchw_to_tbnc(rgb_enh)   # [1,B,N,C]
#         kv_evt = self._tbchw_to_tbnc(evt_enh) # [T,B,N,C]
#         out_rgb = self.rgb_centric_attn(q_rgb, kv_evt, kv_evt)  # [T,B,N,C]
#         out_rgb = self._tbnc_to_bchw(out_rgb, H, W)  # [B,C,H,W]

#         q_evt = kv_evt                              # [T,B,N,C]
#         kv_rgb = self._bchw_to_tbnc(rgb_enh)        # [1,B,N,C]
#         out_evt = self.evt_centric_attn(q_evt, kv_rgb, kv_rgb)  # [T,B,N,C]
#         out_evt = self._tbnc_to_bchw(out_evt, H, W)  # [B,C,H,W]

#         # TAFR - 带有可学习线性投影的AdaIN
#         rgb_refined = self.tafr_rgb_refine(out_rgb, rgb_enh)  # F_f^w作为content，F_f^enh作为style
#         evt_refined = self.tafr_evt_refine(out_evt, evt_enh.mean(0))  # F_e^w作为content，F_e^enh作为style

#         # 使用concat而不是加法，以匹配公式
#         fused = torch.cat([rgb_refined, evt_refined], dim=1)  # [B,2C,H,W]
        
#         # 通过1x1卷积调整通道数
#         fused = self.fusion(fused)  # [B,C,H,W]
        
#         return fused + rgb  # 添加残差连接

