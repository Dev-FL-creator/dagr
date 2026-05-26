import torch
import torch.nn as nn

from dagr.model.networks.image_backbone import ImageBackbone
from dagr.model.layers.fusion_2branch import SpikeCAFR
from dagr.model.backbones.sdt_v3_trilinear_mean import SpikformerV3Extractor


class HybridBackbone(nn.Module):
    #多尺度特征通过SpikeCAFR融合，forward 返回 fused（融合后的特征列表）和 rgb_only（仅 RGB 的特征列表）
    def __init__(self, args, height: int, width: int):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.use_sdt_v3 = str(getattr(args, "backbone_type", "")).lower() == "sdtv3"

        args_local = args
        args_local.use_image = True
        self.rgb = ImageBackbone(args_local, height=height, width=width)

        c2_ch = self.rgb.feature_channels[0]
        c3_ch = self.rgb.feature_channels[1]
        c4_ch = self.rgb.output_channels[0]
        c5_ch = self.rgb.output_channels[1]

        if self.use_sdt_v3:
            self.snn = SpikformerV3Extractor(args, height=height, width=width, pretrained_weight=getattr(args, "load_pretrained_weight", None))
            evt_channels = list(self.snn.out_channels)
            rgb_fuse_channels = [c3_ch, c4_ch, c5_ch]
            self.strides = [8, 16, 32]
            self.out_channels = rgb_fuse_channels

        self.fuse_modules = nn.ModuleList()
        for rgb_ch, evt_ch in zip(rgb_fuse_channels, evt_channels):
            self.fuse_modules.append(SpikeCAFR(rgb_in_channels=rgb_ch, evt_in_channels=evt_ch, out_channels=rgb_ch))

        self.num_scales = len(self.out_channels)
        self.num_classes = self.snn.num_classes
        self.use_image = True

    def get_output_sizes(self):
        sizes = []
        for s in self.strides:
            sizes.append([max(1, self.height // s), max(1, self.width // s)])
        return sizes

    def forward(self, data):
        features, image_outs = self.rgb(data.image)

        rgb_c2 = features[1] if len(features) > 1 else None
        rgb_c3 = features[2] if len(features) > 2 else None
        rgb_c4 = image_outs[0]
        rgb_c5 = image_outs[1]

        if self.use_sdt_v3:
            event_feats = self.snn(data)
            rgb_feats = [rgb_c3, rgb_c4, rgb_c5]
            event_feats_5d = []
            for feat in event_feats:
                if feat.dim() == 4:
                    feat = feat.unsqueeze(0)  # 添加T维度
                event_feats_5d.append(feat)
            event_feats = event_feats_5d
        else:
            snn_feats = self.snn.forward_time(data)
            event_feats = [snn_feats.get("p2"), snn_feats.get("p3"), snn_feats.get("p4"), snn_feats.get("p5")]
            rgb_feats = [rgb_c2, rgb_c3, rgb_c4, rgb_c5]

        fused = []
        for rgb_feat, evt_feat, fuse in zip(rgb_feats, event_feats, self.fuse_modules):
            if rgb_feat is None or evt_feat is None:
                continue
            fused.append(fuse(rgb_feat, evt_feat))

        rgb_only = [x for x in rgb_feats if x is not None]
        return fused, rgb_only
