import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from dagr.model.networks.image_backbone import ImageBackbone
from dagr.model.layers.fusion import SpikeCAFR
from dagr.model.backbones.sdt_v3 import SpikformerV3Extractor


class HybridBackbone(nn.Module):
    """
    RGB backbone with progressive fusion from SNN temporal features via SpikeCAFR.

    Exposes:
      - out_channels = [256, 512]
      - strides = [16, 32]
      - get_output_sizes(height,width) compatible with YOLOXHead
    """

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

        self.use_checkpointing = getattr(args, 'use_checkpointing', False)

        self.num_scales = len(self.out_channels)
        self.num_classes = self.snn.num_classes
        self.use_image = True

    def get_output_sizes(self):
        sizes = []
        for s in self.strides:
            sizes.append([max(1, self.height // s), max(1, self.width // s)])
        return sizes

    def forward(self, data):
        H, W = int(data.height[0]), int(data.width[0])

        H_pad = int(math.ceil(H / 16.0) * 16)
        W_pad = int(math.ceil(W / 16.0) * 16)

        padding = (0, W_pad - W, 0, H_pad - H)

        padded_image = F.pad(data.image, padding, "constant", 0)

        if self.training and self.use_checkpointing:
            features, image_outs = activation_checkpoint(self.rgb, padded_image, use_reentrant=False)
        else:
            features, image_outs = self.rgb(padded_image)

        rgb_c2 = features[1] if len(features) > 1 else None
        rgb_c3 = features[2] if len(features) > 2 else None
        rgb_c4 = image_outs[0]
        rgb_c5 = image_outs[1]

        setattr(data, 'meta_height', H_pad)
        setattr(data, 'meta_width', W_pad)

        if self.use_sdt_v3:
            snn_feats_list = self.snn(data)
            event_feats = snn_feats_list
        else:
            if self.training and self.use_checkpointing:
                snn_feats = activation_checkpoint(self.snn.forward_time, data, use_reentrant=False)
            else:
                snn_feats = self.snn.forward_time(data)
            event_feats = [snn_feats.get("p2"), snn_feats.get("p3"), snn_feats.get("p4"), snn_feats.get("p5")]

        if hasattr(data, 'meta_height'):
            delattr(data, 'meta_height')
        if hasattr(data, 'meta_width'):
            delattr(data, 'meta_width')

        if self.use_sdt_v3:
            rgb_feats = [rgb_c3, rgb_c4, rgb_c5]
        else:
            rgb_feats = [rgb_c2, rgb_c3, rgb_c4, rgb_c5]

        fused = []
        for rgb_feat, evt_feat, fuse in zip(rgb_feats, event_feats, self.fuse_modules):
            if rgb_feat is None or evt_feat is None:
                continue
            if self.training and self.use_checkpointing and not self.use_sdt_v3:
                fused.append(activation_checkpoint(fuse, rgb_feat, evt_feat, use_reentrant=False))
            else:
                fused.append(fuse(rgb_feat, evt_feat))

        rgb_only = [x for x in rgb_feats if x is not None]

        return fused, rgb_only