import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from yolox.models import YOLOX, YOLOXHead, IOUloss

from dagr.model.networks.net import Net
try:
    from dagr.model.backbones.sdt_v3 import SpikformerV3Extractor
except Exception:
    SpikformerV3Extractor = None
from dagr.model.utils import postprocess_network_output, convert_to_evaluation_format, convert_to_training_format


class DAGR(YOLOX):
    def __init__(self, args, height, width):
        self.conf_threshold = 0.001
        self.nms_threshold = 0.65
        self.height = height
        self.width = width

        backbone_type = str(getattr(args, 'backbone_type', '')).lower()
        use_snn = getattr(args, 'use_snn_backbone', False)

        if use_snn and backbone_type == 'sdtv3' and SpikformerV3Extractor is not None:
            backbone = SpikformerV3Extractor(
                args, 
                height=height, 
                width=width, 
                pretrained_weight=getattr(args, "load_pretrained_weight", None)
            )
        else:
            raise ValueError("Event-only mode requires use_snn_backbone=True and valid backbone_type")

        head = YOLOXHead(
            num_classes=backbone.num_classes,
            width=1.0,
            strides=backbone.strides,
            in_channels=backbone.out_channels
        )

        super().__init__(backbone=backbone, head=head)

    def forward(self, x: Data, reset=True, return_targets=True, filtering=True):
        if self.training:
            targets = convert_to_training_format(x.bbox, x.bbox_batch, x.num_graphs)
            outputs = YOLOX.forward(self, x, targets)
            return outputs

        x.reset = reset
        outputs = YOLOX.forward(self, x)

        detections = postprocess_network_output(
            outputs, 
            self.backbone.num_classes, 
            self.conf_threshold, 
            self.nms_threshold, 
            filtering=filtering,
            height=self.height, 
            width=self.width
        )

        ret = [detections]

        if return_targets and hasattr(x, 'bbox'):
            targets = convert_to_evaluation_format(x)
            ret.append(targets)

        return ret