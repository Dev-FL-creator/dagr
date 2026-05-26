import torch

import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from torch_geometric.data import Data
from yolox.models import YOLOX, YOLOXHead, IOUloss
from dagr.model.networks.enhancer import BranchEnhancer
from dagr.model.networks.hybrid_backbone_v3_2branch import HybridBackbone
from dagr.model.backbones.sdt_v3_trilinear_mean import SpikformerV3Extractor
from dagr.model.layers.spline_conv import SplineConvToDense
from dagr.model.layers.conv import ConvBlock
from dagr.model.utils import shallow_copy, init_subnetwork, voxel_size_to_params, postprocess_network_output, convert_to_evaluation_format, init_grid_and_stride, convert_to_training_format


class DAGR(YOLOX):
    #输入为图像和事件数据，输出为检测结果。
    #在训练模式下，计算双分支的总损失；在评估模式下，进行后处理得到最终的检测结果。
    def __init__(self, args, height, width):
        self.conf_threshold = 0.001
        self.nms_threshold = 0.65

        self.height = height
        self.width = width

        use_sdt = str(getattr(args, 'backbone_type', '')).lower() == 'sdtv3'
        use_snn = hasattr(args, 'use_snn_backbone') and getattr(args, 'use_snn_backbone') and (use_sdt is not None)
        print(f"Debug: use_snn: {use_snn}")

        use_image = hasattr(args, 'use_image') and getattr(args, 'use_image')
        print(f"Debug: use_image: {use_image}")
        # 双分支
        if use_snn and getattr(args, 'use_image', False) and HybridBackbone is not None:
            print(f"Debug: running with 2-branch hybrid backbone (Fused, RGB)")
            backbone = HybridBackbone(args, height=height, width=width)

            if getattr(backbone, 'use_sdt_v3', False):
                in_channels_image = backbone.out_channels
            else:
                rgb_all_channels = backbone.rgb.feature_channels + backbone.rgb.output_channels
                in_channels_image = rgb_all_channels
            # 双分支的head，融合分支和RGB分支分别有独立的head计算损失，训练时两者损失相加；评估时两者输出进行融合后处理得到最终检测结果
            head = HybridHead(
                num_classes=backbone.num_classes,
                strides=backbone.strides,
                in_channels=backbone.out_channels,
                in_channels_image=in_channels_image,
                args=args
            )
        # 单分支，仅 SNN
        elif use_snn:
            if use_sdt:
                backbone = SpikformerV3Extractor(args, height=height, width=width, pretrained_weight=getattr(args, "load_pretrained_weight", None))
            head = YOLOXHead(num_classes=backbone.num_classes,
                             width=1.0,
                             strides=backbone.strides,
                             in_channels=backbone.out_channels)
        # 初始化父类 YOLOX，注册 backbone（HybridBackbone或SpikformerV3Extractor） 和 head（HybridHead 或 YOLOXHead）
        super().__init__(backbone=backbone, head=head)

        if "img_net_checkpoint" in args:
            state_dict = torch.load(args.img_net_checkpoint)
            init_subnetwork(self, state_dict['ema'], "backbone.net.", freeze=True)
            init_subnetwork(self, state_dict['ema'], "head.cnn_head.")

    def forward(self, x: Data, reset=True, return_targets=True, filtering=True):

        if not hasattr(self.head, "output_sizes") and hasattr(self.backbone, "get_output_sizes"):
            self.head.output_sizes = self.backbone.get_output_sizes()

        if self.training:
            targets = convert_to_training_format(x.bbox, x.bbox_batch, x.num_graphs)

            if self.backbone.use_image:
                targets0 = convert_to_training_format(x.bbox0, x.bbox0_batch, x.num_graphs)
                targets_tuple = (targets, targets0)
                outputs = YOLOX.forward(self, x, targets_tuple)
            else:
                outputs = YOLOX.forward(self, x, targets)

            return outputs

        x.reset = reset

        outputs = YOLOX.forward(self, x)

        detections = postprocess_network_output(outputs, self.backbone.num_classes, self.conf_threshold, self.nms_threshold, filtering=filtering,
                                                height=self.height, width=self.width)
        #评估模式下，返回检测结果
        ret = [detections]

        if return_targets and hasattr(x, 'bbox'):
            targets = convert_to_evaluation_format(x)
            ret.append(targets)

        return ret

# RGB分支的head，结构与融合分支的head相同，但参数独立，训练时计算RGB分支的损失
class CNNHead(YOLOXHead):
    def __init__(self, num_classes, width=1.0, strides=[8, 16, 32], in_channels=[256, 512, 1024], act="silu", depthwise=False):
        super().__init__(num_classes, width, strides, in_channels, act, depthwise)
    
    def forward(self, xin):
        outputs = dict(cls_output=[], reg_output=[], obj_output=[])

        for k, (cls_conv, reg_conv, x) in enumerate(zip(self.cls_convs, self.reg_convs, xin)):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            reg_feat = reg_conv(reg_x)

            outputs["cls_output"].append(self.cls_preds[k](cls_feat))
            outputs["reg_output"].append(self.reg_preds[k](reg_feat))
            outputs["obj_output"].append(self.obj_preds[k](reg_feat))

        return outputs


class HybridHead(YOLOXHead):
    def __init__(
        self,
        num_classes,
        strides=[8, 16, 32],
        in_channels=[64, 256, 512],
        in_channels_image=None,
        act="silu",
        depthwise=False,
        args=None
    ):

        YOLOXHead.__init__(self, num_classes, 1.0, strides, in_channels, act, depthwise)
        self.strides = strides
        self.num_scales = len(in_channels)

        if in_channels_image is None:
            in_channels_image = in_channels

        self.image_head = CNNHead(
            num_classes=num_classes,
            width=1.0,
            strides=strides,
            in_channels=in_channels_image,
            act=act,
            depthwise=depthwise
        )

        # BranchEnhancer
        self.enhance_fused = BranchEnhancer(
            in_channels=in_channels,
            align_dim=256,
            num_hyperedges=8
        )

        self.enhance_image = BranchEnhancer(
            in_channels=in_channels_image,
            align_dim=256,
            num_hyperedges=8
        )

    def _forward_single(self, xin):
        outputs = dict(cls_output=[], reg_output=[], obj_output=[])
        for k, (cls_conv, reg_conv, x) in enumerate(zip(self.cls_convs, self.reg_convs, xin)):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x
            cls_feat = cls_conv(cls_x)
            reg_feat = reg_conv(reg_x)
            outputs["cls_output"].append(self.cls_preds[k](cls_feat))
            outputs["reg_output"].append(self.reg_preds[k](reg_feat))
            outputs["obj_output"].append(self.obj_preds[k](reg_feat))
        return outputs

    def collect_outputs(self, cls_output, reg_output, obj_output, k, stride_this_level, ret=None):
        if self.training:
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            output, grid = self.get_output_and_grid(
                output, k, stride_this_level, output.type()
            )
            ret['x_shifts'].append(grid[:, :, 0])
            ret['y_shifts'].append(grid[:, :, 1])
            ret['expanded_strides'].append(
                torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(output)
            )
        else:
            output = torch.cat(
                [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
            )

        ret['outputs'].append(output)


    def forward(self, xin, labels=None, imgs=None):
        fused_feats, image_feats = xin

        # 特征增强
        fused_feats = self.enhance_fused(fused_feats)
        image_feats = self.enhance_image(image_feats)

        out_fused = self._forward_single(fused_feats)
        out_image = self.image_head(image_feats)
        #训练模式下，计算双分支的总损失
        if self.training:
            fused_ret = dict(outputs=[], x_shifts=[], y_shifts=[], expanded_strides=[])
            image_ret = dict(outputs=[], x_shifts=[], y_shifts=[], expanded_strides=[])

            for k in range(self.num_scales):
                self.collect_outputs(
                    out_fused["cls_output"][k],
                    out_fused["reg_output"][k],
                    out_fused["obj_output"][k],
                    k,
                    self.strides[k],
                    ret=fused_ret
                )
                self.collect_outputs(
                    out_image["cls_output"][k],
                    out_image["reg_output"][k],
                    out_image["obj_output"][k],
                    k,
                    self.strides[k],
                    ret=image_ret
                )

            if isinstance(labels, tuple) and len(labels) == 2:
                labels_fused, labels_image = labels
            else:
                labels_fused, labels_image = labels, labels

            losses_image = self.get_losses(
                imgs,
                image_ret['x_shifts'],
                image_ret['y_shifts'],
                image_ret['expanded_strides'],
                labels_image,
                torch.cat(image_ret['outputs'], 1),
                [],
                dtype=image_feats[0].dtype,
            )

            losses_fused = self.get_losses(
                imgs,
                fused_ret['x_shifts'],
                fused_ret['y_shifts'],
                fused_ret['expanded_strides'],
                labels_fused,
                torch.cat(fused_ret['outputs'], 1),
                [],
                dtype=fused_feats[0].dtype,
            )

            return tuple(
                l_img + l_fused
                for l_img, l_fused in zip(losses_image, losses_fused)
            )
        #评估模式下，进行后处理得到最终的检测结果
        else:
            fused_ret = dict(outputs=[])
            for k in range(self.num_scales):
                self.collect_outputs(
                    out_fused["cls_output"][k],
                    out_fused["reg_output"][k],
                    out_fused["obj_output"][k],
                    k,
                    self.strides[k],
                    ret=fused_ret
                )

            self.hw = [x.shape[-2:] for x in fused_ret['outputs']]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in fused_ret['outputs']],
                dim=2
            ).permute(0, 2, 1)

            return self.decode_outputs(outputs, dtype=fused_feats[0].type())

