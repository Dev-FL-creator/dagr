import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

try:
    from ultralytics.nn.modules.head import Detect
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.tal import make_anchors, dist2bbox, bbox2dist
except ImportError as e:
    raise ImportError(
        f"YOLOv13 (ultralytics) is not installed. Please install it first:\n"
        f"  cd yolov13\n"
        f"  pip install -r requirements.txt\n"
        f"  pip install -e .\n"
        f"Original error: {e}"
    )

from dagr.model.utils import convert_to_training_format


class YOLOv13LossAdapter:
    
    def __init__(self, detect_head, num_classes, reg_max=16):
        self.detect_head = detect_head
        self.num_classes = num_classes
        self.reg_max = reg_max
        
        class DummyModel(nn.Module):
            def __init__(self, detect_head, num_classes, reg_max):
                super().__init__()
                self.model = nn.ModuleList([detect_head])
                self.args = type('Args', (), {
                    'box': 7.5,
                    'cls': 0.5,
                    'dfl': 1.5,
                })()
        
        dummy_model = DummyModel(detect_head, num_classes, reg_max)
        self.loss_fn = v8DetectionLoss(dummy_model, tal_topk=10)
    
    def __call__(self, yolo13_outputs, labels, imgs, branch_idx=0):
        device = yolo13_outputs[0].device if len(yolo13_outputs) > 0 else (imgs.device if imgs is not None else torch.device('cpu'))
        
        if hasattr(self.loss_fn, 'proj') and self.loss_fn.proj.device != device:
            self.loss_fn.proj = self.loss_fn.proj.to(device)
        if hasattr(self.loss_fn, 'device') and self.loss_fn.device != device:
            self.loss_fn.device = device
        
        batch = self._convert_labels_to_yolov13_format(labels, imgs, branch_idx)
        
        preds = (None, yolo13_outputs)
        
        total_loss, loss_components = self.loss_fn(preds, batch)
        
        loss_dict = {
            'loss_box': loss_components[0],
            'loss_cls': loss_components[1],
            'loss_dfl': loss_components[2] if len(loss_components) > 2 else torch.tensor(0.0, device=loss_components[0].device)
        }
        
        return total_loss, loss_dict
    
    def _convert_labels_to_yolov13_format(self, labels, imgs, branch_idx=0):
        if isinstance(labels, tuple):
            labels = labels[branch_idx]
        
        if labels is None or len(labels) == 0:
            batch_size = imgs.shape[0] if imgs is not None else 1
            return {
                'batch_idx': torch.zeros(0, dtype=torch.long, device=imgs.device if imgs is not None else 'cpu'),
                'cls': torch.zeros(0, dtype=torch.long, device=imgs.device if imgs is not None else 'cpu'),
                'bboxes': torch.zeros(0, 4, device=imgs.device if imgs is not None else 'cpu'),
            }
        
        device = labels.device
        
        if len(labels.shape) == 3:
            batch_size, max_detections, num_cols = labels.shape
            labels_flat = labels.view(-1, num_cols)
            
            non_zero_mask = labels_flat.abs().sum(dim=1) > 0
            labels_filtered = labels_flat[non_zero_mask]
            
            if len(labels_filtered) == 0:
                return {
                    'batch_idx': torch.zeros(0, dtype=torch.long, device=device),
                    'cls': torch.zeros(0, dtype=torch.long, device=device),
                    'bboxes': torch.zeros(0, 4, device=device),
                }
            
            batch_indices = torch.arange(batch_size, device=device).repeat_interleave(max_detections)
            batch_idx = batch_indices[non_zero_mask]
            
            cls = labels_filtered[:, 0].long()
            cx, cy, w, h = labels_filtered[:, 1], labels_filtered[:, 2], labels_filtered[:, 3], labels_filtered[:, 4]
        else:
            batch_size = imgs.shape[0] if imgs is not None else 1
            labels_flat = labels.view(-1, labels.shape[-1])
            
            non_zero_mask = labels_flat.abs().sum(dim=1) > 0
            labels_filtered = labels_flat[non_zero_mask]
            
            if len(labels_filtered) == 0:
                return {
                    'batch_idx': torch.zeros(0, dtype=torch.long, device=device),
                    'cls': torch.zeros(0, dtype=torch.long, device=device),
                    'bboxes': torch.zeros(0, 4, device=device),
                }
            
            if labels_filtered.shape[1] == 6:
                batch_idx = labels_filtered[:, 0].long()
                cls = labels_filtered[:, 1].long()
                cx, cy, w, h = labels_filtered[:, 2], labels_filtered[:, 3], labels_filtered[:, 4], labels_filtered[:, 5]
            else:
                batch_idx = torch.zeros(len(labels_filtered), dtype=torch.long, device=device)
                cls = labels_filtered[:, 0].long()
                cx, cy, w, h = labels_filtered[:, 1], labels_filtered[:, 2], labels_filtered[:, 3], labels_filtered[:, 4]
        
        if imgs is not None:
            _, _, H, W = imgs.shape
        else:
            H, W = 1.0, 1.0 

        cx_norm = cx / W
        cy_norm = cy / H
        w_norm = w / W
        h_norm = h / H
        
        bboxes = torch.stack([cx_norm, cy_norm, w_norm, h_norm], dim=1)
        bboxes = torch.clamp(bboxes, 0.0, 1.0)
        
        return {
            'batch_idx': batch_idx,
            'cls': cls,
            'bboxes': bboxes,
        }


class YOLOv13HeadAdapter(nn.Module):
    
    def __init__(self, num_classes, strides, in_channels, args=None):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.num_scales = len(in_channels)
        
        self.detect_head = Detect(nc=num_classes, ch=in_channels)
        
        if hasattr(self.detect_head, 'stride'):
            self.detect_head.stride = torch.tensor(strides, dtype=torch.float32)
        
        self.loss_fn = YOLOv13LossAdapter(self.detect_head, num_classes, reg_max=self.detect_head.reg_max)
        
        self.use_l1 = False
        self.decode_in_inference = True
        
    def forward(self, xin, labels=None, imgs=None, branch_idx=0):
        yolo13_outputs = self.detect_head(xin)
        
        if self.training:
            total_loss, loss_dict = self.loss_fn(yolo13_outputs, labels, imgs, branch_idx=branch_idx)
            return {
                'total_loss': total_loss,
                **loss_dict
            }
        else:
            if isinstance(yolo13_outputs, tuple):
                decoded_tensor = yolo13_outputs[0]
                return self._decode_outputs(decoded_tensor)
            else:
                return self._decode_outputs(yolo13_outputs)
    
    def _decode_outputs(self, yolo13_outputs):
        if isinstance(yolo13_outputs, torch.Tensor):
            if len(yolo13_outputs.shape) == 3:
                if yolo13_outputs.shape[1] == 4 + self.num_classes:
                    yolo13_outputs = yolo13_outputs.permute(0, 2, 1)
                B, N, C = yolo13_outputs.shape
            else:
                B, C, N = yolo13_outputs.shape
                yolo13_outputs = yolo13_outputs.permute(0, 2, 1)
            
            reg_4d = yolo13_outputs[:, :, :4]
            cls_logits = yolo13_outputs[:, :, 4:]
            
            obj_logits = torch.ones((B, N, 1), dtype=cls_logits.dtype, device=cls_logits.device)
            
            yolox_output = torch.cat([reg_4d, obj_logits, cls_logits], dim=2)
            return yolox_output
        else:
            first_output = yolo13_outputs[0]
            batch_size = first_output.shape[0]
            outputs_list = []
            
            for output in yolo13_outputs:
                if len(output.shape) == 4:
                    B, no, H, W = output.shape
                    output_flat = output.view(B, no, -1)
                    outputs_list.append(output_flat)
                else:
                    raise ValueError(f"Unexpected output shape: {output.shape}")
            
            outputs_cat = torch.cat(outputs_list, dim=2)
            outputs_cat = outputs_cat.permute(0, 2, 1)
            
            reg_max = self.detect_head.reg_max
            reg_dist = outputs_cat[:, :, :reg_max*4]
            cls_logits = outputs_cat[:, :, reg_max*4:]
            
            reg_4d = self._dfl_to_4d(reg_dist)
            
            B, N, _ = reg_4d.shape
            obj_logits = torch.ones((B, N, 1), dtype=cls_logits.dtype, device=cls_logits.device)
            
            yolox_output = torch.cat([reg_4d, obj_logits, cls_logits], dim=2)
            return yolox_output
    
    def _dfl_to_4d(self, reg_dist):
        B, N, _ = reg_dist.shape
        reg_max = self.detect_head.reg_max
        
        reg_dist = reg_dist.view(B, N, 4, reg_max)
        
        proj = torch.arange(reg_max, dtype=reg_dist.dtype, device=reg_dist.device)
        reg_4d = reg_dist.softmax(dim=-1).matmul(proj)
        
        return reg_4d


class HybridHeadV2YOLOv13(nn.Module):
    
    def __init__(self, num_classes, strides, 
                 in_channels_fused, 
                 in_channels_image, 
                 in_channels_mad, 
                 act="silu", depthwise=False, width=1.0, args=None):
        super().__init__()
        
        self.strides = strides
        self.num_scales = len(in_channels_fused)
        self.num_classes = num_classes
        self.use_mad = in_channels_mad is not None
        
        self.fused_head = YOLOv13HeadAdapter(num_classes, strides, in_channels_fused, args)
        self.image_head = YOLOv13HeadAdapter(num_classes, strides, in_channels_image, args)
        if self.use_mad:
            self.mad_head = YOLOv13HeadAdapter(num_classes, strides, in_channels_mad, args)
        else:
            self.mad_head = None
        
        self.use_checkpointing = getattr(args, 'use_checkpointing', False) if args else False
        
    def forward(self, xin, labels=None, imgs=None):
        fused_feats, image_feats, mad_feats = xin
        
        if self.training:
            if self.use_checkpointing:
                def _forward_fused(*feats_tuple):
                    feats_list = list(feats_tuple) if len(feats_tuple) > 1 else list(feats_tuple[0]) if isinstance(feats_tuple[0], (list, tuple)) else [feats_tuple[0]]
                    return self.fused_head.detect_head(feats_list)
                
                def _forward_image(*feats_tuple):
                    feats_list = list(feats_tuple) if len(feats_tuple) > 1 else list(feats_tuple[0]) if isinstance(feats_tuple[0], (list, tuple)) else [feats_tuple[0]]
                    return self.image_head.detect_head(feats_list)
                
                fused_outputs = activation_checkpoint(_forward_fused, *fused_feats, use_reentrant=False)
                image_outputs = activation_checkpoint(_forward_image, *image_feats, use_reentrant=False)
            else:
                fused_outputs = self.fused_head.detect_head(fused_feats)
                image_outputs = self.image_head.detect_head(image_feats)
            
            fused_loss, fused_loss_dict = self.fused_head.loss_fn(
                fused_outputs, labels, imgs, branch_idx=0
            )
            image_loss, image_loss_dict = self.image_head.loss_fn(
                image_outputs, labels, imgs, branch_idx=1
            )
            
            total_loss = fused_loss + image_loss
            
            return {
                'total_loss': total_loss,
                'loss_box': fused_loss_dict['loss_box'] + image_loss_dict['loss_box'],
                'loss_cls': fused_loss_dict['loss_cls'] + image_loss_dict['loss_cls'],
                'loss_dfl': fused_loss_dict['loss_dfl'] + image_loss_dict['loss_dfl'],
                'loss_fused': fused_loss,
                'loss_image': image_loss,
            }
        else:
            return self.fused_head(fused_feats, labels=None, imgs=None, branch_idx=0)