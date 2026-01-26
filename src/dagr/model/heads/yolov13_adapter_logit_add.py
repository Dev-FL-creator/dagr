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
    """
    Converts DAGR labels to YOLOv13 format and computes loss.
    """
    
    def __init__(self, detect_head, num_classes, reg_max=16):
        """
        Args:
            detect_head: YOLOv13 Detect head instance
            num_classes: Number of classes
            reg_max: DFL regression max value (default 16)
        """
        self.detect_head = detect_head
        self.num_classes = num_classes
        self.reg_max = reg_max
        
        # Create a dummy model-like object for v8DetectionLoss
        # v8DetectionLoss needs model.parameters() to get device
        class DummyModel(nn.Module):
            def __init__(self, detect_head, num_classes, reg_max):
                super().__init__()
                self.model = nn.ModuleList([detect_head])  # v8DetectionLoss expects model.model[-1]
                self.args = type('Args', (), {
                    'box': 7.5,  # box loss gain
                    'cls': 0.5,  # cls loss gain
                    'dfl': 1.5,  # dfl loss gain
                })()
        
        dummy_model = DummyModel(detect_head, num_classes, reg_max)
        self.loss_fn = v8DetectionLoss(dummy_model, tal_topk=10)
    
    def __call__(self, yolo13_outputs, labels, imgs):
        """
        Compute YOLOv13 loss from DAGR labels.
        
        Args:
            yolo13_outputs: List of [B, no, H, W] tensors from Detect head
            labels: DAGR format labels (from convert_to_training_format)
            imgs: Image tensors [B, C, H, W] (for shape info)
            
        Returns:
            total_loss: Scalar tensor
            loss_dict: Dictionary with 'loss_box', 'loss_cls', 'loss_dfl'
        """
        # Get device from outputs to ensure loss function uses correct device
        device = yolo13_outputs[0].device if len(yolo13_outputs) > 0 else (imgs.device if imgs is not None else torch.device('cpu'))
        
        # Ensure loss function's proj tensor is on the correct device
        if hasattr(self.loss_fn, 'proj') and self.loss_fn.proj.device != device:
            self.loss_fn.proj = self.loss_fn.proj.to(device)
        if hasattr(self.loss_fn, 'device') and self.loss_fn.device != device:
            self.loss_fn.device = device
        
        # Convert DAGR labels to YOLOv13 format
        batch = self._convert_labels_to_yolov13_format(labels, imgs)
        
        # YOLOv13 outputs are in training format: list of [B, no, H, W]
        # v8DetectionLoss expects (preds, batch) where preds is (y, x) tuple
        # y is inference output, x is training output
        # For training, we pass training outputs as both
        preds = (None, yolo13_outputs)  # (inference_output, training_output)
        
        # Compute loss
        total_loss, loss_components = self.loss_fn(preds, batch)
        
        # loss_components is a tensor [loss_box, loss_cls, loss_dfl]
        loss_dict = {
            'loss_box': loss_components[0],
            'loss_cls': loss_components[1],
            'loss_dfl': loss_components[2] if len(loss_components) > 2 else torch.tensor(0.0, device=loss_components[0].device)
        }
        
        return total_loss, loss_dict
    
    def _convert_labels_to_yolov13_format(self, labels, imgs):
        """
        Convert DAGR labels to YOLOv13 batch format.
        
        DAGR labels format from convert_to_training_format:
        - Shape: [batch_size, max_detections, 5]
        - Columns: [label, cx, cy, w, h] (label is class index, bbox in pixels)
        OR tuple of labels for multi-branch: (fused_labels, image_labels, mad_labels)
        YOLOv13 batch format: dict with 'batch_idx', 'cls', 'bboxes'
        """
        # Handle tuple format (multi-branch): use first element (fused branch)
        if isinstance(labels, tuple):
            labels = labels[0]  # Use fused branch labels
        
        if labels is None or len(labels) == 0:
            # Empty batch
            batch_size = imgs.shape[0] if imgs is not None else 1
            return {
                'batch_idx': torch.zeros(0, dtype=torch.long, device=imgs.device if imgs is not None else 'cpu'),
                'cls': torch.zeros(0, dtype=torch.long, device=imgs.device if imgs is not None else 'cpu'),
                'bboxes': torch.zeros(0, 4, device=imgs.device if imgs is not None else 'cpu'),
            }
        
        device = labels.device
        
        # DAGR labels format: [batch_size, max_detections, 5] = [batch_idx, label, cx, cy, w, h]
        # Need to flatten and filter out zero-padded entries
        if len(labels.shape) == 3:
            # [batch_size, max_detections, 5] -> flatten to [batch_size * max_detections, 5]
            batch_size, max_detections, num_cols = labels.shape
            labels_flat = labels.view(-1, num_cols)
            
            # Filter out zero-padded entries (where all values are zero)
            # Check if any column is non-zero (at least one of label, cx, cy, w, h is non-zero)
            non_zero_mask = labels_flat.abs().sum(dim=1) > 0
            labels_filtered = labels_flat[non_zero_mask]
            
            if len(labels_filtered) == 0:
                # No valid labels
                return {
                    'batch_idx': torch.zeros(0, dtype=torch.long, device=device),
                    'cls': torch.zeros(0, dtype=torch.long, device=device),
                    'bboxes': torch.zeros(0, 4, device=device),
                }
            
            # Reconstruct batch_idx from flattened indices
            batch_indices = torch.arange(batch_size, device=device).repeat_interleave(max_detections)
            batch_idx = batch_indices[non_zero_mask]
            
            # Extract class and bbox coordinates
            # labels_filtered format: [label, cx, cy, w, h]
            cls = labels_filtered[:, 0].long()
            cx, cy, w, h = labels_filtered[:, 1], labels_filtered[:, 2], labels_filtered[:, 3], labels_filtered[:, 4]
        else:
            # Already 2D: [N, 5] format
            batch_size = imgs.shape[0] if imgs is not None else 1
            labels_flat = labels.view(-1, labels.shape[-1])
            
            # Filter out zero-padded entries
            non_zero_mask = labels_flat.abs().sum(dim=1) > 0
            labels_filtered = labels_flat[non_zero_mask]
            
            if len(labels_filtered) == 0:
                return {
                    'batch_idx': torch.zeros(0, dtype=torch.long, device=device),
                    'cls': torch.zeros(0, dtype=torch.long, device=device),
                    'bboxes': torch.zeros(0, 4, device=device),
                }
            
            # Assume first column is batch_idx if available, otherwise infer from shape
            if labels_filtered.shape[1] == 6:
                # Format: [batch_idx, label, cx, cy, w, h]
                batch_idx = labels_filtered[:, 0].long()
                cls = labels_filtered[:, 1].long()
                cx, cy, w, h = labels_filtered[:, 2], labels_filtered[:, 3], labels_filtered[:, 4], labels_filtered[:, 5]
            else:
                # Format: [label, cx, cy, w, h] - need to infer batch_idx
                # This is tricky, we'll use 0 for all (single batch assumption)
                batch_idx = torch.zeros(len(labels_filtered), dtype=torch.long, device=device)
                cls = labels_filtered[:, 0].long()
                cx, cy, w, h = labels_filtered[:, 1], labels_filtered[:, 2], labels_filtered[:, 3], labels_filtered[:, 4]
        
        # Convert from (cx, cy, w, h) to (x1, y1, x2, y2)
        # x1 = cx - w / 2
        # y1 = cy - h / 2
        # x2 = cx + w / 2
        # y2 = cy + h / 2
        
        # # Stack bboxes: [N, 4]
        # bboxes = torch.stack([x1, y1, x2, y2], dim=1)
        # 1. 获取图像尺寸用于归一化
        if imgs is not None:
            _, _, H, W = imgs.shape
        else:
            # 极少见情况，防止除以零
            H, W = 1.0, 1.0 

       # 2. 直接归一化 (保持 cx, cy, w, h 格式)
        cx_norm = cx / W
        cy_norm = cy / H
        w_norm = w / W
        h_norm = h / H
        
        # Stack bboxes: [N, 4] -> [cx, cy, w, h]
        bboxes = torch.stack([cx_norm, cy_norm, w_norm, h_norm], dim=1)
        
        # 3. 防止数值越界 (Clamp to [0, 1])
        bboxes = torch.clamp(bboxes, 0.0, 1.0)
        
        return {
            'batch_idx': batch_idx,
            'cls': cls,
            'bboxes': bboxes,
        }


class YOLOv13HeadAdapter(nn.Module):
    """
    Adapter: Integrates YOLOv13 Detect head into DAGR's YOLOX interface.
    Returns YOLOv13 loss format (dict) instead of YOLOX format (tuple).
    """
    
    def __init__(self, num_classes, strides, in_channels, args=None):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.num_scales = len(in_channels)
        
        # Create YOLOv13 Detect head (imported from ultralytics)
        self.detect_head = Detect(nc=num_classes, ch=in_channels)
        
        # Initialize stride for Detect head
        if hasattr(self.detect_head, 'stride'):
            self.detect_head.stride = torch.tensor(strides, dtype=torch.float32)
        
        # Create YOLOv13 Loss adapter
        self.loss_fn = YOLOv13LossAdapter(self.detect_head, num_classes, reg_max=self.detect_head.reg_max)
        
        # Compatible attributes
        self.use_l1 = False
        self.decode_in_inference = True
        
    def forward(self, xin, labels=None, imgs=None):
        """
        Adapter for YOLOX interface.
        
        Args:
            xin: Multi-scale feature list [feat1, feat2, ...], each is [B, C, H, W]
            labels: Training labels (DAGR format)
            imgs: Images [B, C, H, W] (for loss calculation)
            
        Returns:
            Training: Loss dictionary with 'total_loss', 'loss_box', 'loss_cls', 'loss_dfl'
            Inference: Decoded outputs [B, N, 85] (YOLOX format for compatibility)
        """
        # 1. Forward through YOLOv13 Detect head
        yolo13_outputs = self.detect_head(xin)
        
        if self.training:
            # 2. Training: Compute YOLOv13 loss and return as dict
            # Training mode: detect_head returns list of [B, no, H, W] tensors
            total_loss, loss_dict = self.loss_fn(yolo13_outputs, labels, imgs)
            
            # Return YOLOv13 format loss dict (for DAGR to handle)
            return {
                'total_loss': total_loss,
                **loss_dict
            }
        else:
            # 3. Inference: Convert output format to YOLOX compatible format
            # Inference mode: detect_head returns (decoded_tensor, logits_list) tuple
            # or just decoded_tensor if export=True
            if isinstance(yolo13_outputs, tuple):
                # Non-export mode: (decoded_tensor, logits_list)
                decoded_tensor = yolo13_outputs[0]  # [B, N, 4+nc]
                return self._decode_outputs(decoded_tensor)
            else:
                # Export mode or single tensor: already decoded
                return self._decode_outputs(yolo13_outputs)
    
    def _decode_outputs(self, yolo13_outputs):
        """
        Convert YOLOv13 outputs to YOLOX format [B, N, 85].
        
        YOLOv13 output format in training: [B, no, H, W] list, where no = nc + reg_max*4
        YOLOv13 output format in inference: [B, N, 4+nc] tensor (already decoded)
        YOLOX format: [B, N, 85] where 85 = reg(4) + obj(1) + cls(80)
        """
        # Check if output is already decoded (inference mode) or raw logits (training mode)
        if isinstance(yolo13_outputs, torch.Tensor):
            if yolo13_outputs.shape[1] == 4 + self.num_classes:
                yolo13_outputs = yolo13_outputs.permute(0, 2, 1)
            # Already decoded: [B, N, 4+nc] format
            # YOLOv13 inference output: [B, N, 4+nc] = [dbox(4), cls(nc)]
            # YOLOX format: [B, N, 85] = [reg(4), obj(1), cls(80)]
            B, N, C = yolo13_outputs.shape
            reg_4d = yolo13_outputs[:, :, :4]  # [B, N, 4] - already decoded boxes
            cls_logits = yolo13_outputs[:, :, 4:]  # [B, N, nc] - already sigmoided
            
            # Add obj branch (use max cls as obj approximation)
            obj_logits = cls_logits.max(dim=2)[0].unsqueeze(2)  # [B, N, 1]
            
            # Combine to YOLOX format: [B, N, 85]
            yolox_output = torch.cat([reg_4d, obj_logits, cls_logits], dim=2)
            return yolox_output
        else:
            # Should not reach here in inference mode (handled above)
            # This branch is for training mode raw logits: [B, no, H, W] for each scale
            first_output = yolo13_outputs[0]
            # Raw logits: [B, no, H, W] for each scale
            # 1. Concatenate all scale outputs
            batch_size = first_output.shape[0]
            outputs_list = []
            
            for output in yolo13_outputs:
                # output: [B, no, H, W]
                if len(output.shape) == 4:
                    B, no, H, W = output.shape
                    # Flatten: [B, no, H*W]
                    output_flat = output.view(B, no, -1)
                    outputs_list.append(output_flat)
                else:
                    # Unexpected format, skip or handle error
                    raise ValueError(f"Unexpected output shape: {output.shape}, expected 4D [B, no, H, W] or 3D [B, N, 4+nc]")
            
            # Concatenate: [B, no, total_anchors]
            outputs_cat = torch.cat(outputs_list, dim=2)  # [B, no, N]
            outputs_cat = outputs_cat.permute(0, 2, 1)  # [B, N, no]
            
            # 2. Separate reg and cls
            reg_max = self.detect_head.reg_max
            reg_dist = outputs_cat[:, :, :reg_max*4]  # [B, N, 64]
            cls_logits = outputs_cat[:, :, reg_max*4:]  # [B, N, 80]
            
            # 3. Convert DFL distribution to 4 coordinate values
            reg_4d = self._dfl_to_4d(reg_dist)  # [B, N, 4]
            
            # 4. Add obj branch (extract from cls or use fixed value)
            # Use max cls as obj (approximation)
            obj_logits = cls_logits.max(dim=2)[0].unsqueeze(2)  # [B, N, 1]
            
            # 5. Combine to YOLOX format: [B, N, 85]
            yolox_output = torch.cat([reg_4d, obj_logits, cls_logits], dim=2)
            
            return yolox_output
    
    def _dfl_to_4d(self, reg_dist):
        """
        Convert DFL distribution (reg_max*4) to 4 coordinate values.
        
        Args:
            reg_dist: [B, N, reg_max*4]
            
        Returns:
            reg_4d: [B, N, 4]
        """
        B, N, _ = reg_dist.shape
        reg_max = self.detect_head.reg_max
        
        # Reshape to [B, N, 4, reg_max]
        reg_dist = reg_dist.view(B, N, 4, reg_max)
        
        # Softmax + weighted sum
        proj = torch.arange(reg_max, dtype=reg_dist.dtype, device=reg_dist.device)
        reg_4d = reg_dist.softmax(dim=-1).matmul(proj)  # [B, N, 4]
        
        return reg_4d


class HybridHeadV2YOLOv13(nn.Module):
    """
    Multi-branch fusion head using YOLOv13 heads instead of YOLOX heads.
    Supports Fused + Image branches (NO_MAD=1 case).
    """
    
    def __init__(self, num_classes, strides, 
                 in_channels_fused, 
                 in_channels_image, 
                 in_channels_mad, 
                 act="silu", depthwise=False, width=1.0, args=None):
        super().__init__()
        
        self.strides = strides
        self.num_scales = len(in_channels_fused)
        self.use_mad = in_channels_mad is not None
        
        # Create three YOLOv13 heads
        self.fused_head = YOLOv13HeadAdapter(num_classes, strides, in_channels_fused, args)
        self.image_head = YOLOv13HeadAdapter(num_classes, strides, in_channels_image, args)
        if self.use_mad:
            self.mad_head = YOLOv13HeadAdapter(num_classes, strides, in_channels_mad, args)
        else:
            self.mad_head = None
        
        # Checkpointing support
        self.use_checkpointing = getattr(args, 'use_checkpointing', False) if args else False
        
    def forward(self, xin, labels=None, imgs=None):
        """
        Forward pass for multi-branch fusion.
        
        Args:
            xin: (fused_feats, image_feats, mad_feats) - tuple of feature lists
            labels: Training labels (DAGR format)
            imgs: Images [B, C, H, W]
            
        Returns:
            Training: Loss dictionary with 'total_loss', 'loss_box', 'loss_cls', 'loss_dfl'
            Inference: Decoded outputs [B, N, 85]
        """
        fused_feats, image_feats, mad_feats = xin
        
        if self.training:
            # Training mode: Get raw logits from detect_head and fuse at logits level
            # 1. Get raw outputs from each branch (YOLOv13 format: list of [B, no, H, W])
            if self.use_checkpointing:
                # Detect.forward() takes a single argument x (list of features)
                # activation_checkpoint will unpack arguments, so we need to wrap the list
                # in a tuple and then unpack it in the function
                def _forward_detect_fused(*feats_tuple):
                    # Unpack tuple back to list
                    feats_list = list(feats_tuple) if len(feats_tuple) > 1 else feats_tuple[0]
                    return self.fused_head.detect_head(feats_list)
                
                def _forward_detect_image(*feats_tuple):
                    feats_list = list(feats_tuple) if len(feats_tuple) > 1 else feats_tuple[0]
                    return self.image_head.detect_head(feats_list)
                
                def _forward_detect_mad(*feats_tuple):
                    feats_list = list(feats_tuple) if len(feats_tuple) > 1 else feats_tuple[0]
                    return self.mad_head.detect_head(feats_list)
                
                # Pass each feature as a separate argument, then reconstruct list in function
                fused_outputs = activation_checkpoint(
                    _forward_detect_fused, *fused_feats, use_reentrant=False
                )
                image_outputs = activation_checkpoint(
                    _forward_detect_image, *image_feats, use_reentrant=False
                )
                if self.use_mad and self.mad_head:
                    mad_outputs = activation_checkpoint(
                        _forward_detect_mad, *mad_feats, use_reentrant=False
                    )
                else:
                    mad_outputs = None
            else:
                fused_outputs = self.fused_head.detect_head(fused_feats)
                image_outputs = self.image_head.detect_head(image_feats)
                mad_outputs = self.mad_head.detect_head(mad_feats) if self.use_mad and self.mad_head else None
            
            # 2. Fuse outputs at logits level (add them)
            # YOLOv13 outputs are [B, no, H, W] lists
            fused_outputs_list = []
            for i in range(self.num_scales):
                fused_out = fused_outputs[i]
                image_out = image_outputs[i]
                
                # Check shapes match before adding
                if fused_out.shape != image_out.shape:
                    # If shapes don't match, try to handle gracefully
                    min_batch = min(fused_out.shape[0], image_out.shape[0])
                    if fused_out.shape[1:] != image_out.shape[1:]:
                        # Feature dimensions don't match, only use fused branch
                        fused_out = fused_out[:min_batch]
                    else:
                        # Only batch size differs, truncate to match
                        fused_out = fused_out[:min_batch]
                        image_out = image_out[:min_batch]
                
                if self.use_mad and mad_outputs:
                    mad_out = mad_outputs[i]
                    # Check mad_out shape too
                    if mad_out.shape == fused_out.shape:
                        fused_out = fused_out + image_out.detach() + mad_out.detach()
                    else:
                        fused_out = fused_out + image_out.detach()
                else:
                    # NO_MAD=1 case: only fuse Fused + Image
                    fused_out = fused_out + image_out.detach()
                
                fused_outputs_list.append(fused_out)
            
            # 3. Compute loss
            total_loss, loss_dict = self.fused_head.loss_fn(fused_outputs_list, labels, imgs)
            return {
                'total_loss': total_loss,
                **loss_dict
            }
        else:
            # --- Inference Mode Fix ---
            
            # 1. 获取两个分支的原始 Logits (不要直接调用 forward，因为它会解码)
            # detect_head 返回 (decoded, [logits_list])，我们需要 logits_list (index 1)
            fused_out = self.fused_head.detect_head(fused_feats)
            image_out = self.image_head.detect_head(image_feats)
            
            # 提取 Logits 列表
            fused_logits = fused_out[1] 
            image_logits = image_out[1]
            
            # 2. 执行 Logits 融合 (与训练一致)
            summed_logits = [f + i for f, i in zip(fused_logits, image_logits)]
            
            # 3. 手动解码融合后的 Logits
            head = self.fused_head.detect_head
            shape = summed_logits[0].shape
            
            # 确保 Anchor 已生成
            if head.dynamic or head.shape != shape:
                head.anchors, head.strides = (x.transpose(0, 1) for x in make_anchors(summed_logits, head.stride, 0.5))
                head.shape = shape

            # (a) 拼接所有尺度: [B, 66, Total_Anchors]
            x_cat = torch.cat([xi.view(shape[0], head.no, -1) for xi in summed_logits], 2)
            
            # (b) 分割 Box 和 Cls
            box, cls = x_cat.split((head.reg_max * 4, head.nc), 1)
            
            # (c) 解码 Box (DFL -> Dist -> BBox)
            if head.reg_max > 1:
                # head.dfl 是 DFL 模块，负责 softmax 和卷积
                dbox = dist2bbox(head.dfl(box), head.anchors.unsqueeze(0), xywh=True, dim=1) * head.strides
            else:
                dbox = dist2bbox(box.sigmoid(), head.anchors.unsqueeze(0), xywh=True, dim=1) * head.strides

            # (d) 解码 Cls (Sigmoid)
            cls_scores = cls.sigmoid()
            
            # --- 优化: 使用全 1 作为 Objectness ---
            # 这样 score = 1.0 * cls_score，保持原汁原味
            # 而不是 score = max(cls) * cls_score (会变成平方)
            B, nc, N = cls_scores.shape
            obj_scores = torch.ones((B, 1, N), dtype=cls_scores.dtype, device=cls_scores.device)
            
            # [B, 4, N] -> [B, N, 4]
            dbox = dbox.transpose(1, 2)
            # [B, nc, N] -> [B, N, nc]
            cls_scores = cls_scores.transpose(1, 2)
            # [B, 1, N] -> [B, N, 1]
            obj_scores = obj_scores.transpose(1, 2)
            
            return torch.cat([dbox, obj_scores, cls_scores], dim=2)
