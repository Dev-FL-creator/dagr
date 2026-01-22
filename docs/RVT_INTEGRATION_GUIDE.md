# RVT Backbone 集成指南

本文档说明如何将 RVT backbone 无缝替换 SDTv3 backbone 用于双分支融合检测。

## 架构概述

### 输出维度对齐
两个 backbone 都输出 3 个特征层，维度完全一致：

| Stage | Stride | SDTv3 Channels | RVT Channels | 空间分辨率 (H/stride, W/stride) |
|-------|--------|----------------|--------------|-------------------------------|
| 2     | 8      | 256            | 128          | (H/8, W/8)                    |
| 3     | 16     | 512            | 256          | (H/16, W/16)                  |
| 4     | 32     | 640            | 512          | (H/32, W/32)                  |

**注意**: 可以通过配置 `rvt_embed_dim` 来调整 RVT 的通道数以匹配 SDTv3。

### 融合方式
通过 `SpikeCAFR` 模块逐层融合 RGB 和事件特征：
```
RGB features:   [rgb_c3, rgb_c4, rgb_c5]  # 来自ResNet
Event features: [evt_c2, evt_c3, evt_c4]   # 来自RVT/SDTv3
                    ↓        ↓        ↓
Fusion modules: [fuse1,  fuse2,  fuse3]
                    ↓        ↓        ↓
Fused output:   [out1,   out2,   out3]    # 送入检测头
```

## 快速开始

### 1. 使用 RVT backbone

在训练脚本中设置 `backbone_type="rvt"`:

```yaml
# config/dagr-rvt-dsec.yaml
backbone_type: rvt
use_snn_backbone: true
use_image: true  # 启用双分支

# RVT 配置
rvt_embed_dim: [64, 128, 256, 512, 640]  # 可调整以匹配融合需求
rvt_depths: [2, 2, 6, 2]
rvt_dim_head: 32
rvt_use_lstm: false
```

### 2. 使用 SDTv3 backbone

```yaml
# config/dagr-sdtv3-dsec.yaml
backbone_type: sdtv3
use_snn_backbone: true
use_image: true

# SDTv3 配置
sdt_embed_dim: [128, 256, 512, 640]
sdt_depths: [2, 2, 6, 2]
sdt_T: 4
```

### 3. 运行训练

```bash
# 使用 RVT
python scripts/train_dsec_rvt_2branch.py --config config/dagr-rvt-dsec.yaml

# 使用 SDTv3
python scripts/train_dsec_snn_v3_2branch.py --config config/dagr-sdtv3-dsec.yaml
```

## 关键参数说明

### RVT 参数

- **rvt_embed_dim**: 各阶段的通道数 `[stem, stage2, stage3, stage4, stage5]`
  - 默认: `[64, 128, 256, 512, 640]`
  - 输出阶段是 stage2, stage3, stage4
  
- **rvt_depths**: 各阶段的 block 数量 `[stage2, stage3, stage4, stage5]`
  - 默认: `[2, 2, 6, 2]`
  
- **rvt_dim_head**: Attention head 的维度
  - 默认: 32
  
- **rvt_partition_size**: Window/Grid attention 的分区大小
  - 默认: `(7, 7)`
  
- **rvt_use_lstm**: 是否使用 LSTM 进行时序建模
  - 默认: False
  
- **rvt_return_temporal**: 是否返回时序特征
  - False: 输出 `(B, C, H, W)`
  - True: 输出 `(T, B, C, H, W)`

### 通道数匹配策略

有两种方式确保融合兼容：

#### 方案 1: 调整 RVT 通道数匹配 SDTv3
```yaml
rvt_embed_dim: [128, 256, 512, 640, 768]  # stage2=256, stage3=512, stage4=640
```

#### 方案 2: 在融合模块中自动适配
`SpikeCAFR` 模块会自动处理通道不匹配：
```python
SpikeCAFR(rgb_in_channels=256, evt_in_channels=128, out_channels=256)
```

## 代码结构

```
src/dagr/model/
├── backbones/
│   ├── sdt_v3.py              # SDTv3 backbone
│   └── RvtBackone.py          # RVT backbone + RVTExtractor
├── networks/
│   ├── hybrid_backbone_rvt_2branch.py  # 双分支融合逻辑
│   └── dagr_fusion_seperate_heads_v3_2branch.py  # 检测网络
└── layers/
    └── fusion_2branch.py      # SpikeCAFR 融合模块
```

## 特性对比

| 特性 | SDTv3 | RVT |
|-----|-------|-----|
| 架构类型 | Spiking Transformer | Standard Transformer |
| 时序建模 | 原生支持 (T, B, C, H, W) | 可选 LSTM |
| 注意力机制 | Linear Attention | Window/Grid Attention |
| 内存效率 | 高 (脉冲编码) | 中等 |
| 训练稳定性 | 需要特殊初始化 | 较稳定 |
| 预训练权重 | 支持 | 支持 |

## 调试技巧

### 1. 检查特征维度
```python
# 在 HybridBackbone.forward() 中添加
print(f"Event features shapes: {[f.shape for f in event_feats]}")
print(f"RGB features shapes: {[f.shape for f in rgb_feats]}")
```

### 2. 验证融合输出
```python
# 在融合后
for i, feat in enumerate(fused):
    print(f"Fused stage {i}: {feat.shape}")
```

### 3. 监控梯度流
```python
# 检查是否有梯度消失
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm():.4f}")
```

## 常见问题

### Q1: 通道数不匹配怎么办？
A: `SpikeCAFR` 会自动处理。或者调整 `rvt_embed_dim` 配置。

### Q2: 如何加载预训练权重？
A: 设置 `load_pretrained_weight: "/path/to/checkpoint.pth"` 在配置文件中。

### Q3: 内存不足？
A: 减小 `batch_size` 或使用梯度累积 `accum_steps`。

### Q4: 双分支训练慢？
A: 
- 启用混合精度训练 (注意当前脚本禁用了 AMP)
- 减小 `rvt_depths` 
- 使用更小的 `rvt_partition_size`

## 性能优化建议

1. **通道数配置**: 
   - 小模型: `[64, 128, 256, 384]`
   - 中模型: `[64, 128, 256, 512]`
   - 大模型: `[128, 256, 512, 640]`

2. **Depth 配置**:
   - 快速实验: `[1, 1, 3, 1]`
   - 标准配置: `[2, 2, 6, 2]`
   - 深度模型: `[3, 4, 12, 3]`

3. **训练策略**:
   - 先预训练单分支（只用事件或只用图像）
   - 再微调双分支融合
   - 使用较小的学习率 (1e-4 ~ 2e-4)

## 总结

RVT backbone 提供了与 SDTv3 完全兼容的接口：
- ✅ 相同的输出格式 `[stage2, stage3, stage4]`
- ✅ 相同的 stride `[8, 16, 32]`
- ✅ 可配置的通道数匹配
- ✅ 无缝集成到双分支融合架构
- ✅ 支持事件数据输入 (torch_geometric.Data)

只需修改配置文件中的 `backbone_type`，即可在 RVT 和 SDTv3 之间切换！
