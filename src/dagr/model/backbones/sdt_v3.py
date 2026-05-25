import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

try:
    from timm.models.layers import DropPath
except Exception as exc:
    raise ImportError("timm is required for SpikformerV3Extractor") from exc

try:
    from spikingjelly.clock_driven import functional
except Exception as exc:
    raise ImportError("spikingjelly is required for SpikformerV3Extractor") from exc

try:
    from torch_geometric.data import Data
except Exception:
    Data = None


DEFAULT_SPIKE_NORM = 4.0


class Quant(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, i, min_value, max_value):
        ctx.min = min_value
        ctx.max = max_value
        ctx.save_for_backward(i)
        return torch.round(torch.clamp(i, min=min_value, max=max_value))

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        (i,) = ctx.saved_tensors
        grad_input[i < ctx.min] = 0
        grad_input[i > ctx.max] = 0
        return grad_input, None, None


class MultiSpike(nn.Module):
    def __init__(self, spike_norm=DEFAULT_SPIKE_NORM):
        super().__init__()
        self.spike_norm = spike_norm

    def forward(self, x):
        if self.training:
            return Quant.apply(x, 0.0, self.spike_norm) / self.spike_norm
        else:
            return torch.clamp(x, min=0, max=self.spike_norm).round_() / self.spike_norm


class SepConv(nn.Module):
    def __init__(
        self,
        dim,
        expansion_ratio=2,
        bias=False,
        kernel_size=7,
        padding=3,
        spike_norm=DEFAULT_SPIKE_NORM,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        
        self.spike1 = MultiSpike(spike_norm=spike_norm)
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(med_channels)
        
        self.spike2 = MultiSpike(spike_norm=spike_norm)
        self.dwconv = nn.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=med_channels,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(med_channels)
        
        self.spike3 = MultiSpike(spike_norm=spike_norm)
        self.pwconv2 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        
        x = self.spike1(x)
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W).contiguous()
        
        x = self.spike2(x)
        x = self.bn2(self.dwconv(x.flatten(0, 1))).reshape(T, B, -1, H, W).contiguous()
        
        x = self.spike3(x)
        x = self.bn3(self.pwconv2(x.flatten(0, 1))).reshape(T, B, -1, H, W).contiguous()
        
        return x


class ChannelConv(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, spike_norm=DEFAULT_SPIKE_NORM):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        
        self.spike1 = MultiSpike(spike_norm=spike_norm)
        self.conv1 = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        self.spike2 = MultiSpike(spike_norm=spike_norm)
        self.conv2 = nn.Conv2d(hidden_dim, dim, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        
        x = self.spike1(x)
        x = self.bn1(self.conv1(x.flatten(0, 1))).reshape(T, B, -1, H, W).contiguous()
        
        x = self.spike2(x)
        x = self.bn2(self.conv2(x.flatten(0, 1))).reshape(T, B, C, H, W).contiguous()
        
        return x


class MS_ConvBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, spike_norm=DEFAULT_SPIKE_NORM):
        super().__init__()
        self.sepconv = SepConv(dim=dim, spike_norm=spike_norm)
        self.channel_conv = ChannelConv(dim=dim, mlp_ratio=mlp_ratio, spike_norm=spike_norm)

    def forward(self, x):
        x = x + self.sepconv(x)
        x = x + self.channel_conv(x)
        return x


class ChannelMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, spike_norm=DEFAULT_SPIKE_NORM):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.spike1 = MultiSpike(spike_norm=spike_norm)
        
        self.fc1 = nn.Conv1d(in_features, hidden_features, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        
        self.spike2 = MultiSpike(spike_norm=spike_norm)
        
        self.fc2 = nn.Conv1d(hidden_features, out_features, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        T, B, C, N = x.shape
        
        x = self.spike1(x)
        x = self.bn1(self.fc1(x.flatten(0, 1))).reshape(T, B, self.hidden_features, N).contiguous()
        
        x = self.spike2(x)
        x = self.bn2(self.fc2(x.flatten(0, 1))).reshape(T, B, self.out_features, N).contiguous()
        
        return x


class LinearAttentionFunction(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, q, k, v, scale):
        kv = (k.unsqueeze(-1) * v.unsqueeze(-2)).sum(dim=-3)
        out = (q.unsqueeze(-1) * kv.unsqueeze(-3)).sum(dim=-2) * scale
        ctx.save_for_backward(q, k, v, kv)
        ctx.scale = scale
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_out):
        q, k, v, kv = ctx.saved_tensors
        scale = ctx.scale
        
        grad_out_scaled = grad_out * scale
        
        grad_q = (grad_out_scaled.unsqueeze(-2) * kv.unsqueeze(-3)).sum(dim=-1)
        grad_kv = (q.unsqueeze(-1) * grad_out_scaled.unsqueeze(-2)).sum(dim=-3)
        grad_k = (grad_kv.unsqueeze(-3) * v.unsqueeze(-2)).sum(dim=-1)
        grad_v = (grad_kv.unsqueeze(-3) * k.unsqueeze(-1)).sum(dim=-2)
        
        return grad_q, grad_k, grad_v, None


class MS_Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        spike_norm=DEFAULT_SPIKE_NORM,
        lamda_ratio=4,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.lamda_ratio = lamda_ratio
        self.v_head_dim = (dim * lamda_ratio) // num_heads
        
        self.head_spike = MultiSpike(spike_norm=spike_norm)
        
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=True)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_spike = MultiSpike(spike_norm=spike_norm)
        
        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=True)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_spike = MultiSpike(spike_norm=spike_norm)
        
        self.v_conv = nn.Conv1d(dim, int(dim * lamda_ratio), kernel_size=1, bias=True)
        self.v_bn = nn.BatchNorm1d(int(dim * lamda_ratio))
        self.v_spike = MultiSpike(spike_norm=spike_norm)
        
        self.attn_spike = MultiSpike(spike_norm=spike_norm)
        
        self.proj_conv = nn.Conv1d(int(dim * lamda_ratio), dim, kernel_size=1, bias=True)
        self.proj_bn = nn.BatchNorm1d(dim)

    def forward(self, x, H, W):
        T, B, C, N = x.shape
        C_v = int(C * self.lamda_ratio)
        
        x = self.head_spike(x)
        
        q = self.q_bn(self.q_conv(x.flatten(0, 1)))
        q = q.reshape(T, B, C, N).contiguous()
        q = self.q_spike(q)
        
        k = self.k_bn(self.k_conv(x.flatten(0, 1)))
        k = k.reshape(T, B, C, N).contiguous()
        k = self.k_spike(k)
        
        v = self.v_bn(self.v_conv(x.flatten(0, 1)))
        v = v.reshape(T, B, C_v, N).contiguous()
        v = self.v_spike(v)
        
        q = q.transpose(-1, -2).reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = k.transpose(-1, -2).reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = v.transpose(-1, -2).reshape(T, B, N, self.num_heads, self.v_head_dim).permute(0, 1, 3, 2, 4)
        
        scale = self.scale * 2
        
        with torch.cuda.amp.autocast(enabled=False):
            q_f = q.float().contiguous()
            k_f = k.float().contiguous()
            v_f = v.float().contiguous()
            
            out = LinearAttentionFunction.apply(q_f, k_f, v_f, scale)
        
        out = out.permute(0, 1, 2, 4, 3).reshape(T, B, C_v, N).contiguous()
        out = out.to(x.dtype)
        out = self.attn_spike(out)
        
        out = self.proj_bn(self.proj_conv(out.flatten(0, 1))).reshape(T, B, C, N).contiguous()
        
        return out


class MS_Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        spike_norm=DEFAULT_SPIKE_NORM,
        lamda_ratio=4,
    ):
        super().__init__()
        
        self.sepconv = SepConv(dim=dim, kernel_size=3, padding=1, spike_norm=spike_norm)
        
        self.attn = MS_Attention(
            dim,
            num_heads=num_heads,
            spike_norm=spike_norm,
            lamda_ratio=lamda_ratio,
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ChannelMLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            spike_norm=spike_norm,
        )

    def forward(self, x, H, W):
        T, B, C, N = x.shape
        
        x_2d = x.reshape(T, B, C, H, W).contiguous()
        x_2d = x_2d + self.sepconv(x_2d)
        x = x_2d.reshape(T, B, C, N).contiguous()
        
        x = x + self.drop_path(self.attn(x, H, W))
        x = x + self.drop_path(self.mlp(x))
        
        return x


class MS_DownSampling(nn.Module):
    def __init__(
        self,
        in_channels=2,
        embed_dims=256,
        kernel_size=3,
        stride=2,
        padding=1,
        first_layer=True,
        spike_norm=DEFAULT_SPIKE_NORM,
    ):
        super().__init__()
        self.encode_conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.encode_bn = nn.BatchNorm2d(embed_dims)
        
        if not first_layer:
            self.encode_spike = MultiSpike(spike_norm=spike_norm)

    def forward(self, x):
        T, B, _, _, _ = x.shape
        if hasattr(self, "encode_spike"):
            x = self.encode_spike(x)
        x = self.encode_conv(x.flatten(0, 1))
        _, _, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()
        return x


class SpikformerV3Extractor(nn.Module):
    def __init__(
        self,
        args,
        height,
        width,
        embed_dim=None,
        depths=None,
        num_heads=None,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        pretrained_weight=None,
    ):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.T = int(getattr(args, "sdt_T", 4))
        self.repeat_static = bool(getattr(args, "sdt_repeat_static", False))
        self.input_t_first = bool(getattr(args, "sdt_input_t_first", False))
        self.use_checkpointing = False
        self.in_channels = int(getattr(args, "sdt_in_channels", getattr(args, "in_channels", 2)))
        self.spike_norm = float(getattr(args, "sdt_norm", DEFAULT_SPIKE_NORM))
        self.pretrained_weight = pretrained_weight or getattr(args, "load_pretrained_weight", None)
        self.return_temporal = bool(getattr(args, "sdt_return_temporal", True))

        if depths is None:
            depths = getattr(args, "sdt_depths", [2, 2, 6, 2])
        
        if isinstance(depths, list):
            if len(depths) != 4:
                raise ValueError(f"sdt_depths list must have length 4, got {depths}")
            self.depths = [int(d) for d in depths]
        else:
            raise TypeError(f"Invalid type for depths: {type(depths)}")

        num_heads = num_heads if num_heads is not None else getattr(args, "sdt_num_heads", 8)
        mlp_ratio = getattr(args, "sdt_mlp_ratio", mlp_ratio)
        lamda_ratio = getattr(args, "sdt_lamda_ratio", 4)

        embed_dim = embed_dim or getattr(args, "sdt_embed_dim", [128, 256, 512, 640])
        if len(embed_dim) < 4:
            raise ValueError("embed_dim must provide at least four stage dimensions.")

        self.embed_dim = embed_dim
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depths[2])]

        self.downsample1_1 = MS_DownSampling(
            in_channels=self.in_channels,
            embed_dims=embed_dim[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,
            spike_norm=self.spike_norm,
        )

        self.ConvBlock1_1 = nn.ModuleList([
            MS_ConvBlock(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratio, spike_norm=self.spike_norm)
        ])

        self.downsample1_2 = MS_DownSampling(
            in_channels=embed_dim[0] // 2,
            embed_dims=embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            spike_norm=self.spike_norm,
        )

        self.ConvBlock1_2 = nn.ModuleList([
            MS_ConvBlock(dim=embed_dim[0], mlp_ratio=mlp_ratio, spike_norm=self.spike_norm)
            for _ in range(self.depths[0])
        ])

        self.downsample2 = MS_DownSampling(
            in_channels=embed_dim[0],
            embed_dims=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            spike_norm=self.spike_norm,
        )

        self.ConvBlock2_1 = nn.ModuleList([
            MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratio, spike_norm=self.spike_norm)
        ])

        self.ConvBlock2_2 = nn.ModuleList([
            MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratio, spike_norm=self.spike_norm)
            for _ in range(self.depths[1])
        ])

        self.downsample3 = MS_DownSampling(
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            spike_norm=self.spike_norm,
        )

        self.block3 = nn.ModuleList([
            MS_Block(
                dim=embed_dim[2],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[j],
                spike_norm=self.spike_norm,
                lamda_ratio=lamda_ratio,
            )
            for j in range(self.depths[2])
        ])

        self.downsample4 = MS_DownSampling(
            in_channels=embed_dim[2],
            embed_dims=embed_dim[3],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            spike_norm=self.spike_norm,
        )

        self.block4 = nn.ModuleList([
            MS_Block(
                dim=embed_dim[3],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=0.0,
                spike_norm=self.spike_norm,
                lamda_ratio=lamda_ratio,
            )
            for _ in range(self.depths[3])
        ])

        self.out_channels = [embed_dim[1], embed_dim[2], embed_dim[3]]
        self.strides = [8, 16, 32]
        self.num_scales = 3
        self.use_image = False
        self.is_snn = True
        self.num_classes = getattr(args, "num_classes", getattr(args, "n_classes", 2))

        if self.pretrained_weight:
            self._load_pretrained_weights(self.pretrained_weight)
        else:
            print("[SpikformerV3Extractor] Training from scratch. Applying SNN bias initialization...")
            self._init_snn_weights()

    def _init_snn_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_output_sizes(self):
        sizes = []
        for s in self.strides:
            sizes.append([max(1, self.height // s), max(1, self.width // s)])
        return sizes

    @staticmethod
    def _events_to_frames(data, height, width):
        device = data.x.device
        batch_size = int(data.num_graphs) if hasattr(data, "num_graphs") else 1
        frames = torch.zeros((batch_size, 2, height, width), dtype=torch.float32, device=device)

        if hasattr(data, "batch") and data.batch is not None:
            b = data.batch.long()
        else:
            b = torch.zeros((data.pos.shape[0],), dtype=torch.long, device=device)

        x_norm = data.pos[:, 0]
        y_norm = data.pos[:, 1]
        
        x_pix = (x_norm * (width - 1)).long()
        y_pix = (y_norm * (height - 1)).long()
        x_pix = torch.clamp(x_pix, 0, width - 1)
        y_pix = torch.clamp(y_pix, 0, height - 1)

        p = (data.x[:, 0] > 0).long()
        frames.index_put_((b, p, y_pix, x_pix), torch.ones_like(p, dtype=frames.dtype), accumulate=True)
        
        if torch.rand(1).item() < 0.01:
            print(f"[SDT Input Raw] Max events/pixel: {frames.max().item():.1f}, Mean: {frames.mean().item():.6f}", flush=True)

        frames = torch.clamp(frames, max=3.0) / 3.0
        
        if torch.rand(1).item() < 0.01:
            sat_pct = (frames >= 1.0).float().mean().item() * 100
            print(f"[SDT Input Norm] Max: {frames.max().item():.4f}, Mean: {frames.mean().item():.6f}, Saturated: {sat_pct:.2f}%", flush=True)
        
        return frames.unsqueeze(0)

    def _run_blocks(self, x, blocks, H=None, W=None):
        for blk in blocks:
            if H is not None and W is not None:
                x = blk(x, H, W)
            else:
                x = blk(x)
        return x

    def _prepare_input(self, x):
        if Data is not None and isinstance(x, Data):
            frames = self._events_to_frames(x, self.height, self.width)
            if self.repeat_static and self.T > 1:
                frames = frames.repeat(self.T, 1, 1, 1, 1)
            return frames.contiguous()

        if not torch.is_tensor(x):
            raise TypeError(f"Unsupported input type {type(x)}. Expected torch.Tensor or torch_geometric.data.Data.")

        if x.dim() == 4:
            x = x.unsqueeze(0)
            if self.repeat_static and self.T > 1:
                x = x.repeat(self.T, 1, 1, 1, 1)
            return x.contiguous()

        if x.dim() != 5:
            raise ValueError(f"Expected input with 4 or 5 dims, got {x.dim()}.")

        if self.input_t_first:
            return x.contiguous()

        if x.shape[0] == self.T and x.shape[1] == self.T:
            logging.warning(
                f"SpikformerV3Extractor: Ambiguous input shape (BatchSize==TimeStep={self.T}). "
                "Assuming input is (B, T, C, H, W) and permuting to (T, B, ...) because 'input_t_first' is False. "
                "If input is already (T, B, ...), set 'sdt_input_t_first=True'."
            )
            return x.permute(1, 0, 2, 3, 4).contiguous()

        if x.shape[1] == self.T and x.shape[0] != self.T:
            return x.permute(1, 0, 2, 3, 4).contiguous()
        if x.shape[0] == self.T:
            return x.contiguous()
        return x.permute(1, 0, 2, 3, 4).contiguous()

    def forward(self, x, reset=True):
        if reset:
            functional.reset_net(self)

        x = self._prepare_input(x)

        x = self.downsample1_1(x)
        x = self._run_blocks(x, self.ConvBlock1_1)

        x = self.downsample1_2(x)
        x = self._run_blocks(x, self.ConvBlock1_2)

        x = self.downsample2(x)
        x = self._run_blocks(x, self.ConvBlock2_1)
        stage2 = self._run_blocks(x, self.ConvBlock2_2)

        x = self.downsample3(stage2)
        T, B, C3, h3, w3 = x.shape
        x_tokens = x.reshape(T, B, C3, h3 * w3).contiguous()
        x_tokens = self._run_blocks(x_tokens, self.block3, h3, w3)
        stage3 = x_tokens.reshape(T, B, C3, h3, w3).contiguous()

        stage4 = self.downsample4(stage3)
        T, B, C4, h4, w4 = stage4.shape
        x_tokens = stage4.reshape(T, B, C4, h4 * w4).contiguous()
        x_tokens = self._run_blocks(x_tokens, self.block4, h4, w4)
        stage4 = x_tokens.reshape(T, B, C4, h4, w4).contiguous()

        if self.return_temporal:
            if torch.rand(1).item() < 0.005:
                p3_mean = stage2.mean(dim=0)
                p4_mean = stage3.mean(dim=0)
                p5_mean = stage4.mean(dim=0)
                print(f"[SDT Temporal Features] stage2: T={stage2.shape[0]}, mean={p3_mean.mean().item():.4f}, std={p3_mean.std().item():.4f}", flush=True)
                print(f"[SDT Temporal Features] stage3: T={stage3.shape[0]}, mean={p4_mean.mean().item():.4f}, std={p4_mean.std().item():.4f}", flush=True)
                print(f"[SDT Temporal Features] stage4: T={stage4.shape[0]}, mean={p5_mean.mean().item():.4f}, std={p5_mean.std().item():.4f}", flush=True)
            return [stage2, stage3, stage4]
        else:
            p3 = stage2.mean(dim=0).contiguous()
            p4 = stage3.mean(dim=0).contiguous()
            p5 = stage4.mean(dim=0).contiguous()
            
            if torch.rand(1).item() < 0.005:
                print(f"[SDT Features] p3: mean={p3.mean().item():.4f}, std={p3.std().item():.4f}, max={p3.max().item():.4f}", flush=True)
                print(f"[SDT Features] p4: mean={p4.mean().item():.4f}, std={p4.std().item():.4f}, max={p4.max().item():.4f}", flush=True)
                print(f"[SDT Features] p5: mean={p5.mean().item():.4f}, std={p5.std().item():.4f}, max={p5.max().item():.4f}", flush=True)
            
            return [p3, p4, p5]

    def _load_pretrained_weights(self, weight_path: str):
        try:
            ckpt = torch.load(weight_path, map_location="cpu")
            if isinstance(ckpt, dict):
                if "model" in ckpt:
                    ckpt = ckpt["model"]
                elif "state_dict" in ckpt:
                    ckpt = ckpt["state_dict"]

            if not isinstance(ckpt, dict):
                logging.warning(f"[SpikformerV3Extractor] Unexpected checkpoint format at {weight_path}, skip loading.")
                return

            filtered = {k: v for k, v in ckpt.items() if not k.startswith("head")}

            keys_to_remove = []
            for k in filtered.keys():
                if "downsample1_1" in k or "patch_embed" in k:
                    keys_to_remove.append(k)
            
            if len(keys_to_remove) > 0:
                logging.warning(f"[SpikformerV3Extractor] Force removing {len(keys_to_remove)} keys from first layer (downsample1_1) to avoid BN mismatch.")
                for k in keys_to_remove:
                    del filtered[k]

            missing, unexpected = self.load_state_dict(filtered, strict=False)
            logging.info(f"[SpikformerV3Extractor] Loaded pretrained weights from {weight_path}. "
                         f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        
        except Exception as exc:
            logging.warning(f"[SpikformerV3Extractor] Failed to load pretrained weights from {weight_path}: {exc}")