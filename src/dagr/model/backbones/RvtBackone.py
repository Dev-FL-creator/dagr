import torch
import torch.nn as nn
from timm.models.layers import DropPath
import math


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma
        return x.mul_(gamma) if self.inplace else x * gamma
    
class ConvDownsampling_Cf2Cl(nn.Module):
    def __init__(self, in_dim=5, layer_dim=64, downsample_factor=2,overlap=True,norm_affine=True):
        super().__init__()
        assert downsample_factor in (2, 4, 8)
        if overlap:
            kernel_size = (downsample_factor - 1)*2 + 1
            padding = kernel_size//2
        else:
            kernel_size = downsample_factor
            padding = 0
        self.conv = nn.Conv2d(in_dim, layer_dim, kernel_size=kernel_size, stride=downsample_factor, padding=padding, bias=False)
        self.ln= nn.LayerNorm(normalized_shape=layer_dim, eps=1e-5, elementwise_affine=norm_affine)
        self.in_dim = in_dim
        self.layer_dim = layer_dim
    
    def forward(self, x):
        # Input can be either (N, C, H, W) or (N, H, W, C)
        # Check format by matching expected input channels
        if x.dim() == 4:
            # If last dimension matches in_dim, it's channel-last format
            if x.shape[-1] == self.in_dim or (x.shape[1] != self.in_dim and x.shape[-1] <= 512):
                # Channel-last: (N, H, W, C) -> (N, C, H, W)
                x = x.permute(0, 3, 1, 2)
        
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # N C H W -> N H W C
        x = self.ln(x) # # LN over C_out
        return x #N H W C

class MaxVitAttentionPairCl(nn.Module):
    def __init__(self,
                 dim,
                 skip_first_norm,
                 dim_head=32,
                 partition_size=(7, 7),  # Add partition_size parameter
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 drop=0,
                 drop_path=0,
                 ):
        super().__init__()

        self.att_window = PartitionAttentionCl(dim=dim,
                                               dim_head=dim_head,
                                               partition_type="window",
                                               partition_size=partition_size,
                                               act_layer=act_layer,
                                               norm_layer=norm_layer,
                                               drop_mlp=drop,
                                               drop_path=drop_path,
                                               skip_first_norm=skip_first_norm
                                               )

        self.att_grid = PartitionAttentionCl(dim=dim,
                                             partition_type="grid",
                                             dim_head=dim_head,
                                             partition_size=partition_size,
                                             act_layer=act_layer,
                                             norm_layer=norm_layer,
                                             drop_mlp=drop,
                                             drop_path=drop_path,
                                             skip_first_norm=False
                                             )

    def forward(self, x):
        x = self.att_window(x)
        x = self.att_grid(x)
        return x

class PartitionAttentionCl(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            partition_type,
            partition_size=(11,11),
            drop_mlp=0.,
            drop_path=0.0,
            mlp_expand_ratio=4,
            ls_init_value=1e-5,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            skip_first_norm =False,
            attention_bias = True,
            norm_affine =True 
    ):
        super().__init__()
        self.partition_size = partition_size

        self.partition_type = partition_type

        self.norm1 = nn.Identity() if skip_first_norm else norm_layer(normalized_shape=dim, eps=1e-5, elementwise_affine=norm_affine)
        self.self_attn = SelfAttentionCl(dim,
                                         dim_head=dim_head,
                                         bias=attention_bias)
        self.ls1 = LayerScale(dim=dim, init_values=ls_init_value) if ls_init_value > 0 else nn.Identity()
        self.drop_path1 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(normalized_shape=dim, eps=1e-5, elementwise_affine=norm_affine)
        self.mlp = MLP(dim = dim,
                       channel_last=True,
                       expansion_ratio = mlp_expand_ratio,
                       act_layer = act_layer,
                       drop_prob = drop_mlp)
        self.ls2 = LayerScale(dim=dim, init_values=ls_init_value) if ls_init_value > 0 else nn.Identity()
        self.drop_path2 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()

    def _partition_attn(self, x):
        img_size = x.shape[1:3]
        if self.partition_type == "window":
            partitioned = window_partition(x, self.partition_size)
        else:
            partitioned = grid_partition(x, self.partition_size)

        partitioned = self.self_attn(partitioned)

        if self.partition_type == "window":
            x = window_reverse(partitioned, self.partition_size, (img_size[0], img_size[1]))
        else:
            x = grid_reverse(partitioned, self.partition_size, (img_size[0], img_size[1]))
        return x

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self._partition_attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def window_partition(x, window_size):#x:(B, H, W, C)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows #windows:(B * num_windows, Wh, Ww, C)

def window_reverse(windows, window_size, img_size):
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


def grid_partition(x, grid_size):
    B, H, W, C = x.shape
    x = x.view(B, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1], C)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, grid_size[0], grid_size[1], C)
    return windows


def grid_reverse(windows, grid_size, img_size):
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, H, W, C)
    return x


class SelfAttentionCl(nn.Module):
    """ Channels-last multi-head self-attention - completely manual implementation to avoid CUBLAS """
    def __init__(
            self,
            dim,
            dim_head = 32,
            bias=True):
        super().__init__()
        self.num_heads = dim // dim_head
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.dim = dim

        # Use Conv1d instead of Linear
        self.qkv_conv = nn.Conv1d(dim, dim * 3, kernel_size=1, bias=bias)
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)

    def manual_attention(self, q, k, v):
        """
        Manually compute attention using pure broadcasting operations.
        This avoids einsum/matmul/bmm which may call CUBLAS.
        
        q, k, v: (B, h, N, d)
        Returns: (B, h, N, d)
        """
        B, h, N, d = q.shape
        
        # Compute attention scores: Q @ K^T using broadcasting
        # For each query position i and key position j:
        # score[i,j] = sum_d(q[i,d] * k[j,d])
        
        # Reshape for broadcasting: 
        # q: (B, h, N, 1, d) - query positions
        # k: (B, h, 1, N, d) - key positions
        q_exp = q.unsqueeze(3)  # (B, h, N, 1, d)
        k_exp = k.unsqueeze(2)  # (B, h, 1, N, d)
        
        # Element-wise multiply and sum over d dimension
        # Result: (B, h, N, N) - attention scores
        attn_scores = (q_exp * k_exp).sum(dim=-1) * self.scale
        
        # Handle NaN/Inf
        if torch.isnan(attn_scores).any() or torch.isinf(attn_scores).any():
            attn_scores = torch.nan_to_num(attn_scores, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # Softmax over key positions
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, h, N, N)
        
        # Apply attention to values using broadcasting
        # For each query position i:
        # out[i] = sum_j(attn_weights[i,j] * v[j])
        
        # Reshape for broadcasting:
        # attn_weights: (B, h, N_q, N_kv, 1)
        # v: (B, h, 1, N_kv, d)
        attn_exp = attn_weights.unsqueeze(-1)  # (B, h, N, N, 1)
        v_exp = v.unsqueeze(2)  # (B, h, 1, N, d)
        
        # Element-wise multiply and sum over key positions (dim=3)
        # Result: (B, h, N, d)
        out = (attn_exp * v_exp).sum(dim=3)
        
        return out

    def forward(self, x):
        # x shape: (B, H, W, C) - channel last format
        B, H, W, C = x.shape
        N = H * W
        
        # Reshape to (B, C, N) for Conv1d
        x_seq = x.reshape(B, H * W, C).permute(0, 2, 1).contiguous()  # (B, C, N)
        
        # QKV projection using Conv1d
        qkv = self.qkv_conv(x_seq)  # (B, 3*C, N)
        qkv = qkv.reshape(B, 3, self.num_heads, self.dim_head, N)  # (B, 3, h, d, N)
        qkv = qkv.permute(1, 0, 2, 4, 3).contiguous()  # (3, B, h, N, d)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, h, N, d)
        
        # Use manual attention (pure broadcasting, no matmul/einsum)
        out = self.manual_attention(q, k, v)  # (B, h, N, d)
        
        # Reshape back
        out = out.permute(0, 2, 1, 3).contiguous()  # (B, N, h, d)
        out = out.reshape(B, N, C)  # (B, N, C)
        out = out.permute(0, 2, 1).contiguous()  # (B, C, N) for Conv1d
        
        # Project back using Conv1d
        out = self.proj_conv(out)  # (B, C, N)
        
        # Reshape back to original spatial shape
        out = out.permute(0, 2, 1).reshape(B, H, W, C)  # (B, H, W, C)
        
        return out

class GLU(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 channel_last,
                 act_layer = nn.GELU,
                 bias = True):
        super().__init__()
        proj_out_dim = dim_out*2
        # Use Conv1d for channel-last format to avoid CUBLAS
        if channel_last:
            self.proj = nn.Conv1d(dim_in, proj_out_dim, kernel_size=1, bias=bias)
        else:
            self.proj = nn.Conv2d(dim_in, proj_out_dim, kernel_size=1, stride=1, bias=bias)
        self.channel_dim = 1 if channel_last else 1  # Conv always uses dim=1 for channels
        self.channel_last = channel_last
        self.act_layer = act_layer()

    def forward(self, x: torch.Tensor):
        # If channel_last, convert (B, H, W, C) -> (B, C, N) for Conv1d
        if self.channel_last:
            B, H, W, C = x.shape
            x = x.reshape(B, H * W, C).permute(0, 2, 1)  # (B, C, N)
            x = self.proj(x)
            x, gate = torch.tensor_split(x, 2, dim=self.channel_dim)
            out = x * self.act_layer(gate)
            # Convert back to (B, H, W, C)
            out = out.permute(0, 2, 1).reshape(B, H, W, -1)
            return out
        else:
            x, gate = torch.tensor_split(self.proj(x), 2, dim=self.channel_dim)
            return x * self.act_layer(gate)
    
class MLP(nn.Module):
    def __init__(self,
                 dim,
                 channel_last,
                 expansion_ratio,
                 act_layer,
                 gated = False,
                 bias = True,
                 drop_prob = 0.):
        super().__init__()
        self.channel_last = channel_last
        inner_dim = int(dim * expansion_ratio)
        
        if gated:
            inner_dim = math.floor(inner_dim * 2 / 3 / 32) * 32 # multiple of 32
            proj_in = GLU(dim_in=dim, dim_out=inner_dim, channel_last=channel_last, act_layer=act_layer, bias=bias)
        else:
            # Use Conv1d for channel-last, Conv2d for channel-first
            if channel_last:
                proj_in = nn.Sequential(
                    nn.Conv1d(in_channels=dim, out_channels=inner_dim, kernel_size=1, bias=bias),
                    act_layer(),
                )
            else:
                proj_in = nn.Sequential(
                    nn.Conv2d(in_channels=dim, out_channels=inner_dim, kernel_size=1, stride=1, bias=bias),
                    act_layer(),
                )
        
        # Output projection
        if channel_last:
            proj_out = nn.Conv1d(in_channels=inner_dim, out_channels=dim, kernel_size=1, bias=bias)
        else:
            proj_out = nn.Conv2d(in_channels=inner_dim, out_channels=dim, kernel_size=1, stride=1, bias=bias)
        
        self.net = nn.Sequential(
            proj_in,
            nn.Dropout(p=drop_prob),
            proj_out
        )

    def forward(self, x):
        # If channel_last, convert (B, H, W, C) -> (B, C, N) for Conv1d
        if self.channel_last:
            B, H, W, C = x.shape
            x = x.reshape(B, H * W, C).permute(0, 2, 1)  # (B, C, N)
            x = self.net(x)
            # Convert back to (B, H, W, C)
            x = x.permute(0, 2, 1).reshape(B, H, W, -1)
            return x
        else:
            return self.net(x)

class DWSConvLSTM2d(nn.Module):
    """LSTM with (depthwise-separable) Conv option in NCHW [channel-first] format.
    """

    def __init__(self,
                 dim,
                 dws_conv= True,
                 dws_conv_only_hidden= True,
                 dws_conv_kernel_size = 3,
                 cell_update_dropout = 0.):
        super().__init__()
        assert isinstance(dws_conv, bool)
        assert isinstance(dws_conv_only_hidden, bool)
        self.dim = dim

        xh_dim = dim * 2
        gates_dim = dim * 4
        conv3x3_dws_dim = dim if dws_conv_only_hidden else xh_dim
        self.conv3x3_dws = nn.Conv2d(in_channels=conv3x3_dws_dim,
                                     out_channels=conv3x3_dws_dim,
                                     kernel_size=dws_conv_kernel_size,
                                     padding=dws_conv_kernel_size // 2,
                                     groups=conv3x3_dws_dim) if dws_conv else nn.Identity()
        self.conv1x1 = nn.Conv2d(in_channels=xh_dim,
                                 out_channels=gates_dim,
                                 kernel_size=1)
        self.conv_only_hidden = dws_conv_only_hidden
        self.cell_update_dropout = nn.Dropout(p=cell_update_dropout)

    def forward(self, x, h_and_c_previous = None):
        """
        :param x: (N C H W)
        :param h_and_c_previous: ((N C H W), (N C H W))
        :return: ((N C H W), (N C H W))
        """
        if h_and_c_previous is None:
            # generate zero states
            hidden = torch.zeros_like(x)
            cell = torch.zeros_like(x)
            h_and_c_previous = (hidden, cell)
        h_tm1, c_tm1 = h_and_c_previous

        if self.conv_only_hidden:
            h_tm1 = self.conv3x3_dws(h_tm1)
        xh = torch.cat((x, h_tm1), dim=1)
        if not self.conv_only_hidden:
            xh = self.conv3x3_dws(xh)
        mix = self.conv1x1(xh)

        gates, cell_input = torch.tensor_split(mix, [self.dim * 3], dim=1)
        assert gates.shape[1] == cell_input.shape[1] * 3

        gates = torch.sigmoid(gates)
        forget_gate, input_gate, output_gate = torch.tensor_split(gates, 3, dim=1)
        assert forget_gate.shape == input_gate.shape == output_gate.shape

        cell_input = self.cell_update_dropout(torch.tanh(cell_input))

        c_t = forget_gate * c_tm1 + input_gate * cell_input
        h_t = output_gate * torch.tanh(c_t)

        return h_t, c_t


class RVTExtractor(nn.Module):
    """
    RVT Backbone Extractor for event-based object detection.
    Compatible with SDTv3 interface: outputs 3 stages at strides [8, 16, 32].
    """
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
        
        # Parse args
        self.in_channels = int(getattr(args, "rvt_in_channels", getattr(args, "in_channels", 2)))
        self.return_temporal = bool(getattr(args, "rvt_return_temporal", False))
        self.pretrained_weight = pretrained_weight or getattr(args, "load_pretrained_weight", None)
        
        # Default dimensions to match SDTv3: [256, 512, 640] for stages 2,3,4
        if embed_dim is None:
            embed_dim = getattr(args, "rvt_embed_dim", [64, 128, 256, 512])
        if isinstance(embed_dim, int):
            embed_dim = [64, 128, 256, 512]
        # Don't auto-expand to 5 dimensions - only use what's provided
        if len(embed_dim) < 4:
            raise ValueError(f"embed_dim must have at least 4 values, got {len(embed_dim)}")
        
        self.embed_dim = embed_dim
        
        # Depths for 4 stages
        if depths is None:
            depths = getattr(args, "rvt_depths", [2, 2, 6, 2])
        if isinstance(depths, list) and len(depths) != 4:
            depths = [2, 2, 6, 2]
        self.depths = depths
        
        # Drop path
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Partition size for window/grid attention
        partition_size = getattr(args, "rvt_partition_size", (7, 7))
        dim_head = getattr(args, "rvt_dim_head", 32)
        
        # Stage 1: Stem - downsample by 4x (stride=4)
        self.stem = ConvDownsampling_Cf2Cl(
            in_dim=self.in_channels,
            layer_dim=embed_dim[0],
            downsample_factor=4,
            overlap=True
        )
        
        # Stage 2: stride=4 -> stride=8 (downsample by 2x)
        self.downsample2 = ConvDownsampling_Cf2Cl(
            in_dim=embed_dim[0],
            layer_dim=embed_dim[1],
            downsample_factor=2,
            overlap=True
        )
        cur_dpr = 0
        self.stage2_blocks = nn.ModuleList([
            MaxVitAttentionPairCl(
                dim=embed_dim[1],
                skip_first_norm=(i == 0),
                dim_head=dim_head,
                partition_size=partition_size,
                drop=0.0,
                drop_path=dpr[cur_dpr + i],
            ) for i in range(depths[0])
        ])
        cur_dpr += depths[0]
        
        # Stage 3: stride=8 -> stride=16 (downsample by 2x)
        self.downsample3 = ConvDownsampling_Cf2Cl(
            in_dim=embed_dim[1],
            layer_dim=embed_dim[2],
            downsample_factor=2,
            overlap=True
        )
        self.stage3_blocks = nn.ModuleList([
            MaxVitAttentionPairCl(
                dim=embed_dim[2],
                skip_first_norm=(i == 0),
                dim_head=dim_head,
                partition_size=partition_size,
                drop=0.0,
                drop_path=dpr[cur_dpr + i],
            ) for i in range(depths[1])
        ])
        cur_dpr += depths[1]
        
        # Stage 4: stride=16 -> stride=32 (downsample by 2x)
        self.downsample4 = ConvDownsampling_Cf2Cl(
            in_dim=embed_dim[2],
            layer_dim=embed_dim[3],
            downsample_factor=2,
            overlap=True
        )
        self.stage4_blocks = nn.ModuleList([
            MaxVitAttentionPairCl(
                dim=embed_dim[3],
                skip_first_norm=(i == 0),
                dim_head=dim_head,
                partition_size=partition_size,
                drop=0.0,
                drop_path=dpr[cur_dpr + i],
            ) for i in range(depths[2])
        ])
        cur_dpr += depths[2]
        
        # Optional Stage 5 (for alignment, but not used in output)
        # This allows more depth if needed
        self.use_stage5 = len(embed_dim) > 4 and depths[3] > 0
        if self.use_stage5:
            self.downsample5 = ConvDownsampling_Cf2Cl(
                in_dim=embed_dim[3],
                layer_dim=embed_dim[4],
                downsample_factor=2,
                overlap=True
            )
            self.stage5_blocks = nn.ModuleList([
                MaxVitAttentionPairCl(
                    dim=embed_dim[4],
                    skip_first_norm=(i == 0),
                    dim_head=dim_head,
                    partition_size=partition_size,
                    drop=0.0,
                    drop_path=dpr[cur_dpr + i],
                ) for i in range(depths[3])
            ])
        
        # Output channels match SDTv3: [stage2_dim, stage3_dim, stage4_dim]
        self.out_channels = [embed_dim[1], embed_dim[2], embed_dim[3]]
        self.strides = [8, 16, 32]
        self.num_scales = 3
        self.use_image = False
        self.is_snn = False
        self.num_classes = getattr(args, "num_classes", getattr(args, "n_classes", 2))
        
        # LSTM for temporal modeling (optional)
        self.use_lstm = getattr(args, "rvt_use_lstm", False)
        if self.use_lstm:
            self.lstm_stage2 = DWSConvLSTM2d(embed_dim[1])
            self.lstm_stage3 = DWSConvLSTM2d(embed_dim[2])
            self.lstm_stage4 = DWSConvLSTM2d(embed_dim[3])
            self.lstm_states = {}
        
        # Initialize weights
        if self.pretrained_weight:
            self._load_pretrained_weights(self.pretrained_weight)
        else:
            self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
    
    def get_output_sizes(self):
        sizes = []
        for s in self.strides:
            sizes.append([max(1, self.height // s), max(1, self.width // s)])
        return sizes
    
    @staticmethod
    def _events_to_frames(data, height, width):
        """Convert event data to frame representation."""
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
        
        # Normalize
        frames = torch.clamp(frames, max=3.0) / 3.0
        
        return frames
    
    def _prepare_input(self, x):
        """Prepare input: convert events to frames if needed."""
        try:
            from torch_geometric.data import Data
            if isinstance(x, Data):
                frames = self._events_to_frames(x, self.height, self.width)
                return frames
        except ImportError:
            pass
        
        if not torch.is_tensor(x):
            raise TypeError(f"Unsupported input type {type(x)}.")
        
        if x.dim() == 4:  # B, C, H, W
            return x
        elif x.dim() == 5:  # T, B, C, H, W or B, T, C, H, W
            # Take the last frame or average
            if x.shape[0] < x.shape[1]:  # likely T, B, C, H, W
                return x[-1]  # Take last timestep
            else:  # B, T, C, H, W
                return x[:, -1]
        
        raise ValueError(f"Expected input with 4 or 5 dims, got {x.dim()}.")
    
    def forward(self, x, reset=True):
        """
        Forward pass.
        Returns: list of 3 feature maps at strides [8, 16, 32]
        Each feature map: (B, C, H, W) format to match YOLOX head expectations
        """
        # Prepare input (B, C, H, W)
        x = self._prepare_input(x)
        B = x.shape[0]
        
        # Stage 1: Stem (B, C, H, W) -> (B, H/4, W/4, C1)
        x = self.stem(x)
        
        # Stage 2: (B, H/8, W/8, C2)
        x = self.downsample2(x)
        for blk in self.stage2_blocks:
            x = blk(x)
        
        # Convert to channel-first for LSTM/output: (B, H, W, C) -> (B, C, H, W)
        stage2 = x.permute(0, 3, 1, 2).contiguous()
        
        if self.use_lstm and self.training:
            h_c = self.lstm_states.get('stage2', None)
            stage2, h_c = self.lstm_stage2(stage2, h_c)
            self.lstm_states['stage2'] = (h_c[0].detach(), h_c[1].detach())
        
        # Stage 3: (B, H/16, W/16, C3)
        x = self.downsample3(x)
        for blk in self.stage3_blocks:
            x = blk(x)
        stage3 = x.permute(0, 3, 1, 2).contiguous()
        
        if self.use_lstm and self.training:
            h_c = self.lstm_states.get('stage3', None)
            stage3, h_c = self.lstm_stage3(stage3, h_c)
            self.lstm_states['stage3'] = (h_c[0].detach(), h_c[1].detach())
        
        # Stage 4: (B, H/32, W/32, C4)
        x = self.downsample4(x)
        for blk in self.stage4_blocks:
            x = blk(x)
        stage4 = x.permute(0, 3, 1, 2).contiguous()
        
        if self.use_lstm and self.training:
            h_c = self.lstm_states.get('stage4', None)
            stage4, h_c = self.lstm_stage4(stage4, h_c)
            self.lstm_states['stage4'] = (h_c[0].detach(), h_c[1].detach())
        
        # Optional stage 5 (not returned, for feature extraction depth)
        if self.use_stage5:
            x = self.downsample5(x)
            for blk in self.stage5_blocks:
                x = blk(x)
        
        # Return 3 stages matching SDTv3 interface
        if self.return_temporal:
            # For temporal output, add T dimension
            stage2 = stage2.unsqueeze(0)  # (1, B, C, H, W)
            stage3 = stage3.unsqueeze(0)
            stage4 = stage4.unsqueeze(0)
        
        return [stage2, stage3, stage4]
    
    def _load_pretrained_weights(self, weight_path: str):
        """Load pretrained weights."""
        try:
            import logging
            ckpt = torch.load(weight_path, map_location="cpu")
            if isinstance(ckpt, dict):
                if "model" in ckpt:
                    ckpt = ckpt["model"]
                elif "state_dict" in ckpt:
                    ckpt = ckpt["state_dict"]
            
            if not isinstance(ckpt, dict):
                logging.warning(f"[RVTExtractor] Unexpected checkpoint format, skip loading.")
                return
            
            missing, unexpected = self.load_state_dict(ckpt, strict=False)
            logging.info(f"[RVTExtractor] Loaded pretrained weights from {weight_path}. "
                        f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        except Exception as exc:
            import logging
            logging.warning(f"[RVTExtractor] Failed to load pretrained weights: {exc}")