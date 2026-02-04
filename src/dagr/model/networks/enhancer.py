import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DSConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, d=1, act=True):
        super().__init__()
        if p is None:
            p = autopad(k, None, d)
        self.dw = nn.Conv2d(c1, c1, k, s, p, dilation=d, groups=c1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.pw = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class DSBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5, k1=3, k2=5, d2=1):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = DSConv(c1, c_, k1, s=1, p=None, d=1)
        self.cv2 = DSConv(c_, c2, k2, s=1, p=None, d=d2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class DSC3k(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k1=3, k2=5, d2=1):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(DSBottleneck(c_, c_, shortcut=shortcut, e=1.0, k1=k1, k2=k2, d2=d2) for _ in range(n))
        )


class AdaHyperedgeGen(nn.Module):
    def __init__(self, node_dim, num_hyperedges, num_heads=4, dropout=0.1, context="both"):
        super().__init__()
        self.num_heads = num_heads
        self.num_hyperedges = num_hyperedges
        self.head_dim = node_dim // num_heads
        self.context = context
        self.prototype_base = nn.Parameter(torch.Tensor(num_hyperedges, node_dim))
        nn.init.xavier_uniform_(self.prototype_base)
        if context in ("mean", "max"):
            self.context_net = nn.Linear(node_dim, num_hyperedges * node_dim)
        elif context == "both":
            self.context_net = nn.Linear(2 * node_dim, num_hyperedges * node_dim)
        else:
            raise ValueError(f"Unsupported context '{context}'.")
        self.pre_head_proj = nn.Linear(node_dim, node_dim)
        self.dropout = nn.Dropout(dropout)
        self.scaling = math.sqrt(self.head_dim)

    def forward(self, X):
        B, N, D = X.shape
        if self.context == "mean":
            context_cat = X.mean(dim=1)
        elif self.context == "max":
            context_cat, _ = X.max(dim=1)
        else:
            avg_context = X.mean(dim=1)
            max_context, _ = X.max(dim=1)
            context_cat = torch.cat([avg_context, max_context], dim=-1)
        prototype_offsets = self.context_net(context_cat).view(B, self.num_hyperedges, D)
        prototypes = self.prototype_base.unsqueeze(0) + prototype_offsets
        X_proj = self.pre_head_proj(X)
        X_heads = X_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        proto_heads = prototypes.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        X_heads_flat = X_heads.reshape(B * self.num_heads, N, self.head_dim)
        proto_heads_flat = proto_heads.reshape(B * self.num_heads, self.num_hyperedges, self.head_dim).transpose(1, 2)
        logits = torch.bmm(X_heads_flat, proto_heads_flat) / self.scaling
        logits = logits.view(B, self.num_heads, N, self.num_hyperedges).mean(dim=1)
        logits = self.dropout(logits)
        return F.softmax(logits, dim=1)


class AdaHGConv(nn.Module):
    def __init__(self, embed_dim, num_hyperedges=16, num_heads=4, dropout=0.1, context="both"):
        super().__init__()
        self.edge_generator = AdaHyperedgeGen(embed_dim, num_hyperedges, num_heads, dropout, context)
        self.edge_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())
        self.node_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())

    def forward(self, X):
        A = self.edge_generator(X)
        He = torch.bmm(A.transpose(1, 2), X)
        He = self.edge_proj(He)
        X_new = torch.bmm(A, He)
        X_new = self.node_proj(X_new)
        return X_new + X


class AdaHGComputation(nn.Module):
    def __init__(self, embed_dim, num_hyperedges=16, num_heads=8, dropout=0.1, context="both"):
        super().__init__()
        self.embed_dim = embed_dim
        self.hgnn = AdaHGConv(embed_dim, num_hyperedges, num_heads, dropout, context)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.hgnn(tokens)
        x_out = tokens.transpose(1, 2).view(B, C, H, W)
        return x_out


class C3AH(nn.Module):
    def __init__(self, c1, c2, e=1.0, num_hyperedges=8, context="both"):
        super().__init__()
        c_ = int(c2 * e)
        assert c_ % 16 == 0, "Dimension should be a multiple of 16."
        num_heads = c_ // 16
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = AdaHGComputation(c_, num_hyperedges, num_heads, 0.1, context)
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class ChannelAlign(nn.Module):
    def __init__(self, in_channels, align_dim=256):
        super().__init__()
        self.align_p3 = nn.Conv2d(in_channels[0], align_dim, 1, bias=False) if in_channels[0] != align_dim else nn.Identity()
        self.bn_p3 = nn.BatchNorm2d(align_dim) if in_channels[0] != align_dim else nn.Identity()
        self.align_p4 = nn.Conv2d(in_channels[1], align_dim, 1, bias=False) if in_channels[1] != align_dim else nn.Identity()
        self.bn_p4 = nn.BatchNorm2d(align_dim) if in_channels[1] != align_dim else nn.Identity()
        self.align_p5 = nn.Conv2d(in_channels[2], align_dim, 1, bias=False) if in_channels[2] != align_dim else nn.Identity()
        self.bn_p5 = nn.BatchNorm2d(align_dim) if in_channels[2] != align_dim else nn.Identity()

    def forward(self, feats):
        p3, p4, p5 = feats
        p3_out = self.bn_p3(self.align_p3(p3))
        p4_out = self.bn_p4(self.align_p4(p4))
        p5_out = self.bn_p5(self.align_p5(p5))
        return [p3_out, p4_out, p5_out]


class FuseModule(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_out = Conv(3 * c_in, c_in, 1)

    def forward(self, x):
        p3, p4, p5 = x
        p3_ds = self.downsample(p3)
        p5_up = self.upsample(p5)
        x_cat = torch.cat([p3_ds, p4, p5_up], dim=1)
        return self.conv_out(x_cat)


class HyperACECore(nn.Module):
    def __init__(self, align_dim=256, num_hyperedges=8, n=1, dsc3k=True, shortcut=False, e1=0.5, e2=1.0, context="both"):
        super().__init__()
        self.c = int(align_dim * e1)
        self.fuse = FuseModule(align_dim)
        self.cv1 = Conv(align_dim, 3 * self.c, 1, 1)
        self.cv2 = Conv((4 + n) * self.c, align_dim, 1)
        self.m = nn.ModuleList(
            DSC3k(self.c, self.c, n=2, shortcut=shortcut, g=1, e=1.0, k1=3, k2=7) if dsc3k
            else DSBottleneck(self.c, self.c, shortcut=shortcut, e=1.0, k1=3, k2=7)
            for _ in range(n)
        )
        self.branch1 = C3AH(self.c, self.c, e2, num_hyperedges, context)
        self.branch2 = C3AH(self.c, self.c, e2, num_hyperedges, context)

    def forward(self, aligned_feats):
        x = self.fuse(aligned_feats)
        y = list(self.cv1(x).chunk(3, 1))
        out1 = self.branch1(y[1])
        out2 = self.branch2(y[1])
        y.extend(m(y[-1]) for m in self.m)
        y[1] = out1
        y.append(out2)
        return self.cv2(torch.cat(y, 1))


class FullPAD(nn.Module):
    def __init__(self, in_dim, out_channels):
        super().__init__()
        self.proj_p3 = nn.Sequential(
            nn.Conv2d(in_dim, out_channels[0], 1, bias=False),
            nn.BatchNorm2d(out_channels[0])
        )
        self.proj_p4 = nn.Sequential(
            nn.Conv2d(in_dim, out_channels[1], 1, bias=False),
            nn.BatchNorm2d(out_channels[1])
        )
        self.proj_p5 = nn.Sequential(
            nn.Conv2d(in_dim, out_channels[2], 1, bias=False),
            nn.BatchNorm2d(out_channels[2])
        )
        self.gate3 = nn.Parameter(torch.tensor(0.0))
        self.gate4 = nn.Parameter(torch.tensor(0.0))
        self.gate5 = nn.Parameter(torch.tensor(0.0))

    def forward(self, Y, orig_feats):
        p3, p4, p5 = orig_feats
        h3, w3 = p3.shape[2:]
        h4, w4 = p4.shape[2:]
        h5, w5 = p5.shape[2:]
        H3 = self.proj_p3(F.interpolate(Y, size=(h3, w3), mode='bilinear', align_corners=False))
        H4 = self.proj_p4(F.interpolate(Y, size=(h4, w4), mode='bilinear', align_corners=False))
        H5 = self.proj_p5(F.interpolate(Y, size=(h5, w5), mode='bilinear', align_corners=False))
        P3_out = p3 + self.gate3 * H3
        P4_out = p4 + self.gate4 * H4
        P5_out = p5 + self.gate5 * H5
        return [P3_out, P4_out, P5_out]


class BranchEnhancer(nn.Module):
    def __init__(self, in_channels, align_dim=256, num_hyperedges=8, n=1, dsc3k=True, shortcut=False, e1=0.5, e2=1.0, context="both"):
        super().__init__()
        self.channel_align = ChannelAlign(in_channels, align_dim)
        self.hyper_ace = HyperACECore(align_dim, num_hyperedges, n, dsc3k, shortcut, e1, e2, context)
        self.full_pad = FullPAD(align_dim, in_channels)

    def forward(self, feats):
        orig_feats = feats
        aligned = self.channel_align(feats)
        Y = self.hyper_ace(aligned)
        out = self.full_pad(Y, orig_feats)
        return out


if __name__ == "__main__":
    in_channels = [64, 256, 512]
    B, H, W = 2, 480, 640

    p3 = torch.randn(B, 64, H // 8, W // 8)
    p4 = torch.randn(B, 256, H // 16, W // 16)
    p5 = torch.randn(B, 512, H // 32, W // 32)
    feats = [p3, p4, p5]

    enhancer = BranchEnhancer(in_channels=in_channels, align_dim=256, num_hyperedges=8)
    out = enhancer(feats)

    print("Input shapes:")
    print(f"  P3: {p3.shape}")
    print(f"  P4: {p4.shape}")
    print(f"  P5: {p5.shape}")
    print("\nOutput shapes:")
    print(f"  P3_out: {out[0].shape}")
    print(f"  P4_out: {out[1].shape}")
    print(f"  P5_out: {out[2].shape}")
    print("\nShape consistency check:")
    print(f"  P3 == P3_out: {p3.shape == out[0].shape}")
    print(f"  P4 == P4_out: {p4.shape == out[1].shape}")
    print(f"  P5 == P5_out: {p5.shape == out[2].shape}")

    print("\nGate values (should be 0.0 initially):")
    print(f"  gate3: {enhancer.full_pad.gate3.item()}")
    print(f"  gate4: {enhancer.full_pad.gate4.item()}")
    print(f"  gate5: {enhancer.full_pad.gate5.item()}")