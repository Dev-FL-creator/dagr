#简单的0-1 脉冲函数
import torch
import torch.nn as nn


class _STEClamp01(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(0.0, 1.0)  # Use non-inplace clamp

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x < 0.0] = 0.0
        grad_input[x > 1.0] = 0.0
        return grad_input


def spike(x: torch.Tensor) -> torch.Tensor:
    return _STEClamp01.apply(x)


class CrossAttention(nn.Module):
    """
    Spike-driven cross-attention operating on tokenized sequences.

    Expected shapes:
      - query: [T, B, NQ, C] or [1, B, NQ, C]
      - key  : [T, B, NK, C]
      - value: [T, B, NK, C]

    Returns:
      - out:  [T, B, NQ, C]
    """

    def __init__(self, embed_dims: int, num_heads: int = 8):
        super().__init__()
        assert embed_dims % num_heads == 0
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.scale = (self.head_dim) ** -0.5

        self.q_proj = nn.Sequential(
            nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(embed_dims),
        )
        self.k_proj = nn.Sequential(
            nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(embed_dims),
        )
        self.v_proj = nn.Sequential(
            nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(embed_dims),
        )

        self.out_proj = nn.Sequential(
            nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(embed_dims),
        )

    def _project_tokens(self, x: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        # x: [T, B, N, C] -> conv over channels in (C, N)
        T, B, N, C = x.shape
        x = x.permute(0, 1, 3, 2).contiguous()  # T,B,C,N
        x = proj(x.flatten(0, 1)).reshape(T, B, C, N)
        x = spike(x)
        x = x.permute(0, 1, 3, 2).contiguous()  # T,B,N,C
        return x

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:

        if query.dim() == 4 and key.dim() == 4 and value.dim() == 4:
            pass
        else:
            raise ValueError("CrossAttention expects [T,B,N,C] tensors")

        Tq, B, NQ, C = query.shape
        Tk, Bk, NK, Ck = key.shape
        Tv, Bv, NV, Cv = value.shape
        assert B == Bk == Bv and C == Ck == Cv and NK == NV and Tk == Tv

        if Tq != Tk:
            if Tq == 1 and Tk > 1:
                query = query.expand(Tk, -1, -1, -1)
                Tq = Tk
            elif Tk == 1 and Tq > 1:
                key = key.expand(Tq, -1, -1, -1)
                value = value.expand(Tq, -1, -1, -1)
                Tk = Tq
        T = max(Tq, Tk)

        # 线性spike投影
        q = self._project_tokens(query, self.q_proj)
        k = self._project_tokens(key, self.k_proj)
        v = self._project_tokens(value, self.v_proj)

        # reshape to heads
        def split_heads(t):  # [T,B,N,C] -> [T,B,H,N,D]
            T_, B_, N_, C_ = t.shape
            t = t.view(T_, B_, N_, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4).contiguous()
            return t

        qh = split_heads(q)
        kh = split_heads(k)
        vh = split_heads(v)

        # attention
        attn = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale  # [T,B,H,NQ,NK]
        attn = torch.softmax(attn, dim=-1)
        attn = spike(attn)
        out = torch.matmul(attn, vh)  # [T,B,H,NQ,D]

        # merge heads
        out = out.permute(0, 1, 3, 2, 4).contiguous().view(T, B, NQ, C)

        # output投射
        out = out.permute(0, 1, 3, 2).contiguous()  # T,B,C,N
        out = self.out_proj(out.flatten(0, 1)).reshape(Tq, B, C, NQ)
        out = out.permute(0, 1, 3, 2).contiguous()  # T,B,N,C
        return out





# # 以下为实验的多脉冲量化版本，效果不佳，暂时弃用
# 交叉注意力主要需要确定"关注什么"和"关注程度", 二值 spike 可能足以表达这种关系，多余的表示能力可能引入噪声
# import torch
# import torch.nn as nn


# # 定义多脉冲量化参数
# quant = 4
# T = quant

# class Quant(torch.autograd.Function):
#     @staticmethod
#     @torch.cuda.amp.custom_fwd
#     def forward(ctx, i, min_value=0, max_value=quant):
#         ctx.min = min_value
#         ctx.max = max_value
#         ctx.save_for_backward(i)
#         return torch.round(torch.clamp(i, min=min_value, max=max_value))
    
#     @staticmethod
#     @torch.cuda.amp.custom_fwd
#     def backward(ctx, grad_output):
#         grad_input = grad_output.clone()
#         i, = ctx.saved_tensors
#         grad_input[i < ctx.min] = 0
#         grad_input[i > ctx.max] = 0
#         return grad_input, None, None

# # 全局单例，避免每个模块创建新的实例
# _QUANT_FUNC = Quant()

# class MultiSpike_norm4(nn.Module):
#     def __init__(
#             self,
#             Vth=1.0,
#             T=T,
#     ):
#         super().__init__()
#         self.register_buffer('T_value', torch.tensor(float(T)))
#         self.register_buffer('Vth_value', torch.tensor(float(Vth)))
    
#     def forward(self, x):
#         # 使用全局_QUANT_FUNC而不是实例成员
#         return _QUANT_FUNC.apply(x) / self.T_value


# class CrossAttention(nn.Module):
#     """
#     Spike-driven cross-attention operating on tokenized sequences.
#     Using MultiSpike_norm4 for firing mechanism.

#     Expected shapes:
#       - query: [T, B, NQ, C] or [1, B, NQ, C]
#       - key  : [T, B, NK, C]
#       - value: [T, B, NK, C]

#     Returns:
#       - out:  [T, B, NQ, C]
#     """

#     def __init__(self, embed_dims: int, num_heads: int = 8):
#         super().__init__()
#         assert embed_dims % num_heads == 0
#         self.embed_dims = embed_dims
#         self.num_heads = num_heads
#         self.head_dim = embed_dims // num_heads
#         self.scale = (self.head_dim) ** -0.5

#         # 初始化MultiSpike激活函数
#         self.multispike = MultiSpike_norm4()

#         self.q_proj = nn.Sequential(
#             nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm1d(embed_dims),
#         )
#         self.k_proj = nn.Sequential(
#             nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm1d(embed_dims),
#         )
#         self.v_proj = nn.Sequential(
#             nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm1d(embed_dims),
#         )

#         self.out_proj = nn.Sequential(
#             nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm1d(embed_dims),
#         )

#     def _project_tokens(self, x: torch.Tensor, proj: nn.Module) -> torch.Tensor:
#         # x: [T, B, N, C] -> conv over channels in (C, N)
#         T, B, N, C = x.shape
#         x = x.permute(0, 1, 3, 2).contiguous()  # T,B,C,N
#         x = proj(x.flatten(0, 1)).reshape(T, B, C, N)
#         x = self.multispike(x)  # 使用MultiSpike_norm4代替原来的spike
#         x = x.permute(0, 1, 3, 2).contiguous()  # T,B,N,C
#         return x

#     def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
#         if query.dim() == 4 and key.dim() == 4 and value.dim() == 4:
#             pass
#         else:
#             raise ValueError("CrossAttention expects [T,B,N,C] tensors")

#         Tq, B, NQ, C = query.shape
#         Tk, Bk, NK, Ck = key.shape
#         Tv, Bv, NV, Cv = value.shape
#         assert B == Bk == Bv and C == Ck == Cv and NK == NV and Tk == Tv

#         if Tq != Tk:
#             if Tq == 1 and Tk > 1:
#                 query = query.expand(Tk, -1, -1, -1)
#                 Tq = Tk
#             elif Tk == 1 and Tq > 1:
#                 key = key.expand(Tq, -1, -1, -1)
#                 value = value.expand(Tq, -1, -1, -1)
#                 Tk = Tq
#         T = max(Tq, Tk)

#         # 线性multispike投影
#         q = self._project_tokens(query, self.q_proj)
#         k = self._project_tokens(key, self.k_proj)
#         v = self._project_tokens(value, self.v_proj)

#         # reshape to heads
#         def split_heads(t):  # [T,B,N,C] -> [T,B,H,N,D]
#             T_, B_, N_, C_ = t.shape
#             t = t.view(T_, B_, N_, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4).contiguous()
#             return t

#         qh = split_heads(q)
#         kh = split_heads(k)
#         vh = split_heads(v)

#         # attention
#         attn = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale  # [T,B,H,NQ,NK]
#         attn = torch.softmax(attn, dim=-1)
#         attn = self.multispike(attn)  # 使用MultiSpike_norm4代替原来的spike
#         out = torch.matmul(attn, vh)  # [T,B,H,NQ,D]

#         # merge heads
#         out = out.permute(0, 1, 3, 2, 4).contiguous().view(T, B, NQ, C)

#         # output投射
#         out = out.permute(0, 1, 3, 2).contiguous()  # T,B,C,N
#         out = self.out_proj(out.flatten(0, 1)).reshape(Tq, B, C, NQ)
#         out = out.permute(0, 1, 3, 2).contiguous()  # T,B,N,C
#         return out




