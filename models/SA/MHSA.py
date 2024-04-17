import torch.nn as nn


class MHSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., info = None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.info = info



    def forward(self, x, size=None, _attn=False):
        B, N, C = x.shape
        if type(_attn) == bool:
            if _attn:
                qk = (x @ self.qkv.weight.T[:, : 2 * C] + self.qkv.bias[:2 * C]).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                q, k = qk[0], qk[1]
            else:
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T1, T2)
            if size is not None:
                attn = attn + size.log()[:, None, None, :, 0]

            attn = attn.softmax(dim=-1)

            attn_ = attn

            if _attn:
                return attn_
        else:
            v = (x @ self.qkv.weight.T[:, 2 * C:] + self.qkv.bias[2 * C:]).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            attn = _attn
            attn_ = attn.detach()

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_