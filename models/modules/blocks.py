import torch
import torch.nn as nn

from models.SA.MHSA import MHSA
from timm.models.layers import Mlp, DropPath, trunc_normal_

from utils.mctf import mctf


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block_DEIT(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 info = None, layer = 0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MHSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, info=info)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.info = info
        self.layer = layer

    def forward(self, x, prefix):
        size = self.info[prefix + "size"] if self.info["prop_attn"] else None
        if self.info["use_mctf"] and self.layer in self.info["activate"] and self.info["one_step_ahead"]:
            if self.info["one_step_ahead"] == 1:
                attn = self.attn(self.norm1(x), size, _attn=True)
                x, _attn = mctf(x, attn, self.info, prefix, self.training)
                x_attn, _ = self.attn(self.norm1(x), size, _attn=_attn)
            elif self.info["one_step_ahead"] == 2: # Precise
                with torch.no_grad():
                    attn = self.attn(self.norm1(x), size, _attn=True)
                x, _ = mctf(x, attn, self.info, prefix, self.training)
                size = self.info[prefix + "size"] if self.info["prop_attn"] else None
                x_attn, _ = self.attn(self.norm1(x), size)
        else:
            x_attn, attn = self.attn(self.norm1(x), size)

        x = x + self.drop_path1(x_attn)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        if self.info["use_mctf"] and self.layer in self.info["activate"] and not self.info["one_step_ahead"]: # W/O One-Step Ahead
            x, _ = mctf(x, attn, self.info, prefix, self.training)
        return x
