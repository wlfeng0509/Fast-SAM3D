# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import *
import torch
import torch.nn as nn
from ..basic import SparseTensor
from ..attention import SparseMultiHeadAttention, SerializeMode
from ...norm import LayerNorm32
from .blocks import SparseFeedForwardNet

from taylor_utils_ss import derivative_approximation, taylor_cache_init, taylor_formula


class ModulatedSparseTransformerBlock(nn.Module):
    """
    Sparse Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal[
            "full", "shift_window", "shift_sequence", "shift_order", "swin"
        ] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
                6, dim=1
            )
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(mod).chunk(6, dim=1)
            )
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, mod, use_reentrant=False
            )
        else:
            return self._forward(x, mod)


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal[
            "full", "shift_window", "shift_sequence", "shift_order", "swin"
        ] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True)
            )


    # ⭐⭐⭐
    def _forward(
        self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor
    ) -> SparseTensor:
        
        # Block输入 torch.Size([1, 1024]) torch.Size([5248, 1024]) torch.Size([5248, 4]) torch.Size([1, 1024]) torch.Size([1, 5496, 1024])
        # print("Block input", x.shape, x.feats.shape, x.coords.shape, mod.shape, context.shape)

        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
                6, dim=1
            )
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(mod).chunk(6, dim=1)
            )
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        # self-attn
        h = self.self_attn(h)

        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        # cross-attn
        h = self.cross_attn(h, context)

        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        # mlp
        h = self.mlp(h)

        h = h * gate_mlp
        x = x + h
        return x



    def forward(
        self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor
    ) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, mod, context, use_reentrant=False
            )
        else:
            return self._forward(x, mod, context)


# 😊😊😊子类
class ModulatedSparseTransformerCrossBlock_T(ModulatedSparseTransformerCrossBlock):

    def __init__(self, *args, **kwargs):
        # 调用父类参数
        super().__init__(*args, **kwargs)

    # 修改前向传播
    def _forward(
        self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor,current, cache_dic
    ) -> SparseTensor:
        # print("sparse")

        # FLOPs 初始化
        B, N, C = x.shape  # 获取输入 x 的 shape
        flops = 0
        test_FLOPs = cache_dic.get('test_FLOPs', False)  # 检查是否启用 FLOPs 测量
        
        # 嵌入分解
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
                6, dim=1
            )
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(mod).chunk(6, dim=1)
            )
        # ❤️
        if current['type'] == 'full':
            # 计算计算量
            if test_FLOPs:
                flops += 2 * B * N * C
            # AdaLN FLOPs (SiLU and Linear)
            if test_FLOPs:
                flops += B * C  # SiLU FLOPs
                flops += B * C * 6 * C  # Linear FLOPs in adaLN_modulation

            h = x.replace(self.norm1(x.feats))
            h = h * (1 + scale_msa) + shift_msa

            # 1.准备attn模块
            current['module'] = 'attn'
            taylor_cache_init(cache_dic, current)
            h = self.self_attn(h)
            derivative_approximation(cache_dic, current, h)
            h = h * gate_msa
            x = x + h

            # 2. 准备 cross-attn 模块
            current['module'] = 'cross-attn'
            taylor_cache_init(cache_dic, current)
            h = x.replace(self.norm2(x.feats))
            h = self.cross_attn(h, context)
            derivative_approximation(cache_dic, current, h)
            x = x + h
            h = x.replace(self.norm3(x.feats))
            h = h * (1 + scale_mlp) + shift_mlp

            # 3. 准备 mlp模块
            current['module'] = 'mlp'
            taylor_cache_init(cache_dic, current)
            h = self.mlp(h)
            derivative_approximation(cache_dic, current, h)
            h = h * gate_mlp
            x = x + h

        # ❤️
        elif current['type'] == 'taylor':
            # AdaLN FLOPs (SiLU and Linear)
            if test_FLOPs:
                flops += B * C  # SiLU FLOPs
                flops += B * C * 6 * C  # Linear FLOPs in adaLN_modulation
            current['module'] = 'attn'
            x= x + taylor_formula(cache_dic, current) * gate_msa

            current['module'] = 'cross-attn'
            x = x + taylor_formula(cache_dic, current)
            h = x.replace(self.norm3(x.feats))
            h = h * (1 + scale_mlp) + shift_mlp

            current['module'] = 'mlp'
            h = taylor_formula(cache_dic, current)* gate_mlp
            x = x + h

        return x
    
    # 修改前向传播
    def forward(
        self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor, current, cache_dic
    ) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, mod, context, use_reentrant=False
            )
        else:
            return self._forward(x, mod, context)
