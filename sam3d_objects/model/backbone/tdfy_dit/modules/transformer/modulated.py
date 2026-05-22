# Copyright (c) Meta Platforms, Inc. and affiliates.
from functools import partial
from typing import *
from torch.utils import _pytree
import torch
import torch.nn as nn
from ..attention import MultiHeadAttention, MOTMultiHeadSelfAttention
from ..norm import LayerNorm32
from .blocks import FeedForwardNet


class ModulatedTransformerBlock(nn.Module):
    """
    Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
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
        self.attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
                6, dim=1
            )
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(mod).chunk(6, dim=1)
            )
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.attn(h)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, mod, use_reentrant=False
            )
        else:
            return self._forward(x, mod)


class ModulatedTransformerCrossBlock(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
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
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
                6, dim=1
            )
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(mod).chunk(6, dim=1)
            )
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.self_attn(h)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

  
    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, mod, context, use_reentrant=False
            )
        else:
            return self._forward(x, mod, context)

# ⭐
class MOTModulatedTransformerCrossBlock(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        latent_names: List = None,
        freeze_shared_parameters: bool = False,
    ):
        
        
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        # print("MOTlatent_names", latent_names)
        self.norm1 = torch.nn.ModuleDict(
            {
                latent_name: LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
                for latent_name in latent_names
            }
        )
        self.norm2 = torch.nn.ModuleDict(
            {
                latent_name: LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
                for latent_name in latent_names
            }
        )
        self.norm3 = torch.nn.ModuleDict(
            {
                latent_name: LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
                for latent_name in latent_names
            }
        )
        # ⭐
        self.self_attn = MOTMultiHeadSelfAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
            latent_names=latent_names,
        )
        # ⭐
        self.cross_attn = torch.nn.ModuleDict(
            {
                latent_name: MultiHeadAttention(
                    channels,
                    ctx_channels=ctx_channels,
                    num_heads=num_heads,
                    type="cross",
                    attn_mode="full",
                    qkv_bias=qkv_bias,
                    qk_rms_norm=qk_rms_norm_cross,
                )
                for latent_name in latent_names
            }
        )
        # ⭐
        self.mlp = torch.nn.ModuleDict(
            {
                latent_name: FeedForwardNet(
                    channels,
                    mlp_ratio=mlp_ratio,
                )
                for latent_name in latent_names
            }
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True)
            )
            if freeze_shared_parameters:
                self.adaLN_modulation.eval()
                self.adaLN_modulation.requires_grad_(False)

    def _apply_module(self, h, module):
        return module(h)

    def _apply_cross_attn(self, h, cross_attn, context):
        return cross_attn(h, context)

    def _apply_msa(self, h, scale_msa, shift_msa):
        return h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

    def _apply_mlp(self, h, scale_mlp, shift_mlp):
        return h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)

    def _apply_add(self, x, h):
        return x + h

    def _apply_multiplication(self, h, multiplier):
        return h * multiplier.unsqueeze(1)

    # This is stupid, _pytree does not support ModuleDict
    # 🛠️ 修复 1: 让 _moduledict_to_dict 自动对齐 x 的 keys
    def _moduledict_to_dict(self, module_dict, ref_dict=None):
        if isinstance(module_dict, torch.nn.ModuleDict):
            raw_dict = {k: v for k, v in module_dict.items()}
        else:
            raw_dict = module_dict
        
        # 如果传入了参考字典 x，只返回 x 里有的 key
        # 这样 tree_map 就永远不会因为结构不匹配而报错了！
        if ref_dict is not None and isinstance(ref_dict, dict):
            return {k: raw_dict[k] for k in ref_dict.keys() if k in raw_dict}
        return raw_dict


    # def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor):
    #     if self.share_mod:
    #         shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
    #             6, dim=1
    #         )
    #     else:
    #         shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
    #             self.adaLN_modulation(mod).chunk(6, dim=1)
    #         )
    #     h = _pytree.tree_map(self._apply_module, x, self._moduledict_to_dict(self.norm1))
    #     h = _pytree.tree_map(
    #         partial(self._apply_msa, scale_msa=scale_msa, shift_msa=shift_msa),
    #         h
    #     )
    #     h = self.self_attn(h)
    #     h = _pytree.tree_map(
    #         partial(self._apply_multiplication, multiplier=gate_msa),
    #         h
    #     )
    #     x = _pytree.tree_map(
    #         self._apply_add,
    #         x,
    #         h
    #     )
    #     h = _pytree.tree_map(self._apply_module, x, self._moduledict_to_dict(self.norm2))
    #     h = _pytree.tree_map(
    #         partial(self._apply_cross_attn, context=context),
    #         h,
    #         self._moduledict_to_dict(self.cross_attn),
    #     )
    #     x = _pytree.tree_map(
    #         self._apply_add,
    #         x,
    #         h
    #     )
    #     h = _pytree.tree_map(self._apply_module, x, self._moduledict_to_dict(self.norm3))
    #     h = _pytree.tree_map(
    #         partial(self._apply_mlp, scale_mlp=scale_mlp, shift_mlp=shift_mlp),
    #         h
    #     )
    #     h = _pytree.tree_map(self._apply_module, h, self._moduledict_to_dict(self.mlp))
    #     h = _pytree.tree_map(
    #         partial(self._apply_multiplication, multiplier=gate_mlp),
    #         h
    #     )
    #     x = _pytree.tree_map(
    #         self._apply_add,
    #         x,
    #         h
    #     )
    #     return x

    def _forward_token_fast(self, x: Dict, mod: torch.Tensor, context: torch.Tensor):
            # 定义计时器
            t_prep = CudaTimer("1. Prep")
            t_msa  = CudaTimer("2. MSA Block")
            t_mca  = CudaTimer("3. MCA Block")
            t_mlp  = CudaTimer("4. MLP Block")

            # ================= 1. Prep (准备参数) =================
            with t_prep:
                if self.share_mod:
                    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
                else:
                    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                        self.adaLN_modulation(mod).chunk(6, dim=1)
                    )

                # Global Unsqueeze
                scale_msa = scale_msa.unsqueeze(1)
                shift_msa = shift_msa.unsqueeze(1)
                gate_msa  = gate_msa.unsqueeze(1)
                scale_mlp = scale_mlp.unsqueeze(1)
                shift_mlp = shift_mlp.unsqueeze(1)
                gate_mlp  = gate_mlp.unsqueeze(1)

                # 提取数据
                h_shape = x.get('shape')
                h_rot = x.get('6drotation_normalized')

            # ================= 2. MSA (Self-Attention) =================
            with t_msa:
                # Shape 分支
                if h_shape is not None:
                    res_shape = h_shape
                    h_shape = self.norm1['shape'](h_shape)
                    # h * (1 + scale) + shift
                    h_shape = h_shape * (1 + scale_msa) + shift_msa
                
                # Rot 分支
                if h_rot is not None:
                    res_rot = h_rot
                    h_rot = self.norm1['6drotation_normalized'](h_rot)
                    h_rot = h_rot * (1 + scale_msa) + shift_msa

                # Attention Core
                attn_in = {}
                if h_shape is not None: attn_in['shape'] = h_shape
                if h_rot is not None: attn_in['6drotation_normalized'] = h_rot
                
                # 这里是真正的大计算
                attn_out = self.self_attn(attn_in)
                
                # Residual + Gate (融合计算)
                if h_shape is not None:
                    h_shape = attn_out['shape']
                    h_shape = torch.addcmul(res_shape, h_shape, gate_msa)
                if h_rot is not None:
                    h_rot = attn_out['6drotation_normalized']
                    h_rot = torch.addcmul(res_rot, h_rot, gate_msa)

            # ================= 3. MCA (Cross-Attention) =================
            with t_mca:
                # Shape
                if h_shape is not None:
                    res_shape = h_shape
                    h_shape = self.norm2['shape'](h_shape)
                    h_shape = self.cross_attn['shape'](h_shape, context)
                    h_shape = res_shape + h_shape
                # Rot
                if h_rot is not None:
                    res_rot = h_rot
                    h_rot = self.norm2['6drotation_normalized'](h_rot)
                    h_rot = self.cross_attn['6drotation_normalized'](h_rot, context)
                    h_rot = res_rot + h_rot

            # ================= 4. MLP (Feed Forward) =================
            with t_mlp:
                # Shape
                if h_shape is not None:
                    res_shape = h_shape
                    h_shape = self.norm3['shape'](h_shape)
                    h_shape = h_shape * (1 + scale_mlp) + shift_mlp
                    h_shape = self.mlp['shape'](h_shape)
                    h_shape = torch.addcmul(res_shape, h_shape, gate_mlp)
                # Rot
                if h_rot is not None:
                    res_rot = h_rot
                    h_rot = self.norm3['6drotation_normalized'](h_rot)
                    h_rot = h_rot * (1 + scale_mlp) + shift_mlp
                    h_rot = self.mlp['6drotation_normalized'](h_rot)
                    h_rot = torch.addcmul(res_rot, h_rot, gate_mlp)

            # ================= 打印报告 (调试用) =================
            # ⚠️ 注意：report() 会触发 synchronize，这会轻微拖慢整体流水线
            # 正式训练时建议注释掉打印
            # print(f"[FastPath] Prep: {t_prep.report():.3f} | MSA: {t_msa.report():.3f} | MCA: {t_mca.report():.3f} | MLP: {t_mlp.report():.3f} | Total: {t_prep.report()+t_msa.report()+t_mca.report()+t_mlp.report():.3f} ms")

            # 返回
            out = {}
            if h_shape is not None: out['shape'] = h_shape
            if h_rot is not None: out['6drotation_normalized'] = h_rot
            return out


    # 原来的通用 _forward (修复了报错)
    def _forward(self, x: Dict, mod: torch.Tensor, context: torch.Tensor):
        # ... (AdaLN chunk 代码同上) ...
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(mod).chunk(6, dim=1)
            )

        # 🛠️ 关键修改：传入 x 作为参考，过滤 norm1 的 keys
        h = _pytree.tree_map(self._apply_module, x, self._moduledict_to_dict(self.norm1, ref_dict=x))
        
        h = _pytree.tree_map(
            partial(self._apply_msa, scale_msa=scale_msa, shift_msa=shift_msa),
            h
        )
        h = self.self_attn(h)
        h = _pytree.tree_map(
            partial(self._apply_multiplication, multiplier=gate_msa),
            h
        )
        x = _pytree.tree_map(self._apply_add, x, h)
        
        # MCA
        # 🛠️ 关键修改：传入 h 或 x 作为参考
        h = _pytree.tree_map(self._apply_module, x, self._moduledict_to_dict(self.norm2, ref_dict=x))
        h = _pytree.tree_map(
            partial(self._apply_cross_attn, context=context),
            h,
            self._moduledict_to_dict(self.cross_attn, ref_dict=h),
        )
        x = _pytree.tree_map(self._apply_add, x, h)

        # MLP
        # 🛠️ 关键修改：传入 x 作为参考
        h = _pytree.tree_map(self._apply_module, x, self._moduledict_to_dict(self.norm3, ref_dict=x))
        h = _pytree.tree_map(
            partial(self._apply_mlp, scale_mlp=scale_mlp, shift_mlp=shift_mlp),
            h
        )
        h = _pytree.tree_map(self._apply_module, h, self._moduledict_to_dict(self.mlp, ref_dict=h))
        h = _pytree.tree_map(
            partial(self._apply_multiplication, multiplier=gate_mlp),
            h
        )
        x = _pytree.tree_map(self._apply_add, x, h)
        return x
    



    def forward(self, x: Dict, mod: torch.Tensor, context: torch.Tensor):
            # Token pruning mode can pass a small key subset, so use the fast path.
            # 这里的判断条件 loose 一点，只要是字典且小于3个key就走 fast path
            if not self.use_checkpoint and isinstance(x, dict) and len(x) <= 2:
                # 确保只有 shape 或 rot 才能进 fast path，防止其他 key 报错
                valid_keys = {'shape', '6drotation_normalized'}
                if all(k in valid_keys for k in x.keys()):
                    return self._forward_token_fast(x, mod, context)

            # 兜底：走修复后的通用通道
            if self.use_checkpoint:
                return torch.utils.checkpoint.checkpoint(
                    self._forward, x, mod, context, use_reentrant=False
                )
            else:
                return self._forward(x, mod, context)

class CudaTimer:
    def __init__(self, name):
        self.name = name
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end.record()
        
    def report(self):
        torch.cuda.synchronize() # 强制同步，确保时间准确
        return self.start.elapsed_time(self.end)
