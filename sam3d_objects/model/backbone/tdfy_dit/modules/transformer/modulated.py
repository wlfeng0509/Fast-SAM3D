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
    # Fix 1: make _moduledict_to_dict automatically align with x keys
    def _moduledict_to_dict(self, module_dict, ref_dict=None):
        if isinstance(module_dict, torch.nn.ModuleDict):
            raw_dict = {k: v for k, v in module_dict.items()}
        else:
            raw_dict = module_dict
        
        # If a reference dictionary x is passed, return only keys present in x
        # This prevents tree_map from failing due to structure mismatch
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
            # Define timer
            t_prep = CudaTimer("1. Prep")
            t_msa  = CudaTimer("2. MSA Block")
            t_mca  = CudaTimer("3. MCA Block")
            t_mlp  = CudaTimer("4. MLP Block")

            # ================= 1. Prep (prepare parameters) =================
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

                # Extract data
                h_shape = x.get('shape')
                h_rot = x.get('6drotation_normalized')

            # ================= 2. MSA (Self-Attention) =================
            with t_msa:
                # Shape branch
                if h_shape is not None:
                    res_shape = h_shape
                    h_shape = self.norm1['shape'](h_shape)
                    # h * (1 + scale) + shift
                    h_shape = h_shape * (1 + scale_msa) + shift_msa
                
                # Rot branch
                if h_rot is not None:
                    res_rot = h_rot
                    h_rot = self.norm1['6drotation_normalized'](h_rot)
                    h_rot = h_rot * (1 + scale_msa) + shift_msa

                # Attention Core
                attn_in = {}
                if h_shape is not None: attn_in['shape'] = h_shape
                if h_rot is not None: attn_in['6drotation_normalized'] = h_rot
                
                # This is the actual heavy computation
                attn_out = self.self_attn(attn_in)
                
                # Residual + Gate (fusion computation)
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

            # ================= Print report (for debugging) =================
            # Note: report() triggers synchronize, which slightly slows the overall pipeline
            # Comment out printing for formal training
            # print(f"[FastPath] Prep: {t_prep.report():.3f} | MSA: {t_msa.report():.3f} | MCA: {t_mca.report():.3f} | MLP: {t_mlp.report():.3f} | Total: {t_prep.report()+t_msa.report()+t_mca.report()+t_mlp.report():.3f} ms")

            # Return
            out = {}
            if h_shape is not None: out['shape'] = h_shape
            if h_rot is not None: out['6drotation_normalized'] = h_rot
            return out


    # Original generic _forward (fixed the error)
    def _forward(self, x: Dict, mod: torch.Tensor, context: torch.Tensor):
        # ... (AdaLN chunk code is the same as above) ...
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(mod).chunk(6, dim=1)
            )

        # Key change: pass x as a reference and filter norm1 keys
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
        # Key change: pass h or x as a reference
        h = _pytree.tree_map(self._apply_module, x, self._moduledict_to_dict(self.norm2, ref_dict=x))
        h = _pytree.tree_map(
            partial(self._apply_cross_attn, context=context),
            h,
            self._moduledict_to_dict(self.cross_attn, ref_dict=h),
        )
        x = _pytree.tree_map(self._apply_add, x, h)

        # MLP
        # Key change: pass x as a reference
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
            # The condition here is looser: use the fast path as long as it is a dictionary with fewer than 3 keys
            if not self.use_checkpoint and isinstance(x, dict) and len(x) <= 2:
                # Ensure only shape or rot enters the fast path to prevent other keys from failing
                valid_keys = {'shape', '6drotation_normalized'}
                if all(k in valid_keys for k in x.keys()):
                    return self._forward_token_fast(x, mod, context)

            # Fallback: use the fixed generic path
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
        torch.cuda.synchronize() # Force synchronization to ensure accurate timing
        return self.start.elapsed_time(self.end)
