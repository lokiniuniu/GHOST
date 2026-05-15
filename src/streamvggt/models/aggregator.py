# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any

from streamvggt.layers import PatchEmbed
from streamvggt.layers.block import Block
from streamvggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from streamvggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from streamvggt.eviction.importance_eviction import (
    compute_frame_importance,
    compute_token_importance,
    compute_combined_importance,
    compute_importance_incremental,
    compute_importance_for_current_frame_only,
    compute_patch_saliency,
)

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
IMPORTANCE_EVICTION_MODES = frozenset({"importance"})
EVICTION_MODES_ANCHOR1_ONLY = frozenset({"importance"})

# Default importance weights: configs/importance_weights_default.json
# Legacy sigmoid combo2_det exact: configs/importance_weights_sigmoid_combo2_camgeo05623_deterministic.json
_DEFAULT_IMPORTANCE_WEIGHTS: Dict[str, float] = {
    "w_camera": 0.55,
    "w_geometry": 0.55,
    "w_temporal": 0.25,
    "w_saliency": 0.28,
    "w_depth_conf": 0.45,
    "w_pts_conf": 0.35,
    "w_frame": 0.5,
    "w_token": 0.5,
    "special_token_boost": 0.3,
    "special_token_tiebreak_eps": 1e-6,
    "special_token_noise_scale": 0.0,
}

# KV cache mode: (num_anchor_frames, num_special_tokens, special_token_offset, fixed_exempt_from_budget)
# - anchor1_camera: fix first frame all + camera (1 token) per subsequent frame
# - anchor1_register: fix first frame all + register (4 tokens) per subsequent frame
# - anchor1_camera_register: fix first frame all + camera+register (5 tokens) per subsequent frame
# - anchor1_only: fix first frame only, no special preservation (used by importance eviction)
# - anchor3_only: fix first 3 frames all, no camera/register fixed for subsequent frames
# - anchor3_camera_register: fix first 3 frames all + camera+register (5 tokens) per subsequent frame
# - anchor3_camera_register_exempt: same as anchor3 but fixed tokens don't count toward budget
KV_MODE_CONFIG = {
    "anchor1_camera": (1, 1, 0, False),
    "anchor1_register": (1, 4, 1, False),
    "anchor1_camera_register": (1, 5, 0, False),
    "anchor1_only": (1, 0, 0, False),
    "anchor3_only": (3, 0, 0, False),
    "anchor3_camera_register": (3, 5, 0, False),
    "anchor3_camera_register_exempt": (3, 5, 0, True),
}


def _resolve_kv_mode(kv_mode: Optional[str], patch_start_idx: int):
    """Resolve kv_mode string to (num_anchor_frames, num_special_tokens, special_token_offset, fixed_exempt)."""
    if kv_mode is None:
        return (3, patch_start_idx, 0, False)
    cfg = KV_MODE_CONFIG.get(kv_mode)
    if cfg is None:
        raise ValueError(f"Unknown kv_mode={kv_mode!r}. Valid: {list(KV_MODE_CONFIG.keys())}")
    return cfg


_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.


    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).reshape(1, 1, 3, 1, 1),
                persistent=False,
            )
        self.last_scores = torch.zeros(self.depth)
        # Optional: precomputed budget proportions from compute_kv_budget_from_cosine_sim.py
        self.budget_proportions = None  # [depth] tensor or None
        # Gradient checkpointing for OOM reduction (e.g. Fisher/Gradient/Hessian strategies)
        self.gradient_checkpointing = False
        # Cache static frame-attention sparse masks by (tokens_per_frame, window_size, device, dtype).
        self._frame_sparse_mask_cache: Dict[Tuple[int, int, str, str], torch.Tensor] = {}


    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(
        self,
        images: torch.Tensor,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0,
        total_budget=0,
        profile_timing=False,
        qvg_manager=None,
        timing_dict=None,
        kv_mode: Optional[str] = None,
        collect_attn_weights: bool = False,
        collect_attn_layers: Optional[List[int]] = None,
        frame_metadata: Optional[List[Dict[str, Any]]] = None,
        eviction_mode: str = "importance",
        importance_cache: Optional[Dict[str, Any]] = None,
        importance_weights: Optional[Dict[str, float]] = None,
        use_importance_in_attn: bool = False,
        softmax_importance_before_k: bool = False,
        debug_importance_in_attn: bool = False,
        kv_share_cfg: Optional[Dict[str, Any]] = None,
        frame_sparse_cfg: Optional[Dict[str, Any]] = None,
        global_sparse_cfg: Optional[Dict[str, Any]] = None,
        use_flex_attention: bool = False,
        flex_block_size: int = 128,
        flex_compile_mode: str = "fullgraph",
        motivation_kv_probe: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            global_sparse_cfg: Optional dict for no-cache global attention only (use_cache=False).
                When mode is ``sparse_vggt`` and layer schedule matches, replaces the dense
                frame-causal mask with Sparse-VGGT-style block sparsity (see ``Attention``).

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape
        patch_h_img = H // self.patch_size
        patch_w_img = W // self.patch_size

        if use_cache and past_key_values[0] is not None:
            # _, _, S_true, _, _ = past_key_values[0][0].shape
            S_true = past_frame_idx + 1
        else:
            S_true = S
        
        if use_cache and S > 1:
            print(f"Use KV cache expects S=1, got S={S}")

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean.to(images.device)) / self._resnet_std.to(images.device)

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.reshape(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        if use_cache:
            camera_token_full = slice_expand_and_flatten(self.camera_token, B, S_true)
            camera_token = camera_token_full[-1:, :, :]
            
            register_token_full = slice_expand_and_flatten(self.register_token, B, S_true)
            register_token = register_token_full[-1:, :, :]
        else:
            camera_token = slice_expand_and_flatten(self.camera_token, B, S)
            register_token = slice_expand_and_flatten(self.register_token, B, S)
        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        # [Importance eviction] Compute importance scores when using importance-based eviction.
        # Per-layer cache: when layers have different budgets (e.g. cosine_budget), each layer
        # keeps different token counts, so we need cached_per_layer[layer_idx] to avoid size mismatch.
        if use_cache and eviction_mode in IMPORTANCE_EVICTION_MODES and importance_cache is not None:
            importance_cache.setdefault("cached_per_layer", {})
        if use_cache and eviction_mode in IMPORTANCE_EVICTION_MODES and frame_metadata is not None:
            patch_h, patch_w = H // self.patch_size, W // self.patch_size
            n_patch = patch_h * patch_w
            # Precompute full importance (for layers without per-layer cache) and new-frame-only
            _iw = {**_DEFAULT_IMPORTANCE_WEIGHTS, **(importance_weights or {})}
            full_importance_scores, updated_cache = compute_importance_incremental(
                frame_metadata,
                patch_tokens,
                past_frame_idx,
                patch_h,
                patch_w,
                self.patch_start_idx,
                importance_cache=importance_cache,
                cached_importance_scores=None,  # always full recompute for "full" fallback
                device=tokens.device,
                w_camera=_iw["w_camera"],
                w_geometry=_iw["w_geometry"],
                w_temporal=_iw["w_temporal"],
                w_saliency=_iw["w_saliency"],
                w_depth_conf=_iw["w_depth_conf"],
                w_pts_conf=_iw["w_pts_conf"],
                w_frame=_iw["w_frame"],
                w_token=_iw["w_token"],
                special_token_boost=_iw["special_token_boost"],
                special_token_tiebreak_eps=_iw["special_token_tiebreak_eps"],
                special_token_noise_scale=_iw["special_token_noise_scale"],
            )
            if importance_cache is not None and importance_cache.get("_full_cache_initialized") is not True:
                saved_per_layer = importance_cache.get("cached_per_layer", {})
                saved_profile_raw = importance_cache.get("_profile_raw")
                saved_profile_min_frames = importance_cache.get("_profile_min_frames")
                importance_cache.clear()
                importance_cache.update(updated_cache)
                importance_cache["_full_cache_initialized"] = True
                importance_cache["cached_per_layer"] = saved_per_layer
                if saved_profile_raw is not None:
                    importance_cache["_profile_raw"] = saved_profile_raw
                if saved_profile_min_frames is not None:
                    importance_cache["_profile_min_frames"] = saved_profile_min_frames
            new_frame_importance = compute_importance_for_current_frame_only(
                patch_tokens, frame_metadata, past_frame_idx,
                patch_h, patch_w, self.patch_start_idx,
                device=tokens.device,
                w_frame=_iw["w_frame"],
                w_token=_iw["w_token"],
                special_token_boost=_iw["special_token_boost"],
                special_token_tiebreak_eps=_iw["special_token_tiebreak_eps"],
                special_token_noise_scale=_iw["special_token_noise_scale"],
                w_saliency=_iw["w_saliency"],
                w_depth_conf=_iw["w_depth_conf"],
                w_pts_conf=_iw["w_pts_conf"],
            )
        else:
            full_importance_scores = None
            new_frame_importance = None

        frame_idx = 0
        global_idx = 0
        output_list = []
        attn_weights_collected = []  # per-layer attention when collect_attn_weights
        current_budgets = self._calculate_dynamic_budgets(total_budget)
        scores = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos,
                        patch_h=H // self.patch_size,
                        patch_w=W // self.patch_size,
                        frame_sparse_cfg=frame_sparse_cfg,
                        patch_tokens=patch_tokens,
                        use_flex_attention=use_flex_attention,
                        flex_block_size=flex_block_size,
                        flex_compile_mode=flex_compile_mode,
                    )
                elif attn_type == "global":
                    if use_cache:
                        # [Per-layer importance] Use cached_per_layer when available to avoid size mismatch
                        # (cosine_budget gives different budgets per layer, so each layer keeps different token counts)
                        layer_importance_scores = full_importance_scores if eviction_mode in IMPORTANCE_EVICTION_MODES else None
                        if eviction_mode in IMPORTANCE_EVICTION_MODES and importance_cache is not None:
                            cached_per_layer = importance_cache.get("cached_per_layer", {})
                            cached = cached_per_layer.get(global_idx)
                            if cached is not None and new_frame_importance is not None:
                                layer_importance_scores = torch.cat([
                                    cached.to(new_frame_importance.device).to(new_frame_importance.dtype),
                                    new_frame_importance,
                                ], dim=0)
                            elif full_importance_scores is not None:
                                layer_importance_scores = full_importance_scores
                            importance_cache["_current_layer"] = global_idx
                        # [QVG] Use manager (quantized) or raw list (BF16)
                        use_qvg = qvg_manager is not None and qvg_manager.config.enabled
                        if use_qvg:
                            kv_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                            if timing_dict is not None and torch.cuda.is_available():
                                torch.cuda.synchronize()
                            t_decode0 = time.perf_counter() if timing_dict is not None else 0
                            past_key_values_block = qvg_manager.get_past_key_values_for_layer(
                                global_idx, images.device, kv_dtype, timing_dict=timing_dict
                            )
                            if timing_dict is not None and torch.cuda.is_available():
                                torch.cuda.synchronize()
                            if timing_dict is not None:
                                timing_dict["decode"] += time.perf_counter() - t_decode0
                        else:
                            past_key_values_block = past_key_values[global_idx] if past_key_values[global_idx] is not None else None
                        chunk_size = qvg_manager.config.chunk_size if use_qvg else 1
                        # anchor1_only: single-frame anchor; camera/register not fixed per frame (ranked with patches)
                        effective_kv_mode = (
                            "anchor1_only" if eviction_mode in EVICTION_MODES_ANCHOR1_ONLY else kv_mode
                        )
                        proc_out = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos,
                            past_key_values_block=past_key_values_block,
                            prev_layer_kv=(
                                past_key_values[global_idx - 1]
                                if (not use_qvg and global_idx > 0 and past_key_values[global_idx - 1] is not None)
                                else None
                            ),
                            use_cache=True,
                            past_frame_idx=past_frame_idx,
                            cache_budget=current_budgets[global_idx].item(),
                            chunk_size=chunk_size,
                            timing_dict=timing_dict,
                            kv_mode=effective_kv_mode,
                            return_attn_weights=collect_attn_weights,
                            importance_scores=layer_importance_scores,
                            importance_cache=importance_cache,
                            use_importance_in_attn=use_importance_in_attn,
                            softmax_importance_before_k=softmax_importance_before_k,
                            debug_importance_in_attn=debug_importance_in_attn,
                            kv_share_cfg=kv_share_cfg,
                            motivation_kv_probe=motivation_kv_probe,
                        )
                        if collect_attn_weights and len(proc_out) == 6:
                            tokens, global_idx, global_intermediates, new_kv, current_scores, layer_attn = proc_out
                            if collect_attn_layers is None or (global_idx - 1) in collect_attn_layers:
                                attn_weights_collected.append(layer_attn)
                            else:
                                del layer_attn
                        else:
                            tokens, global_idx, global_intermediates, new_kv, current_scores = proc_out[:5]
                        # [QVG] Store via manager (quantized) or in list (raw)
                        layer_idx = global_idx - 1
                        num_anchor = self.global_blocks[layer_idx].attn.num_anchor_tokens
                        if use_qvg:
                            if timing_dict is not None and torch.cuda.is_available():
                                torch.cuda.synchronize()
                            t_encode0 = time.perf_counter() if timing_dict is not None else 0
                            qvg_manager.update_layer_cache(
                                layer_idx, new_kv[0], new_kv[1], num_anchor,
                                frame_idx=past_frame_idx,
                                timing_dict=timing_dict,
                            )
                            if timing_dict is not None and torch.cuda.is_available():
                                torch.cuda.synchronize()
                            if timing_dict is not None:
                                timing_dict["encode"] += time.perf_counter() - t_encode0
                        else:
                            past_key_values[layer_idx] = new_kv
                        del past_key_values_block  # Release dequantized KV to reduce peak memory
                        if current_scores is not None: # pruning happened
                            scores.append(current_scores)
                        else:
                            scores.append(self.last_scores[global_idx-1].item())
                    else:
                        tokens, global_idx, global_intermediates = self._process_global_attention(
                            tokens,
                            B,
                            S,
                            P,
                            C,
                            global_idx,
                            pos=pos,
                            global_sparse_cfg=global_sparse_cfg,
                            patch_h=patch_h_img,
                            patch_w=patch_w_img,
                            use_flex_attention=use_flex_attention,
                            flex_block_size=flex_block_size,
                            flex_compile_mode=flex_compile_mode,
                        )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")
            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)
        if scores: # update scores
            self.last_scores = torch.tensor(scores, device=self.last_scores.device, dtype=self.last_scores.dtype)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        if use_cache:
            if collect_attn_weights and attn_weights_collected:
                return output_list, self.patch_start_idx, past_key_values, attn_weights_collected
            return output_list, self.patch_start_idx, past_key_values
        return output_list, self.patch_start_idx

    def _build_static_frame_sparse_mask(
        self,
        tokens_per_frame: int,
        patch_h: int,
        patch_w: int,
        window_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        num_special = self.patch_start_idx
        n_patch = patch_h * patch_w
        if tokens_per_frame != num_special + n_patch:
            return None
        window_size = max(1, int(window_size))
        if window_size % 2 == 0:
            window_size += 1
        radius = window_size // 2

        allow = torch.zeros((tokens_per_frame, tokens_per_frame), dtype=torch.bool, device=device)
        # Special tokens are global hubs (both query and key side).
        allow[:, :num_special] = True
        allow[:num_special, :] = True

        for p in range(n_patch):
            q_idx = num_special + p
            py, px = divmod(p, patch_w)
            y0 = max(0, py - radius)
            y1 = min(patch_h, py + radius + 1)
            x0 = max(0, px - radius)
            x1 = min(patch_w, px + radius + 1)
            for y in range(y0, y1):
                base = y * patch_w
                for x in range(x0, x1):
                    k_idx = num_special + base + x
                    allow[q_idx, k_idx] = True

        mask = torch.full(
            (tokens_per_frame, tokens_per_frame),
            torch.finfo(dtype).min,
            device=device,
            dtype=dtype,
        )
        mask[allow] = 0
        return mask

    def _get_static_frame_sparse_mask(
        self,
        tokens_per_frame: int,
        patch_h: int,
        patch_w: int,
        window_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        key = (tokens_per_frame, int(window_size), str(device), str(dtype))
        mask = self._frame_sparse_mask_cache.get(key)
        if mask is None:
            mask = self._build_static_frame_sparse_mask(
                tokens_per_frame=tokens_per_frame,
                patch_h=patch_h,
                patch_w=patch_w,
                window_size=window_size,
                device=device,
                dtype=dtype,
            )
            if mask is None:
                return None
            self._frame_sparse_mask_cache[key] = mask
        return mask

    def _apply_dynamic_topk_queries_to_mask(
        self,
        base_mask: torch.Tensor,
        patch_h: int,
        patch_w: int,
        topk_ratio: float,
        patch_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        n_patch = patch_h * patch_w
        num_special = self.patch_start_idx
        if n_patch <= 0:
            return base_mask
        ratio = max(0.0, min(1.0, float(topk_ratio)))
        if ratio <= 0:
            return base_mask
        k = max(1, min(n_patch, int(round(n_patch * ratio))))
        if k >= n_patch:
            # If all patch queries are promoted, all patch-to-patch connections are visible.
            out = base_mask.clone()
            out[num_special : num_special + n_patch, num_special : num_special + n_patch] = 0
            return out

        if patch_tokens is not None and patch_tokens.dim() == 3 and patch_tokens.shape[1] == n_patch:
            saliency = compute_patch_saliency(patch_tokens, patch_h, patch_w).flatten()
            top_idx = torch.topk(saliency, k=k, largest=True).indices.to(torch.long)
        else:
            top_idx = torch.arange(k, device=base_mask.device, dtype=torch.long)

        out = base_mask.clone()
        for idx in top_idx.tolist():
            q = num_special + idx
            out[q, num_special : num_special + n_patch] = 0
        return out

    def _process_frame_attention(
        self,
        tokens,
        B,
        S,
        P,
        C,
        frame_idx,
        pos=None,
        patch_h: Optional[int] = None,
        patch_w: Optional[int] = None,
        frame_sparse_cfg: Optional[Dict[str, Any]] = None,
        patch_tokens: Optional[torch.Tensor] = None,
        use_flex_attention: bool = False,
        flex_block_size: int = 128,
        flex_compile_mode: str = "fullgraph",
    ):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.reshape(B, S, P, 2).reshape(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            attn_mask = None
            sparse_kv_cfg = None
            if frame_sparse_cfg is not None:
                mode = str(frame_sparse_cfg.get("mode", "none"))
                if mode in ("static_window", "dynamic_topk") and patch_h is not None and patch_w is not None:
                    start_layer = max(0, int(frame_sparse_cfg.get("start_layer", 0)))
                    apply_every = max(1, int(frame_sparse_cfg.get("apply_every", 1)))
                    if frame_idx >= start_layer and (frame_idx - start_layer) % apply_every == 0:
                        window_size = int(frame_sparse_cfg.get("window_size", 7))
                        attn_mask = self._get_static_frame_sparse_mask(
                            tokens_per_frame=P,
                            patch_h=patch_h,
                            patch_w=patch_w,
                            window_size=window_size,
                            device=tokens.device,
                            dtype=tokens.dtype,
                        )
                        if attn_mask is not None and mode == "dynamic_topk":
                            topk_ratio = float(frame_sparse_cfg.get("topk_ratio", 0.1))
                            attn_mask = self._apply_dynamic_topk_queries_to_mask(
                                base_mask=attn_mask,
                                patch_h=patch_h,
                                patch_w=patch_w,
                                topk_ratio=topk_ratio,
                                patch_tokens=patch_tokens,
                            )
                elif mode == "sparsity" and patch_h is not None and patch_w is not None:
                    # Sparse K/V for frame-attention: keep all queries and special tokens,
                    # subsample patch K/V on a uniform grid; preserve diagonal; optional mean-fill.
                    start_layer = max(0, int(frame_sparse_cfg.get("start_layer", 8)))
                    apply_every = max(1, int(frame_sparse_cfg.get("apply_every", 1)))
                    if frame_idx >= start_layer and (frame_idx - start_layer) % apply_every == 0:
                        sparse_kv_cfg = {
                            "num_special_tokens": self.patch_start_idx,
                            "patch_h": int(patch_h),
                            "patch_w": int(patch_w),
                            "stride_h": int(frame_sparse_cfg.get("stride_h", 2)),
                            "stride_w": int(frame_sparse_cfg.get("stride_w", 2)),
                            "preserve_diagonal": bool(frame_sparse_cfg.get("preserve_diagonal", True)),
                            "use_mean_fill": bool(frame_sparse_cfg.get("use_mean_fill", True)),
                            "debug_sparse_stats": bool(frame_sparse_cfg.get("debug_sparse_stats", False)),
                            "debug_print_every": int(frame_sparse_cfg.get("debug_print_every", 1)),
                            "layer_idx": int(frame_idx),
                        }
                elif mode == "sparse_vggt" and patch_h is not None and patch_w is not None:
                    # Sparse-VGGT-style frame attention:
                    # use pooled proxy attention to pick sparse patch blocks.
                    start_layer = max(0, int(frame_sparse_cfg.get("start_layer", 8)))
                    apply_every = max(1, int(frame_sparse_cfg.get("apply_every", 1)))
                    if frame_idx >= start_layer and (frame_idx - start_layer) % apply_every == 0:
                        sparse_kv_cfg = {
                            "mode": "sparse_vggt",
                            "num_special_tokens": self.patch_start_idx,
                            "patch_h": int(patch_h),
                            "patch_w": int(patch_w),
                            "preserve_diagonal": bool(frame_sparse_cfg.get("preserve_diagonal", True)),
                            "svggt_sparse_ratio": frame_sparse_cfg.get("svggt_sparse_ratio", 0.75),
                            "svggt_cdf_threshold": frame_sparse_cfg.get("svggt_cdf_threshold", None),
                            "svggt_topk_blocks": frame_sparse_cfg.get("svggt_topk_blocks", None),
                            "svggt_pool_mode": str(frame_sparse_cfg.get("svggt_pool_mode", "avg")),
                            "svggt_ks_q": int(frame_sparse_cfg.get("svggt_ks_q", 128)),
                            "svggt_ks_k": int(frame_sparse_cfg.get("svggt_ks_k", 64)),
                            "svggt_use_sparge_kernel": bool(
                                frame_sparse_cfg.get("svggt_use_sparge_kernel", True)
                            ),
                            "debug_sparse_stats": bool(frame_sparse_cfg.get("debug_sparse_stats", False)),
                            "debug_print_every": int(frame_sparse_cfg.get("debug_print_every", 1)),
                            "layer_idx": int(frame_idx),
                        }

            tokens = self.frame_blocks[frame_idx](
                tokens,
                pos=pos,
                attn_mask=attn_mask,
                sparse_kv_cfg=sparse_kv_cfg,
                use_flex_attention=use_flex_attention,
                flex_block_size=flex_block_size,
                flex_compile_mode=flex_compile_mode,
            )
            frame_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(
        self,
        tokens,
        B,
        S,
        P,
        C,
        global_idx,
        pos=None,
        past_key_values_block=None,
        prev_layer_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        past_frame_idx=0,
        cache_budget=None,
        chunk_size=1,
        timing_dict=None,
        kv_mode: Optional[str] = None,
        return_attn_weights: bool = False,
        importance_scores: Optional[torch.Tensor] = None,
        importance_cache: Optional[Dict[str, Any]] = None,
        use_importance_in_attn: bool = False,
        softmax_importance_before_k: bool = False,
        debug_importance_in_attn: bool = False,
        kv_share_cfg: Optional[Dict[str, Any]] = None,
        global_sparse_cfg: Optional[Dict[str, Any]] = None,
        patch_h: Optional[int] = None,
        patch_w: Optional[int] = None,
        use_flex_attention: bool = False,
        flex_block_size: int = 128,
        flex_compile_mode: str = "fullgraph",
        motivation_kv_probe: Optional[Dict[str, Any]] = None,
    ) -> Union[Tuple[torch.Tensor, int, List[torch.Tensor]], Tuple[torch.Tensor, int, List[torch.Tensor], List]]:
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
                """
        
        if tokens.shape != (B, S * P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.reshape(B, S, P, 2).reshape(B, S * P, 2)
            
        intermediates = []

        for _ in range(self.aa_block_size):
            sparse_kv_global = None
            if not use_cache:
                L = S * P
                frame_ids = torch.arange(L, device=tokens.device) // P  # [0,0,...,1,1,...,S-1]
                future_frame = frame_ids.unsqueeze(1) < frame_ids.unsqueeze(0)
                attn_mask = future_frame.to(tokens.dtype) * torch.finfo(tokens.dtype).min
                if (
                    global_sparse_cfg is not None
                    and patch_h is not None
                    and patch_w is not None
                ):
                    mode = str(global_sparse_cfg.get("mode", "none"))
                    if mode == "sparse_vggt":
                        start_layer = max(0, int(global_sparse_cfg.get("start_layer", 0)))
                        apply_every = max(1, int(global_sparse_cfg.get("apply_every", 1)))
                        if global_idx >= start_layer and (global_idx - start_layer) % apply_every == 0:
                            sparse_kv_global = {
                                "mode": "sparse_vggt",
                                "is_global_sparse": True,
                                "num_frames": S,
                                "tokens_per_frame": P,
                                "num_special_tokens": self.patch_start_idx,
                                "patch_h": int(patch_h),
                                "patch_w": int(patch_w),
                                "frame_causal": bool(global_sparse_cfg.get("frame_causal", True)),
                                "svggt_sparse_ratio": global_sparse_cfg.get(
                                    "svggt_sparse_ratio", 0.75
                                ),
                                "svggt_cdf_threshold": global_sparse_cfg.get(
                                    "svggt_cdf_threshold", None
                                ),
                                "svggt_topk_blocks": global_sparse_cfg.get(
                                    "svggt_topk_blocks", None
                                ),
                                "svggt_pool_mode": str(
                                    global_sparse_cfg.get("svggt_pool_mode", "avg")
                                ),
                                "svggt_ks_q": int(global_sparse_cfg.get("svggt_ks_q", 128)),
                                "svggt_ks_k": int(global_sparse_cfg.get("svggt_ks_k", 64)),
                                "svggt_use_sparge_kernel": bool(
                                    global_sparse_cfg.get("svggt_use_sparge_kernel", True)
                                ),
                                "debug_sparse_stats": bool(
                                    global_sparse_cfg.get("debug_sparse_stats", False)
                                ),
                                "debug_print_every": int(
                                    global_sparse_cfg.get("debug_print_every", 1)
                                ),
                                "layer_idx": int(global_idx),
                            }
                            attn_mask = None
            else:
                attn_mask = None

            attn_mask_eff = attn_mask

            scores = None
            layer_attn = None
            if use_cache:
                num_anchor_frames, num_special_tokens, special_token_offset, fixed_exempt = (
                    _resolve_kv_mode(kv_mode, self.patch_start_idx)
                )
                out = self.global_blocks[global_idx](
                    tokens,
                    pos=pos,
                    attn_mask=attn_mask,
                    past_key_values=past_key_values_block,
                    use_cache=True,
                    cache_budget=cache_budget,
                    frame_idx=past_frame_idx,
                    chunk_size=chunk_size,
                    timing_dict=timing_dict,
                    num_special_tokens=num_special_tokens,
                    num_anchor_frames=num_anchor_frames,
                    special_token_offset=special_token_offset,
                    fixed_exempt_from_budget=fixed_exempt,
                    return_attn_weights=return_attn_weights,
                    importance_scores=importance_scores,
                    importance_cache=importance_cache,
                    use_importance_in_attn=use_importance_in_attn,
                    softmax_importance_before_k=softmax_importance_before_k,
                    debug_importance_in_attn=debug_importance_in_attn,
                    layer_idx=global_idx,
                    prev_layer_kv=prev_layer_kv,
                    kv_share_cfg=kv_share_cfg,
                    motivation_kv_probe=motivation_kv_probe,
                )
                if return_attn_weights and len(out) == 4:
                    tokens, block_kv, scores, layer_attn = out
                else:
                    tokens, block_kv, scores = out
            else:
                if getattr(self, "gradient_checkpointing", False) and self.training:
                    block = self.global_blocks[global_idx]
                    # Bind loop-scoped values into defaults so checkpoint recompute
                    # doesn't read a mutated global_idx from later iterations.
                    def _block_forward(
                        t,
                        _block=block,
                        _pos=pos,
                        _attn_mask=attn_mask_eff,
                        _svc=sparse_kv_global,
                        _uf=use_flex_attention,
                        _fbs=flex_block_size,
                        _fcm=flex_compile_mode,
                    ):
                        return _block(
                            t,
                            pos=_pos,
                            attn_mask=_attn_mask,
                            sparse_kv_cfg=_svc,
                            use_flex_attention=_uf,
                            flex_block_size=_fbs,
                            flex_compile_mode=_fcm,
                        )

                    tokens = torch.utils.checkpoint.checkpoint(
                        _block_forward, tokens, use_reentrant=False
                    )
                else:
                    tokens = self.global_blocks[global_idx](
                        tokens,
                        pos=pos,
                        attn_mask=attn_mask_eff,
                        sparse_kv_cfg=sparse_kv_global,
                        use_flex_attention=use_flex_attention,
                        flex_block_size=flex_block_size,
                        flex_compile_mode=flex_compile_mode,
                    )

            global_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

            # if self.use_causal_global:
            #     del attn_mask
        if use_cache:
            if return_attn_weights and layer_attn is not None:
                return tokens, global_idx, intermediates, block_kv, scores, layer_attn
            return tokens, global_idx, intermediates, block_kv, scores
        return tokens, global_idx, intermediates
        
    def _calculate_dynamic_budgets(self, total_budget):
        """Allocate total_budget across layers. Uses budget_proportions if set (from
        compute_kv_budget_from_cosine_sim.py), else uses last_scores (improved_importance)."""
        with torch.no_grad():
            if self.budget_proportions is not None:
                proportions = self.budget_proportions.to(self.last_scores.device)
            else:
                diversity_scores = 1.0 - self.last_scores
                scaled_scores = diversity_scores / 0.5
                proportions = torch.softmax(scaled_scores, dim=0)
            if total_budget < 0:
                total_budget = 0
            budgets = proportions * total_budget

        return budgets.int()


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.reshape(B * S, *combined.shape[2:])
    return combined
