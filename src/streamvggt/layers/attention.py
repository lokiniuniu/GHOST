import logging
import os
import time
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Union, Tuple, Dict, Optional, Any, List
try:
    from sparse_vggt.utils.sparse_wrapper import block_sparse_attn_cuda as _sparge_block_sparse_attn_cuda
    SPARGE_KERNEL_AVAILABLE = True
except Exception:
    try:
        from streamvggt.kernels.sparge_wrapper import block_sparse_attn_cuda as _sparge_block_sparse_attn_cuda
        SPARGE_KERNEL_AVAILABLE = True
    except Exception:
        _sparge_block_sparse_attn_cuda = None
        SPARGE_KERNEL_AVAILABLE = False
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except Exception:
    FLEX_ATTENTION_AVAILABLE = False
_COMPILED_FLEX_ATTENTION = {}

XFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self.num_anchor_tokens = 0
        self._tokens_per_frame = None  # set on first frame
        self._last_kept_attn_importance = None
        self._flex_block_mask_cache = {}

    def _append_motivation_kv_probe(
        self,
        q: Tensor,
        k: Tensor,
        *,
        past_key_values: Optional[Tuple[Tensor, Tensor]],
        frame_idx: Optional[int],
        layer_idx: Optional[int],
        motivation_kv_probe: Optional[Dict[str, Any]],
        patch_start_idx: int,
    ) -> None:
        """Record mean cos(K_patch^s, mean(Q_patch^current))) for each past source frame s."""
        if motivation_kv_probe is None or not motivation_kv_probe.get("enabled"):
            return
        tgt = int(motivation_kv_probe.get("layer_idx", -1))
        if layer_idx is None or int(layer_idx) != tgt:
            return
        if past_key_values is None:
            motivation_kv_probe.setdefault("records", []).append(
                {"past_frame_idx": int(frame_idx) if frame_idx is not None else -1, "past_frame_sims": []}
            )
            return
        if frame_idx is None or self._tokens_per_frame is None:
            return
        T = int(frame_idx)
        num_anchor = int(self.num_anchor_tokens)
        P = int(self._tokens_per_frame)
        ps = int(patch_start_idx)
        if P <= 0 or ps >= P or q.shape[2] != P:
            return
        q_patch = q[:, :, ps:, :]
        q_mean = F.normalize(q_patch.mean(dim=2, keepdim=True), dim=-1)
        _, _, nk, _ = k.shape
        past_sims: List[float] = []
        for s in range(T):
            if s == 0:
                pos0, pos1 = 0, min(num_anchor, nk)
            else:
                pos0 = num_anchor + (s - 1) * P
                pos1 = pos0 + P
            if pos1 > nk or pos0 >= pos1:
                break
            k_block = k[:, :, pos0:pos1, :]
            if k_block.shape[2] <= ps:
                past_sims.append(float("nan"))
                continue
            k_patch = k_block[:, :, ps:, :]
            kn = F.normalize(k_patch, dim=-1)
            cos = (kn * q_mean).sum(dim=-1)
            past_sims.append(float(cos.mean().detach().cpu()))
        motivation_kv_probe.setdefault("records", []).append(
            {"past_frame_idx": T, "layer_idx": int(layer_idx), "past_frame_sims": past_sims}
        )

    def _reset_cache_state(self):
        self.num_anchor_tokens = 0
        self._tokens_per_frame = None
        self._last_kept_attn_importance = None
        self._flex_block_mask_cache.clear()

    def _run_flex_attention(
        self,
        q,
        k,
        v,
        block_mask,
        *,
        compile_mode: str = "fullgraph",
    ):
        # FlexAttention requires query/key/value to share the exact dtype.
        # Align q/k to v.dtype to keep activation memory low in AMP (typically bf16).
        flex_dtype = v.dtype
        if q.dtype != flex_dtype:
            q = q.to(flex_dtype)
        if k.dtype != flex_dtype:
            k = k.to(flex_dtype)

        global _COMPILED_FLEX_ATTENTION
        if compile_mode in ("none", "disabled"):
            return flex_attention(q, k, v, block_mask=block_mask)

        key = compile_mode
        compiled = _COMPILED_FLEX_ATTENTION.get(key)
        if compiled is None:
            try:
                if compile_mode == "fullgraph":
                    compiled = torch.compile(flex_attention, fullgraph=True)
                elif compile_mode == "default":
                    compiled = torch.compile(flex_attention)
                elif compile_mode == "reduce-overhead":
                    compiled = torch.compile(flex_attention, mode="reduce-overhead")
                else:
                    compiled = torch.compile(flex_attention)
            except Exception:
                compiled = flex_attention
            _COMPILED_FLEX_ATTENTION[key] = compiled
        return compiled(q, k, v, block_mask=block_mask)

    def _get_flex_block_mask(self, attn_mask: torch.Tensor, block_size: int):
        if attn_mask.dim() == 3:
            dense_mask = attn_mask[0]
        else:
            dense_mask = attn_mask
        if dense_mask.dim() != 2:
            return None
        q_len, kv_len = dense_mask.shape[-2], dense_mask.shape[-1]
        cache_key = (int(dense_mask.data_ptr()), q_len, kv_len, int(block_size), str(dense_mask.device))
        cached = self._flex_block_mask_cache.get(cache_key)
        if cached is not None:
            return cached
        allow = dense_mask == 0

        def _mask_mod(b, h, q_idx, kv_idx):
            return allow[q_idx, kv_idx]

        block_mask = create_block_mask(
            _mask_mod,
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=dense_mask.device,
            BLOCK_SIZE=max(16, int(block_size)),
        )
        self._flex_block_mask_cache[cache_key] = block_mask
        return block_mask

    def _subsample_along_tokens(self, x: torch.Tensor, stride: int) -> torch.Tensor:
        if stride <= 1 or x.shape[2] <= 1:
            return x
        return x[:, :, ::stride, :]

    def _gather_tokens(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        if idx.numel() == 0:
            return x[:, :, :0, :]
        idx = idx.to(device=x.device, dtype=torch.long)
        expanded = idx.view(1, 1, -1, 1).expand(x.shape[0], x.shape[1], idx.numel(), x.shape[3])
        return torch.gather(x, 2, expanded)

    def _build_grid_patch_keep_indices(
        self,
        patch_h: int,
        patch_w: int,
        stride_h: int,
        stride_w: int,
        device: torch.device,
    ) -> torch.Tensor:
        stride_h = max(1, int(stride_h))
        stride_w = max(1, int(stride_w))
        ys = torch.arange(0, max(1, int(patch_h)), stride_h, device=device, dtype=torch.long)
        xs = torch.arange(0, max(1, int(patch_w)), stride_w, device=device, dtype=torch.long)
        if ys.numel() == 0:
            ys = torch.zeros(1, device=device, dtype=torch.long)
        if xs.numel() == 0:
            xs = torch.zeros(1, device=device, dtype=torch.long)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return (gy * int(patch_w) + gx).reshape(-1).to(torch.long)

    def _build_flex_block_mask_from_allow(
        self,
        allow: torch.Tensor,
        *,
        block_size: int,
    ):
        if not FLEX_ATTENTION_AVAILABLE:
            return None
        q_len, kv_len = allow.shape[-2], allow.shape[-1]
        cache_key = (
            "sparse",
            int(allow.data_ptr()),
            q_len,
            kv_len,
            int(block_size),
            str(allow.device),
        )
        cached = self._flex_block_mask_cache.get(cache_key)
        if cached is not None:
            return cached

        def _mask_mod(b, h, q_idx, kv_idx):
            return allow[q_idx, kv_idx]

        block_mask = create_block_mask(
            _mask_mod,
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=allow.device,
            BLOCK_SIZE=max(16, int(block_size)),
        )
        self._flex_block_mask_cache[cache_key] = block_mask
        return block_mask

    def _predict_attention_pooled(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        ks_q: int,
        ks_k: int,
        pool_mode: str,
    ) -> torch.Tensor:
        # query/key: [B, H, T, D]
        if pool_mode not in ("avg", "max"):
            pool_mode = "avg"
        pooling_fn = F.avg_pool1d if pool_mode == "avg" else F.max_pool1d
        B, H, Tq, D = query.shape
        Tk = key.shape[-2]
        ks_q = max(1, int(ks_q))
        ks_k = max(1, int(ks_k))

        q1 = query.reshape(B * H, D, Tq)
        k1 = key.reshape(B * H, D, Tk)
        q_pool = pooling_fn(q1, kernel_size=ks_q, ceil_mode=True).reshape(B, H, D, -1).transpose(-2, -1)
        k_pool = pooling_fn(k1, kernel_size=ks_k, ceil_mode=True).reshape(B, H, D, -1).transpose(-2, -1)

        score = torch.matmul(q_pool, k_pool.transpose(-1, -2)) * (D ** -0.5)
        return F.softmax(score.float(), dim=-1)

    def _build_sparse_vggt_allow_map(
        self,
        *,
        n_tokens: int,
        num_special_tokens: int,
        n_patch: int,
        pooled_score: torch.Tensor,
        ks_q: int,
        ks_k: int,
        sparse_ratio: Optional[float],
        cdf_threshold: Optional[float],
        topk_blocks: Optional[int],
        preserve_diagonal: bool,
    ) -> torch.Tensor:
        device = pooled_score.device
        allow = torch.zeros((n_tokens, n_tokens), dtype=torch.bool, device=device)
        num_special = max(0, min(int(num_special_tokens), n_tokens))
        if num_special > 0:
            allow[:, :num_special] = True
            allow[:num_special, :] = True

        # Use averaged pooled score across batch/heads to build a stable global mask.
        pooled = pooled_score.mean(dim=(0, 1))  # [q_blk, k_blk]
        q_blk, k_blk = pooled.shape
        if q_blk <= 0 or k_blk <= 0:
            return allow

        if sparse_ratio is not None:
            sparse_ratio = float(max(0.0, min(1.0, sparse_ratio)))
        if cdf_threshold is not None:
            cdf_threshold = float(max(0.0, min(1.0, cdf_threshold)))
        if topk_blocks is not None:
            topk_blocks = int(max(1, min(k_blk, int(topk_blocks))))

        if topk_blocks is None and sparse_ratio is None and cdf_threshold is None:
            sparse_ratio = 0.75

        ks_q = max(1, int(ks_q))
        ks_k = max(1, int(ks_k))

        for qb in range(q_blk):
            row = pooled[qb]
            sort_vals, sort_idx = torch.sort(row, descending=True)

            num_sel = None
            if cdf_threshold is not None:
                cdf = torch.cumsum(sort_vals, dim=-1)
                num_sel = int(torch.searchsorted(cdf, torch.tensor(cdf_threshold, device=device), right=True).item())
                num_sel = max(1, min(k_blk, num_sel))
            if sparse_ratio is not None:
                keep_from_ratio = int(round(k_blk * (1.0 - sparse_ratio)))
                keep_from_ratio = max(1, min(k_blk, keep_from_ratio))
                num_sel = keep_from_ratio if num_sel is None else max(num_sel, keep_from_ratio)
            if topk_blocks is not None:
                num_sel = topk_blocks if num_sel is None else max(num_sel, topk_blocks)
            if num_sel is None:
                num_sel = k_blk

            selected = sort_idx[:num_sel]
            q_start = qb * ks_q
            q_end = min((qb + 1) * ks_q, n_patch)
            q_rows = torch.arange(q_start, q_end, device=device, dtype=torch.long) + num_special
            if q_rows.numel() <= 0:
                continue
            for kb in selected.tolist():
                k_start = kb * ks_k
                k_end = min((kb + 1) * ks_k, n_patch)
                k_cols = torch.arange(k_start, k_end, device=device, dtype=torch.long) + num_special
                if k_cols.numel() > 0:
                    allow[q_rows.unsqueeze(1), k_cols.unsqueeze(0)] = True

        if preserve_diagonal and n_patch > 0:
            diag = torch.arange(n_patch, device=device, dtype=torch.long) + num_special
            allow[diag, diag] = True
        return allow

    def _sparse_attention_sparse_vggt_style(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        num_special_tokens: int,
        patch_h: int,
        patch_w: int,
        sparse_ratio: Optional[float],
        cdf_threshold: Optional[float],
        topk_blocks: Optional[int],
        pool_mode: str,
        ks_q: int,
        ks_k: int,
        preserve_diagonal: bool,
        use_sparge_kernel: bool,
        flex_block_size: int,
        flex_compile_mode: str,
        attn_drop_p: float,
    ) -> Optional[torch.Tensor]:
        B, H, N, _ = q.shape
        n_patch = int(patch_h) * int(patch_w)
        if n_patch <= 0 or N != int(num_special_tokens) + n_patch:
            return None

        num_special = max(0, min(int(num_special_tokens), N))
        q_patch = q[:, :, num_special:, :]
        k_patch = k[:, :, num_special:, :]
        v_patch = v[:, :, num_special:, :]
        if q_patch.shape[2] <= 0 or k_patch.shape[2] <= 0:
            return None

        pooled = self._predict_attention_pooled(
            q_patch,
            k_patch,
            ks_q=ks_q,
            ks_k=ks_k,
            pool_mode=pool_mode,
        )

        # Use real SpargeAttn block-sparse kernel when available.
        if use_sparge_kernel and SPARGE_KERNEL_AVAILABLE and _sparge_block_sparse_attn_cuda is not None:
            x_special = None
            if num_special > 0:
                q_special = q[:, :, :num_special, :]
                x_special = F.scaled_dot_product_attention(
                    q_special,
                    k,
                    v,
                    dropout_p=attn_drop_p,
                )
            key_all = torch.cat([k_patch, k[:, :, :num_special, :]], dim=2) if num_special > 0 else k_patch
            value_all = torch.cat([v_patch, v[:, :, :num_special, :]], dim=2) if num_special > 0 else v_patch
            x_patch = _sparge_block_sparse_attn_cuda(
                query=q_patch,
                key=key_all,
                value=value_all,
                pooled_score=pooled,
                topk=topk_blocks,
                sparse_ratio=sparse_ratio,
                cdf_threshold=cdf_threshold,
                return_sparsity=False,
            )
            if num_special > 0:
                return torch.cat([x_special, x_patch], dim=2)
            return x_patch

        allow = self._build_sparse_vggt_allow_map(
            n_tokens=N,
            num_special_tokens=num_special,
            n_patch=n_patch,
            pooled_score=pooled,
            ks_q=ks_q,
            ks_k=ks_k,
            sparse_ratio=sparse_ratio,
            cdf_threshold=cdf_threshold,
            topk_blocks=topk_blocks,
            preserve_diagonal=preserve_diagonal,
        )

        if FLEX_ATTENTION_AVAILABLE:
            block_mask = self._build_flex_block_mask_from_allow(
                allow,
                block_size=flex_block_size,
            )
            if block_mask is not None:
                return self._run_flex_attention(
                    q,
                    k,
                    v,
                    block_mask,
                    compile_mode=flex_compile_mode,
                )

        dense_mask = torch.full(
            (N, N),
            torch.finfo(q.dtype).min,
            device=q.device,
            dtype=q.dtype,
        )
        dense_mask[allow] = 0
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=dense_mask,
            dropout_p=attn_drop_p,
        )

    def _sparse_attention_sparse_vggt_style_global(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        num_frames: int,
        tokens_per_frame: int,
        num_special_tokens: int,
        patch_h: int,
        patch_w: int,
        frame_causal: bool,
        sparse_ratio: Optional[float],
        cdf_threshold: Optional[float],
        topk_blocks: Optional[int],
        pool_mode: str,
        ks_q: int,
        ks_k: int,
        preserve_diagonal: bool,
        use_sparge_kernel: bool,
        flex_block_size: int,
        flex_compile_mode: str,
        attn_drop_p: float,
    ) -> Optional[torch.Tensor]:
        """
        Sparse-VGGT-style attention over the full global sequence (all frames flattened).

        Layout per frame (length P = tokens_per_frame): [special x num_special][patch x n_patch],
        repeated S times, total N = S * P. Patch-only proxy scores are over length S * n_patch,
        then lifted to N x N with special-token hubs. When frame_causal is True, matches the
        existing global no-cache mask: query at frame fq may attend to keys only in frames <= fq.

        Note: Sparge CUDA path is not used here (multi-frame causal + layout); FlexAttention or
        dense SDPA mask is used instead.
        """
        del use_sparge_kernel  # reserved; global path uses flex / dense mask only
        B, H, N, _ = q.shape
        S = int(num_frames)
        P = int(tokens_per_frame)
        n_patch = int(patch_h) * int(patch_w)
        num_special = int(num_special_tokens)
        if S <= 0 or P <= 0 or n_patch <= 0 or N != S * P or P != num_special + n_patch:
            return None

        L = N
        device = q.device
        q4 = q.view(B, H, S, P, -1)
        k4 = k.view(B, H, S, P, -1)
        q_patch = q4[:, :, :, num_special:, :].reshape(B, H, S * n_patch, -1)
        k_patch = k4[:, :, :, num_special:, :].reshape(B, H, S * n_patch, -1)
        if q_patch.shape[2] <= 0:
            return None

        pooled = self._predict_attention_pooled(
            q_patch,
            k_patch,
            ks_q=ks_q,
            ks_k=ks_k,
            pool_mode=pool_mode,
        )
        Spi = S * n_patch
        allow_patch = self._build_sparse_vggt_allow_map(
            n_tokens=Spi,
            num_special_tokens=0,
            n_patch=Spi,
            pooled_score=pooled,
            ks_q=ks_q,
            ks_k=ks_k,
            sparse_ratio=sparse_ratio,
            cdf_threshold=cdf_threshold,
            topk_blocks=topk_blocks,
            preserve_diagonal=preserve_diagonal,
        )

        idx_flat = torch.arange(Spi, device=device, dtype=torch.long)
        frame_of_patch = idx_flat // n_patch
        local_patch = idx_flat % n_patch
        GI = frame_of_patch * P + num_special + local_patch

        allow_full = torch.zeros((L, L), dtype=torch.bool, device=device)
        gi = GI.view(-1, 1)
        gj = GI.view(1, -1)
        if frame_causal:
            frame_ids = torch.arange(L, device=device, dtype=torch.long) // P
            fq = frame_ids[gi]
            fk = frame_ids[gj]
            patch_ok = fq >= fk
            allow_full[gi, gj] = allow_patch & patch_ok
        else:
            allow_full[gi, gj] = allow_patch

        if num_special > 0:
            pos1d = torch.arange(L, device=device, dtype=torch.long)
            frame = pos1d // P
            off = pos1d % P
            spec_positions = torch.nonzero(off < num_special, as_tuple=False).squeeze(-1)
            Ns = int(spec_positions.numel())
            if Ns > 0:
                if frame_causal:
                    Fspec = frame[spec_positions]
                    Fr = Fspec.unsqueeze(1)
                    Fk = frame.view(1, -1)
                    hub_row = Fk <= Fr
                    rs = spec_positions.unsqueeze(1).expand(-1, L)
                    ks_idx = torch.arange(L, device=device).view(1, -1).expand(Ns, -1)
                    allow_full[rs, ks_idx] |= hub_row

                    Fc = Fspec.unsqueeze(1)
                    Fq = frame.view(1, -1)
                    hub_col = Fq >= Fc
                    qs_idx = torch.arange(L, device=device).view(1, -1).expand(Ns, -1)
                    cs = spec_positions.unsqueeze(1).expand(-1, L)
                    allow_full[qs_idx, cs] |= hub_col
                else:
                    allow_full[:, spec_positions] = True
                    allow_full[spec_positions, :] = True

        if preserve_diagonal:
            allow_full.fill_diagonal_(True)

        if FLEX_ATTENTION_AVAILABLE:
            block_mask = self._build_flex_block_mask_from_allow(
                allow_full,
                block_size=flex_block_size,
            )
            if block_mask is not None:
                return self._run_flex_attention(
                    q,
                    k,
                    v,
                    block_mask,
                    compile_mode=flex_compile_mode,
                )

        dense_limit = 4096
        if L > dense_limit:
            logger.warning(
                "global sparse_vggt: sequence length %d > %d and FlexAttention unavailable; "
                "falling back to full (dense) attention.",
                L,
                dense_limit,
            )
            return F.scaled_dot_product_attention(
                q, k, v, dropout_p=attn_drop_p
            )

        dense_mask = torch.full(
            (L, L),
            torch.finfo(q.dtype).min,
            device=device,
            dtype=q.dtype,
        )
        dense_mask[allow_full] = 0
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=dense_mask,
            dropout_p=attn_drop_p,
        )

    def _sparse_attention_with_flex_meanfill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        num_special_tokens: int,
        patch_h: int,
        patch_w: int,
        stride_h: int,
        stride_w: int,
        preserve_diagonal: bool,
        use_mean_fill: bool,
        flex_block_size: int,
        flex_compile_mode: str,
        debug_sparse_stats: bool = False,
        debug_layer_idx: Optional[int] = None,
        debug_print_every: int = 1,
    ) -> Optional[torch.Tensor]:
        # q/k/v: [B, H, N, D], N = num_special + patch_h*patch_w
        B, H, N, D = q.shape
        n_patch = int(patch_h) * int(patch_w)
        if n_patch <= 0 or N != int(num_special_tokens) + n_patch:
            return None

        num_special = max(0, min(int(num_special_tokens), N))
        keep_patch_local = self._build_grid_patch_keep_indices(
            patch_h=patch_h,
            patch_w=patch_w,
            stride_h=stride_h,
            stride_w=stride_w,
            device=q.device,
        )
        if keep_patch_local.numel() == 0:
            keep_patch_local = torch.zeros(1, device=q.device, dtype=torch.long)
        keep_patch_local = torch.unique(keep_patch_local, sorted=True)
        keep_patch_local = keep_patch_local[keep_patch_local < n_patch]
        if keep_patch_local.numel() <= 0:
            keep_patch_local = torch.zeros(1, device=q.device, dtype=torch.long)

        keep_global = torch.cat(
            [
                torch.arange(num_special, device=q.device, dtype=torch.long),
                keep_patch_local + num_special,
            ],
            dim=0,
        )
        keep_global = torch.unique(keep_global, sorted=True)
        if keep_global.numel() >= N:
            return None

        keep_mask = torch.zeros(N, device=q.device, dtype=torch.bool)
        keep_mask[keep_global] = True
        drop_global = torch.arange(N, device=q.device, dtype=torch.long)[~keep_mask]

        # Build K/V as [kept tokens] (+ [mean token]) (+ [diag tokens for dropped]).
        exp_keep = keep_global.view(1, 1, -1, 1).expand(B, H, keep_global.numel(), D)
        k_keep = torch.gather(k, 2, exp_keep)
        v_keep = torch.gather(v, 2, exp_keep)
        k_eff = k_keep
        v_eff = v_keep
        has_mean_token = bool(use_mean_fill and drop_global.numel() > 0)
        if has_mean_token:
            exp_drop = drop_global.view(1, 1, -1, 1).expand(B, H, drop_global.numel(), D)
            k_drop = torch.gather(k, 2, exp_drop)
            v_drop = torch.gather(v, 2, exp_drop)
            k_mean = k_drop.mean(dim=2, keepdim=True)
            v_mean = v_drop.mean(dim=2, keepdim=True)
            k_eff = torch.cat([k_keep, k_mean], dim=2)
            v_eff = torch.cat([v_keep, v_mean], dim=2)
        has_diag_cols = bool(preserve_diagonal and drop_global.numel() > 0)
        if has_diag_cols:
            exp_drop_diag = drop_global.view(1, 1, -1, 1).expand(B, H, drop_global.numel(), D)
            k_diag = torch.gather(k, 2, exp_drop_diag)
            v_diag = torch.gather(v, 2, exp_drop_diag)
            k_eff = torch.cat([k_eff, k_diag], dim=2)
            v_eff = torch.cat([v_eff, v_diag], dim=2)

        # Build sparse allow map for Flex:
        # - All queries can attend to special K tokens.
        # - Special queries attend to all kept K tokens.
        # - Patch queries attend to kept grid K tokens.
        # - Preserve diagonal q_i -> k_i by dedicated dropped-token diag columns.
        # - Mean token (if any) is visible to all queries.
        kv_eff = k_eff.shape[2]
        allow = torch.zeros((N, kv_eff), dtype=torch.bool, device=q.device)

        special_keep_count = num_special
        patch_keep_start = special_keep_count
        patch_keep_end = patch_keep_start + keep_patch_local.numel()
        mean_col = patch_keep_end if has_mean_token else -1
        diag_start = patch_keep_end + (1 if has_mean_token else 0)

        if special_keep_count > 0:
            allow[:, :special_keep_count] = True
        if num_special > 0:
            allow[:num_special, :patch_keep_end] = True

        if patch_keep_end > patch_keep_start and N > num_special:
            allow[num_special:, patch_keep_start:patch_keep_end] = True

        if has_diag_cols:
            dropped_patch = drop_global[drop_global >= num_special]
            if dropped_patch.numel() > 0:
                # Map each dropped token query to its dedicated diagonal KV column.
                diag_cols = (
                    torch.arange(dropped_patch.numel(), device=q.device, dtype=torch.long)
                    + diag_start
                )
                allow[dropped_patch, diag_cols] = True

        if has_mean_token:
            allow[:, mean_col] = True

        block_mask = self._build_flex_block_mask_from_allow(allow, block_size=flex_block_size)
        if block_mask is None:
            return None
        if debug_sparse_stats:
            print_every = max(1, int(debug_print_every))
            lid = -1 if debug_layer_idx is None else int(debug_layer_idx)
            if lid < 0 or lid % print_every == 0:
                kept = int(keep_global.numel())
                dropped = int(drop_global.numel())
                kv_cols = int(kv_eff)
                approx_density = float(kv_cols) / float(max(1, N))
                print(
                    (
                        "[sparsity-flex] "
                        f"layer={lid} N={int(N)} kept={kept} dropped={dropped} "
                        f"mean_token={int(has_mean_token)} diag_cols={int(has_diag_cols) * dropped} "
                        f"kv_cols={kv_cols} approx_density={approx_density:.4f}"
                    ),
                    flush=True,
                )
        return self._run_flex_attention(
            q,
            k_eff,
            v_eff,
            block_mask,
            compile_mode=flex_compile_mode,
        )

    def _sparse_attention_with_diag_and_meanfill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        num_special_tokens: int,
        patch_h: int,
        patch_w: int,
        stride_h: int,
        stride_w: int,
        preserve_diagonal: bool,
        use_mean_fill: bool,
        attn_drop_p: float,
    ) -> torch.Tensor:
        # q/k/v: [B, H, N, D], N = num_special + patch_h*patch_w
        B, H, N, _ = q.shape
        n_patch = int(patch_h) * int(patch_w)
        if n_patch <= 0 or N != int(num_special_tokens) + n_patch:
            return F.scaled_dot_product_attention(
                q, k, v, dropout_p=attn_drop_p
            )

        num_special = max(0, min(int(num_special_tokens), N))
        keep_patch_local = self._build_grid_patch_keep_indices(
            patch_h=patch_h,
            patch_w=patch_w,
            stride_h=stride_h,
            stride_w=stride_w,
            device=q.device,
        )
        if keep_patch_local.numel() == 0:
            keep_patch_local = torch.zeros(1, device=q.device, dtype=torch.long)
        keep_patch_local = torch.unique(keep_patch_local, sorted=True)
        keep_patch_local = keep_patch_local[keep_patch_local < n_patch]
        if keep_patch_local.numel() <= 0:
            keep_patch_local = torch.zeros(1, device=q.device, dtype=torch.long)

        patch_global_offset = num_special
        keep_global = torch.cat(
            [
                torch.arange(num_special, device=q.device, dtype=torch.long),
                keep_patch_local + patch_global_offset,
            ],
            dim=0,
        )
        keep_global = torch.unique(keep_global, sorted=True)
        if keep_global.numel() >= N:
            return F.scaled_dot_product_attention(
                q, k, v, dropout_p=attn_drop_p
            )

        keep_mask = torch.zeros(N, device=q.device, dtype=torch.bool)
        keep_mask[keep_global] = True
        drop_global = torch.arange(N, device=q.device, dtype=torch.long)[~keep_mask]

        expanded_keep = keep_global.view(1, 1, -1, 1).expand(B, H, keep_global.numel(), q.shape[-1])
        k_keep = torch.gather(k, 2, expanded_keep)
        v_keep = torch.gather(v, 2, expanded_keep)

        q_scaled = q * self.scale
        logits_keep = torch.matmul(q_scaled, k_keep.transpose(-2, -1))

        logits_parts = [logits_keep]
        value_parts = [v_keep]

        if use_mean_fill and drop_global.numel() > 0:
            expanded_drop = drop_global.view(1, 1, -1, 1).expand(B, H, drop_global.numel(), q.shape[-1])
            k_drop = torch.gather(k, 2, expanded_drop)
            v_drop = torch.gather(v, 2, expanded_drop)
            k_mean = k_drop.mean(dim=2, keepdim=True)
            v_mean = v_drop.mean(dim=2, keepdim=True)
            logits_mean = torch.matmul(q_scaled, k_mean.transpose(-2, -1))
            logits_parts.append(logits_mean)
            value_parts.append(v_mean)

        if preserve_diagonal:
            diag_logits = (q_scaled * k).sum(dim=-1, keepdim=True)
            diag_mask = ~keep_mask
            if not bool(diag_mask.any()):
                diag_logits = None
            else:
                diag_logits = diag_logits.masked_fill(
                    ~diag_mask.view(1, 1, N, 1), torch.finfo(diag_logits.dtype).min
                )
            if diag_logits is not None:
                logits_parts.append(diag_logits)

        logits = torch.cat(logits_parts, dim=-1)
        attn = F.softmax(logits, dim=-1)
        if attn_drop_p > 0:
            attn = F.dropout(attn, p=attn_drop_p, training=self.training)

        out = torch.matmul(attn[..., : value_parts[0].shape[2]], value_parts[0])
        col_start = value_parts[0].shape[2]
        for vp in value_parts[1:]:
            out = out + attn[..., col_start : col_start + vp.shape[2]] * vp
            col_start += vp.shape[2]

        if preserve_diagonal and len(logits_parts) > len(value_parts):
            out = out + attn[..., col_start : col_start + 1] * v
        return out

    def _apply_layer_group_share(
        self,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
        prev_layer_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
        layer_idx: Optional[int],
        cfg: Dict[str, object],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if prev_layer_kv is None or layer_idx is None:
            return past_k, past_v
        group_size = max(2, int(cfg.get("group_size", 2)))
        # Share on the second layer in each group (e.g. 1,3,5... for group_size=2)
        if layer_idx % group_size != (group_size - 1):
            return past_k, past_v
        prev_k, prev_v = prev_layer_kv
        if prev_k is None or prev_v is None:
            return past_k, past_v
        min_tokens = min(past_k.shape[2], prev_k.shape[2])
        if min_tokens <= 0:
            return past_k, past_v
        share_heads_ratio = float(cfg.get("share_heads_ratio", 0.5))
        share_token_ratio = float(cfg.get("share_token_ratio", 0.5))
        share_heads = max(1, min(past_k.shape[1], int(round(past_k.shape[1] * share_heads_ratio))))
        share_tokens = max(1, min(min_tokens, int(round(min_tokens * share_token_ratio))))

        out_k = past_k.clone()
        out_v = past_v.clone()
        out_k[:, :share_heads, :share_tokens, :] = prev_k[:, :share_heads, :share_tokens, :].to(out_k.dtype)
        out_v[:, :share_heads, :share_tokens, :] = prev_v[:, :share_heads, :share_tokens, :].to(out_v.dtype)
        return out_k, out_v

    def _apply_coarse_share(
        self,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
        layer_idx: Optional[int],
        tokens_per_frame: int,
        cfg: Dict[str, object],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx is None:
            return past_k, past_v
        start_layer = int(cfg.get("coarse_start_layer", 12))
        if layer_idx < start_layer:
            return past_k, past_v
        if tokens_per_frame <= 0:
            return past_k, past_v

        near_frames = max(1, int(cfg.get("coarse_near_frames", 4)))
        stride = max(2, int(cfg.get("coarse_stride", 4)))
        near_tokens = min(past_k.shape[2], near_frames * tokens_per_frame)
        far_len = max(0, past_k.shape[2] - near_tokens)
        if far_len <= 0:
            return past_k, past_v

        far_k = past_k[:, :, :far_len, :]
        far_v = past_v[:, :, :far_len, :]
        near_k = past_k[:, :, far_len:, :]
        near_v = past_v[:, :, far_len:, :]
        far_k_coarse = self._subsample_along_tokens(far_k, stride)
        far_v_coarse = self._subsample_along_tokens(far_v, stride)
        return torch.cat([far_k_coarse, near_k], dim=2), torch.cat([far_v_coarse, near_v], dim=2)

    def _apply_delta_share(
        self,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
        prev_layer_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
        layer_idx: Optional[int],
        cfg: Dict[str, object],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if prev_layer_kv is None or layer_idx is None:
            return past_k, past_v
        start_layer = int(cfg.get("delta_start_layer", 1))
        if layer_idx < start_layer:
            return past_k, past_v
        prev_k, prev_v = prev_layer_kv
        if prev_k is None or prev_v is None:
            return past_k, past_v

        min_tokens = min(past_k.shape[2], prev_k.shape[2])
        if min_tokens <= 0:
            return past_k, past_v
        keep_ratio = float(cfg.get("delta_keep_ratio", 0.5))
        keep_ratio = max(0.05, min(1.0, keep_ratio))
        keep_tokens = max(1, int(round(min_tokens * keep_ratio)))

        delta = (past_k[:, :, :min_tokens, :] - prev_k[:, :, :min_tokens, :].to(past_k.dtype)).abs().mean(dim=(0, 1, 3))
        top_idx = torch.topk(delta, k=keep_tokens, largest=True).indices.sort().values
        kept_k = self._gather_tokens(past_k[:, :, :min_tokens, :], top_idx)
        kept_v = self._gather_tokens(past_v[:, :, :min_tokens, :], top_idx)
        # Keep tail tokens that don't overlap the min shared window.
        tail_k = past_k[:, :, min_tokens:, :]
        tail_v = past_v[:, :, min_tokens:, :]
        if tail_k.shape[2] > 0:
            kept_k = torch.cat([kept_k, tail_k], dim=2)
            kept_v = torch.cat([kept_v, tail_v], dim=2)
        return kept_k, kept_v

    def _apply_target_ratio_cap(
        self,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
        target_ratio: float,
        tokens_per_frame: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if target_ratio >= 1.0:
            return past_k, past_v
        n = past_k.shape[2]
        if n <= 1:
            return past_k, past_v
        target_n = max(1, int(round(n * target_ratio)))
        if target_n >= n:
            return past_k, past_v

        # Preserve recent history (most useful for streaming) and subsample old tokens.
        keep_recent = max(1, min(tokens_per_frame, target_n))
        old_n = n - keep_recent
        keep_old = target_n - keep_recent
        if keep_old <= 0:
            return past_k[:, :, -keep_recent:, :], past_v[:, :, -keep_recent:, :]
        idx_old = torch.linspace(0, old_n - 1, steps=keep_old, device=past_k.device).round().long()
        old_k = past_k[:, :, :old_n, :]
        old_v = past_v[:, :, :old_n, :]
        kept_old_k = self._gather_tokens(old_k, idx_old)
        kept_old_v = self._gather_tokens(old_v, idx_old)
        recent_k = past_k[:, :, -keep_recent:, :]
        recent_v = past_v[:, :, -keep_recent:, :]
        return torch.cat([kept_old_k, recent_k], dim=2), torch.cat([kept_old_v, recent_v], dim=2)

    def _apply_kv_sharing(
        self,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
        *,
        layer_idx: Optional[int],
        prev_layer_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
        kv_share_cfg: Optional[Dict[str, object]],
        tokens_per_frame: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if kv_share_cfg is None:
            return past_k, past_v
        method = str(kv_share_cfg.get("method", "none"))
        if method == "none":
            return past_k, past_v
        if method == "layer_group":
            past_k, past_v = self._apply_layer_group_share(past_k, past_v, prev_layer_kv, layer_idx, kv_share_cfg)
        elif method == "coarse":
            past_k, past_v = self._apply_coarse_share(past_k, past_v, layer_idx, tokens_per_frame, kv_share_cfg)
        elif method == "delta":
            past_k, past_v = self._apply_delta_share(past_k, past_v, prev_layer_kv, layer_idx, kv_share_cfg)
        target_ratio = kv_share_cfg.get("target_ratio", None)
        if target_ratio is not None:
            past_k, past_v = self._apply_target_ratio_cap(
                past_k,
                past_v,
                float(target_ratio),
                tokens_per_frame=tokens_per_frame,
            )
        return past_k, past_v

    def _expand_patch_importance_to_candidates(
        self,
        importance_scores: torch.Tensor,
        num_candidates: int,
        tokens_per_frame: int,
        num_special_tokens: int,
        special_token_offset: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if importance_scores is None or importance_scores.dim() != 1:
            return None
        if num_candidates <= 0:
            return torch.empty(0, device=device, dtype=dtype)
        if num_special_tokens <= 0:
            return None

        patch_indices = []
        for i in range(0, num_candidates, tokens_per_frame):
            for j in range(tokens_per_frame):
                if j < special_token_offset or j >= special_token_offset + num_special_tokens:
                    if i + j < num_candidates:
                        patch_indices.append(i + j)
        if len(patch_indices) != importance_scores.numel():
            return None

        expanded = torch.ones(num_candidates, device=device, dtype=dtype)
        expanded[torch.tensor(patch_indices, device=device, dtype=torch.long)] = (
            importance_scores.to(device).to(dtype)
        )
        return expanded

    def eviction(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_budget: int,
        num_anchor_tokens: int,
        num_special_tokens: int = 0,
        tokens_per_frame: Optional[int] = None,
        special_token_offset: int = 0,
        fixed_exempt_from_budget: bool = False,
        importance_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, Optional[torch.Tensor]]:
        """
        Evicts tokens from the key-value cache based on key cosine similarity.

        Args:
            k (torch.Tensor): The key tensor of shape [B, H, N, D].
            v (torch.Tensor): The value tensor of shape [B, H, N, D].
            cache_budget (int): The maximum number of tokens to retain (or budget for
                evictable part only when fixed_exempt_from_budget=True).
            num_anchor_tokens (int): The number of initial tokens to preserve.
            num_special_tokens (int): If > 0, absolutely preserve special tokens in the
                candidate segment. Each frame block has special tokens at
                [offset, offset+count) within the frame.
            tokens_per_frame (int, optional): Size of one frame for special token logic.
            special_token_offset (int): Start index of special tokens within each frame
                (0=camera first, 1=register only for tokens 1-4).
            fixed_exempt_from_budget (bool): If True, anchor+special don't count toward
                cache_budget; budget applies only to evictable patch tokens.

        Returns:
            A tuple of (pruned_k, pruned_v, avg_scores).
        """
        B, H, N, D = k.shape

        self._last_kept_attn_importance = None
        if N <= cache_budget and not fixed_exempt_from_budget:
            return k, v, 0.0, None

        anchor_k, candidate_k = k.split([num_anchor_tokens, N - num_anchor_tokens], dim=2)
        anchor_v, candidate_v = v.split([num_anchor_tokens, N - num_anchor_tokens], dim=2)

        num_candidates = N - num_anchor_tokens
        if fixed_exempt_from_budget:
            num_to_keep_from_candidates = cache_budget
        else:
            num_to_keep_from_candidates = cache_budget - num_anchor_tokens

        # Handle edge cases: budget < anchors, or keep more than we have
        if not fixed_exempt_from_budget and num_to_keep_from_candidates <= 0:
            return k[:, :, :cache_budget, :], v[:, :, :cache_budget, :], 0.0, None
        if num_to_keep_from_candidates >= num_candidates:
            return k, v, 0.0, None

        # [Special tokens] Absolutely preserve special tokens in candidate segment.
        frame_size = tokens_per_frame if tokens_per_frame is not None else num_anchor_tokens
        if num_special_tokens > 0 and num_anchor_tokens > 0:
            tokens_per_frame = frame_size
            special_indices = []
            patch_indices = []
            for i in range(0, num_candidates, tokens_per_frame):
                for j in range(num_special_tokens):
                    idx = i + special_token_offset + j
                    if idx < num_candidates and idx < i + tokens_per_frame:
                        special_indices.append(idx)
                for j in range(tokens_per_frame):
                    if j < special_token_offset or j >= special_token_offset + num_special_tokens:
                        if i + j < num_candidates:
                            patch_indices.append(i + j)

            n_special = len(special_indices)
            n_patch = len(patch_indices)
            num_to_keep_from_patches = num_to_keep_from_candidates - n_special

            if num_to_keep_from_patches <= 0:
                # Budget too small: absolutely keep anchor + all special (may exceed budget)
                kept_special_idx = torch.tensor(
                    special_indices, device=k.device, dtype=torch.long
                )
                expanded = kept_special_idx.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                expanded = expanded.expand(B, H, n_special, D)
                kept_k = torch.gather(candidate_k, 2, expanded)
                kept_v = torch.gather(candidate_v, 2, expanded)
                final_k = torch.cat([anchor_k, kept_k], dim=2)
                final_v = torch.cat([anchor_v, kept_v], dim=2)
                self._last_kept_attn_importance = torch.ones(
                    n_special, device=k.device, dtype=k.dtype
                )
                return final_k, final_v, 0.0, None

            if n_patch == 0:
                kept_special_idx = torch.tensor(special_indices, device=k.device, dtype=torch.long)
                expanded = kept_special_idx.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                expanded = expanded.expand(B, H, n_special, D)
                kept_k = torch.gather(candidate_k, 2, expanded)
                kept_v = torch.gather(candidate_v, 2, expanded)
                final_k = torch.cat([anchor_k, kept_k], dim=2)
                final_v = torch.cat([anchor_v, kept_v], dim=2)
                self._last_kept_attn_importance = torch.ones(
                    n_special, device=k.device, dtype=k.dtype
                )
                return final_k, final_v, 0.0, None

            # Evict only from patch tokens
            patch_indices_tensor = torch.tensor(
                patch_indices, device=k.device, dtype=torch.long,
            )
            patch_k = candidate_k[:, :, patch_indices, :].contiguous()

            if importance_scores is not None and importance_scores.numel() == n_patch:
                imp_scores = importance_scores.to(k.device).to(k.dtype)
                if imp_scores.dim() == 1:
                    imp_scores = imp_scores.unsqueeze(0).unsqueeze(0).expand(B, H, -1)
                scores = imp_scores
                avg_scores = scores.mean().item()
                num_keep_patch = min(num_to_keep_from_patches, n_patch)
                _, top_in_patch = torch.topk(scores, k=num_keep_patch, dim=-1)
            else:
                # No valid importance scores: keep the most recent patch tokens (FIFO).
                num_keep_patch = min(num_to_keep_from_patches, n_patch)
                top_in_patch = torch.arange(
                    n_patch - num_keep_patch, n_patch, device=k.device, dtype=torch.long
                ).unsqueeze(0).unsqueeze(0).expand(B, H, -1)
                avg_scores = 0.0
            top_in_patch = top_in_patch.sort(dim=-1).values
            # Map from patch-local index to original candidate index
            top_patch_orig = patch_indices_tensor[top_in_patch]

            kept_special_idx = torch.tensor(
                special_indices, device=k.device, dtype=torch.long,
            )
            # Merge special + kept patches, sort by original index to preserve causal order
            special_expanded = kept_special_idx.unsqueeze(0).unsqueeze(0).expand(
                B, H, n_special
            )
            merged_idx = torch.cat([special_expanded, top_patch_orig], dim=-1)
            merged_idx, _ = torch.sort(merged_idx, dim=-1)

            expanded_merged = merged_idx.unsqueeze(-1).expand(
                B, H, merged_idx.shape[-1], D
            )
            kept_candidate_k = torch.gather(candidate_k, 2, expanded_merged)
            kept_candidate_v = torch.gather(candidate_v, 2, expanded_merged)

            final_k = torch.cat([anchor_k, kept_candidate_k], dim=2)
            final_v = torch.cat([anchor_v, kept_candidate_v], dim=2)
            attn_imp = torch.ones(num_candidates, device=k.device, dtype=k.dtype)
            if importance_scores is not None and importance_scores.numel() == n_patch:
                attn_imp[patch_indices_tensor] = importance_scores.to(k.device).to(k.dtype)
            merged_for_attn = merged_idx[0, 0, :]
            self._last_kept_attn_importance = attn_imp[merged_for_attn]
            # Special token path: importance_scores has n_patch; no simple kept_imp mapping
            return final_k, final_v, avg_scores, None

        # Original eviction (no special token preservation)
        # When cache was evicted, importance_scores come from concat(cached_importance, new_frame)
        # so they match num_candidates. No fallback needed.
        if importance_scores is not None and importance_scores.numel() == num_candidates:
            imp_scores = importance_scores.to(k.device).to(k.dtype)
            if imp_scores.dim() == 1:
                imp_scores = imp_scores.unsqueeze(0).unsqueeze(0).expand(B, H, -1)
            scores = imp_scores
            avg_scores = scores.mean().item()
            _, top_indices = torch.topk(scores, k=num_to_keep_from_candidates, dim=-1)
        else:
            # No valid importance scores: keep the most recent candidates (FIFO).
            top_indices = torch.arange(
                num_candidates - num_to_keep_from_candidates, num_candidates,
                device=k.device, dtype=torch.long,
            ).unsqueeze(0).unsqueeze(0).expand(B, H, -1)
            avg_scores = 0.0
        top_indices = top_indices.sort(dim=-1).values

        expanded_indices = top_indices.unsqueeze(-1).expand(B, H, num_to_keep_from_candidates, D)
        kept_candidate_k = torch.gather(candidate_k, 2, expanded_indices)
        kept_candidate_v = torch.gather(candidate_v, 2, expanded_indices)

        final_k = torch.cat([anchor_k, kept_candidate_k], dim=2)
        final_v = torch.cat([anchor_v, kept_candidate_v], dim=2)

        kept_imp = None
        if importance_scores is not None and importance_scores.numel() == num_candidates:
            idx = top_indices[0, 0, :]
            kept_imp = importance_scores[idx].to(importance_scores.dtype)
            self._last_kept_attn_importance = kept_imp.to(k.device).to(k.dtype)
        return final_k, final_v, avg_scores, kept_imp

    def forward(
        self,
        x: torch.Tensor,
        pos=None,
        attn_mask=None,
        past_key_values=None,
        use_cache=False,
        cache_budget=None,
        frame_idx: Optional[int] = None,
        chunk_size: int = 1,
        timing_dict: Optional[dict] = None,
        num_special_tokens: int = 0,
        num_anchor_frames: int = 1,
        special_token_offset: int = 0,
        fixed_exempt_from_budget: bool = False,
        return_attn_weights: bool = False,
        importance_scores: Optional[torch.Tensor] = None,
        importance_cache: Optional[Dict] = None,
        use_importance_in_attn: bool = False,
        # If True with use_importance_in_attn: softmax(candidate importance) then scale by n_cand before K *= w.
        softmax_importance_before_k: bool = False,
        debug_importance_in_attn: bool = False,
        layer_idx: Optional[int] = None,
        prev_layer_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        kv_share_cfg: Optional[Dict[str, object]] = None,
        use_flex_attention: bool = False,
        flex_block_size: int = 128,
        flex_compile_mode: str = "fullgraph",
        sparse_kv_cfg: Optional[Dict[str, object]] = None,
        motivation_kv_probe: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = None
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        self._last_kept_attn_importance = None
        key_importance_on_k = None
        debug_reason = "use_importance_in_attn_disabled_or_no_cache"
        debug_num_candidates = 0
        if use_cache:
            debug_reason = "num_candidates_zero"
            if past_key_values is not None:
                past_k, past_v = past_key_values
                past_k, past_v = self._apply_kv_sharing(
                    past_k,
                    past_v,
                    layer_idx=layer_idx,
                    prev_layer_kv=prev_layer_kv,
                    kv_share_cfg=kv_share_cfg,
                    tokens_per_frame=k.shape[2],
                )
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            # [Anchor frames] Set num_anchor_tokens: first num_anchor_frames as anchor
            if self._tokens_per_frame is None:
                self._tokens_per_frame = k.shape[2]
            target_anchor = num_anchor_frames * self._tokens_per_frame
            if self.num_anchor_tokens < target_anchor:
                self.num_anchor_tokens = min(k.shape[2], target_anchor)
            if motivation_kv_probe is not None:
                ps_idx = int(motivation_kv_probe.get("patch_start_idx", 0))
                self._append_motivation_kv_probe(
                    q,
                    k,
                    past_key_values=past_key_values,
                    frame_idx=frame_idx,
                    layer_idx=layer_idx,
                    motivation_kv_probe=motivation_kv_probe,
                    patch_start_idx=ps_idx,
                )
            # [QVG delta] Evict only at chunk boundary (or when over budget for safety)
            at_chunk_boundary = chunk_size <= 1 or (
                frame_idx is not None and (frame_idx + 1) % chunk_size == 0
            )
            if cache_budget is not None and (k.shape[2] > cache_budget or at_chunk_boundary):
                if timing_dict is not None and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_ev0 = time.perf_counter() if timing_dict is not None else 0
                k, v, scores, kept_importance_scores = self.eviction(
                    k, v, cache_budget, self.num_anchor_tokens,
                    num_special_tokens=num_special_tokens,
                    tokens_per_frame=self._tokens_per_frame,
                    special_token_offset=special_token_offset,
                    fixed_exempt_from_budget=fixed_exempt_from_budget,
                    importance_scores=importance_scores,
                )
                if timing_dict is not None and torch.cuda.is_available():
                    torch.cuda.synchronize()
                if timing_dict is not None:
                    timing_dict["eviction"] += time.perf_counter() - t_ev0
                if importance_cache is not None and kept_importance_scores is not None:
                    layer_idx = importance_cache.get("_current_layer", 0)
                    importance_cache.setdefault("cached_per_layer", {})[layer_idx] = kept_importance_scores

            new_kv = (k, v)
            if use_importance_in_attn:
                num_anchor = min(self.num_anchor_tokens, k.shape[2])
                num_candidates = k.shape[2] - num_anchor
                debug_num_candidates = num_candidates
                if num_candidates > 0:
                    candidate_imp = None
                    if (
                        self._last_kept_attn_importance is not None
                        and self._last_kept_attn_importance.numel() == num_candidates
                    ):
                        candidate_imp = self._last_kept_attn_importance
                        debug_reason = "using_last_kept_attn_importance"
                    elif importance_scores is not None and importance_scores.numel() == num_candidates:
                        candidate_imp = importance_scores.to(k.device).to(k.dtype)
                        debug_reason = "using_importance_scores_direct"
                    elif importance_scores is not None:
                        frame_size = (
                            self._tokens_per_frame
                            if self._tokens_per_frame is not None
                            else max(1, num_anchor)
                        )
                        candidate_imp = self._expand_patch_importance_to_candidates(
                            importance_scores=importance_scores,
                            num_candidates=num_candidates,
                            tokens_per_frame=frame_size,
                            num_special_tokens=num_special_tokens,
                            special_token_offset=special_token_offset,
                            device=k.device,
                            dtype=k.dtype,
                        )
                        debug_reason = (
                            "expanded_patch_importance"
                            if candidate_imp is not None
                            else "expand_patch_importance_failed"
                        )
                    else:
                        debug_reason = "importance_scores_none"
                    if candidate_imp is not None and candidate_imp.numel() == num_candidates:
                        anchor_imp = torch.ones(num_anchor, device=k.device, dtype=k.dtype)
                        cand = candidate_imp.to(k.device).to(k.dtype)
                        if softmax_importance_before_k:
                            n_c = max(1, cand.numel())
                            cand = (
                                F.softmax(cand.float(), dim=0).to(cand.dtype)
                                * n_c
                            )
                            debug_reason = "applied_key_importance_on_k_softmax_n"
                        else:
                            cand = cand.clamp_min(1e-6)
                            debug_reason = "applied_key_importance_on_k"
                        # Reweight keys directly so we can still use fused attention kernels.
                        key_importance_on_k = torch.cat([anchor_imp, cand], dim=0).view(
                            1, 1, -1, 1
                        )
                    elif candidate_imp is not None:
                        debug_reason = "candidate_importance_size_mismatch"
            else:
                debug_reason = "use_importance_in_attn_false"

        if key_importance_on_k is not None:
            k = k * key_importance_on_k
        if debug_importance_in_attn and use_cache:
            if key_importance_on_k is not None:
                imp_flat = key_importance_on_k.reshape(-1)
                print(
                    (
                        "[importance_in_attn] "
                        f"frame={frame_idx} layer={layer_idx} applied=1 "
                        f"n_keys={imp_flat.numel()} n_candidates={debug_num_candidates} "
                        f"imp_min={imp_flat.min().item():.6f} "
                        f"imp_max={imp_flat.max().item():.6f} "
                        f"imp_mean={imp_flat.mean().item():.6f} "
                        f"reason={debug_reason}"
                    ),
                    flush=True,
                )
            else:
                print(
                    (
                        "[importance_in_attn] "
                        f"frame={frame_idx} layer={layer_idx} applied=0 "
                        f"n_candidates={debug_num_candidates} reason={debug_reason}"
                    ),
                    flush=True,
                )

        attn_weights = None
        if (
            use_flex_attention
            and not use_cache
            and not return_attn_weights
            and attn_mask is not None
            and FLEX_ATTENTION_AVAILABLE
        ):
            block_mask = self._get_flex_block_mask(attn_mask, block_size=flex_block_size)
            if block_mask is not None:
                x = self._run_flex_attention(
                    q,
                    k,
                    v,
                    block_mask,
                    compile_mode=flex_compile_mode,
                )
            else:
                x = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )
        elif self.fused_attn and not return_attn_weights:
            use_sparse_kv = (
                sparse_kv_cfg is not None
                and not use_cache
                and attn_mask is None
            )
            if use_sparse_kv:
                sparse_mode = str(sparse_kv_cfg.get("mode", "sparsity"))
                num_special = int(sparse_kv_cfg.get("num_special_tokens", 0))
                patch_h = int(sparse_kv_cfg.get("patch_h", 0))
                patch_w = int(sparse_kv_cfg.get("patch_w", 0))
                stride_h = int(sparse_kv_cfg.get("stride_h", 2))
                stride_w = int(sparse_kv_cfg.get("stride_w", 2))
                preserve_diagonal = bool(sparse_kv_cfg.get("preserve_diagonal", True))
                use_mean_fill = bool(sparse_kv_cfg.get("use_mean_fill", True))
                debug_sparse_stats = bool(sparse_kv_cfg.get("debug_sparse_stats", False))
                debug_layer_idx = sparse_kv_cfg.get("layer_idx", None)
                debug_print_every = int(sparse_kv_cfg.get("debug_print_every", 1))
                x = None
                if sparse_mode == "sparse_vggt":
                    if bool(sparse_kv_cfg.get("is_global_sparse")):
                        x = self._sparse_attention_sparse_vggt_style_global(
                            q,
                            k,
                            v,
                            num_frames=int(sparse_kv_cfg.get("num_frames", 0)),
                            tokens_per_frame=int(sparse_kv_cfg.get("tokens_per_frame", 0)),
                            num_special_tokens=num_special,
                            patch_h=patch_h,
                            patch_w=patch_w,
                            frame_causal=bool(sparse_kv_cfg.get("frame_causal", True)),
                            sparse_ratio=sparse_kv_cfg.get("svggt_sparse_ratio", 0.75),
                            cdf_threshold=sparse_kv_cfg.get("svggt_cdf_threshold", None),
                            topk_blocks=sparse_kv_cfg.get("svggt_topk_blocks", None),
                            pool_mode=str(sparse_kv_cfg.get("svggt_pool_mode", "avg")),
                            ks_q=int(sparse_kv_cfg.get("svggt_ks_q", 128)),
                            ks_k=int(sparse_kv_cfg.get("svggt_ks_k", 64)),
                            preserve_diagonal=preserve_diagonal,
                            use_sparge_kernel=bool(
                                sparse_kv_cfg.get("svggt_use_sparge_kernel", True)
                            ),
                            flex_block_size=flex_block_size,
                            flex_compile_mode=flex_compile_mode,
                            attn_drop_p=self.attn_drop.p if self.training else 0.0,
                        )
                    else:
                        x = self._sparse_attention_sparse_vggt_style(
                            q,
                            k,
                            v,
                            num_special_tokens=num_special,
                            patch_h=patch_h,
                            patch_w=patch_w,
                            sparse_ratio=sparse_kv_cfg.get("svggt_sparse_ratio", 0.75),
                            cdf_threshold=sparse_kv_cfg.get("svggt_cdf_threshold", None),
                            topk_blocks=sparse_kv_cfg.get("svggt_topk_blocks", None),
                            pool_mode=str(sparse_kv_cfg.get("svggt_pool_mode", "avg")),
                            ks_q=int(sparse_kv_cfg.get("svggt_ks_q", 128)),
                            ks_k=int(sparse_kv_cfg.get("svggt_ks_k", 64)),
                            preserve_diagonal=preserve_diagonal,
                            use_sparge_kernel=bool(
                                sparse_kv_cfg.get("svggt_use_sparge_kernel", True)
                            ),
                            flex_block_size=flex_block_size,
                            flex_compile_mode=flex_compile_mode,
                            attn_drop_p=self.attn_drop.p if self.training else 0.0,
                        )
                else:
                    if FLEX_ATTENTION_AVAILABLE:
                        x = self._sparse_attention_with_flex_meanfill(
                            q,
                            k,
                            v,
                            num_special_tokens=num_special,
                            patch_h=patch_h,
                            patch_w=patch_w,
                            stride_h=stride_h,
                            stride_w=stride_w,
                            preserve_diagonal=preserve_diagonal,
                            use_mean_fill=use_mean_fill,
                            flex_block_size=flex_block_size,
                            flex_compile_mode=flex_compile_mode,
                            debug_sparse_stats=debug_sparse_stats,
                            debug_layer_idx=debug_layer_idx,
                            debug_print_every=debug_print_every,
                        )
                    if x is None:
                        # Fallback path when FlexAttention is unavailable.
                        x = self._sparse_attention_with_diag_and_meanfill(
                            q,
                            k,
                            v,
                            num_special_tokens=num_special,
                            patch_h=patch_h,
                            patch_w=patch_w,
                            stride_h=stride_h,
                            stride_w=stride_w,
                            preserve_diagonal=preserve_diagonal,
                            use_mean_fill=use_mean_fill,
                            attn_drop_p=self.attn_drop.p if self.training else 0.0,
                        )
                if x is None:
                    x = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=attn_mask,
                        dropout_p=self.attn_drop.p if self.training else 0.0,
                    )
            else:
                x = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )
        else:
            # Use non-fused path when return_attn_weights or fused_attn=False
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            # Mask
            if attn_mask is not None:
                assert attn_mask.shape[-2:] == (N, N), f"Expected mask shape [..., {N}, {N}], got {attn_mask.shape}"
                attn = attn + attn_mask

            attn = attn.softmax(dim=-1)
            attn_weights = attn if return_attn_weights else None
            attn = self.attn_drop(attn)

            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if use_cache:
            if return_attn_weights and attn_weights is not None:
                return x, new_kv, scores, attn_weights
            return x, new_kv, scores
        if return_attn_weights and attn_weights is not None:
            return x, attn_weights
        return x


class MemEffAttention(Attention):
    def forward(
        self,
        x: Tensor,
        attn_bias=None,
        pos=None,
        attn_mask=None,
        sparse_kv_cfg: Optional[Dict[str, object]] = None,
        **kwargs,
    ) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(
                x,
                pos=pos,
                attn_mask=attn_mask,
                sparse_kv_cfg=sparse_kv_cfg,
            )

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        return x