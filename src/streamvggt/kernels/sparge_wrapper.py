import math
from collections import namedtuple
from typing import Optional

import torch
import spas_sage_attn._qattn as qattn
from spas_sage_attn.quant_per_block import per_block_int8
from spas_sage_attn.utils import block_map_lut_triton, fill_block_map_triton, hyperparameter_check

SortResult = namedtuple("SortResult", ["values", "indices"])


def _int32_idx(sort_result):
    return SortResult(
        sort_result.values,
        sort_result.indices.to(torch.int32),
    )


def _mem_eff_sort(t, chunks=4, dim=1):
    sorted_chunks = [
        _int32_idx(torch.sort(tt, dim=-1, descending=True))
        for tt in torch.chunk(t, chunks, dim=dim)
    ]
    values = torch.cat([s.values for s in sorted_chunks], dim=dim)
    indices = torch.cat([s.indices for s in sorted_chunks], dim=dim)
    return SortResult(values, indices)


def _check_sparse_mode(topk: Optional[int], sparse_ratio: Optional[float], cdf_threshold: Optional[float]):
    use_topk = topk is not None
    use_ratio = sparse_ratio is not None
    use_cdf = cdf_threshold is not None
    only_use_topk = use_topk and (not use_ratio) and (not use_cdf)
    only_use_ratio = (not use_topk) and use_ratio and (not use_cdf)
    only_use_cdf = (not use_topk) and (not use_ratio) and use_cdf
    use_ratio_and_cdf = use_ratio and use_cdf and (not use_topk)
    assert (
        only_use_topk + only_use_ratio + only_use_cdf + use_ratio_and_cdf == 1
    ), f"Current: {topk=}, {sparse_ratio=}, {cdf_threshold=}"


def get_block_mask(
    pooled_score: torch.Tensor,
    sink_blocks: int,
    topk: Optional[int] = None,
    sparse_ratio: Optional[float] = None,
    cdf_threshold: Optional[float] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    _check_sparse_mode(topk, sparse_ratio, cdf_threshold)
    B, nh, q_blk, k_blk = pooled_score.shape
    assert 0 <= sink_blocks <= k_blk

    if sparse_ratio is not None:
        assert 0 <= sparse_ratio <= 1
        topk = int(k_blk * (1 - sparse_ratio))
    if topk is not None:
        assert 0 <= topk <= k_blk
    if cdf_threshold is not None:
        assert 0 <= cdf_threshold <= 1

    if pooled_score.numel() < 2e8:
        sorted_score = torch.sort(pooled_score, dim=-1, descending=True)
    else:
        sorted_score = _mem_eff_sort(pooled_score)

    num_to_select = None
    if cdf_threshold is not None:
        cdf = torch.cumsum(sorted_score.values, dim=-1)
        cdf_thresh = hyperparameter_check(cdf_threshold, nh, pooled_score.device)
        cdf_thresh = cdf_thresh.view(1, nh, 1, 1) + eps
        cdf_thresh = cdf_thresh.expand(B, -1, q_blk, 1).contiguous()
        num_to_select = torch.searchsorted(cdf, cdf_thresh, right=True).squeeze(-1)

    if topk is not None:
        if num_to_select is None:
            num_to_select = torch.full((B, nh, q_blk), topk, device=pooled_score.device)
        else:
            num_to_select = torch.clamp(num_to_select, min=topk)

    final_map = torch.zeros_like(pooled_score, dtype=torch.bool)
    final_map = fill_block_map_triton(final_map, num_to_select, sorted_score.indices)

    if sink_blocks > 0:
        ones_shape = list(final_map.shape)
        ones_shape[-1] = sink_blocks
        trailing_ones = torch.ones(ones_shape, device=final_map.device, dtype=torch.bool)
        final_map = torch.cat([final_map, trailing_ones], dim=-1)
    return final_map


def block_sparse_attn_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    pooled_score: torch.Tensor,
    topk: Optional[int] = None,
    sparse_ratio: Optional[float] = None,
    cdf_threshold: Optional[float] = None,
    return_sparsity: bool = False,
    dtype: torch.dtype = torch.float16,
):
    out_dtype = query.dtype
    _is_causal = 0
    KBLK = 64
    pvthreshd = 1e10
    pvthreshd = hyperparameter_check(pvthreshd, query.size(-3), query.device)

    Tk = key.shape[-2]
    orig_Kblk = pooled_score.shape[-1]
    total_Kblk = math.ceil(Tk / KBLK)
    sink_blocks = total_Kblk - orig_Kblk
    final_map = get_block_mask(
        pooled_score,
        sink_blocks=sink_blocks,
        topk=topk,
        sparse_ratio=sparse_ratio,
        cdf_threshold=cdf_threshold,
    )
    lut, valid_block_num = block_map_lut_triton(final_map)

    query, key, value = (
        query.contiguous().to(dtype),
        key.contiguous().to(dtype),
        value.contiguous().to(dtype),
    )

    km = key.mean(dim=-2, keepdim=True)
    q_int8, q_scale, k_int8, k_scale = per_block_int8(query, key - km)
    q_scale = q_scale.squeeze(-1)
    k_scale = k_scale.squeeze(-1)

    hd = query.shape[-1]
    scale = 1.0 / (hd**0.5)
    out = torch.empty_like(query)
    qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
        q_int8,
        k_int8,
        value,
        out,
        lut,
        valid_block_num,
        pvthreshd,
        q_scale,
        k_scale,
        1,
        _is_causal,
        1,
        scale,
        0,
    )
    out = out.to(out_dtype)
    if return_sparsity:
        sparsity = 1 - final_map.float().mean()
        return out, sparsity
    return out
