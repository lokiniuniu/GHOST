import time
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from streamvggt.models.aggregator import Aggregator
from streamvggt.heads.camera_head import CameraHead
from streamvggt.heads.dpt_head import DPTHead
from streamvggt.heads.track_head import TrackHead
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple, List, Any, Callable, Dict
from dataclasses import dataclass

# [QVG] KV Cache Quantisation – import manager and config
try:
    from streamvggt.quantization import (
        QVGConfig,
        QVGInfiniteVGGTKVCacheManager,
        DiversityQuantConfig,
        DiversityKVCacheManager,
    )
    _QVG_AVAILABLE = True
except ImportError:
    _QVG_AVAILABLE = False
    DiversityQuantConfig = None
    DiversityKVCacheManager = None

IMPORTANCE_EVICTION_MODES = frozenset({"importance"})

def _get_kv_cache_size_bytes(past_key_values_list) -> int:
    """Compute total memory (bytes) of KV cache from a list of (k, v) tuples."""
    total = 0
    if past_key_values_list is None:
        return 0
    for item in past_key_values_list:
        if item is not None:
            k, v = item
            if isinstance(k, torch.Tensor):
                total += k.numel() * k.element_size()
            if isinstance(v, torch.Tensor):
                total += v.numel() * v.element_size()
    return total


@dataclass
class StreamVGGTOutput(ModelOutput):
    ress: Optional[List[dict]] = None
    views: Optional[torch.Tensor] = None
    kv_cache_mem_bytes: Optional[int] = None
    sequence_state: Optional[Dict[str, Any]] = None

class StreamVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, total_budget=1200000):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
        self.total_budget = total_budget
    


    def forward(
        self,
        views,
        query_points: torch.Tensor = None,
        history_info: Optional[dict] = None,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0
    ):
        images = torch.stack(
            [view["img"] for view in views], dim=0
        ).permute(1, 0, 2, 3, 4)    # B S C H W

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        if history_info is None:
            history_info = {"token": None}

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.track_head is not None and query_points is not None:
                track_list, vis, conf = self.track_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                predictions["track"] = track_list[-1]  # track of the last iteration
                predictions["vis"] = vis
                predictions["conf"] = conf
            predictions["images"] = images

            B, S = images.shape[:2]
            ress = []
            for s in range(S):
                res = {
                    'pts3d_in_other_view': predictions['world_points'][:, s],  # [B, H, W, 3]
                    'conf': predictions['world_points_conf'][:, s],  # [B, H, W]

                    'depth': predictions['depth'][:, s],  # [B, H, W, 1]
                    'depth_conf': predictions['depth_conf'][:, s],  # [B, H, W]
                    'camera_pose': predictions['pose_enc'][:, s, :],  # [B, 9]

                    **({'valid_mask': views[s]["valid_mask"]}
                    if 'valid_mask' in views[s] else {}),  # [B, H, W]

                    **({'track': predictions['track'][:, s],  # [B, N, 2]
                        'vis': predictions['vis'][:, s],  # [B, N]
                        'track_conf': predictions['conf'][:, s]}
                    if 'track' in predictions else {})
                }
                ress.append(res)
            return StreamVGGTOutput(ress=ress, views=views)  # [S] [B, C, H, W]
    
    def inference(
        self, 
        frames, 
        query_points: torch.Tensor = None, 
        past_key_values=None, 
        frame_writer: Optional[Callable[[int, dict, dict], None]] = None,
        cache_results: bool = True,
        total_budget=None,
        # [QVG] Quantisation options – pass a QVGConfig or preset name string
        qvg_config: Optional[Any] = None,
        # [Diversity] Diversity-aware quant: first frame full prec, K channel-wise, V token-wise, top 10-20% exempt
        diversity_quant_config: Optional[Any] = None,
        # [Importance eviction] Use importance-based eviction (camera/geometry/temporal + token saliency/conf)
        eviction_mode: str = "importance",
        # [QVG] Debug: log detailed memory breakdown each frame (for peak analysis)
        debug_mem: bool = False,
        # Profile inference time: per-step breakdown (aggregator, heads, etc.)
        debug_time: bool = False,
        # KV cache mode: anchor1_camera, anchor1_register, anchor3_camera_register, anchor3_camera_register_exempt
        kv_mode: Optional[str] = None,
        # Precomputed budget proportions from compute_kv_budget_from_cosine_sim.py
        budget_proportions_path: Optional[str] = None,
        # Importance eviction: dict with w_camera, w_geometry, w_temporal, w_saliency, w_depth_conf, w_pts_conf, w_frame, w_token
        importance_weights: Optional[Dict[str, float]] = None,
        # Keep default importance eviction, but reweight K by importance before attention.
        use_importance_in_attn: bool = False,
        # If True with use_importance_in_attn: softmax(candidate imp) * n_cand, then multiply K.
        softmax_importance_before_k: bool = False,
        # Debug logging for importance-in-attention at attention layer granularity.
        debug_importance_in_attn: bool = False,
        # Profile: print raw importance stats and sigmoid coefficients (for subset debug)
        profile_importance_raw: bool = False,
        profile_min_frames: Optional[int] = None,
        # Experimental KV sharing across adjacent layers.
        kv_share_cfg: Optional[Dict[str, Any]] = None,
        # Optional frame-attention sparse config.
        frame_sparse_cfg: Optional[Dict[str, Any]] = None,
        # Use FlexAttention for frame-attention with explicit masks.
        use_flex_attention: bool = False,
        flex_block_size: int = 128,
        flex_compile_mode: str = "fullgraph",
        sequence_state: Optional[Dict[str, Any]] = None,
        return_sequence_state: bool = False,
        motivation_dump_path: Optional[str] = None,
        motivation_probe_layer: int = -1,
    ):
        if sequence_state is None:
            sequence_state = {}

        resumed = bool(sequence_state.get("initialized", False))
        state_past_key_values = sequence_state.get("past_key_values")
        state_past_key_values_camera = sequence_state.get("past_key_values_camera")
        state_importance_cache = sequence_state.get("importance_cache")
        state_frame_metadata = sequence_state.get("frame_metadata")
        frame_offset = int(sequence_state.get("frame_offset", 0))

        # [QVG] Build manager when quantisation is requested
        qvg_manager = None
        if diversity_quant_config is not None and DiversityKVCacheManager is not None:
            if isinstance(diversity_quant_config, str):
                cfg_fn = getattr(DiversityQuantConfig, diversity_quant_config, None)
                diversity_quant_config = cfg_fn() if cfg_fn is not None else DiversityQuantConfig()
            if diversity_quant_config.enabled:
                qvg_manager = DiversityKVCacheManager(
                    config=diversity_quant_config,
                    num_layers=self.aggregator.depth,
                )
                if past_key_values is None:
                    past_key_values = [None] * self.aggregator.depth
            else:
                if past_key_values is None:
                    past_key_values = [None] * self.aggregator.depth
        elif qvg_config is not None and _QVG_AVAILABLE:
            if isinstance(qvg_config, str):
                cfg_fn = getattr(QVGConfig, qvg_config, None)
                qvg_config = cfg_fn() if cfg_fn is not None else QVGConfig()
            if qvg_config.enabled:
                qvg_manager = QVGInfiniteVGGTKVCacheManager(
                    config=qvg_config,
                    num_layers=self.aggregator.depth,
                    num_special_tokens=self.aggregator.patch_start_idx,
                )
                if past_key_values is None:
                    past_key_values = [None] * self.aggregator.depth
            else:
                if past_key_values is None:
                    past_key_values = [None] * self.aggregator.depth
        else:
            if past_key_values is None:
                past_key_values = [None] * self.aggregator.depth

        if state_past_key_values is not None:
            past_key_values = state_past_key_values
        if state_past_key_values_camera is not None:
            past_key_values_camera = state_past_key_values_camera
        else:
            past_key_values_camera = [None] * self.camera_head.trunk_depth
        total_budget = self.total_budget

        # Reset attention cache state only for a new sequence.
        if not resumed:
            for block in self.aggregator.global_blocks:
                if hasattr(block.attn, "_reset_cache_state"):
                    block.attn._reset_cache_state()
        
        all_ress = []
        processed_frames = []
        peak_kv_cache_bytes = 0
        _debug_mem_prev_peak = 0.0  # for debug_mem: log when peak increases

        # [debug_time] Accumulate seconds per step across all frames
        _time_agg = 0.0
        _time_camera = 0.0
        _time_depth = 0.0
        _time_point = 0.0
        _time_track = 0.0
        _time_other = 0.0
        # [debug_time] QVG-specific: decode, encode (incl. kmeans), eviction
        _timing_dict = {"decode": 0.0, "encode": 0.0, "eviction": 0.0} if debug_time else None
        _time_decode = _time_encode = _time_eviction = 0.0  # accumulate across all frames for final summary
        _time_decode_unpack = _time_decode_dequant = _time_encode_sas = _time_encode_quantize = 0.0
        _time_decode_dequant_triton = _time_decode_dequant_sym = _time_decode_centroid_add = 0.0
        _time_encode_kmeans = _time_encode_centroid_sub = 0.0
        # [debug_time] Per-chunk accumulation for ratio (chunk_size from QVG or default 8)
        _chunk_size = 8
        if qvg_manager is not None and qvg_manager.config.enabled:
            _chunk_size = qvg_manager.config.chunk_size
        _chunk_decode, _chunk_encode, _chunk_eviction, _chunk_agg = 0.0, 0.0, 0.0, 0.0

        if debug_mem:
            print("[MEM] Debug memory logging enabled (frames 0,1,7,8,15, every 50, on peak increase, last)", flush=True)

        if eviction_mode in IMPORTANCE_EVICTION_MODES:
            importance_cache = state_importance_cache if state_importance_cache is not None else {}
        else:
            importance_cache = None
        if importance_cache is not None and profile_importance_raw:
            importance_cache["_profile_raw"] = True
            if profile_min_frames is not None:
                importance_cache["_profile_min_frames"] = int(profile_min_frames)

        motivation_kv_probe: Optional[Dict[str, Any]] = None
        if motivation_dump_path:
            L = self.aggregator.depth
            li = motivation_probe_layer if motivation_probe_layer >= 0 else L - 1
            motivation_kv_probe = sequence_state.setdefault(
                "motivation_kv_probe",
                {
                    "enabled": True,
                    "layer_idx": int(li),
                    "patch_start_idx": int(self.aggregator.patch_start_idx),
                    "records": [],
                },
            )
            motivation_kv_probe["enabled"] = True
            motivation_kv_probe["layer_idx"] = int(li)
            motivation_kv_probe["patch_start_idx"] = int(self.aggregator.patch_start_idx)
            if not resumed:
                motivation_kv_probe["records"] = []

        prev_frame_metadata = state_frame_metadata if isinstance(state_frame_metadata, list) else []

        # Load precomputed budget proportions if provided
        if budget_proportions_path is not None:
            import json
            with open(budget_proportions_path) as f:
                cfg = json.load(f)
            proportions = torch.tensor(cfg["proportions"], dtype=torch.float32)
            self.aggregator.budget_proportions = proportions
        else:
            self.aggregator.budget_proportions = None

        from tqdm import tqdm
        for i, frame in tqdm(enumerate(frames), total=len(frames), desc="Frames", unit="frame", position=1, leave=False):

            images = frame["img"].unsqueeze(0)
            if debug_time:
                _timing_dict["decode"] = 0.0
                _timing_dict["encode"] = 0.0
                _timing_dict["eviction"] = 0.0
                if i % _chunk_size == 0:
                    _timing_dict["decode_unpack"] = 0.0
                    _timing_dict["decode_dequant"] = 0.0
                    _timing_dict["decode_dequant_triton"] = 0.0
                    _timing_dict["decode_dequant_sym"] = 0.0
                    _timing_dict["decode_centroid_add"] = 0.0
                    _timing_dict["encode_sas"] = 0.0
                    _timing_dict["encode_kmeans"] = 0.0
                    _timing_dict["encode_centroid_sub"] = 0.0
                    _timing_dict["encode_quantize"] = 0.0
            if debug_time and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            if all_ress:
                frame_metadata = [
                    {"camera_pose": r["camera_pose"], "depth": r["depth"],
                     "depth_conf": r["depth_conf"], "conf": r.get("conf")}
                    for r in all_ress
                ]
            else:
                frame_metadata = []
            if prev_frame_metadata:
                frame_metadata = prev_frame_metadata + frame_metadata

            aggregator_output = self.aggregator(
                images,
                past_key_values=past_key_values,
                use_cache=True,
                past_frame_idx=frame_offset + i,
                total_budget=total_budget,
                qvg_manager=qvg_manager,   # [QVG] pass manager (None = disabled)
                timing_dict=_timing_dict if debug_time else None,
                kv_mode=kv_mode,
                frame_metadata=frame_metadata,
                eviction_mode=eviction_mode,
                importance_cache=importance_cache,
                importance_weights=importance_weights,
                use_importance_in_attn=use_importance_in_attn,
                softmax_importance_before_k=softmax_importance_before_k,
                debug_importance_in_attn=debug_importance_in_attn,
                kv_share_cfg=kv_share_cfg,
                frame_sparse_cfg=frame_sparse_cfg,
                use_flex_attention=use_flex_attention,
                flex_block_size=flex_block_size,
                flex_compile_mode=flex_compile_mode,
                motivation_kv_probe=motivation_kv_probe,
            )

            if debug_time and torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            if debug_time:
                agg_sec = t1 - t0
                _time_agg += agg_sec
                d, e, v = _timing_dict["decode"], _timing_dict["encode"], _timing_dict["eviction"]
                _time_decode += d
                _time_encode += e
                _time_eviction += v
                # Per-frame timing
                print(
                    f"[TIME] frame={i:4d}  decode={d*1000:6.1f}ms  encode={e*1000:6.1f}ms  eviction={v*1000:6.1f}ms  "
                    f"agg_total={agg_sec*1000:6.1f}ms",
                    flush=True,
                )
                # Per-chunk accumulation
                _chunk_decode += d
                _chunk_encode += e
                _chunk_eviction += v
                _chunk_agg += agg_sec
                # At chunk boundary: print ratio for this chunk (decode, encode, eviction separately)
                if (i + 1) % _chunk_size == 0:
                    chunk_id = (i + 1) // _chunk_size - 1
                    start_f, end_f = chunk_id * _chunk_size, (chunk_id + 1) * _chunk_size - 1
                    r_d = 100.0 * _chunk_decode / _chunk_agg if _chunk_agg > 0 else 0.0
                    r_e = 100.0 * _chunk_encode / _chunk_agg if _chunk_agg > 0 else 0.0
                    r_v = 100.0 * _chunk_eviction / _chunk_agg if _chunk_agg > 0 else 0.0
                    r_extra = r_d + r_e + r_v
                    print(
                        f"[CHUNK] chunk {chunk_id} (frame {start_f}-{end_f})  agg={_chunk_agg*1000:.0f}ms  "
                        f"decode={r_d:.1f}%  encode={r_e:.1f}%  eviction={r_v:.1f}%  extra_total={r_extra:.1f}%",
                        flush=True,
                    )
                    # Sub-step breakdown for this chunk
                    du = _timing_dict.get("decode_unpack", 0.0)
                    dd_t = _timing_dict.get("decode_dequant_triton", 0.0)
                    dd_s = _timing_dict.get("decode_dequant_sym", 0.0)
                    dc = _timing_dict.get("decode_centroid_add", 0.0)
                    ek = _timing_dict.get("encode_kmeans", 0.0)
                    ec = _timing_dict.get("encode_centroid_sub", 0.0)
                    es = _timing_dict.get("encode_sas", 0.0)
                    eq = _timing_dict.get("encode_quantize", 0.0)
                    _time_decode_unpack += du
                    _time_decode_dequant += dd_t + dd_s + dc
                    _time_decode_dequant_triton += dd_t
                    _time_decode_dequant_sym += dd_s
                    _time_decode_centroid_add += dc
                    _time_encode_sas += es
                    _time_encode_kmeans += ek
                    _time_encode_centroid_sub += ec
                    _time_encode_quantize += eq
                    dd_total = dd_t + dd_s + dc
                    if du + dd_total > 0:
                        print(f"        [decode] dequant+centroid (反量化+质心):", flush=True)
                        pct_unpack = 100 * du / (du + dd_total)
                        print(f"          unpack (bit解包):     {du*1000:.1f}ms ({pct_unpack:.1f}%)", flush=True)
                        if dd_t > 0:
                            pct_t = 100 * dd_t / dd_total
                            print(f"          dequant_triton:    {dd_t*1000:.1f}ms ({pct_t:.1f}% of dequant+centroid)", flush=True)
                        if dd_s > 0:
                            pct_s = 100 * dd_s / dd_total
                            print(f"          dequant_sym:       {dd_s*1000:.1f}ms ({pct_s:.1f}% of dequant+centroid)", flush=True)
                        if dc > 0:
                            pct_c = 100 * dc / dd_total
                            print(f"          centroid_add:      {dc*1000:.1f}ms ({pct_c:.1f}% of dequant+centroid)", flush=True)
                    if es + eq > 0:
                        print(f"        [encode] sas (kmeans+质心减):", flush=True)
                        if ek + ec > 0:
                            pct_k = 100 * ek / (ek + ec)
                            pct_c = 100 * ec / (ek + ec)
                            print(f"          kmeans:            {ek*1000:.1f}ms ({pct_k:.1f}% of sas)", flush=True)
                            print(f"          centroid_sub:      {ec*1000:.1f}ms ({pct_c:.1f}% of sas)", flush=True)
                        pct_q = 100 * eq / (es + eq)
                        print(f"          quantize:          {eq*1000:.1f}ms ({pct_q:.1f}% of encode)", flush=True)
                    _chunk_decode = _chunk_encode = _chunk_eviction = _chunk_agg = 0.0

            if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                aggregated_tokens, patch_start_idx, past_key_values = aggregator_output
            else:
                aggregated_tokens, patch_start_idx = aggregator_output

            with torch.cuda.amp.autocast(enabled=False):
                if self.camera_head is not None:
                    if debug_time and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_cam0 = time.perf_counter()
                    pose_enc, past_key_values_camera = self.camera_head(aggregated_tokens, past_key_values_camera=past_key_values_camera, use_cache=True)
                    pose_enc = pose_enc[-1]
                    camera_pose = pose_enc[:, 0, :]
                    if debug_time and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _time_camera += time.perf_counter() - t_cam0

                if self.depth_head is not None:
                    if debug_time and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_dep0 = time.perf_counter()
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    depth = depth[:, 0]
                    depth_conf = depth_conf[:, 0]
                    if debug_time and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _time_depth += time.perf_counter() - t_dep0

                if self.point_head is not None:
                    if debug_time and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_pt0 = time.perf_counter()
                    pts3d, pts3d_conf = self.point_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    pts3d = pts3d[:, 0]
                    pts3d_conf = pts3d_conf[:, 0]
                    if debug_time and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _time_point += time.perf_counter() - t_pt0

                if self.track_head is not None and query_points is not None:
                    if debug_time and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_tr0 = time.perf_counter()
                    track_list, vis, conf = self.track_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                    )
                    track = track_list[-1][:, 0]
                    query_points = track
                    vis = vis[:, 0]
                    track_conf = conf[:, 0]
                    if debug_time and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _time_track += time.perf_counter() - t_tr0

            if debug_time and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_oth0 = time.perf_counter()

            res_gpu = {
                "pts3d_in_other_view": pts3d,
                "conf": pts3d_conf,
                "depth": depth,
                "depth_conf": depth_conf,
                "camera_pose": camera_pose,
                **({"valid_mask": frame["valid_mask"]} if "valid_mask" in frame else {}),
                **(
                    {"track": track, "vis": vis, "track_conf": track_conf}
                    if query_points is not None
                    else {}
                ),
            }
            res_cpu = {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in res_gpu.items()
            }
            if frame_writer is not None:
                frame_writer(i, frame, res_cpu)

            if cache_results:
                all_ress.append(res_cpu)
                processed_frames.append(
                    {nk: nv.detach().cpu() if isinstance(nv, torch.Tensor) else nv for nk, nv in frame.items()}
                )

            # [QVG] Track peak KV cache memory
            if qvg_manager is not None:
                # Quantised memory from manager + camera head (unquantised)
                kv_cache_bytes = qvg_manager.total_nbytes() + _get_kv_cache_size_bytes(past_key_values_camera)
            else:
                kv_cache_bytes = _get_kv_cache_size_bytes(past_key_values) + _get_kv_cache_size_bytes(past_key_values_camera)
            peak_kv_cache_bytes = max(peak_kv_cache_bytes, kv_cache_bytes)

            # [QVG] Debug memory: log detailed breakdown at key frames or when peak increases
            if debug_mem and torch.cuda.is_available():
                alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                camera_bytes = _get_kv_cache_size_bytes(past_key_values_camera)
                is_last = i == len(frames) - 1
                log_frame = (
                    i in (0, 1, 7, 8, 15) or
                    (i > 0 and i % 50 == 0) or
                    (peak_gb > _debug_mem_prev_peak) or
                    is_last
                )
                if log_frame:
                    _debug_mem_prev_peak = peak_gb
                    if qvg_manager is not None:
                        mb = qvg_manager.memory_breakdown()
                        qvg_q = mb["qvg_quantized_bytes"] / (1024 ** 2)
                        qvg_b = mb["qvg_bf16_buffer_bytes"] / (1024 ** 2)
                        cam_mb = camera_bytes / (1024 ** 2)
                        print(
                            f"[MEM] frame={i:4d} "
                            f"alloc={alloc_gb:.3f}GB peak={peak_gb:.3f}GB | "
                            f"QVG_quant={qvg_q:.1f}MB QVG_bf16buf={qvg_b:.1f}MB "
                            f"camera={cam_mb:.1f}MB",
                            flush=True,
                        )
                    else:
                        agg_bytes = _get_kv_cache_size_bytes(past_key_values)
                        print(
                            f"[MEM] frame={i:4d} "
                            f"alloc={alloc_gb:.3f}GB peak={peak_gb:.3f}GB | "
                            f"aggregator={agg_bytes/(1024**2):.1f}MB camera={camera_bytes/(1024**2):.1f}MB",
                            flush=True,
                        )

            del res_gpu
            torch.cuda.empty_cache()
            if debug_time and torch.cuda.is_available():
                torch.cuda.synchronize()
            if debug_time:
                _time_other += time.perf_counter() - t_oth0

        # [debug_time] Print last partial chunk if any
        if debug_time and _chunk_agg > 0:
            nf = len(frames)
            start_f = (nf // _chunk_size) * _chunk_size
            r_d = 100.0 * _chunk_decode / _chunk_agg if _chunk_agg > 0 else 0.0
            r_e = 100.0 * _chunk_encode / _chunk_agg if _chunk_agg > 0 else 0.0
            r_v = 100.0 * _chunk_eviction / _chunk_agg if _chunk_agg > 0 else 0.0
            r_extra = r_d + r_e + r_v
            print(
                f"[CHUNK] chunk {nf // _chunk_size} (frame {start_f}-{nf-1}, partial)  agg={_chunk_agg*1000:.0f}ms  "
                f"decode={r_d:.1f}%  encode={r_e:.1f}%  eviction={r_v:.1f}%  extra_total={r_extra:.1f}%",
                flush=True,
            )
            du = _timing_dict.get("decode_unpack", 0.0)
            dd_t = _timing_dict.get("decode_dequant_triton", 0.0)
            dd_s = _timing_dict.get("decode_dequant_sym", 0.0)
            dc = _timing_dict.get("decode_centroid_add", 0.0)
            ek = _timing_dict.get("encode_kmeans", 0.0)
            ec = _timing_dict.get("encode_centroid_sub", 0.0)
            es = _timing_dict.get("encode_sas", 0.0)
            eq = _timing_dict.get("encode_quantize", 0.0)
            _time_decode_unpack += du
            _time_decode_dequant += dd_t + dd_s + dc
            _time_decode_dequant_triton += dd_t
            _time_decode_dequant_sym += dd_s
            _time_decode_centroid_add += dc
            _time_encode_sas += es
            _time_encode_kmeans += ek
            _time_encode_centroid_sub += ec
            _time_encode_quantize += eq
            dd_total = dd_t + dd_s + dc
            if du + dd_total > 0:
                print(f"        decode: unpack={100*du/(du+dd_total):.1f}%", flush=True)
                if dd_t > 0:
                    print(f"          dequant+centroid: triton_fused={100*dd_t/dd_total:.1f}%", flush=True)
                if dd_s + dc > 0:
                    print(f"          dequant_sym={100*dd_s/dd_total:.1f}%  centroid_add={100*dc/dd_total:.1f}%", flush=True)
            if es + eq > 0:
                print(f"        encode: sas={100*es/(es+eq):.1f}%  quantize={100*eq/(es+eq):.1f}%", flush=True)
                if ek + ec > 0:
                    print(f"          sas: kmeans={100*ek/(ek+ec):.1f}%  centroid_sub={100*ec/(ek+ec):.1f}%", flush=True)

        # [debug_time] Print per-step time breakdown
        if debug_time:
            total = _time_agg + _time_camera + _time_depth + _time_point + _time_track + _time_other
            nf = len(frames)
            print("\n" + "─" * 60, flush=True)
            print("  [TIME] Inference time breakdown (seconds, % of total)", flush=True)
            print("─" * 60, flush=True)
            for name, sec in [
                ("aggregator", _time_agg),
                ("camera_head", _time_camera),
                ("depth_head", _time_depth),
                ("point_head", _time_point),
                ("track_head", _time_track),
                ("other (copy/cache)", _time_other),
            ]:
                pct = 100.0 * sec / total if total > 0 else 0.0
                ms_per_frame = 1000.0 * sec / nf if nf > 0 else 0.0
                print(f"  {name:20s} : {sec:8.3f}s ({pct:5.1f}%)  →  {ms_per_frame:.1f} ms/frame", flush=True)
            if _timing_dict and (qvg_manager is not None and qvg_manager.config.enabled):
                print("  " + "─" * 56, flush=True)
                print("  [TIME] QVG breakdown (inside aggregator, accumulated over all frames):", flush=True)
                for name, sec in [
                    ("  decode (反量化)", _time_decode),
                    ("  encode (量化+kmeans)", _time_encode),
                    ("  eviction (TopK淘汰)", _time_eviction),
                ]:
                    pct = 100.0 * sec / total if total > 0 else 0.0
                    pct_agg = 100.0 * sec / _time_agg if _time_agg > 0 else 0.0
                    ms = 1000.0 * sec / nf if nf > 0 else 0.0
                    print(f"  {name:25s} : {sec:8.3f}s ({pct:5.1f}% total, {pct_agg:5.1f}% of agg)  →  {ms:.1f} ms/frame", flush=True)
                # Sub-step breakdown for decode and encode (accumulated over all chunks)
                d_unpack = _time_decode_unpack
                d_dequant = _time_decode_dequant
                d_triton = _time_decode_dequant_triton
                d_sym = _time_decode_dequant_sym
                d_cent = _time_decode_centroid_add
                e_sas = _time_encode_sas
                e_kmeans = _time_encode_kmeans
                e_cent_sub = _time_encode_centroid_sub
                e_quant = _time_encode_quantize
                if d_unpack + d_dequant > 0 or e_sas + e_quant > 0:
                    print("  " + "─" * 56, flush=True)
                    print("  [TIME] Decode sub-steps:", flush=True)
                    for name, sec in [
                        ("    unpack (bit解包)", d_unpack),
                        ("    dequant_triton (Triton融合)", d_triton),
                        ("    dequant_sym (对称反量化)", d_sym),
                        ("    centroid_add (质心加回)", d_cent),
                    ]:
                        if sec > 0:
                            pct_d = 100.0 * sec / (d_unpack + d_dequant) if (d_unpack + d_dequant) > 0 else 0.0
                            ms = 1000.0 * sec / nf if nf > 0 else 0.0
                            print(f"  {name:35s} : {sec:8.3f}s ({pct_d:5.1f}% of decode)  →  {ms:.1f} ms/frame", flush=True)
                    print("  [TIME] Encode sub-steps:", flush=True)
                    for name, sec in [
                        ("    kmeans (K均值聚类)", e_kmeans),
                        ("    centroid_sub (质心相减)", e_cent_sub),
                        ("    quantize (对称量化)", e_quant),
                    ]:
                        if sec > 0:
                            pct_e = 100.0 * sec / (e_sas + e_quant) if (e_sas + e_quant) > 0 else 0.0
                            ms = 1000.0 * sec / nf if nf > 0 else 0.0
                            print(f"  {name:35s} : {sec:8.3f}s ({pct_e:5.1f}% of encode)  →  {ms:.1f} ms/frame", flush=True)
            print("─" * 60, flush=True)
            print(f"  TOTAL : {total:.3f}s  ({nf} frames)  →  {1000.0*total/nf:.1f} ms/frame", flush=True)
            print("─" * 60 + "\n", flush=True)

        if motivation_dump_path:
            from pathlib import Path as _Path

            dump_path = _Path(motivation_dump_path)
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            probe_records = (
                (motivation_kv_probe or {}).get("records", [])
                if motivation_kv_probe is not None
                else []
            )
            frames_dump = []
            n = len(all_ress) if cache_results else 0
            for j in range(n):
                r = all_ress[j]
                img = None
                if cache_results and j < len(processed_frames) and "img" in processed_frames[j]:
                    img = processed_frames[j]["img"].float().cpu().numpy()
                kv_rec = probe_records[j] if j < len(probe_records) else {}
                frames_dump.append(
                    {
                        "camera_pose": r["camera_pose"].float().numpy(),
                        "depth": r["depth"].float().numpy(),
                        "depth_conf": r["depth_conf"].float().numpy(),
                        "conf": r["conf"].float().numpy() if "conf" in r else None,
                        "img_chw": img,
                        "kv_probe": {
                            "past_frame_idx": kv_rec.get("past_frame_idx"),
                            "past_frame_sims": kv_rec.get("past_frame_sims", []),
                        },
                    }
                )
            torch.save(
                {
                    "meta": {
                        "patch_start_idx": int(self.aggregator.patch_start_idx),
                        "img_size": int(getattr(self.aggregator.patch_embed, "img_size", 518)),
                        "patch_size": int(self.aggregator.patch_size),
                        "probe_layer": int(
                            (motivation_kv_probe or {}).get("layer_idx", self.aggregator.depth - 1)
                        ),
                        "eviction_mode": eviction_mode,
                        "num_frames": n,
                    },
                    "frames": frames_dump,
                },
                dump_path,
            )

        next_sequence_state = None
        if return_sequence_state:
            chunk_frame_metadata = [
                {"camera_pose": r["camera_pose"], "depth": r["depth"], "depth_conf": r["depth_conf"], "conf": r.get("conf")}
                for r in all_ress
            ]
            next_sequence_state = {
                "initialized": True,
                "past_key_values": past_key_values,
                "past_key_values_camera": past_key_values_camera,
                "importance_cache": importance_cache,
                "frame_metadata": prev_frame_metadata + chunk_frame_metadata,
                "frame_offset": frame_offset + len(frames),
                **({"motivation_kv_probe": motivation_kv_probe} if motivation_kv_probe is not None else {}),
            }

        return StreamVGGTOutput(
            ress=all_ress if cache_results else None,
            views=processed_frames if cache_results else None,
            kv_cache_mem_bytes=peak_kv_cache_bytes,
            sequence_state=next_sequence_state,
        )