"""
streamvggt.quantization
=======================
QVG-style training-free KV Cache quantisation for InfiniteVGGT.

Public API:
    QVGConfig                         – configuration dataclass (presets available)
    QVGInfiniteVGGTKVCacheManager     – end-to-end quantise/dequantise manager
    SemanticAwareSmoothing            – SAS module (QVG Sec 4.1)
    ProgressiveResidualQuantization   – PRQ module (QVG Sec 4.2)
    fused_dequant                     – Triton/PyTorch fused dequant dispatch
    TRITON_AVAILABLE                  – bool, whether Triton kernel is usable
"""

from .semantic_aware_smoothing import SemanticAwareSmoothing
from .progressive_residual_quantization import (
    ProgressiveResidualQuantization,
    PRQQuantizedCache,
)
from .kv_cache_manager import (
    QVGConfig,
    QVGInfiniteVGGTKVCacheManager,
)
from .triton_kernels import fused_dequant, TRITON_AVAILABLE
from .diversity_aware_quantization import (
    DiversityQuantConfig,
    quantize_diversity_aware,
    dequantize_diversity_aware,
)
from .diversity_kv_cache_manager import DiversityKVCacheManager

__all__ = [
    "QVGConfig",
    "QVGInfiniteVGGTKVCacheManager",
    "SemanticAwareSmoothing",
    "ProgressiveResidualQuantization",
    "PRQQuantizedCache",
    "fused_dequant",
    "TRITON_AVAILABLE",
    "DiversityQuantConfig",
    "DiversityKVCacheManager",
    "quantize_diversity_aware",
    "dequantize_diversity_aware",
]
