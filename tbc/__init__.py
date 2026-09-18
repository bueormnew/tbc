"""TBC — Ternary Behavioral Compilation. Paquete principal."""
from .config import TBCConfig, parse_budget
from .memory import MemoryMonitor, MemoryBudgetExceeded
from .cache import HierarchicalCache, CircularBuffer, Summary, summarize_tensor
from .trace import SparseCheckpointScheduler, TraceGenerator
from .sensitivity import SensitivityAnalyzer
from .search import (
    initial_ternarization,
    compute_alpha_analytic,
    compute_alpha_per_group,
    dequantize_ternary,
    CoordinateSearchEngine,
    BeamSearch,
)
from .equivalence import EquivalenceChecker, EquivalenceResult
from .compiler import TBC_COMPILE, TBCModel, detect_architecture, get_ternarize_targets
from .pack import pack_i2_s_tl2, unpack_i2_s_tl2, regroup_to_group_size
from .linalg import weight_out_in, write_weight_out_in, forward_out_in, is_conv1d
from .export import export_bitnet_cpp_compatible, bitnet_cpp_can_load, build_manifest

__all__ = [
    "TBCConfig", "parse_budget",
    "MemoryMonitor", "MemoryBudgetExceeded",
    "HierarchicalCache", "CircularBuffer", "Summary", "summarize_tensor",
    "SparseCheckpointScheduler", "TraceGenerator",
    "SensitivityAnalyzer",
    "initial_ternarization", "compute_alpha_analytic", "compute_alpha_per_group",
    "dequantize_ternary", "CoordinateSearchEngine", "BeamSearch",
    "EquivalenceChecker", "EquivalenceResult",
    "TBC_COMPILE", "TBCModel", "detect_architecture", "get_ternarize_targets",
    "pack_i2_s_tl2", "unpack_i2_s_tl2", "regroup_to_group_size",
    "export_bitnet_cpp_compatible", "bitnet_cpp_can_load", "build_manifest",
]
__version__ = "tbc-1.0.0"
