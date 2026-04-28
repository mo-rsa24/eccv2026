"""
compositions/repair_methods/
=============================
Sixteen inference-time PoE repair methods for compositional latent diffusion.

Each method is self-contained in its own file.  All share the same interface:
  - a compose_* function for single-step use
  - a run_* function for a full denoising loop

Quick-reference
---------------
01  method_01_adaptive_weighting       Adaptive timestep/state-dependent weights
02  method_02_gradient_surgery         PCGrad-style conflict projection on deltas
03  method_03_trust_region             Constrained QP — min progress per expert
04  method_04_mask_gated_poe           Attention-derived spatial mask gating
05  method_05_residual_adapter         Learnable r̂_t interaction-term corrector
06  method_06_curvature_preconditioning Second-order / curvature-aware step
07  method_07_mcmc_corrector           Annealed Langevin / SMC corrector
08  method_08_branch_and_select        Mixture-of-joints branch sampling
09  method_09_probe_energy             Failure-energy gradient steering
10  method_10_adaptive_diagnostics     Diagnostic-driven meta-controller
11  method_11_tweedie_poe_corrector    Tweedie-space r_t surrogate (diverge+proj+spatial)
12  method_12_overlap_penalty_corrector Localized overlap penalty + preservation
13  method_13_spatial_routing          Phase-Aware Spatial Routing (softmax router)
14  method_14_poe_anchored_contrastive PoE-Anchored Contrastive Tweedie (CO3 without joint prompt)
15  method_15_corrected_spatial_masking Corrected Tweedie-space masking (fixes 4 bugs in naïve version)
16  method_16_ir_poe                   Interaction-Recovered PoE (two-phase, frozen masks, gated CFG)

Usage
-----
from compositions.repair_methods.method_01_adaptive_weighting import (
    AdaptiveWeightingConfig, run_adaptive_weighting
)
"""

from .method_01_adaptive_weighting      import AdaptiveWeightingConfig,       run_adaptive_weighting
from .method_02_gradient_surgery        import GradientSurgeryConfig,         run_gradient_surgery
from .method_03_trust_region            import TrustRegionConfig,             run_trust_region
from .method_04_mask_gated_poe          import MaskGatedPoEConfig,            run_mask_gated_poe
from .method_05_residual_adapter        import ResidualAdapterConfig,         run_residual_adapter,  ResidualAdapterMLP
from .method_06_curvature_preconditioning import CurvaturePrecondConfig,      run_curvature_preconditioning
from .method_07_mcmc_corrector          import MCMCCorrectorConfig,           run_mcmc_corrector
from .method_08_branch_and_select       import BranchAndSelectConfig,         run_branch_and_select
from .method_09_probe_energy            import ProbeEnergyConfig,             run_probe_energy
from .method_10_adaptive_diagnostics    import AdaptiveDiagnosticsConfig,     run_adaptive_diagnostics
from .method_11_tweedie_poe_corrector   import TweediePoeConfig,              run_tweedie_poe_corrector
from .method_12_overlap_penalty_corrector import OverlapPenaltyConfig,        run_overlap_penalty_corrector
from .method_13_spatial_routing         import SpatialRoutingConfig,          run_spatial_routing
from .method_14_poe_anchored_contrastive import PoeAnchoredContrastiveConfig,  run_poe_anchored_contrastive
from .method_15_corrected_spatial_masking import CorrectedSpatialMaskingConfig, run_corrected_spatial_masking
from .method_16_ir_poe                    import IRPoEConfig,                   run_ir_poe

__all__ = [
    # Method 01
    "AdaptiveWeightingConfig", "run_adaptive_weighting",
    # Method 02
    "GradientSurgeryConfig", "run_gradient_surgery",
    # Method 03
    "TrustRegionConfig", "run_trust_region",
    # Method 04
    "MaskGatedPoEConfig", "run_mask_gated_poe",
    # Method 05
    "ResidualAdapterConfig", "ResidualAdapterMLP", "run_residual_adapter",
    # Method 06
    "CurvaturePrecondConfig", "run_curvature_preconditioning",
    # Method 07
    "MCMCCorrectorConfig", "run_mcmc_corrector",
    # Method 08
    "BranchAndSelectConfig", "run_branch_and_select",
    # Method 09
    "ProbeEnergyConfig", "run_probe_energy",
    # Method 10
    "AdaptiveDiagnosticsConfig", "run_adaptive_diagnostics",
    # Method 11
    "TweediePoeConfig", "run_tweedie_poe_corrector",
    # Method 12
    "OverlapPenaltyConfig", "run_overlap_penalty_corrector",
    # Method 13
    "SpatialRoutingConfig", "run_spatial_routing",
    # Method 14
    "PoeAnchoredContrastiveConfig", "run_poe_anchored_contrastive",
    # Method 15
    "CorrectedSpatialMaskingConfig", "run_corrected_spatial_masking",
    # Method 16
    "IRPoEConfig", "run_ir_poe",
]
