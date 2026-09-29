"""Structural contracts for the result-blind AbdomenCT Validation-2 preflight."""

from .contracts import (
    FEATURE_COLUMNS,
    REQUIRED_LABELS,
    ContractError,
    dice_for_required_organ,
    make_metadata_split,
    normalize_label_values,
    validate_feature_columns,
)
from .endpoints import (
    compute_r5,
    compute_r8,
    compute_validation2_baseline,
    r8_decision,
    segmentation_competence,
)
from .features import extract_qc_features
from .freeze import make_fixed_quota_split, make_segmentation_fold_assignments
from .qc_model import fit_qc_ridge, predict_qc
from .quality import four_organ_quality

__all__ = [
    "FEATURE_COLUMNS",
    "REQUIRED_LABELS",
    "ContractError",
    "dice_for_required_organ",
    "make_metadata_split",
    "normalize_label_values",
    "validate_feature_columns",
    "compute_r5",
    "compute_r8",
    "compute_validation2_baseline",
    "extract_qc_features",
    "fit_qc_ridge",
    "four_organ_quality",
    "make_fixed_quota_split",
    "make_segmentation_fold_assignments",
    "predict_qc",
    "r8_decision",
    "segmentation_competence",
]
