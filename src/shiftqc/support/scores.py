"""Source-only support-score estimators for frozen TASK-011 features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class SupportInputError(ValueError):
    """Raised when source-support inputs violate the frozen contract."""


@dataclass(frozen=True)
class FittedSourceSupport:
    """Support estimators fitted on QC_TRAIN plus CALIBRATION only."""

    scaler: StandardScaler
    neighbors: NearestNeighbors
    ledoit_wolf: LedoitWolf
    standardized_reference: np.ndarray
    reference_patient_ids: tuple[str, ...]
    feature_columns: tuple[str, ...]
    k_values: tuple[int, ...]


def validate_feature_artifact(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    """Require exact metadata plus frozen 15-feature schema."""
    feature_columns = list(config["feature_columns"])
    expected_schema = ["patient_id", "domain", "split", *feature_columns]
    if list(frame.columns) != expected_schema:
        raise SupportInputError(
            f"feature schema mismatch: expected={expected_schema}, observed={list(frame.columns)}"
        )
    if len(feature_columns) != 15 or len(set(feature_columns)) != 15:
        raise SupportInputError("TASK-011 requires exactly 15 unique features")
    if frame["patient_id"].duplicated().any():
        raise SupportInputError("feature artifact contains duplicate patient IDs")
    if frame["split"].eq("OOD_FINAL").any():
        raise SupportInputError("OOD_FINAL is forbidden during TASK-011")
    matrix = frame[feature_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise SupportInputError("feature artifact contains NaN or infinity")


def fit_source_support(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> FittedSourceSupport:
    """Fit all source-support parameters on the frozen 187-patient reference."""
    validate_feature_artifact(frame, config)
    reference_splits = tuple(str(value) for value in config["reference_splits"])
    reference = frame.loc[frame["split"].isin(reference_splits)].copy()
    observed_counts = {
        str(name): int(value)
        for name, value in reference["split"].value_counts().sort_index().items()
    }
    expected_counts = {
        str(name): int(value)
        for name, value in config["reference_split_counts"].items()
    }
    if observed_counts != dict(sorted(expected_counts.items())):
        raise SupportInputError(
            f"reference split mismatch: expected={expected_counts}, observed={observed_counts}"
        )
    if len(reference) != int(config["reference_count"]):
        raise SupportInputError(
            f"reference count mismatch: expected={config['reference_count']}, "
            f"observed={len(reference)}"
        )
    if not reference["domain"].eq("source").all():
        raise SupportInputError("source reference contains a non-source domain patient")
    feature_columns = tuple(str(value) for value in config["feature_columns"])
    matrix = reference[list(feature_columns)].to_numpy(dtype=np.float64)

    standardization = config["standardization"]
    scaler = StandardScaler(
        with_mean=bool(standardization["with_mean"]),
        with_std=bool(standardization["with_std"]),
    )
    standardized = scaler.fit_transform(matrix)
    k_values = tuple(int(value) for value in config["knn"]["k_values"])
    if tuple(sorted(set(k_values))) != k_values or min(k_values) <= 0:
        raise SupportInputError("k values must be unique positive integers in ascending order")
    if max(k_values) > len(reference):
        raise SupportInputError("largest k exceeds source-reference count")
    neighbors = NearestNeighbors(
        n_neighbors=max(k_values),
        metric=str(config["knn"]["metric"]),
        algorithm=str(config["knn"]["algorithm"]),
    )
    neighbors.fit(standardized)

    mahalanobis = config["mahalanobis"]
    ledoit_wolf = LedoitWolf(
        store_precision=bool(mahalanobis["store_precision"]),
        assume_centered=bool(mahalanobis["assume_centered"]),
    )
    ledoit_wolf.fit(standardized)
    return FittedSourceSupport(
        scaler=scaler,
        neighbors=neighbors,
        ledoit_wolf=ledoit_wolf,
        standardized_reference=standardized,
        reference_patient_ids=tuple(str(value) for value in reference["patient_id"]),
        feature_columns=feature_columns,
        k_values=k_values,
    )


def compute_held_out_support_scores(
    frame: pd.DataFrame,
    fitted: FittedSourceSupport,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Apply unchanged source-fitted parameters to ID_TEST and OOD_DEV."""
    validate_feature_artifact(frame, config)
    evaluation_splits = set(str(value) for value in config["evaluation_counts"])
    evaluation = frame.loc[frame["split"].isin(evaluation_splits)].copy()
    observed_counts = {
        str(name): int(value)
        for name, value in evaluation["split"].value_counts().sort_index().items()
    }
    expected_counts = {
        str(name): int(value) for name, value in config["evaluation_counts"].items()
    }
    if observed_counts != dict(sorted(expected_counts.items())):
        raise SupportInputError(
            f"evaluation split mismatch: expected={expected_counts}, observed={observed_counts}"
        )
    matrix = evaluation[list(fitted.feature_columns)].to_numpy(dtype=np.float64)
    standardized = fitted.scaler.transform(matrix)
    distances, _indices = fitted.neighbors.kneighbors(
        standardized,
        n_neighbors=max(fitted.k_values),
        return_distance=True,
    )
    output = evaluation[["patient_id", "domain", "split"]].copy()
    for k in fitted.k_values:
        output[f"support_knn_k{k}"] = np.mean(
            distances[:, :k],
            axis=1,
            dtype=np.float64,
        )
    output["support_mahalanobis"] = fitted.ledoit_wolf.mahalanobis(standardized)
    score_columns = [column for column in output.columns if column.startswith("support_")]
    scores = output[score_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all():
        raise SupportInputError("support scores contain NaN or infinity")
    knn_columns = [column for column in score_columns if column.startswith("support_knn_")]
    if np.any(output[knn_columns].to_numpy(dtype=np.float64) < 0):
        raise SupportInputError("kNN support distance is negative")
    tolerance = float(config["mahalanobis_nonnegative_tolerance"])
    if np.any(output["support_mahalanobis"].to_numpy(dtype=np.float64) < -tolerance):
        raise SupportInputError("Mahalanobis score is negative beyond numerical tolerance")
    if output["patient_id"].duplicated().any():
        raise SupportInputError("support-score output contains duplicate patient IDs")
    if output["split"].eq("OOD_FINAL").any():
        raise SupportInputError("support-score output contains OOD_FINAL")
    return output.sort_values(
        ["split", "patient_id"], kind="stable", ignore_index=True
    )
