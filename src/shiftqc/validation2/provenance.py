"""Deterministic, outcome-free image-identity utilities for TASK-032."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import nibabel as nib
import numpy as np


AUTHORITATIVE_MATCH_STATUSES = frozenset(
    {
        "EXACT_CONTENT_MATCH",
        "DETERMINISTIC_VOXEL_EQUIVALENT_MATCH",
        "DOCUMENTED_TRANSFORM_EQUIVALENT_MATCH",
    }
)

FORBIDDEN_SCIENTIFIC_FIELDS = frozenset(
    {
        "dice",
        "q_true",
        "q_hat",
        "r5",
        "r8",
        "qc_mae",
        "qc_residual",
        "aurc",
        "risk_coverage",
        "d014",
        "adjexcess",
    }
)


class ProvenanceError(ValueError):
    """A provenance authority or uniqueness contract was violated."""


def numeric_array_sha256(array: np.ndarray) -> str:
    """Hash decoded numeric values independent of source dtype/endianness.

    Values are normalized to little-endian float64 in C order. Shape is part of
    the domain separator, so equal byte streams with different shapes cannot
    collide under this fingerprint contract.
    """

    values = np.asarray(array)
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ProvenanceError("voxel arrays must be finite numeric values")
    normalized = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(b"SHIFTQC_TASK032_NUMERIC_ARRAY_V1\0")
    digest.update(np.asarray(normalized.shape, dtype="<i8").tobytes())
    digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def canonical_ras_array(image: nib.spatialimages.SpatialImage) -> tuple[np.ndarray, np.ndarray]:
    """Return a losslessly reoriented RAS+ voxel array and affine."""

    if len(image.shape) != 3:
        raise ProvenanceError("only three-dimensional source volumes are supported")
    canonical = nib.as_closest_canonical(image, enforce_diag=False)
    if nib.aff2axcodes(canonical.affine) != ("R", "A", "S"):
        raise ProvenanceError("canonical orientation did not resolve to RAS+")
    values = np.asanyarray(canonical.dataobj)
    if not np.isfinite(values).all():
        raise ProvenanceError("canonical voxels contain NaN or infinity")
    return values, np.asarray(canonical.affine, dtype=np.float64)


def dicom_pixels_to_hu(
    pixels: np.ndarray,
    *,
    rescale_slope: float,
    rescale_intercept: float,
) -> np.ndarray:
    """Apply the DICOM linear rescale transform deterministically."""

    values = np.asarray(pixels)
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ProvenanceError("DICOM pixels must be finite numeric values")
    if not np.isfinite(rescale_slope) or not np.isfinite(rescale_intercept):
        raise ProvenanceError("DICOM rescale parameters must be finite")
    if rescale_slope == 0:
        raise ProvenanceError("DICOM RescaleSlope must be nonzero")
    return values.astype(np.float64) * float(rescale_slope) + float(rescale_intercept)


@dataclass(frozen=True)
class IdentityRecord:
    identifier: str
    canonical_voxel_hash: str
    raw_voxel_hash: str | None = None


def unique_hash_matches(
    abdomen: Sequence[IdentityRecord],
    source: Sequence[IdentityRecord],
) -> dict[str, tuple[str, str]]:
    """Classify exact one-to-one hash matches, rejecting every collision."""

    if len({row.identifier for row in abdomen}) != len(abdomen):
        raise ProvenanceError("duplicate AbdomenCT identifier")
    if len({row.identifier for row in source}) != len(source):
        raise ProvenanceError("duplicate source identifier")
    abdomen_by_hash: dict[str, list[IdentityRecord]] = defaultdict(list)
    source_by_hash: dict[str, list[IdentityRecord]] = defaultdict(list)
    for row in abdomen:
        abdomen_by_hash[row.canonical_voxel_hash].append(row)
    for row in source:
        source_by_hash[row.canonical_voxel_hash].append(row)

    result: dict[str, tuple[str, str]] = {}
    for row in abdomen:
        candidates = source_by_hash.get(row.canonical_voxel_hash, [])
        reverse = abdomen_by_hash[row.canonical_voxel_hash]
        if len(candidates) == 1 and len(reverse) == 1:
            source_row = candidates[0]
            stage = (
                "EXACT_CONTENT_MATCH"
                if row.raw_voxel_hash is not None
                and source_row.raw_voxel_hash is not None
                and row.raw_voxel_hash == source_row.raw_voxel_hash
                else "DETERMINISTIC_VOXEL_EQUIVALENT_MATCH"
            )
            result[row.identifier] = (source_row.identifier, stage)
        elif candidates:
            result[row.identifier] = ("", "AMBIGUOUS")
        else:
            result[row.identifier] = ("", "NO_MATCH")
    return result


def duplicate_patient_ids(rows: Iterable[Mapping[str, str]]) -> dict[str, list[str]]:
    """Return source-scoped patient IDs represented by multiple case IDs."""

    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        dataset = str(row["source_dataset"])
        patient = str(row["source_patient_id"])
        case = str(row["source_case_id"])
        if not dataset or not patient or not case:
            raise ProvenanceError("patient audit rows require non-empty identities")
        grouped[f"{dataset}:{patient}"].append(case)
    return {
        key: sorted(cases)
        for key, cases in sorted(grouped.items())
        if len(cases) > 1
    }


def cross_source_patient_overlaps(
    rows: Iterable[Mapping[str, str]],
) -> dict[str, list[str]]:
    """Return explicit global patient IDs represented in multiple datasets."""

    datasets_by_global_id: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        global_id = str(row.get("global_patient_id", ""))
        dataset = str(row.get("source_dataset", ""))
        if global_id and dataset:
            datasets_by_global_id[global_id].add(dataset)
    return {
        patient: sorted(datasets)
        for patient, datasets in sorted(datasets_by_global_id.items())
        if len(datasets) > 1
    }


def reject_scientific_fields(fieldnames: Iterable[str]) -> tuple[str, ...]:
    """Enforce that provenance outputs contain no scientific-result fields."""

    names = tuple(str(name) for name in fieldnames)
    normalized = {name.lower() for name in names}
    forbidden = sorted(normalized & FORBIDDEN_SCIENTIFIC_FIELDS)
    if forbidden:
        raise ProvenanceError(f"scientific fields are forbidden: {forbidden}")
    return names
