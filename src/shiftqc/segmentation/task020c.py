"""Frozen integrity, image-only inference, and artifact checks for TASK-020C."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

from shiftqc.data.mnms_nnunet import _save_3d_volume, visibility_audit
from shiftqc.data.nnunet_export import sha256_file


INFERENCE_ROLES = ("QC_TRAIN", "CALIBRATION", "M3_ID_EVAL", "M3_OOD_EVAL")


class Task020CFailure(RuntimeError):
    def __init__(self, code: str, details: Any = None):
        self.code = code
        self.details = details
        super().__init__(f"{code}: {details}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_upstream(repository: Path, config: dict[str, Any]) -> dict[str, str]:
    checks = dict(config["upstream"]["task020b"])
    for name in ("m3_split", "task020a_plans", "conversion_manifest", "splits_final"):
        item = config["upstream"][name]
        checks[item["path"]] = item["sha256"]
    observed: dict[str, str] = {}
    for relative, expected in checks.items():
        path = repository / relative
        if not path.is_file():
            raise Task020CFailure("frozen_upstream_missing", relative)
        digest = sha256_file(path)
        if digest != expected:
            raise Task020CFailure(
                "frozen_upstream_hash_mismatch",
                {"path": relative, "expected": expected, "observed": digest},
            )
        observed[relative] = digest
    return observed


def storage_preflight(repository: Path, config: dict[str, Any]) -> dict[str, Any]:
    minimum = int(config["minimum_free_bytes"])
    paths = {
        "nnUNet_results": repository / config["nnunet_roots"]["results"],
        "inference_outputs": repository / config["nnunet_roots"]["prediction_output"],
        "EXP-0005_outputs": repository / config["outputs"]["directory"],
    }
    result: dict[str, Any] = {}
    for name, path in paths.items():
        probe = path if path.exists() else next(parent for parent in path.parents if parent.exists())
        usage = shutil.disk_usage(probe)
        result[name] = {
            "path_provenance": path.relative_to(repository).as_posix(),
            "filesystem_probe": probe.relative_to(repository).as_posix() or ".",
            "total_bytes": int(usage.total),
            "used_bytes": int(usage.used),
            "free_bytes": int(usage.free),
            "minimum_required_free_bytes": minimum,
            "sufficient": int(usage.free) >= minimum,
        }
    if not all(record["sufficient"] for record in result.values()):
        raise Task020CFailure("insufficient_storage", result)
    return result


def verify_frozen_training_state(
    repository: Path, config: dict[str, Any], fold: int = 0
) -> dict[str, Any]:
    dataset_base = repository / config["nnunet_roots"]["preprocessed"] / config["dataset"]
    plans = load_json(dataset_base / f"{config['plans_identifier']}.json")
    planned = plans["configurations"][config["configuration"]]
    architecture = planned["architecture"]
    kwargs = architecture["arch_kwargs"]
    observed_plan = {
        "target_spacing": planned["spacing"],
        "patch_size": planned["patch_size"],
        "batch_size": int(planned["batch_size"]),
        "architecture_suffix": architecture["network_class_name"].split(".")[-1],
        "stages": int(kwargs["n_stages"]),
        "features": kwargs["features_per_stage"],
        "network_normalization": kwargs["norm_op"],
        "intensity_normalization": planned["normalization_schemes"],
    }
    if observed_plan != config["frozen_plan"]:
        raise Task020CFailure(
            "frozen_plan_mismatch", {"expected": config["frozen_plan"], "observed": observed_plan}
        )
    split_path = dataset_base / "splits_final.json"
    if sha256_file(split_path) != config["upstream"]["splits_final"]["sha256"]:
        raise Task020CFailure("active_custom_split_hash_mismatch", None)
    splits = load_json(split_path)
    expected_folds = int(config.get("num_folds", 1))
    if len(splits) != expected_folds:
        raise Task020CFailure(
            "custom_split_fold_count_mismatch",
            {"expected": expected_folds, "observed": len(splits)},
        )
    if not 0 <= fold < len(splits):
        raise Task020CFailure("fold_out_of_range", {"fold": fold, "folds": len(splits)})
    train = set(splits[fold]["train"])
    validation = set(splits[fold]["val"])
    train_patients = {case.rsplit("__", 1)[0] for case in train}
    validation_patients = {case.rsplit("__", 1)[0] for case in validation}
    counts = {
        "train_cases": len(train),
        "validation_cases": len(validation),
        "train_patients": len(train_patients),
        "validation_patients": len(validation_patients),
        "case_overlap": len(train & validation),
        "patient_overlap": len(train_patients & validation_patients),
    }
    expected = config["expected_counts"]
    expected_counts = {
        "train_cases": int(expected["train_cases"]),
        "validation_cases": int(expected["validation_cases"]),
        "train_patients": int(expected["train_patients"]),
        "validation_patients": int(expected["validation_patients"]),
        "case_overlap": 0,
        "patient_overlap": 0,
    }
    if counts != expected_counts:
        raise Task020CFailure(
            "frozen_split_count_mismatch", {"expected": expected_counts, "observed": counts}
        )
    conversion = pd.read_parquet(repository / config["upstream"]["conversion_manifest"]["path"])
    leakage = visibility_audit(
        repository,
        conversion,
        {"frozen_split": {"parquet": config["upstream"]["m3_split"]["path"]}},
    )
    if leakage["total_forbidden_visible"] != 0:
        raise Task020CFailure("forbidden_training_visibility", leakage)
    return {"plan": observed_plan, "split": counts, "leakage": leakage}


def _orthonormal_affine(affine: np.ndarray) -> tuple[np.ndarray, float]:
    """Snap near-orthonormal direction cosines onto the nearest orthonormal frame.

    Non-orthonormal direction cosines are widespread in the M&Ms headers rather
    than confined to a few files: over the 230 inference patients the largest
    element-wise deviation has median ``1.5e-06`` but a long tail, exceeding
    ``1e-04`` for 37 patients (almost all vendor C). ITK refuses to read a file
    whose frame is too far off perpendicular (``ITK only supports orthonormal
    direction cosines``), and three patients - C8J7L5, C8O0P2, E3F5U2 - land just
    over that line while R1R6Y8 at ``3.79e-04`` lands just under it. Correcting
    only the rejected files would therefore draw an arbitrary boundary through a
    continuum, so the normalisation is applied cohort-wide.

    The deviation is float noise in the header, not an oblique acquisition: the
    worst case is ``4.5e-04`` mm against a ``1.25`` mm voxel, a relative error of
    ``3.6e-04``. The affected patients are corrected rather than dropped because
    the population is frozen.

    Voxel spacing and the translation are preserved exactly; only the rotation is
    replaced, by the nearest orthogonal matrix under the Frobenius norm (the polar
    factor, obtained from the SVD). Returns the corrected affine and the largest
    absolute change to any element of the 3x3 linear part.
    """
    affine = np.asarray(affine, dtype=np.float64)
    linear = affine[:3, :3]
    spacing = np.linalg.norm(linear, axis=0)
    if not np.all(np.isfinite(linear)) or np.any(spacing == 0.0):
        raise Task020CFailure("degenerate_input_affine", np.asarray(affine).tolist())
    direction = linear / spacing
    u, _, vt = np.linalg.svd(direction)
    rotation = u @ vt
    # The nearest orthogonal matrix can flip handedness when the input is far from
    # orthonormal. A flip would mirror the volume, so refuse instead of correcting.
    if np.sign(np.linalg.det(rotation)) != np.sign(np.linalg.det(direction)):
        raise Task020CFailure("affine_orthonormalization_handedness_flip", np.asarray(affine).tolist())
    corrected = affine.copy()
    corrected[:3, :3] = rotation @ np.diag(spacing)
    return corrected, float(np.abs(corrected[:3, :3] - linear).max())


def _orthonormalized_source(
    source_image: nib.spatialimages.SpatialImage,
) -> tuple[nib.Nifti1Image, float]:
    """Return ``source_image`` re-expressed on an orthonormal frame.

    ``_save_3d_volume`` is shared with the TASK-020A training conversion and is
    deliberately left untouched, so the correction is applied by handing it a
    source whose affine, qform and sform already carry the orthonormal frame.
    """
    corrected, delta = _orthonormal_affine(source_image.affine)
    fixed = nib.Nifti1Image(
        source_image.dataobj, corrected, source_image.header.copy()
    )
    _, qcode = source_image.get_qform(coded=True)
    _, scode = source_image.get_sform(coded=True)
    fixed.set_qform(corrected, int(qcode) if qcode else 1)
    fixed.set_sform(corrected, int(scode) if scode else 1)
    return fixed, delta


def _source_path(dataset_root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise Task020CFailure("unsafe_source_path", relative)
    root = dataset_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise Task020CFailure("source_image_missing", relative)
    return path


def inference_patients(repository: Path, config: dict[str, Any]) -> pd.DataFrame:
    # Deliberately excludes gt_path_relative so inference preparation cannot read GT.
    frame = pd.read_parquet(
        repository / config["upstream"]["m3_split"]["path"],
        columns=[
            "patient_id", "vendor", "official_partition", "scientific_role",
            "source_split", "image_path_relative", "ed_frame", "es_frame",
            "exclusion_reason",
        ],
    )
    source = frame.loc[
        frame["scientific_role"].eq("SOURCE")
        & frame["source_split"].isin(("QC_TRAIN", "CALIBRATION"))
    ].copy()
    source["inference_role"] = source["source_split"].astype(str)
    target = frame.loc[frame["scientific_role"].isin(("M3_ID_EVAL", "M3_OOD_EVAL"))].copy()
    target["inference_role"] = target["scientific_role"].astype(str)
    patients = pd.concat([source, target], ignore_index=True)
    if patients["patient_id"].duplicated().any():
        raise Task020CFailure("duplicate_inference_patient_id", None)
    # D-011/D-013: fold 0 covers every role; folds 1-4 supply only the agreement
    # signal and therefore run on the evaluation roles alone.
    selected_roles = tuple(config["inference"].get("roles", INFERENCE_ROLES))
    unknown = set(selected_roles) - set(INFERENCE_ROLES)
    if unknown:
        raise Task020CFailure("unknown_inference_role", sorted(unknown))
    patients = patients.loc[patients["inference_role"].isin(selected_roles)].copy()
    counts = patients["inference_role"].value_counts().to_dict()
    expected = {
        role: int(config["expected_counts"]["inference"][role]["patients"])
        for role in selected_roles
    }
    if counts != expected:
        raise Task020CFailure("inference_patient_count_mismatch", {"expected": expected, "observed": counts})
    if patients["exclusion_reason"].notna().any():
        raise Task020CFailure("excluded_patient_selected_for_inference", None)
    return patients.sort_values(["inference_role", "patient_id"], kind="stable", ignore_index=True)


def build_inference_case_table(patients: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in patients.itertuples(index=False):
        for phase, frame in (("ED", row.ed_frame), ("ES", row.es_frame)):
            records.append(
                {
                    "patient_id": str(row.patient_id),
                    "case_id": f"{row.patient_id}__{phase}",
                    "phase": phase,
                    "vendor": str(row.vendor),
                    "scientific_role": str(row.inference_role),
                    "official_partition": str(row.official_partition),
                    "image_path_relative": str(row.image_path_relative),
                    "phase_frame": int(frame),
                }
            )
    cases = pd.DataFrame.from_records(records)
    if cases["case_id"].duplicated().any() or cases["patient_id"].nunique() * 2 != len(cases):
        raise Task020CFailure("inference_case_identity_failure", None)
    phase_sets = cases.groupby("patient_id", observed=True)["phase"].agg(lambda values: set(values))
    if not phase_sets.eq({"ED", "ES"}).all():
        raise Task020CFailure("inference_phase_pair_failure", None)
    return cases.sort_values("case_id", kind="stable", ignore_index=True)


def _case_table_hash(cases: pd.DataFrame) -> str:
    payload = "\n".join(
        f"{row.case_id}|{row.patient_id}|{row.phase}|{row.scientific_role}|{row.image_path_relative}|{row.phase_frame}"
        for row in cases.itertuples(index=False)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare_inference_inputs(
    repository: Path,
    dataset_root: Path,
    input_root: Path,
    manifest_path: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    cases = build_inference_case_table(inference_patients(repository, config))
    expected_hash = _case_table_hash(cases)
    if input_root.exists():
        if not manifest_path.is_file():
            raise Task020CFailure("existing_inference_input_without_manifest", None)
        existing = pd.read_parquet(manifest_path)
        if _case_table_hash(existing) != expected_hash:
            raise Task020CFailure("existing_inference_input_manifest_mismatch", None)
        missing = [
            row.input_image_path for row in existing.itertuples(index=False)
            if not (repository / row.input_image_path).is_file()
        ]
        if missing:
            raise Task020CFailure("existing_inference_input_missing_files", missing[:10])
        return existing

    staging = input_root.with_name(input_root.name + ".partial")
    if staging.exists():
        raise Task020CFailure("partial_inference_input_exists", staging.relative_to(repository).as_posix())
    staging.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    try:
        for row in cases.itertuples(index=False):
            source_path = _source_path(dataset_root, row.image_path_relative)
            source_image = nib.load(source_path)
            if len(source_image.shape) != 4 or not (0 <= row.phase_frame < source_image.shape[3]):
                raise Task020CFailure(
                    "invalid_inference_phase_frame",
                    {"case_id": row.case_id, "shape": source_image.shape, "frame": row.phase_frame},
                )
            volume = np.asanyarray(source_image.dataobj[..., row.phase_frame])
            if not np.isfinite(volume).all():
                raise Task020CFailure("nonfinite_inference_input", row.case_id)
            destination = staging / f"{row.case_id}_0000.nii.gz"
            # Applied to every case, not only the three that ITK rejects. The
            # deviation is a continuum across the cohort and the rejected files sit
            # barely above a threshold others sit barely below, so a conditional
            # correction would be arbitrary. Every staged file therefore differs
            # from a naive copy of the source header, by a median of 1.5e-06 mm;
            # the per-case size is recorded below rather than assumed negligible.
            written_image, affine_delta = _orthonormalized_source(source_image)
            _save_3d_volume(
                written_image,
                volume,
                destination,
                dtype=np.dtype(source_image.get_data_dtype()),
            )
            records.append(
                {
                    **row._asdict(),
                    "input_image_path": (
                        input_root.relative_to(repository) / destination.name
                    ).as_posix(),
                    "input_sha256": sha256_file(destination),
                    "input_shape": list(volume.shape),
                    "input_affine": np.asarray(written_image.affine).tolist(),
                    "source_affine": np.asarray(source_image.affine).tolist(),
                    "affine_orthonormal_delta": affine_delta,
                }
            )
        staging.replace(input_root)
    except BaseException:
        # Keep the partial directory for explicit forensic inspection and safe resume decisions.
        raise
    result = pd.DataFrame.from_records(records)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(manifest_path, index=False, engine="pyarrow")
    return result


def validate_predictions(
    repository: Path,
    input_manifest: pd.DataFrame,
    prediction_root: Path,
    checkpoint_sha256: str,
    save_probabilities: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate one fold's predictions against its frozen input manifest.

    ``save_probabilities`` mirrors the per-fold setting in the D-013 config. nnU-Net
    writes ``.npz`` and ``.pkl`` only when probabilities are requested
    (``nnunetv2/inference/export_prediction.py``), so folds 1-4 legitimately have
    neither. Hard-prediction checks (geometry, label set, finiteness) are never
    relaxed; only the probability checks are skipped when no probabilities exist.
    """
    records: list[dict[str, Any]] = []
    max_sum_error = 0.0
    probability_min = float("inf")
    probability_max = float("-inf")
    observed_label_union: set[int] = set()
    for row in input_manifest.itertuples(index=False):
        hard_path = prediction_root / f"{row.case_id}.nii.gz"
        probability_path = prediction_root / f"{row.case_id}.npz"
        properties_path = prediction_root / f"{row.case_id}.pkl"
        required = (hard_path, probability_path, properties_path) if save_probabilities else (hard_path,)
        missing = [p.name for p in required if not p.is_file()]
        if missing:
            raise Task020CFailure("prediction_artifact_missing", {"case_id": row.case_id, "missing": missing})
        input_image = nib.load(repository / row.input_image_path)
        hard_image = nib.load(hard_path)
        if hard_image.shape != input_image.shape or not np.allclose(
            hard_image.affine, input_image.affine, rtol=0.0, atol=1e-5
        ):
            raise Task020CFailure("hard_prediction_geometry_mismatch", row.case_id)
        hard = np.asanyarray(hard_image.dataobj)
        if not np.isfinite(hard).all():
            raise Task020CFailure("nonfinite_hard_prediction", row.case_id)
        labels = {int(value) for value in np.unique(hard)}
        if not labels.issubset({0, 1, 2, 3}):
            raise Task020CFailure("unexpected_hard_prediction_label", {"case_id": row.case_id, "labels": sorted(labels)})
        observed_label_union.update(labels)
        if save_probabilities:
            with np.load(probability_path) as archive:
                if "probabilities" not in archive.files:
                    raise Task020CFailure("probability_array_missing", row.case_id)
                probabilities = archive["probabilities"]
            expected_shape = (4, *tuple(reversed(input_image.shape)))
            if probabilities.shape != expected_shape:
                raise Task020CFailure(
                    "probability_shape_mismatch",
                    {"case_id": row.case_id, "expected": expected_shape, "observed": probabilities.shape},
                )
            if not np.isfinite(probabilities).all():
                raise Task020CFailure("nonfinite_probability", row.case_id)
            sums = probabilities.sum(axis=0)
            sum_error = float(np.max(np.abs(sums - 1.0)))
            if sum_error > 1e-4:
                raise Task020CFailure("probability_sum_mismatch", {"case_id": row.case_id, "max_error": sum_error})
            max_sum_error = max(max_sum_error, sum_error)
            probability_min = min(probability_min, float(probabilities.min()))
            probability_max = max(probability_max, float(probabilities.max()))
        records.append(
            {
                "patient_id": row.patient_id,
                "case_id": row.case_id,
                "phase": row.phase,
                "vendor": row.vendor,
                "scientific_role": row.scientific_role,
                "official_partition": row.official_partition,
                "hard_prediction_path": hard_path.relative_to(repository).as_posix(),
                "probability_path": (
                    probability_path.relative_to(repository).as_posix() if save_probabilities else None
                ),
                "prediction_properties_path": (
                    properties_path.relative_to(repository).as_posix() if save_probabilities else None
                ),
                "hard_prediction_sha256": sha256_file(hard_path),
                "probability_sha256": sha256_file(probability_path) if save_probabilities else None,
                "prediction_checkpoint_sha256": checkpoint_sha256,
            }
        )
    result = pd.DataFrame.from_records(records).sort_values("case_id", ignore_index=True)
    counts = result.groupby("scientific_role", observed=True).agg(
        patients=("patient_id", "nunique"), cases=("case_id", "size")
    )
    return result, {
        "rows": len(result),
        "unique_patients": int(result["patient_id"].nunique()),
        "counts": {
            role: {"patients": int(counts.loc[role, "patients"]), "cases": int(counts.loc[role, "cases"])}
            # D-013: folds 1-4 carry only the evaluation roles, so report the roles
            # actually present rather than the four-role constant.
            for role in sorted(counts.index)
        },
        "probabilities_saved": save_probabilities,
        "probability_class_count": 4 if save_probabilities else None,
        "probability_values_finite": True if save_probabilities else None,
        "max_voxelwise_probability_sum_error": max_sum_error if save_probabilities else None,
        "probability_min": probability_min if save_probabilities else None,
        "probability_max": probability_max if save_probabilities else None,
        "hard_prediction_label_union": sorted(observed_label_union),
        "hard_prediction_geometry_valid": True,
        "two_phases_per_patient": bool(
            result.groupby("patient_id", observed=True)["phase"].agg(lambda x: set(x)).eq({"ED", "ES"}).all()
        ),
    }
