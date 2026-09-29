"""Synthetic tests for the frozen TASK-022/023 QC model and deployed calibration.

Construction only. Nothing here evaluates the model: no skill, no correlation, no
coverage. Section 18 of the task packet asks for deliberate firewall attacks, and
those are the four tests at the end.
"""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shiftqc.qc.baselines import QCBaselineError
from shiftqc.qc.m3_deployment import (
    CALIBRATION_POPULATION,
    TRAINING_POPULATION,
    Task022Failure,
    deployed_intervals,
    feature_columns_sha256,
    fit_deployed_conformal,
    fit_qc_model,
    predict_q_hat,
    unconditional_source_interval,
)
from shiftqc.qc.m3_inputs import FROZEN_FEATURE_ORDER

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (REPOSITORY / "configs/templates/qc_baselines.json").read_text())
ALPHA_LEVELS = {"cp80": 0.20, "cp90": 0.10, "cp95": 0.05}


def _features(n: int, population: str, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({c: rng.uniform(0.1, 0.9, n) for c in FROZEN_FEATURE_ORDER})
    frame.insert(0, "vendor", "A")
    frame.insert(0, "population", population)
    frame.insert(0, "patient_id", [f"{population}_{i:03d}" for i in range(n)])
    return frame


def _target(frame: pd.DataFrame, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.uniform(0.5, 0.95, len(frame)), index=frame.index, name="dice_macro")


def _fitted():
    f = _features(30, TRAINING_POPULATION)
    return fit_qc_model(f, _target(f), CONFIG), f


# --------------------------------------------------------------- QC model

def test_canonical_feature_order_is_required() -> None:
    f = _features(30, TRAINING_POPULATION)
    shuffled = f[["patient_id", "population", "vendor", *reversed(FROZEN_FEATURE_ORDER)]]
    with pytest.raises(Task022Failure) as e:
        fit_qc_model(shuffled, _target(shuffled), CONFIG)
    assert e.value.code == "feature_columns_out_of_canonical_order"


def test_missing_and_extra_columns_fail() -> None:
    f = _features(30, TRAINING_POPULATION)
    with pytest.raises((Task022Failure, QCBaselineError)):
        fit_qc_model(f.drop(columns=["entropy_p90"]), _target(f), CONFIG)
    extra = f.copy(); extra["dice_macro"] = 0.5
    model = fit_qc_model(extra, _target(extra), CONFIG)   # a target column present is ignored
    assert model["feature_columns"] == list(FROZEN_FEATURE_ORDER)


def test_non_finite_feature_fails() -> None:
    f = _features(30, TRAINING_POPULATION); f.loc[0, "entropy_mean"] = np.nan
    with pytest.raises(QCBaselineError):
        fit_qc_model(f, _target(f), CONFIG)


def test_only_qc_train_reaches_the_fit() -> None:
    for population in ("CALIBRATION", "M3_ID_EVAL", "M3_OOD_EVAL", "SEG_DEV"):
        f = _features(30, population)
        with pytest.raises(Task022Failure) as e:
            fit_qc_model(f, _target(f), CONFIG)
        assert e.value.code == "fit_population_not_qc_train"


def test_mixed_population_frame_is_rejected() -> None:
    mixed = pd.concat([_features(15, TRAINING_POPULATION),
                       _features(15, "M3_OOD_EVAL", seed=3)], ignore_index=True)
    with pytest.raises(Task022Failure) as e:
        fit_qc_model(mixed, _target(mixed), CONFIG)
    assert e.value.code == "fit_population_not_qc_train"


def test_wrong_training_count_and_duplicates_fail() -> None:
    f = _features(29, TRAINING_POPULATION)
    with pytest.raises(Task022Failure) as e:
        fit_qc_model(f, _target(f), CONFIG)
    assert e.value.code == "unexpected_population_patient_count"

    d = _features(30, TRAINING_POPULATION); d.loc[1, "patient_id"] = d.loc[0, "patient_id"]
    with pytest.raises(Task022Failure) as e:
        fit_qc_model(d, _target(d), CONFIG)
    assert e.value.code == "duplicate_patient_id"


def test_alpha_selection_and_fit_are_deterministic() -> None:
    a, _ = _fitted(); b, _ = _fitted()
    assert a["selected_alpha"] == b["selected_alpha"]
    assert a["ridge_coefficients"] == b["ridge_coefficients"]
    assert a["scaler_mean"] == b["scaler_mean"]
    assert a["selected_alpha"] in CONFIG["qc1_ridge"]["alpha_grid"]


def test_cv_definition_is_the_frozen_one() -> None:
    model, _ = _fitted()
    assert model["cv"] == CONFIG["qc1_ridge"]["cv"]
    assert model["cv"]["n_splits"] == 5 and model["cv"]["random_state"] == 2026
    assert model["cv"]["scoring"] == "neg_mean_absolute_error"
    assert model["alpha_grid"] == [float(a) for a in CONFIG["qc1_ridge"]["alpha_grid"]]


def test_scaler_and_ridge_are_fit_on_qc_train_only() -> None:
    model, f = _fitted()
    matrix = f[list(FROZEN_FEATURE_ORDER)].to_numpy(dtype=np.float64)
    assert np.allclose(model["scaler_mean"], matrix.mean(axis=0))
    assert model["n_train"] == 30 and model["training_population"] == TRAINING_POPULATION


def test_serialisation_round_trip_gives_identical_predictions(tmp_path: Path) -> None:
    joblib = pytest.importorskip("joblib")
    model, f = _fitted()
    path = tmp_path / "qc_model.joblib"
    joblib.dump(model["estimator"], path)
    other = _features(20, "M3_OOD_EVAL", seed=11)
    a = predict_q_hat(model["estimator"], other)
    b = predict_q_hat(joblib.load(path), other)
    assert np.array_equal(a, b)


def test_prediction_signature_admits_no_label() -> None:
    params = set(inspect.signature(predict_q_hat).parameters)
    assert params == {"model", "features"}
    for forbidden in ("q_true", "target", "label", "dice", "ground"):
        assert not any(forbidden in p for p in params)


def test_patient_identity_is_preserved() -> None:
    model, _ = _fitted()
    other = _features(20, "M3_ID_EVAL", seed=5)
    q_hat = predict_q_hat(model["estimator"], other)
    assert len(q_hat) == len(other)
    reordered = other.iloc[::-1].reset_index(drop=True)
    assert np.allclose(predict_q_hat(model["estimator"], reordered), q_hat[::-1])


def test_feature_columns_hash_is_stable_and_order_sensitive() -> None:
    assert feature_columns_sha256(FROZEN_FEATURE_ORDER) == feature_columns_sha256(FROZEN_FEATURE_ORDER)
    assert feature_columns_sha256(FROZEN_FEATURE_ORDER) != feature_columns_sha256(
        tuple(reversed(FROZEN_FEATURE_ORDER)))


# -------------------------------------------------------------- conformal

def _calibration(n: int = 30, seed: int = 4, population: str = CALIBRATION_POPULATION):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "patient_id": [f"CAL_{i:03d}" for i in range(n)],
        "population": population, "vendor": "A",
        "q_true": rng.uniform(0.6, 0.95, n), "q_hat": rng.uniform(0.6, 0.95, n)})


def test_finite_sample_rank_is_exact_for_thirty_calibration_patients() -> None:
    out = fit_deployed_conformal(_calibration(), alpha_levels=ALPHA_LEVELS)
    assert out["n_calibration"] == 30
    assert out["levels"]["cp90"]["finite_sample_rank"] == 28      # ceil(31 * 0.90)
    assert out["levels"]["cp80"]["finite_sample_rank"] == 25      # ceil(31 * 0.80)
    assert out["levels"]["cp95"]["finite_sample_rank"] == 30      # ceil(31 * 0.95)


def test_known_residuals_give_the_known_order_statistic() -> None:
    cal = _calibration()
    cal["q_true"] = 0.5
    cal["q_hat"] = 0.5 + np.arange(30) / 100.0                    # residuals 0.00 .. 0.29
    out = fit_deployed_conformal(cal, alpha_levels={"cp90": 0.10})
    assert out["levels"]["cp90"]["residual_threshold"] == pytest.approx(0.27)  # 28th smallest


def test_calibration_only_and_count_are_enforced() -> None:
    for population in ("M3_ID_EVAL", "M3_OOD_EVAL", "QC_TRAIN"):
        with pytest.raises(Task022Failure) as e:
            fit_deployed_conformal(_calibration(population=population), alpha_levels=ALPHA_LEVELS)
        assert e.value.code == "calibration_population_not_calibration"
    with pytest.raises(Task022Failure) as e:
        fit_deployed_conformal(_calibration(n=29), alpha_levels=ALPHA_LEVELS)
    assert e.value.code == "unexpected_population_patient_count"


def test_duplicate_and_non_finite_calibration_fail() -> None:
    d = _calibration(); d.loc[1, "patient_id"] = d.loc[0, "patient_id"]
    with pytest.raises(Task022Failure) as e:
        fit_deployed_conformal(d, alpha_levels=ALPHA_LEVELS)
    assert e.value.code == "duplicate_patient_id"
    n = _calibration(); n.loc[0, "q_true"] = np.nan
    with pytest.raises(Exception):
        fit_deployed_conformal(n, alpha_levels=ALPHA_LEVELS)


def test_intervals_are_symmetric_then_clipped() -> None:
    q_hat = np.array([-0.20, 0.50, 1.10])
    out = deployed_intervals(q_hat, 0.25)
    assert np.allclose(out["lower_raw"], q_hat - 0.25)
    assert np.allclose(out["upper_raw"], q_hat + 0.25)
    assert np.allclose(out["lower"], np.clip(q_hat - 0.25, 0, 1))
    assert np.allclose(out["upper"], np.clip(q_hat + 0.25, 0, 1))
    assert np.all(out["lower"] <= out["upper"])


def test_threshold_is_deterministic_and_hashable() -> None:
    a = fit_deployed_conformal(_calibration(), alpha_levels=ALPHA_LEVELS)
    b = fit_deployed_conformal(_calibration(), alpha_levels=ALPHA_LEVELS)
    assert a["levels"] == b["levels"] and a["residual_sha256"] == b["residual_sha256"]


def test_unconditional_source_interval_uses_calibration_only() -> None:
    values = np.linspace(0.5, 0.9, 30)
    out = unconditional_source_interval(values, alpha_levels={"cp90": 0.10})
    assert out["source_population"] == CALIBRATION_POPULATION and out["n_calibration"] == 30
    level = out["cp90"] if "cp90" in out else out["levels"]["cp90"]
    assert level["lower_unclipped"] == pytest.approx(np.quantile(values, 0.05))
    assert level["upper_unclipped"] == pytest.approx(np.quantile(values, 0.95))


# ------------------------------------------------- section 18 firewall attacks

def test_attack_a_evaluation_labels_into_base_fit_is_refused() -> None:
    """A: hand M3_ID_EVAL q_true to the base fit."""
    f = _features(70, "M3_ID_EVAL")
    with pytest.raises(Task022Failure) as e:
        fit_qc_model(f, _target(f), CONFIG)
    assert e.value.code == "fit_population_not_qc_train"


def test_attack_b_evaluation_residuals_into_conformal_is_refused() -> None:
    """B: hand M3_OOD_EVAL residuals to the deployed threshold."""
    with pytest.raises(Task022Failure) as e:
        fit_deployed_conformal(_calibration(n=100, population="M3_OOD_EVAL"),
                               alpha_levels=ALPHA_LEVELS)
    assert e.value.code == "calibration_population_not_calibration"


def test_attack_c_unexpected_population_into_training_api_is_refused() -> None:
    """C: an invented population label."""
    f = _features(30, "SOMETHING_ELSE")
    with pytest.raises(Task022Failure) as e:
        fit_qc_model(f, _target(f), CONFIG)
    assert e.value.code == "fit_population_not_qc_train"


def test_attack_d_mutating_evaluation_labels_changes_nothing() -> None:
    """D: mutate evaluation q_true after fitting; model, alpha, threshold, q_hat unchanged."""
    model, train = _fitted()
    cal = _calibration()
    conformal = fit_deployed_conformal(cal, alpha_levels=ALPHA_LEVELS)
    evaluation = _features(100, "M3_OOD_EVAL", seed=9)
    q_hat_before = predict_q_hat(model["estimator"], evaluation)

    evaluation_labels = _target(evaluation, seed=13)
    evaluation_labels.loc[:] = 0.0                       # destroy every evaluation label

    model_after, _ = fit_qc_model(train, _target(train), CONFIG), None
    conformal_after = fit_deployed_conformal(cal, alpha_levels=ALPHA_LEVELS)
    q_hat_after = predict_q_hat(model["estimator"], evaluation)

    assert model_after["selected_alpha"] == model["selected_alpha"]
    assert model_after["ridge_coefficients"] == model["ridge_coefficients"]
    assert conformal_after["levels"] == conformal["levels"]
    assert np.array_equal(q_hat_before, q_hat_after)
