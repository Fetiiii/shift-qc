import numpy as np

from shiftqc.validation2.qc_model import fit_qc_ridge, predict_qc


def test_qc_ridge_fits_reproducibly_on_synthetic_inputs():
    rng = np.random.default_rng(2026)
    features = rng.normal(size=(10, 18))
    weights = np.linspace(-0.2, 0.2, 18)
    quality = np.clip(0.5 + features @ weights / 8.0, 0.0, 1.0)
    patient_ids = [f"synthetic-{index:02d}" for index in range(len(features))]
    roles = ["QC_TRAIN"] * len(features)

    first = fit_qc_ridge(features, quality, patient_ids, roles)
    second = fit_qc_ridge(features, quality, patient_ids, roles)
    first_prediction = predict_qc(first, features)
    second_prediction = predict_qc(second, features)

    assert first.best_params_ == second.best_params_
    assert first_prediction.shape == (10,)
    assert np.isfinite(first_prediction).all()
    np.testing.assert_allclose(first_prediction, second_prediction, rtol=0, atol=1e-12)
