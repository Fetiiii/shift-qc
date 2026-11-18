"""TASK-052 / EXP-0007: synthetic design and estimator-equivalence gate."""

from __future__ import annotations

import numpy as np
import pytest

from shiftqc.confirmatory.r5 import compute_r5
from shiftqc.confirmatory.r8 import compute_r8
from shiftqc.exploratory.synthetic_r8 import (
    Q_MEAN,
    Q_SD,
    RECALIBRATORS,
    SyntheticDesignError,
    frozen_folds,
    patient_ids,
    point_estimates,
    simulate_domain,
)


def _dataset(scenario: str = "A", rho: float = 0.5, n_id: int = 66, n_ood: int = 245):
    rng = np.random.default_rng(7)
    id_q_hat, id_q = simulate_domain(n_id, rho, rng, scenario=scenario, out_of_domain=False)
    ood_q_hat, ood_q = simulate_domain(n_ood, rho, rng, scenario=scenario, out_of_domain=True)
    return id_q_hat, id_q, ood_q_hat, ood_q


def test_point_estimates_equal_the_frozen_endpoint_functions() -> None:
    """Protocol Section 7 gate: the driver returns the registered point estimates."""
    id_q_hat, id_q, ood_q_hat, ood_q = _dataset()
    id_ids, ood_ids = patient_ids(66, "i"), patient_ids(245, "o")
    mine = point_estimates(id_q_hat, id_q, ood_q_hat, ood_q,
                           frozen_folds(66, "i"), frozen_folds(245, "o"))
    r5 = compute_r5(id_ids, id_q_hat, id_ids, id_q, ood_ids, ood_q_hat, ood_ids, ood_q,
                    bootstrap_B=2)
    assert mine["R5.S_ID"] == r5["S_ID"]
    assert mine["R5.S_OOD"] == r5["S_OOD"]
    assert mine["R5.Delta"] == r5["Delta5"]
    for name in RECALIBRATORS:
        r8 = compute_r8(id_ids, id_q_hat, id_ids, id_q, ood_ids, ood_q_hat, ood_ids, ood_q,
                        recalibrator=name, bootstrap_B=2)
        assert mine[f"R8.{name}.S_ID"] == r8["S_ID"]
        assert mine[f"R8.{name}.S_OOD"] == r8["S_OOD"]
        assert mine[f"R8.{name}.Delta"] == r8["Delta8"]


def test_moments_and_population_correlation() -> None:
    rng = np.random.default_rng(0)
    q_hat, q = simulate_domain(400_000, 0.5, rng, scenario="A", out_of_domain=False)
    assert np.mean(q) == pytest.approx(Q_MEAN, abs=2e-3)
    assert np.std(q) == pytest.approx(Q_SD, abs=2e-3)
    assert np.corrcoef(q_hat, q)[0, 1] == pytest.approx(0.5, abs=5e-3)
    assert np.all((q >= 0.0) & (q <= 1.0))


def test_scenarios_differ_only_out_of_domain() -> None:
    for scenario in "BCDEF":
        a = simulate_domain(50, 0.5, np.random.default_rng(1), scenario="A", out_of_domain=False)
        b = simulate_domain(50, 0.5, np.random.default_rng(1), scenario=scenario,
                            out_of_domain=False)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_scenario_transforms() -> None:
    def pair(scenario: str):
        return simulate_domain(200_000, 0.5, np.random.default_rng(3), scenario=scenario,
                               out_of_domain=True)
    base_hat, base_q = simulate_domain(200_000, 0.5, np.random.default_rng(3), scenario="A",
                                       out_of_domain=True)
    b_hat, b_q = pair("B")
    assert np.allclose(b_hat, 0.3 + 0.5 * base_hat) and np.array_equal(b_q, base_q)
    c_hat, c_q = pair("C")
    assert np.corrcoef(c_hat, c_q)[0, 1] == pytest.approx(-0.5, abs=1e-2)
    e_hat, e_q = pair("E")
    assert abs(np.corrcoef(e_hat, e_q)[0, 1]) < 1e-2
    _f_hat, f_q = pair("F")
    assert np.std(f_q) > np.std(base_q) * 1.5
    d_hat, _d_q = pair("D")
    assert np.all(np.abs(d_hat - Q_MEAN) <= Q_SD)


def test_noninformative_cell_has_spread_and_no_information() -> None:
    q_hat, q = simulate_domain(200_000, 0.0, np.random.default_rng(5), scenario="A",
                               out_of_domain=False)
    assert np.std(q_hat) == pytest.approx(0.5 * Q_SD, rel=2e-2)
    assert abs(np.corrcoef(q_hat, q)[0, 1]) < 1e-2


def test_design_violations_fail() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(SyntheticDesignError):
        simulate_domain(10, 0.5, rng, scenario="Z", out_of_domain=False)
    with pytest.raises(SyntheticDesignError):
        simulate_domain(10, 0.0, rng, scenario="B", out_of_domain=True)
    with pytest.raises(SyntheticDesignError):
        simulate_domain(10, 1.0, rng, scenario="A", out_of_domain=False)


def test_folds_are_fixed_per_sample_size() -> None:
    assert np.array_equal(frozen_folds(66, "i"), frozen_folds(66, "i"))
    assert frozen_folds(66, "i").shape == (10, 66)
