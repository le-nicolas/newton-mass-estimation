import unittest

import numpy as np

from main import (
    ExperimentConfig,
    fit_mass_closed_form,
    fit_mass_with_optimizer,
    r2_score,
    simulate_data,
)


class MassEstimationTests(unittest.TestCase):
    def test_simulation_is_reproducible_for_same_seed(self) -> None:
        config = ExperimentConfig(seed=123)
        accel_1, true_1, observed_1 = simulate_data(config)
        accel_2, true_2, observed_2 = simulate_data(config)
        self.assertTrue(np.array_equal(accel_1, accel_2))
        self.assertTrue(np.array_equal(true_1, true_2))
        self.assertTrue(np.array_equal(observed_1, observed_2))

    def test_closed_form_recovers_true_mass_when_noise_is_zero(self) -> None:
        config = ExperimentConfig(true_mass=3.7, noise_std=0.0, seed=99)
        acceleration, _, observed_force = simulate_data(config)
        estimate = fit_mass_closed_form(acceleration, observed_force)
        self.assertAlmostEqual(estimate, config.true_mass, places=12)

    def test_optimizer_recovers_true_mass_when_noise_is_zero(self) -> None:
        config = ExperimentConfig(true_mass=7.2, noise_std=0.0, seed=11)
        acceleration, _, observed_force = simulate_data(config)
        estimate, _ = fit_mass_with_optimizer(
            acceleration=acceleration,
            observed_force=observed_force,
            initial_guess=0.3,
        )
        self.assertAlmostEqual(estimate, config.true_mass, places=7)

    def test_r2_is_one_for_perfect_predictions(self) -> None:
        observed = np.array([2.0, 4.0, 6.0, 8.0], dtype=float)
        predicted = observed.copy()
        self.assertAlmostEqual(r2_score(observed, predicted), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
