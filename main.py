from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class ExperimentConfig:
    true_mass: float = 5.0
    min_acceleration: float = 0.0
    max_acceleration: float = 10.0
    sample_count: int = 50
    noise_std: float = 5.0
    seed: int = 42


def simulate_data(config: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic acceleration-force data using F = m*a with Gaussian noise."""
    rng = np.random.default_rng(config.seed)
    acceleration = np.linspace(
        config.min_acceleration, config.max_acceleration, config.sample_count
    )
    true_force = config.true_mass * acceleration
    noise = rng.normal(loc=0.0, scale=config.noise_std, size=acceleration.shape)
    observed_force = true_force + noise
    return acceleration, true_force, observed_force


def sum_squared_error(
    mass: float, acceleration: np.ndarray, observed_force: np.ndarray
) -> float:
    residuals = observed_force - mass * acceleration
    return float(np.dot(residuals, residuals))


def fit_mass_closed_form(acceleration: np.ndarray, observed_force: np.ndarray) -> float:
    """Least-squares estimate for y = m*x constrained through origin."""
    denominator = float(np.dot(acceleration, acceleration))
    if np.isclose(denominator, 0.0):
        raise ValueError("Acceleration values cannot all be zero.")
    numerator = float(np.dot(acceleration, observed_force))
    return numerator / denominator


def fit_mass_with_optimizer(
    acceleration: np.ndarray, observed_force: np.ndarray, initial_guess: float
) -> Tuple[float, float]:
    def objective(params: np.ndarray) -> float:
        return sum_squared_error(float(params[0]), acceleration, observed_force)

    result = minimize(
        objective,
        x0=np.array([initial_guess], dtype=float),
        method="L-BFGS-B",
        bounds=[(0.0, None)],
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return float(result.x[0]), float(result.fun)


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def r2_score(observed: np.ndarray, predicted: np.ndarray) -> float:
    total_variance = float(np.dot(observed - observed.mean(), observed - observed.mean()))
    if np.isclose(total_variance, 0.0):
        return 1.0
    unexplained = float(np.dot(observed - predicted, observed - predicted))
    return 1.0 - (unexplained / total_variance)


def plot_results(
    acceleration: np.ndarray,
    observed_force: np.ndarray,
    true_force: np.ndarray,
    optimized_force: np.ndarray,
    true_mass: float,
    optimized_mass: float,
    output_path: Path | None,
    show_plot: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        acceleration,
        observed_force,
        label="Observed force (noisy)",
        color="#1f77b4",
        alpha=0.7,
    )
    ax.plot(
        acceleration,
        true_force,
        label=f"True force (m={true_mass:.2f})",
        color="#2ca02c",
        linestyle="--",
    )
    ax.plot(
        acceleration,
        optimized_force,
        label=f"Estimated force (m={optimized_mass:.2f})",
        color="#d62728",
        linewidth=2,
    )
    ax.set_xlabel("Acceleration (m/s^2)")
    ax.set_ylabel("Force (N)")
    ax.set_title("Estimating mass from noisy force measurements")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"Saved plot to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate mass from noisy observations using F = m*a."
    )
    parser.add_argument("--true-mass", type=float, default=5.0, help="Ground-truth mass.")
    parser.add_argument(
        "--min-acceleration",
        type=float,
        default=0.0,
        help="Minimum acceleration used in simulation.",
    )
    parser.add_argument(
        "--max-acceleration",
        type=float,
        default=10.0,
        help="Maximum acceleration used in simulation.",
    )
    parser.add_argument(
        "--samples", type=int, default=50, help="Number of data points to simulate."
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=5.0,
        help="Standard deviation of Gaussian measurement noise.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--initial-guess",
        type=float,
        default=1.0,
        help="Initial optimizer guess for mass.",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        default=None,
        help="Optional image path to save the plot (for example outputs/result.png).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable the interactive plot window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 2:
        raise ValueError("--samples must be at least 2.")
    if args.noise_std < 0:
        raise ValueError("--noise-std must be non-negative.")
    if args.max_acceleration <= args.min_acceleration:
        raise ValueError("--max-acceleration must be greater than --min-acceleration.")
    if args.initial_guess < 0:
        raise ValueError("--initial-guess must be non-negative.")

    config = ExperimentConfig(
        true_mass=args.true_mass,
        min_acceleration=args.min_acceleration,
        max_acceleration=args.max_acceleration,
        sample_count=args.samples,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    acceleration, true_force, observed_force = simulate_data(config)

    closed_form_mass = fit_mass_closed_form(acceleration, observed_force)
    optimized_mass, optimizer_sse = fit_mass_with_optimizer(
        acceleration, observed_force, args.initial_guess
    )
    optimized_force = optimized_mass * acceleration

    print("=== Mass Estimation Results ===")
    print(f"True mass:            {config.true_mass:.4f}")
    print(f"Closed-form estimate: {closed_form_mass:.4f}")
    print(f"Optimizer estimate:   {optimized_mass:.4f}")
    print(f"Absolute error:       {abs(optimized_mass - config.true_mass):.4f}")
    print(f"Optimizer SSE:        {optimizer_sse:.4f}")
    print(f"Optimizer RMSE:       {rmse(observed_force, optimized_force):.4f}")
    print(f"Optimizer R^2:        {r2_score(observed_force, optimized_force):.4f}")

    plot_results(
        acceleration=acceleration,
        observed_force=observed_force,
        true_force=true_force,
        optimized_force=optimized_force,
        true_mass=config.true_mass,
        optimized_mass=optimized_mass,
        output_path=args.save_plot,
        show_plot=not args.no_show,
    )


if __name__ == "__main__":
    main()
