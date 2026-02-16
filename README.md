# ML_Newton

Estimate mass from noisy force and acceleration measurements using Newton's second law:

`F = m * a`

This project simulates data, fits mass with two methods (closed-form least squares and numerical optimization), reports quality metrics, and visualizes the result.

## Why this project

The original script demonstrated the idea quickly. This version makes it reproducible, testable, and easier to run with different settings.

## Features

- Reproducible synthetic data generation (`numpy` random generator + seed)
- Two estimators for mass:
  - closed-form least-squares solution for `F = m*a`
  - optimizer-based solution using `scipy.optimize.minimize`
- Metrics: SSE, RMSE, and `R^2`
- Configurable CLI arguments
- Optional plot saving for reports/README usage
- Unit tests for core behavior

## Project structure

```text
ML_Newton/
|- main.py
|- tests/
|  |- test_main.py
|- requirements.txt
|- README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Default run

```bash
python main.py
```

### Headless run + save figure

```bash
python main.py --no-show --save-plot outputs/result.png
```

### Custom experiment

```bash
python main.py --true-mass 8 --samples 120 --noise-std 2.5 --seed 7 --initial-guess 0.5 --no-show
```

## Example output

```text
=== Mass Estimation Results ===
True mass:            5.0000
Closed-form estimate: 5.0447
Optimizer estimate:   5.0447
Absolute error:       0.0447
Optimizer SSE:        874.5116
Optimizer RMSE:       4.1821
Optimizer R^2:        0.9095
```

Values will vary with noise and seed.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Notes

- This is a simple 1-parameter linear model constrained through the origin.
- In this setup, the closed-form and optimizer estimates should closely match.
