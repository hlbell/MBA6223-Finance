"""Monte Carlo retirement simulation engine."""
from __future__ import annotations
import numpy as np


# Return (mean, std_dev) interpolated linearly between anchor points
_ANCHORS = {
    1:  (0.040, 0.050),
    5:  (0.065, 0.100),
    10: (0.110, 0.180),
}


def _return_params(risk_level: int) -> tuple[float, float]:
    """Linear interpolation of return params from risk level anchors."""
    if risk_level <= 1:
        return _ANCHORS[1]
    if risk_level >= 10:
        return _ANCHORS[10]
    if risk_level <= 5:
        t = (risk_level - 1) / (5 - 1)
        mu = _ANCHORS[1][0] + t * (_ANCHORS[5][0] - _ANCHORS[1][0])
        sigma = _ANCHORS[1][1] + t * (_ANCHORS[5][1] - _ANCHORS[1][1])
    else:
        t = (risk_level - 5) / (10 - 5)
        mu = _ANCHORS[5][0] + t * (_ANCHORS[10][0] - _ANCHORS[5][0])
        sigma = _ANCHORS[5][1] + t * (_ANCHORS[10][1] - _ANCHORS[5][1])
    return mu, sigma


def run_monte_carlo(
    current_age: int,
    retirement_age: int,
    life_expectancy: int,
    current_savings: float,
    annual_contribution: float,
    annual_expenses: float,
    risk_level: int,
    inflation: float = 0.03,
    n_simulations: int = 1000,
    seed: int | None = None,
) -> dict:
    """
    Run Monte Carlo retirement simulation.

    Returns dict with:
      - probability_of_success
      - percentile_10, percentile_50, percentile_90  (full year time series)
      - years (list of ages)
      - histogram_bins, histogram_counts
      - success_threshold
      - median_at_retirement, p10_at_retirement, p90_at_retirement
    """
    rng = np.random.default_rng(seed)

    mu, sigma = _return_params(risk_level)
    # Real return for distribution phase (inflation-adjusted)
    real_mu = (1 + mu) / (1 + inflation) - 1

    acc_years = max(0, retirement_age - current_age)
    dist_years = max(0, life_expectancy - retirement_age)
    total_years = acc_years + dist_years

    if total_years == 0:
        return _empty_result(current_savings, current_age)

    # Generate all returns at once: shape (n_simulations, total_years)
    all_returns = rng.normal(
        loc=np.concatenate([
            np.full(acc_years, mu),
            np.full(dist_years, real_mu),
        ]),
        scale=sigma,
        size=(n_simulations, total_years),
    ).astype(np.float32)

    # Track portfolio at each year: shape (n_simulations, total_years)
    paths = np.zeros((n_simulations, total_years), dtype=np.float32)
    portfolio = np.full(n_simulations, float(current_savings), dtype=np.float64)

    # Accumulation phase
    for y in range(acc_years):
        portfolio = portfolio * (1 + all_returns[:, y]) + annual_contribution
        paths[:, y] = portfolio

    # Distribution phase
    for y in range(dist_years):
        withdrawal = annual_expenses * ((1 + inflation) ** y)
        portfolio = portfolio * (1 + all_returns[:, acc_years + y]) - withdrawal
        np.maximum(portfolio, 0, out=portfolio)
        paths[:, acc_years + y] = portfolio

    # Success: portfolio still > 0 at final year
    final_values = paths[:, -1]
    probability_of_success = float(np.mean(final_values > 0))

    # Percentile time series
    p10 = np.percentile(paths, 10, axis=0).tolist()
    p50 = np.percentile(paths, 50, axis=0).tolist()
    p90 = np.percentile(paths, 90, axis=0).tolist()

    # Retirement-year index
    ret_idx = acc_years - 1 if acc_years > 0 else 0
    median_at_ret = float(np.median(paths[:, ret_idx]))
    p10_at_ret = float(np.percentile(paths[:, ret_idx], 10))
    p90_at_ret = float(np.percentile(paths[:, ret_idx], 90))

    # Histogram of final portfolio values (clip at p99 to suppress outliers)
    clip_max = float(np.percentile(final_values, 99))
    hist_counts, hist_edges = np.histogram(
        np.clip(final_values, 0, clip_max),
        bins=40,
        range=(0, max(clip_max, 1.0)),
    )
    histogram_bins = hist_edges[:-1].tolist()
    histogram_counts = hist_counts.tolist()

    years = list(range(current_age, current_age + total_years))

    return {
        "probability_of_success": probability_of_success,
        "years": years,
        "percentile_10": p10,
        "percentile_50": p50,
        "percentile_90": p90,
        "histogram_bins": histogram_bins,
        "histogram_counts": histogram_counts,
        "success_threshold": 0.0,
        "median_at_retirement": median_at_ret,
        "p10_at_retirement": p10_at_ret,
        "p90_at_retirement": p90_at_ret,
        "years_of_projection": total_years,
    }


def _empty_result(current_savings: float, current_age: int) -> dict:
    return {
        "probability_of_success": 1.0,
        "years": [current_age],
        "percentile_10": [current_savings],
        "percentile_50": [current_savings],
        "percentile_90": [current_savings],
        "histogram_bins": [0.0],
        "histogram_counts": [1],
        "success_threshold": 0.0,
        "median_at_retirement": current_savings,
        "p10_at_retirement": current_savings,
        "p90_at_retirement": current_savings,
        "years_of_projection": 0,
    }
