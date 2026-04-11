"""Behavioral finance insights generator."""
from __future__ import annotations
from models import InsightItem, COLInfo
from simulation import run_monte_carlo

# Lower-COL retirement city alternatives for location optimization
_ALTERNATIVES = [
    ("Phoenix, AZ", 97),
    ("Asheville, NC", 94),
    ("Tucson, AZ", 88),
    ("Raleigh, NC", 102),
    ("San Antonio, TX", 91),
    ("Knoxville, TN", 83),
    ("Chattanooga, TN", 87),
    ("Greenville, SC", 92),
    ("Albuquerque, NM", 90),
    ("Oklahoma City, OK", 84),
]


def generate_insights(
    current_age: int,
    retirement_age: int,
    life_expectancy: int,
    current_savings: float,
    annual_contribution: float,
    annual_expenses: float,
    risk_level: int,
    inflation: float,
    n_simulations: int,
    probability_of_success: float,
    p50_final: float,
    p10_at_retirement: float,
    p50_at_retirement: float,
    col_info: COLInfo | None,
) -> tuple[list[InsightItem], float, float]:
    """
    Generate behavioral finance insights.

    Returns (insights_list, boost_probability, boost_portfolio_gain).
    """
    insights: list[InsightItem] = []

    # 1. Sustainability Check (always first if critical)
    sustainability = _sustainability_insight(
        probability_of_success, annual_contribution, annual_expenses, retirement_age, current_age
    )
    if sustainability:
        insights.append(sustainability)

    # 2. Sequence-of-Returns Risk (always included)
    seq_risk = _sequence_of_returns_insight(
        current_age, retirement_age, p50_at_retirement, p10_at_retirement
    )
    insights.append(seq_risk)

    # 3. Loss Aversion
    loss_aversion = _loss_aversion_insight(current_age, risk_level)
    if loss_aversion:
        insights.append(loss_aversion)

    # 4. Contribution Boost (re-run simulation with +$6000/yr)
    boost_prob, boost_gain = _run_contribution_boost(
        current_age, retirement_age, life_expectancy,
        current_savings, annual_contribution, annual_expenses,
        risk_level, inflation, n_simulations, probability_of_success, p50_final
    )
    contribution_boost = _contribution_boost_insight(
        probability_of_success, boost_prob, boost_gain
    )
    if contribution_boost:
        insights.append(contribution_boost)

    # 5. Location Optimization
    if col_info is not None:
        location = _location_insight(col_info)
        if location:
            insights.append(location)

    # Sort by impact priority
    _impact_order = {"high": 0, "medium": 1, "low": 2}
    insights.sort(key=lambda i: _impact_order[i.impact])

    return insights, boost_prob, boost_gain


def _sustainability_insight(
    prob: float,
    annual_contribution: float,
    annual_expenses: float,
    retirement_age: int,
    current_age: int,
) -> InsightItem | None:
    if prob >= 0.80:
        return InsightItem(
            id="sustainability",
            title="Retirement Plan on Track",
            body=(
                f"Your current plan shows a {prob*100:.0f}% probability of success. "
                "Continue your saving strategy and review annually. "
                "Consider increasing contributions when income grows to build an even larger safety margin."
            ),
            impact="low",
            icon="✅",
        )
    elif prob >= 0.70:
        return InsightItem(
            id="sustainability",
            title="Good Progress — Room to Improve",
            body=(
                f"Your success probability is {prob*100:.0f}% — solid, but below the recommended 80% threshold. "
                "Consider one of: increasing annual contributions, delaying retirement by 1–2 years, "
                "or reducing planned expenses by 10–15%."
            ),
            impact="medium",
            icon="📋",
        )
    else:
        years_to_retire = retirement_age - current_age
        extra_needed = annual_expenses * 0.15
        return InsightItem(
            id="sustainability",
            title="Retirement Plan Needs Attention",
            body=(
                f"Your success probability is {prob*100:.0f}% — below the 70% minimum threshold. "
                f"Recommended actions: (1) Increase annual contributions by ${extra_needed:,.0f}/yr, "
                f"(2) Delay retirement by 2–3 years, or (3) Reduce planned expenses by 15–20%. "
                f"You have {years_to_retire} years to close this gap."
            ),
            impact="high",
            icon="⚠️",
        )


def _sequence_of_returns_insight(
    current_age: int,
    retirement_age: int,
    p50_at_retirement: float,
    p10_at_retirement: float,
) -> InsightItem:
    years_to_retirement = retirement_age - current_age
    gap = max(0, p50_at_retirement - p10_at_retirement)
    impact = "high" if years_to_retirement <= 10 else "medium"

    if years_to_retirement <= 10:
        body = (
            f"With only {years_to_retirement} years to retirement, a market downturn now could "
            f"reduce your portfolio by ${gap:,.0f} below the median projection. "
            "Consider a 'bond tent' strategy: gradually shift 10–20% into bonds "
            "5 years before retirement, then slowly reduce bonds in early retirement. "
            "A 2-year cash buffer can also protect against forced selling in downturns."
        )
    else:
        body = (
            f"The difference between your median (${p50_at_retirement:,.0f}) and "
            f"worst-case 10th percentile (${p10_at_retirement:,.0f}) at retirement is ${gap:,.0f}. "
            "As you approach retirement, gradually shift toward more stable assets "
            "to reduce sequence risk. Avoid large equity allocations in the final 5 years before retirement."
        )

    return InsightItem(
        id="sequence_risk",
        title="Sequence-of-Returns Risk",
        body=body,
        impact=impact,
        icon="📉",
    )


def _loss_aversion_insight(current_age: int, risk_level: int) -> InsightItem | None:
    age_appropriate = max(1, min(10, round((110 - current_age) / 10)))
    if risk_level >= age_appropriate - 1:
        return None  # Risk level is appropriate or above

    gap = age_appropriate - risk_level
    return InsightItem(
        id="loss_aversion",
        title="Loss Aversion Check",
        body=(
            f"At age {current_age}, a typical portfolio would carry risk level {age_appropriate} "
            f"(based on the 110-minus-age rule). Your current setting of {risk_level} is "
            f"{gap} level{'s' if gap > 1 else ''} below the age-appropriate benchmark. "
            "This under-allocation to growth assets may significantly reduce your terminal wealth "
            "over a long horizon due to compounding effects. Consider a gradual risk increase."
        ),
        impact="medium",
        icon="🧠",
    )


def _run_contribution_boost(
    current_age, retirement_age, life_expectancy,
    current_savings, annual_contribution, annual_expenses,
    risk_level, inflation, n_simulations,
    original_prob, original_p50_final,
) -> tuple[float, float]:
    """Re-run simulation with +$6,000/yr contribution and return (new_prob, median_gain)."""
    try:
        boosted = run_monte_carlo(
            current_age=current_age,
            retirement_age=retirement_age,
            life_expectancy=life_expectancy,
            current_savings=current_savings,
            annual_contribution=annual_contribution + 6000,
            annual_expenses=annual_expenses,
            risk_level=risk_level,
            inflation=inflation,
            n_simulations=n_simulations,
        )
        boost_prob = boosted["probability_of_success"]
        new_p50_final = boosted["percentile_50"][-1] if boosted["percentile_50"] else 0
        boost_gain = max(0, new_p50_final - original_p50_final)
        return boost_prob, boost_gain
    except Exception:
        return original_prob, 0.0


def _contribution_boost_insight(
    original_prob: float,
    boost_prob: float,
    boost_gain: float,
) -> InsightItem | None:
    prob_gain = boost_prob - original_prob
    if prob_gain < 0.005 and boost_gain < 1000:
        return None  # Negligible impact — don't clutter insights

    return InsightItem(
        id="contribution_boost",
        title="Contribution Boost Opportunity",
        body=(
            f"Adding $500/month ($6,000/yr) to your annual contributions would "
            f"increase your success probability from {original_prob*100:.0f}% to {boost_prob*100:.0f}% "
            f"(+{prob_gain*100:.1f} percentage points) and add a projected "
            f"${boost_gain:,.0f} to your median retirement portfolio through compounding."
        ),
        impact="medium",
        icon="💰",
    )


def _location_insight(col_info: COLInfo) -> InsightItem | None:
    if col_info.target_col_index <= col_info.current_col_index:
        return None

    # Find cheaper alternatives than the target city
    cheaper = [
        (city, idx)
        for city, idx in _ALTERNATIVES
        if idx < col_info.target_col_index
    ][:3]

    if not cheaper:
        return None

    alt_text = ", ".join(f"{city} (index {idx})" for city, idx in cheaper)
    pct_more_expensive = (col_info.target_col_index / col_info.current_col_index - 1) * 100

    return InsightItem(
        id="location_optimization",
        title="Location Optimization Opportunity",
        body=(
            f"{col_info.target_city} has a COL index of {col_info.target_col_index:.0f}, "
            f"which is {pct_more_expensive:.0f}% more expensive than your current location "
            f"({col_info.current_city}, index {col_info.current_col_index:.0f}). "
            f"Consider lower-cost alternatives: {alt_text}. "
            "Retiring in a cheaper city can meaningfully reduce your required savings and improve success probability."
        ),
        impact="medium",
        icon="📍",
    )
