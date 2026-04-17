"""GeoRetire FastAPI backend."""
from __future__ import annotations
import sys
import os

# Allow imports from the backend directory itself
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from models import (
    SimulationRequest,
    SimulationResponse,
    CityResponse,
    COLInfo,
    HealthResponse,
)
from simulation import run_monte_carlo
from col_service import lookup_both_zips, lookup_zip_full
from insights import generate_insights

app = FastAPI(title="GeoRetire API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health():
    return {"status": "ok"}


@app.get("/api/city-from-zip/{zip_code}", response_model=CityResponse)
async def city_from_zip(zip_code: str):
    result = await lookup_zip_full(zip_code)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Zip code {zip_code} not found")
    city, state, col_index = result
    return CityResponse(zip=zip_code, city=city, state=state, col_index=col_index)


@app.post("/api/simulate", response_model=SimulationResponse)
async def simulate(req: SimulationRequest):
    if req.current_age >= req.retirement_age:
        raise HTTPException(
            status_code=422,
            detail="current_age must be less than retirement_age",
        )
    if req.life_expectancy <= req.retirement_age:
        # Degenerate case: already past life expectancy — return 100% success
        req = req.model_copy(update={"life_expectancy": req.retirement_age + 1})

    # ── COL lookup (concurrent) ──────────────────────────────────────
    col_info: COLInfo | None = None
    if req.current_zip and req.target_zip:
        current_result, target_result = await lookup_both_zips(
            req.current_zip, req.target_zip
        )
        if current_result is not None and target_result is not None:
            c_city, c_state, c_col = current_result
            t_city, t_state, t_col = target_result
            # Fall back to baseline (100) when Teleport/table doesn't cover the city
            c_col = c_col if c_col is not None else 100.0
            t_col = t_col if t_col is not None else 100.0
            col_ratio = t_col / c_col if c_col > 0 else 1.0
            adjusted_expenses = req.annual_expenses * col_ratio
            col_info = COLInfo(
                current_city=f"{c_city}, {c_state}",
                target_city=f"{t_city}, {t_state}",
                current_col_index=c_col,
                target_col_index=t_col,
                adjusted_expenses=adjusted_expenses,
                col_ratio=col_ratio,
            )

    # Use COL-adjusted expenses for simulation if available
    effective_expenses = col_info.adjusted_expenses if col_info else req.annual_expenses

    # ── Monte Carlo simulation ───────────────────────────────────────
    sim = run_monte_carlo(
        current_age=req.current_age,
        retirement_age=req.retirement_age,
        life_expectancy=req.life_expectancy,
        current_savings=req.current_savings,
        annual_contribution=req.annual_contribution,
        annual_expenses=effective_expenses,
        risk_level=req.risk_level,
        inflation=req.inflation,
        n_simulations=req.n_simulations,
    )

    # ── Behavioral finance insights ──────────────────────────────────
    p50_final = sim["percentile_50"][-1] if sim["percentile_50"] else 0.0

    insights, boost_prob, boost_gain = generate_insights(
        current_age=req.current_age,
        retirement_age=req.retirement_age,
        life_expectancy=req.life_expectancy,
        current_savings=req.current_savings,
        annual_contribution=req.annual_contribution,
        annual_expenses=effective_expenses,
        risk_level=req.risk_level,
        inflation=req.inflation,
        n_simulations=req.n_simulations,
        probability_of_success=sim["probability_of_success"],
        p50_final=p50_final,
        p10_at_retirement=sim["p10_at_retirement"],
        p50_at_retirement=sim["median_at_retirement"],
        col_info=col_info,
    )

    return SimulationResponse(
        probability_of_success=sim["probability_of_success"],
        years=sim["years"],
        percentile_10=sim["percentile_10"],
        percentile_50=sim["percentile_50"],
        percentile_90=sim["percentile_90"],
        histogram_bins=sim["histogram_bins"],
        histogram_counts=sim["histogram_counts"],
        success_threshold=sim["success_threshold"],
        median_at_retirement=sim["median_at_retirement"],
        p10_at_retirement=sim["p10_at_retirement"],
        p90_at_retirement=sim["p90_at_retirement"],
        years_of_projection=sim["years_of_projection"],
        col_info=col_info,
        insights=insights,
        boost_probability=boost_prob,
        boost_portfolio_gain=boost_gain,
    )


# ── Serve frontend static files ──────────────────────────────────────
# Mount AFTER API routes so /api/* is matched first.
# When frozen by PyInstaller, static files are extracted to sys._MEIPASS.
if getattr(sys, "frozen", False):
    _frontend_dir = sys._MEIPASS
else:
    _frontend_dir = os.path.join(os.path.dirname(__file__), "..")
app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="static")
