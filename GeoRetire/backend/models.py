from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class SimulationRequest(BaseModel):
    current_age: int = Field(..., ge=18, le=80)
    retirement_age: int = Field(..., ge=40, le=90)
    life_expectancy: int = Field(90, ge=50, le=110)
    current_savings: float = Field(..., ge=0)
    annual_contribution: float = Field(..., ge=0)
    annual_expenses: float = Field(..., ge=0)
    risk_level: int = Field(..., ge=1, le=10)
    inflation: float = Field(0.03, ge=0.0, le=0.15)
    n_simulations: int = Field(1000, ge=100, le=10000)
    current_zip: str = Field("", max_length=10)
    target_zip: str = Field("", max_length=10)


class COLInfo(BaseModel):
    current_city: str
    target_city: str
    current_col_index: float
    target_col_index: float
    adjusted_expenses: float
    col_ratio: float


class InsightItem(BaseModel):
    id: str
    title: str
    body: str
    impact: Literal["high", "medium", "low"]
    icon: str


class SimulationResponse(BaseModel):
    probability_of_success: float
    years: list[int]
    percentile_10: list[float]
    percentile_50: list[float]
    percentile_90: list[float]
    histogram_bins: list[float]
    histogram_counts: list[int]
    success_threshold: float
    median_at_retirement: float
    p10_at_retirement: float
    p90_at_retirement: float
    years_of_projection: int
    col_info: COLInfo | None
    insights: list[InsightItem]
    boost_probability: float
    boost_portfolio_gain: float


class CityResponse(BaseModel):
    zip: str
    city: str
    state: str
    col_index: float | None


class HealthResponse(BaseModel):
    status: str
