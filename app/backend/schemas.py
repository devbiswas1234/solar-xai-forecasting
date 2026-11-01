"""
schemas.py — Pydantic models for solar energy forecasting and explainable AI API.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class SolarInputFeatures(BaseModel):
    # Primary Meteorological & Sensor Inputs
    pyranometer: float = Field(..., description="Solar Irradiance / Pyranometer reading (Watt/m2)", ge=0.0, le=1600.0)
    wind_direction: float = Field(180.0, description="Wind Direction (Degree) [0, 360)", ge=0.0, le=360.0)
    relative_humidity: float = Field(50.0, description="Relative Humidity (%) [0, 100]", ge=0.0, le=100.0)
    temp1: float = Field(35.0, description="Solar Panel Module Temperature 1 (degC)", ge=-10.0, le=95.0)
    temp2: float = Field(35.0, description="Solar Panel Module Temperature 2 (degC)", ge=-10.0, le=95.0)
    wind_speed: float = Field(1.5, description="Wind Speed (m/s)", ge=0.0, le=40.0)
    air_temp: float = Field(28.0, description="Ambient Air Temperature (degC)", ge=-10.0, le=60.0)
    hour: float = Field(12.0, description="Decimal Hour of day (e.g. 12.5 = 12:30)", ge=0.0, le=24.0)
    month: int = Field(6, description="Month of year (1-12)", ge=1, le=12)

    # Autoregressive Lag Features
    pyranometer_lag1: Optional[float] = Field(None, description="Pyranometer reading at t-1 (Watt/m2)")
    power_output_lag1: Optional[float] = Field(None, description="Actual Power Output at t-1 (Watts)")
    pyranometer_lag2: Optional[float] = Field(None, description="Pyranometer reading at t-2 (Watt/m2)")
    power_output_lag2: Optional[float] = Field(None, description="Actual Power Output at t-2 (Watts)")

    # Soiling baseline proxy
    rolling_soiling_factor: Optional[float] = Field(0.0085, description="Rolling soiling degradation factor (median ratio)")

    # Optional pre-computed derived features
    hour_sin: Optional[float] = None
    hour_cos: Optional[float] = None
    month_sin: Optional[float] = None
    month_cos: Optional[float] = None
    temp_diff: Optional[float] = None
    expected_baseline_power: Optional[float] = None


class PredictionRequest(BaseModel):
    features: SolarInputFeatures
    model_name: Optional[str] = Field("xgboost", description="Model to use: 'xgboost', 'random_forest', 'linear_regression', 'lstm', or 'all'")


class RampAnalysisResult(BaseModel):
    delta_w: float = Field(..., description="Change in power output relative to t-1 (Watts)")
    abs_delta_w: float = Field(..., description="Absolute change in power output (Watts)")
    threshold_w: float = Field(3.98, description="Critical ramp threshold (90th percentile = 3.98 W)")
    is_critical: bool = Field(..., description="Whether this change constitutes a critical ramp event")
    direction: str = Field(..., description="'SURGE' (ramp up), 'DROP' (ramp down), or 'STABLE'")
    alert_level: str = Field(..., description="'NORMAL', 'MODERATE', or 'CRITICAL'")
    recommendation: str = Field(..., description="Grid stabilization recommendation for system operators")


class PredictionResponse(BaseModel):
    champion_model: str
    primary_prediction_w: float
    multi_model_predictions: Dict[str, float]
    ramp_analysis: RampAnalysisResult
    derived_features: Dict[str, float]
    feature_vector: Dict[str, float]


class SHAPContribution(BaseModel):
    feature: str
    feature_label: str
    feature_value: float
    shap_value: float
    impact_direction: str  # "increases_forecast" | "decreases_forecast" | "neutral"
    abs_shap: float


class ExplainRequest(BaseModel):
    features: SolarInputFeatures


class ExplainResponse(BaseModel):
    base_value: float = Field(..., description="Expected baseline prediction (E[f(x)]) before feature impacts")
    predicted_value: float = Field(..., description="Final model prediction (Watts)")
    net_shap_sum: float = Field(..., description="Sum of all SHAP values")
    contributions: List[SHAPContribution]
    top_positive: List[SHAPContribution]
    top_negative: List[SHAPContribution]


class BatchPredictRequest(BaseModel):
    rows: List[SolarInputFeatures]
    model_name: Optional[str] = "xgboost"


class BatchPredictResponse(BaseModel):
    count: int
    predictions: List[Dict[str, Any]]
    critical_ramps_detected: int
    mean_predicted_w: float
    max_predicted_w: float
    min_predicted_w: float


class PresetScenario(BaseModel):
    id: str
    name: str
    category: str
    description: str
    features: Dict[str, Any]
