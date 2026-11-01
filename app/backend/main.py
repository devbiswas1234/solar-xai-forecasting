"""
main.py — FastAPI Application for Solar Energy Forecasting and Explainable AI (XAI).
"""

import os
import sys
import tempfile

# Prevent matplotlib cache warning
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_solar")

from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import io

from .schemas import (
    SolarInputFeatures,
    PredictionRequest,
    PredictionResponse,
    ExplainRequest,
    ExplainResponse,
    BatchPredictRequest,
    BatchPredictResponse,
)
from .model_service import ModelService, FEATURE_METADATA, FEATURE_ORDER, RAMP_THRESHOLD_W

app = FastAPI(
    title="Solar Energy Forecasting & XAI Grid Intelligence API",
    description="Full-stack API serving machine learning forecasters (XGBoost, Random Forest, LR, LSTM), real-time SHAP explainability, and PV ramp event detection for solar grid management.",
    version="1.0.0",
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML Model Service
model_service = ModelService()

# Directory paths
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(BACKEND_DIR)
FRONTEND_DIR = os.path.join(APP_DIR, "frontend")
RESULTS_DIR = model_service.results_dir

# Mount static directories
if os.path.exists(RESULTS_DIR):
    app.mount("/research-artifacts", StaticFiles(directory=RESULTS_DIR), name="research-artifacts")

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ── Root & Static UI ────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Solar Forecasting & XAI API is running. Access /docs for interactive documentation."}


# ── System Health & Metadata ────────────────────────────────────────────────
@app.get("/api/status", summary="System Health & Active Models")
def get_system_status():
    return {
        "status": "online",
        "service": "Solar XAI Forecasting & Grid Management Engine",
        "active_models": {
            "xgboost": model_service.xgb_model is not None,
            "random_forest": model_service.rf_model is not None,
            "linear_regression": model_service.lr_model is not None,
            "lstm": model_service.lstm_model is not None,
        },
        "shap_explainer_ready": model_service.explainer is not None,
        "test_dataset_loaded": model_service.test_X is not None,
        "ramp_threshold_w": RAMP_THRESHOLD_W,
    }


@app.get("/api/features", summary="List Model Feature Metadata")
def get_feature_definitions():
    return {
        "feature_order": FEATURE_ORDER,
        "features": FEATURE_METADATA,
        "total_features": len(FEATURE_ORDER),
    }


@app.get("/api/presets", summary="Retrieve Preset Operational Scenarios")
def get_presets():
    return {
        "presets": model_service.get_preset_scenarios()
    }


# ── Core Forecasting & XAI Endpoints ─────────────────────────────────────────
@app.post("/api/predict", response_model=PredictionResponse, summary="Predict Solar Power Output & Ramp Dynamics")
def predict_solar_power(payload: PredictionRequest):
    try:
        result = model_service.predict(payload.features, selected_model=payload.model_name or "xgboost")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/explain", response_model=ExplainResponse, summary="Generate Real-time SHAP Waterfall Explanation")
def explain_solar_prediction(payload: ExplainRequest):
    try:
        explanation = model_service.explain(payload.features)
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP explanation error: {str(e)}")


@app.get("/api/metrics", summary="Model Comparison Performance Metrics")
def get_comparison_metrics():
    metrics = model_service.get_model_metrics()
    return {
        "champion_model": "XGBoost",
        "champion_r2": 0.8687,
        "champion_mae_w": 0.9626,
        "metrics": metrics,
        "split_method": "Monthly-Block Chronological Split (70% Train, 15% Val, 15% Test)",
        "features_excluded_for_leakage": ["Voltage (V)", "Current (A)", "Power_Output (W)"],
    }


# ── Grid Ramp Management & Historical Replay ────────────────────────────────
@app.get("/api/ramp-report", summary="PV Ramp Event Statistics & Test Log")
def get_ramp_event_report(limit: int = Query(50, ge=1, le=500)):
    if model_service.ramp_report_df is None:
        return {"total_events": 0, "critical_events": 0, "events": []}

    df = model_service.ramp_report_df
    critical_df = df[df["Critical_Event"] == True]

    records = df.head(limit).to_dict(orient="records")
    return {
        "threshold_w": RAMP_THRESHOLD_W,
        "total_test_steps": len(df),
        "total_critical_events": int(len(critical_df)),
        "detection_precision": 0.710,
        "detection_recall": 0.510,
        "detection_f1": 0.594,
        "sample_records": records,
    }


@app.get("/api/test-sequence", summary="Historical Test Set Replay Time Series")
def get_test_time_series(limit: int = Query(150, ge=10, le=500)):
    return model_service.get_test_sequence(limit=limit)


# ── Batch Ingestion ──────────────────────────────────────────────────────────
@app.post("/api/batch-predict", response_model=BatchPredictResponse, summary="Batch Prediction on Multiple Points")
def batch_predict(payload: BatchPredictRequest):
    predictions = []
    crit_count = 0
    preds_vals = []

    for item in payload.rows:
        res = model_service.predict(item, selected_model=payload.model_name or "xgboost")
        p_val = res["primary_prediction_w"]
        preds_vals.append(p_val)
        if res["ramp_analysis"]["is_critical"]:
            crit_count += 1
        predictions.append({
            "predicted_power_w": p_val,
            "ramp_delta_w": res["ramp_analysis"]["delta_w"],
            "alert_level": res["ramp_analysis"]["alert_level"],
            "multi_model": res["multi_model_predictions"],
        })

    return {
        "count": len(predictions),
        "predictions": predictions,
        "critical_ramps_detected": crit_count,
        "mean_predicted_w": round(float(np.mean(preds_vals)), 3) if preds_vals else 0.0,
        "max_predicted_w": round(float(np.max(preds_vals)), 3) if preds_vals else 0.0,
        "min_predicted_w": round(float(np.min(preds_vals)), 3) if preds_vals else 0.0,
    }


@app.post("/api/upload-csv", summary="Upload CSV for Batch Prediction & Analysis")
async def upload_csv_forecast(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Check required or fuzzy match
        results = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            # Map columns to schema
            mapped = {
                "pyranometer": float(row_dict.get("Pyranometer (Watt/m2)", row_dict.get("pyranometer", 500.0))),
                "wind_direction": float(row_dict.get("Wind Direction (Degree)", row_dict.get("wind_direction", 180.0))),
                "relative_humidity": float(row_dict.get("Relative Humidity (%)", row_dict.get("relative_humidity", 50.0))),
                "temp1": float(row_dict.get("Temp1 (degC)", row_dict.get("temp1", 35.0))),
                "temp2": float(row_dict.get("Temp2 (degC)", row_dict.get("temp2", 35.0))),
                "wind_speed": float(row_dict.get("Wind Speed (m/s)", row_dict.get("wind_speed", 1.5))),
                "air_temp": float(row_dict.get("Air Temp (degC)", row_dict.get("air_temp", 28.0))),
                "hour": float(row_dict.get("Hour", row_dict.get("hour", 12.0))),
                "month": int(row_dict.get("month", 6)),
                "pyranometer_lag1": row_dict.get("Pyranometer_lag1"),
                "power_output_lag1": row_dict.get("Power_Output_lag1"),
                "pyranometer_lag2": row_dict.get("Pyranometer_lag2"),
                "power_output_lag2": row_dict.get("Power_Output_lag2"),
                "rolling_soiling_factor": float(row_dict.get("Rolling_Soiling_Factor", 0.0085)),
            }
            res = model_service.predict(mapped, selected_model="xgboost")
            results.append({
                "predicted_w": res["primary_prediction_w"],
                "ramp_w": res["ramp_analysis"]["delta_w"],
                "alert": res["ramp_analysis"]["alert_level"],
            })

        return {
            "rows_processed": len(results),
            "results": results[:100],  # Return preview
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {str(e)}")


def run():
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    run()
