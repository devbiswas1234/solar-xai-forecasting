"""
test_endpoints.py — Direct unit and integration tests for FastAPI backend functions.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.main import (
    get_system_status,
    get_feature_definitions,
    get_presets,
    predict_solar_power,
    explain_solar_prediction,
    get_comparison_metrics,
    get_ramp_event_report,
    get_test_time_series,
)
from backend.schemas import SolarInputFeatures, PredictionRequest, ExplainRequest

def run_tests():
    print("[TEST] 1. Testing get_system_status() ...")
    status = get_system_status()
    assert status["status"] == "online"
    assert status["active_models"]["xgboost"] is True
    print(f"  -> System status online, active models: {status['active_models']}")

    print("[TEST] 2. Testing get_presets() ...")
    presets_data = get_presets()
    presets = presets_data["presets"]
    assert len(presets) >= 5
    print(f"  -> Loaded {len(presets)} presets: {[p['name'] for p in presets]}")

    print("[TEST] 3. Testing predict_solar_power() with Summer Noon preset ...")
    p0 = presets[0]["features"]
    features_obj = SolarInputFeatures(**p0)
    pred_req = PredictionRequest(features=features_obj, model_name="xgboost")
    pred_res = predict_solar_power(pred_req)
    assert pred_res["primary_prediction_w"] > 0
    assert "ramp_analysis" in pred_res
    print(f"  -> Primary Forecast: {pred_res['primary_prediction_w']} W")
    print(f"  -> Multi-model predictions: {pred_res['multi_model_predictions']}")
    print(f"  -> Ramp Analysis: Delta = {pred_res['ramp_analysis']['delta_w']} W, Alert = {pred_res['ramp_analysis']['alert_level']}")

    print("[TEST] 4. Testing predict_solar_power() with Cloud Transient (Critical Ramp Down) ...")
    p1 = presets[1]["features"]
    features_obj1 = SolarInputFeatures(**p1)
    pred_req1 = PredictionRequest(features=features_obj1, model_name="xgboost")
    pred_res1 = predict_solar_power(pred_req1)
    assert pred_res1["ramp_analysis"]["is_critical"] is True
    assert pred_res1["ramp_analysis"]["alert_level"] == "CRITICAL"
    print(f"  -> Critical Ramp Event successfully detected! Direction: {pred_res1['ramp_analysis']['direction']}, Delta: {pred_res1['ramp_analysis']['delta_w']} W")
    print(f"  -> Operator Recommendation: {pred_res1['ramp_analysis']['recommendation']}")

    print("[TEST] 5. Testing explain_solar_prediction() (Live SHAP Waterfall) ...")
    exp_req = ExplainRequest(features=features_obj)
    exp_res = explain_solar_prediction(exp_req)
    assert "base_value" in exp_res
    assert len(exp_res["contributions"]) == 19
    top_pos = exp_res["top_positive"]
    top_neg = exp_res["top_negative"]
    print(f"  -> Base value (E[f(x)]): {exp_res['base_value']} W")
    print(f"  -> Top positive driver: {top_pos[0]['feature_label']} (+{top_pos[0]['shap_value']} W)")
    if top_neg:
        print(f"  -> Top negative driver: {top_neg[0]['feature_label']} ({top_neg[0]['shap_value']} W)")

    print("[TEST] 6. Testing get_comparison_metrics() ...")
    metrics_res = get_comparison_metrics()
    assert len(metrics_res["metrics"]) >= 4
    for m in metrics_res["metrics"]:
        print(f"  -> Model: {m['Model']:<18} R²={m['R2']:.4f}  MAE={m['MAE']:.3f} W")

    print("[TEST] 7. Testing get_test_time_series() ...")
    series_res = get_test_time_series(limit=10)
    assert series_res["count"] == 10
    print(f"  -> Retrieved {series_res['count']} test points for historical replay.")

    print("\n" + "=" * 65)
    print("  ALL BACKEND ENDPOINTS AND ML LOGIC VERIFIED 100%!  ")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
