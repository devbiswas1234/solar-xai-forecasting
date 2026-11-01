"""
model_service.py — Model loading, inference pipeline, SHAP explanation engine, and ramp detection.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch
import shap

from typing import Dict, List, Any, Optional, Tuple

# Human-readable labels and descriptions for all 19 model features
FEATURE_METADATA = {
    "Pyranometer (Watt/m2)": {
        "label": "Solar Irradiance",
        "category": "Solar Radiation",
        "unit": "W/m²",
        "description": "Global Horizontal Irradiance (GHI) measured by the pyranometer."
    },
    "Wind Direction (Degree)": {
        "label": "Wind Direction",
        "category": "Atmospheric",
        "unit": "°",
        "description": "Wind direction in degrees, wrapped to [0, 360) range."
    },
    "Relative Humidity (%)": {
        "label": "Relative Humidity",
        "category": "Atmospheric",
        "unit": "%",
        "description": "Atmospheric relative humidity, calibrated to [0, 100]%."
    },
    "Temp1 (degC)": {
        "label": "Module Temp 1",
        "category": "Thermal",
        "unit": "°C",
        "description": "Surface temperature probe 1 on the photovoltaic panel."
    },
    "Temp2 (degC)": {
        "label": "Module Temp 2",
        "category": "Thermal",
        "unit": "°C",
        "description": "Surface temperature probe 2 on the photovoltaic panel."
    },
    "Wind Speed (m/s)": {
        "label": "Wind Speed",
        "category": "Atmospheric",
        "unit": "m/s",
        "description": "Ambient wind velocity aiding panel convective cooling."
    },
    "Air Temp (degC)": {
        "label": "Ambient Air Temp",
        "category": "Thermal",
        "unit": "°C",
        "description": "Ambient air temperature surrounding the installation."
    },
    "Hour": {
        "label": "Decimal Hour",
        "category": "Temporal",
        "unit": "h",
        "description": "Time of day in decimal format (e.g. 13.5 = 1:30 PM)."
    },
    "Hour_sin": {
        "label": "Diurnal Cycle (sin)",
        "category": "Cyclic Temporal",
        "unit": "rad",
        "description": "Sine cyclical transform of the 24-hour solar diurnal cycle."
    },
    "Hour_cos": {
        "label": "Diurnal Cycle (cos)",
        "category": "Cyclic Temporal",
        "unit": "rad",
        "description": "Cosine cyclical transform of the 24-hour solar diurnal cycle."
    },
    "Month_sin": {
        "label": "Seasonal Cycle (sin)",
        "category": "Cyclic Temporal",
        "unit": "rad",
        "description": "Sine cyclical transform representing annual seasonal solar altitude."
    },
    "Month_cos": {
        "label": "Seasonal Cycle (cos)",
        "category": "Cyclic Temporal",
        "unit": "rad",
        "description": "Cosine cyclical transform representing annual seasonal solar altitude."
    },
    "Temp_Diff": {
        "label": "Thermal Differential",
        "category": "Thermal",
        "unit": "°C",
        "description": "Module Temp minus Ambient Air Temp (Temp1 - Air Temp). Captures thermal efficiency derating."
    },
    "Pyranometer_lag1": {
        "label": "Irradiance Lag-1",
        "category": "Autoregressive",
        "unit": "W/m²",
        "description": "Solar irradiance measured at the previous time interval (t-1)."
    },
    "Power_Output_lag1": {
        "label": "Power Output Lag-1",
        "category": "Autoregressive",
        "unit": "W",
        "description": "Actual PV power output at the previous time interval (t-1)."
    },
    "Pyranometer_lag2": {
        "label": "Irradiance Lag-2",
        "category": "Autoregressive",
        "unit": "W/m²",
        "description": "Solar irradiance measured 2 steps prior (t-2)."
    },
    "Power_Output_lag2": {
        "label": "Power Output Lag-2",
        "category": "Autoregressive",
        "unit": "W",
        "description": "Actual PV power output measured 2 steps prior (t-2)."
    },
    "Rolling_Soiling_Factor": {
        "label": "Soiling Factor",
        "category": "Physical State",
        "unit": "ratio",
        "description": "Rolling median ratio of power to irradiance, capturing dust accumulation and panel degradation."
    },
    "Expected_Baseline_Power": {
        "label": "Expected Baseline Power",
        "category": "Physical Model",
        "unit": "W",
        "description": "Theoretical clean-panel output baseline: Pyranometer × Rolling_Soiling_Factor."
    },
}

FEATURE_ORDER = list(FEATURE_METADATA.keys())
RAMP_THRESHOLD_W = 3.98  # 90th percentile threshold from research paper


# LSTM Neural Architecture matching train_models.py
class LSTMForecaster(torch.nn.Module):
    def __init__(self, input_size=19, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                                 batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)


class ModelService:
    def __init__(self, project_root: Optional[str] = None):
        if project_root is None:
            self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        else:
            self.project_root = os.path.abspath(project_root)

        # Robust resolution for models and results directories
        candidate_model_dirs = [
            os.path.join(self.project_root, "models"),
            os.path.join(self.project_root, "solar-xai-forecasting", "models"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "solar-xai-forecasting", "models")),
        ]
        self.models_dir = next((d for d in candidate_model_dirs if os.path.isdir(d)), candidate_model_dirs[0])

        candidate_results_dirs = [
            os.path.join(self.project_root, "results"),
            os.path.join(self.project_root, "solar-xai-forecasting", "results"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "solar-xai-forecasting", "results")),
        ]
        self.results_dir = next((d for d in candidate_results_dirs if os.path.isdir(d)), candidate_results_dirs[0])

        self.xgb_model = None
        self.rf_model = None
        self.lr_model = None
        self.lstm_model = None
        self.scaler = None
        self.explainer = None
        self.metrics_df = None
        self.ramp_report_df = None
        self.test_X = None
        self.test_y = None

        self._load_all_artifacts()

    def _load_all_artifacts(self):
        print(f"[ModelService] Loading trained models from {self.models_dir} ...")

        # 1. XGBoost (Champion)
        xgb_path = os.path.join(self.models_dir, "xgboost_final.pkl")
        if os.path.exists(xgb_path):
            self.xgb_model = joblib.load(xgb_path)
            print("[ModelService] Loaded XGBoost Champion model.")
            try:
                # Pre-initialize TreeExplainer for instantaneous SHAP inference
                self.explainer = shap.TreeExplainer(self.xgb_model)
                print("[ModelService] Initialized SHAP TreeExplainer.")
            except Exception as e:
                print(f"[ModelService] Warning: SHAP explainer initialization error: {e}")

        # 2. Random Forest
        rf_path = os.path.join(self.models_dir, "random_forest_final.pkl")
        if os.path.exists(rf_path):
            self.rf_model = joblib.load(rf_path)
            print("[ModelService] Loaded Random Forest model.")

        # 3. Linear Regression & Scaler
        lr_path = os.path.join(self.models_dir, "linear_regression_final.pkl")
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        if os.path.exists(lr_path):
            self.lr_model = joblib.load(lr_path)
            print("[ModelService] Loaded Linear Regression model.")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print("[ModelService] Loaded StandardScaler.")

        # 4. PyTorch LSTM
        lstm_path = os.path.join(self.models_dir, "lstm_final.pth")
        if os.path.exists(lstm_path):
            try:
                self.lstm_model = LSTMForecaster(input_size=19, hidden_size=64, num_layers=2, dropout=0.2)
                self.lstm_model.load_state_dict(torch.load(lstm_path, map_location="cpu"))
                self.lstm_model.eval()
                print("[ModelService] Loaded PyTorch LSTM Forecaster.")
            except Exception as e:
                print(f"[ModelService] Warning: Could not load LSTM weights: {e}")

        # 5. Metrics & Reports
        metrics_path = os.path.join(self.results_dir, "final_model_metrics.csv")
        if os.path.exists(metrics_path):
            self.metrics_df = pd.read_csv(metrics_path)

        ramp_path = os.path.join(self.results_dir, "pv_ramp_events_report.csv")
        if os.path.exists(ramp_path):
            self.ramp_report_df = pd.read_csv(ramp_path)

        # 6. Test splits for historical replay
        test_x_path = os.path.join(self.results_dir, "X_test.csv")
        test_y_path = os.path.join(self.results_dir, "y_test.csv")
        if os.path.exists(test_x_path) and os.path.exists(test_y_path):
            self.test_X = pd.read_csv(test_x_path)
            self.test_y = pd.read_csv(test_y_path)
            print(f"[ModelService] Loaded test dataset ({len(self.test_X)} rows).")

    def build_feature_vector(self, raw_features: Any) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Transforms raw user input into the complete 19-dimensional feature dataframe.
        Derives cyclic time terms, thermal differentials, lags, and baseline power.
        """
        if hasattr(raw_features, "model_dump"):
            data = raw_features.model_dump()
        elif isinstance(raw_features, dict):
            data = raw_features.copy()
        else:
            data = dict(raw_features)

        # Core sensors
        pyranometer = float(data.get("pyranometer", 500.0))
        wind_dir = float(data.get("wind_direction", 180.0)) % 360.0
        humidity = float(np.clip(data.get("relative_humidity", 50.0), 0.0, 100.0))
        temp1 = float(data.get("temp1", 35.0))
        temp2 = float(data.get("temp2", temp1))
        wind_speed = float(max(0.0, data.get("wind_speed", 1.5)))
        air_temp = float(data.get("air_temp", 28.0))
        hour = float(data.get("hour", 12.0))
        month = int(data.get("month", 6))

        # Lag handling: If not supplied, generate physically reasonable autoregressive defaults
        pyr_lag1 = data.get("pyranometer_lag1")
        if pyr_lag1 is None:
            pyr_lag1 = pyranometer * 0.98
        else:
            pyr_lag1 = float(pyr_lag1)

        pyr_lag2 = data.get("pyranometer_lag2")
        if pyr_lag2 is None:
            pyr_lag2 = pyr_lag1 * 0.98
        else:
            pyr_lag2 = float(pyr_lag2)

        # Estimate historical power lag if not supplied (typical system scale ~0.015 W per W/m2)
        pwr_lag1 = data.get("power_output_lag1")
        if pwr_lag1 is None:
            pwr_lag1 = max(0.0, (pyr_lag1 / 1000.0) * 14.5)
        else:
            pwr_lag1 = float(pwr_lag1)

        pwr_lag2 = data.get("power_output_lag2")
        if pwr_lag2 is None:
            pwr_lag2 = max(0.0, pwr_lag1 * 0.98)
        else:
            pwr_lag2 = float(pwr_lag2)

        soiling_factor = float(data.get("rolling_soiling_factor", 0.0085))

        # Derived mathematical & physical features
        hour_sin = data.get("hour_sin") or float(np.sin(2 * np.pi * hour / 24.0))
        hour_cos = data.get("hour_cos") or float(np.cos(2 * np.pi * hour / 24.0))
        month_sin = data.get("month_sin") or float(np.sin(2 * np.pi * month / 12.0))
        month_cos = data.get("month_cos") or float(np.cos(2 * np.pi * month / 12.0))
        temp_diff = data.get("temp_diff") or float(temp1 - air_temp)
        expected_baseline = data.get("expected_baseline_power") or float(pyranometer * soiling_factor)

        feature_dict = {
            "Pyranometer (Watt/m2)": pyranometer,
            "Wind Direction (Degree)": wind_dir,
            "Relative Humidity (%)": humidity,
            "Temp1 (degC)": temp1,
            "Temp2 (degC)": temp2,
            "Wind Speed (m/s)": wind_speed,
            "Air Temp (degC)": air_temp,
            "Hour": hour,
            "Hour_sin": hour_sin,
            "Hour_cos": hour_cos,
            "Month_sin": month_sin,
            "Month_cos": month_cos,
            "Temp_Diff": temp_diff,
            "Pyranometer_lag1": pyr_lag1,
            "Power_Output_lag1": pwr_lag1,
            "Pyranometer_lag2": pyr_lag2,
            "Power_Output_lag2": pwr_lag2,
            "Rolling_Soiling_Factor": soiling_factor,
            "Expected_Baseline_Power": expected_baseline,
        }

        df = pd.DataFrame([feature_dict])[FEATURE_ORDER]
        return df, feature_dict

    def predict(self, raw_features: Any, selected_model: str = "xgboost") -> Dict[str, Any]:
        """Runs predictions across available models and analyzes ramp dynamics."""
        X_df, feature_dict = self.build_feature_vector(raw_features)
        predictions = {}

        # 1. XGBoost
        if self.xgb_model is not None:
            xgb_val = float(self.xgb_model.predict(X_df)[0])
            predictions["XGBoost (Champion)"] = max(0.0, round(xgb_val, 3))

        # 2. Random Forest
        if self.rf_model is not None:
            rf_val = float(self.rf_model.predict(X_df)[0])
            predictions["Random Forest"] = max(0.0, round(rf_val, 3))

        # 3. Linear Regression
        if self.lr_model is not None and self.scaler is not None:
            X_sc = self.scaler.transform(X_df)
            lr_val = float(self.lr_model.predict(X_sc)[0])
            predictions["Linear Regression"] = max(0.0, round(lr_val, 3))

        # 4. PyTorch LSTM
        if self.lstm_model is not None and self.scaler is not None:
            try:
                # Construct synthetic sequence of length 6 leading to the current point
                X_sc = self.scaler.transform(X_df)[0]
                seq = np.tile(X_sc, (1, 6, 1)).astype(np.float32)
                with torch.no_grad():
                    lstm_tensor = torch.tensor(seq)
                    lstm_val = float(self.lstm_model(lstm_tensor).item())
                    predictions["LSTM (Deep Learning)"] = max(0.0, round(lstm_val, 3))
            except Exception as e:
                print(f"[ModelService] LSTM predict error: {e}")

        # Choose primary forecast
        if selected_model == "random_forest" and "Random Forest" in predictions:
            primary_val = predictions["Random Forest"]
            champion_label = "Random Forest"
        elif selected_model == "linear_regression" and "Linear Regression" in predictions:
            primary_val = predictions["Linear Regression"]
            champion_label = "Linear Regression"
        elif selected_model == "lstm" and "LSTM (Deep Learning)" in predictions:
            primary_val = predictions["LSTM (Deep Learning)"]
            champion_label = "LSTM"
        else:
            primary_val = predictions.get("XGBoost (Champion)", 0.0)
            champion_label = "XGBoost (R² = 0.8687)"

        # Ramp analysis relative to t-1
        pwr_lag1 = feature_dict["Power_Output_lag1"]
        delta_w = round(primary_val - pwr_lag1, 3)
        abs_delta = abs(delta_w)
        is_critical = abs_delta >= RAMP_THRESHOLD_W

        if delta_w > 0.5:
            direction = "SURGE" if is_critical else "RAMP_UP"
        elif delta_w < -0.5:
            direction = "DROP" if is_critical else "RAMP_DOWN"
        else:
            direction = "STABLE"

        if is_critical:
            alert_level = "CRITICAL"
            if direction == "DROP":
                recommendation = (
                    f"CRITICAL RAMP-DOWN DETECTED (-{abs_delta:.2f} W). Severe solar drop likely caused by cloud transient. "
                    "Action: Dispatch fast-start spinning reserves and trigger battery storage (BESS) discharge."
                )
            else:
                recommendation = (
                    f"CRITICAL RAMP-UP DETECTED (+{abs_delta:.2f} W). Rapid solar surge. "
                    "Action: Activate inverter ramp-rate limiting or route surplus generation to battery charging."
                )
        elif abs_delta >= 2.0:
            alert_level = "MODERATE"
            recommendation = (
                f"Moderate generation rate change ({delta_w:+.2f} W). "
                "Normal automated generation control (AGC) will buffer the fluctuation."
            )
        else:
            alert_level = "NORMAL"
            recommendation = "Stable solar output within standard grid power envelope. No reserve intervention required."

        ramp_result = {
            "delta_w": delta_w,
            "abs_delta_w": abs_delta,
            "threshold_w": RAMP_THRESHOLD_W,
            "is_critical": is_critical,
            "direction": direction,
            "alert_level": alert_level,
            "recommendation": recommendation,
        }

        return {
            "champion_model": champion_label,
            "primary_prediction_w": primary_val,
            "multi_model_predictions": predictions,
            "ramp_analysis": ramp_result,
            "derived_features": {
                "Hour_sin": round(feature_dict["Hour_sin"], 4),
                "Hour_cos": round(feature_dict["Hour_cos"], 4),
                "Month_sin": round(feature_dict["Month_sin"], 4),
                "Month_cos": round(feature_dict["Month_cos"], 4),
                "Temp_Diff": round(feature_dict["Temp_Diff"], 2),
                "Expected_Baseline_Power": round(feature_dict["Expected_Baseline_Power"], 3),
            },
            "feature_vector": {k: round(v, 4) for k, v in feature_dict.items()},
        }

    def explain(self, raw_features: Any) -> Dict[str, Any]:
        """Computes local SHAP explanation for the provided input state using TreeExplainer."""
        if self.explainer is None:
            if self.xgb_model is None:
                raise RuntimeError("XGBoost model not loaded.")
            self.explainer = shap.TreeExplainer(self.xgb_model)

        X_df, _ = self.build_feature_vector(raw_features)
        shap_vals = self.explainer.shap_values(X_df)[0]
        base_val = self.explainer.expected_value
        if isinstance(base_val, np.ndarray):
            base_val = float(base_val[0])
        else:
            base_val = float(base_val)

        pred_val = float(self.xgb_model.predict(X_df)[0])

        contributions = []
        for feat_name, shap_v in zip(FEATURE_ORDER, shap_vals):
            feat_val = float(X_df[feat_name].iloc[0])
            s_val = float(shap_v)
            abs_s = abs(s_val)

            if s_val > 0.01:
                direction = "increases_forecast"
            elif s_val < -0.01:
                direction = "decreases_forecast"
            else:
                direction = "neutral"

            meta = FEATURE_METADATA.get(feat_name, {})
            label = meta.get("label", feat_name)

            contributions.append({
                "feature": feat_name,
                "feature_label": label,
                "feature_value": round(feat_val, 3),
                "shap_value": round(s_val, 4),
                "impact_direction": direction,
                "abs_shap": round(abs_s, 4),
            })

        # Sort contributions by absolute SHAP impact
        contributions.sort(key=lambda x: x["abs_shap"], reverse=True)

        top_pos = [c for c in contributions if c["impact_direction"] == "increases_forecast"]
        top_neg = [c for c in contributions if c["impact_direction"] == "decreases_forecast"]

        return {
            "base_value": round(base_val, 3),
            "predicted_value": round(max(0.0, pred_val), 3),
            "net_shap_sum": round(float(np.sum(shap_vals)), 4),
            "contributions": contributions,
            "top_positive": top_pos[:5],
            "top_negative": top_neg[:5],
        }

    def get_model_metrics(self) -> List[Dict[str, Any]]:
        """Returns the comparative metrics table for all 4 trained models."""
        if self.metrics_df is not None:
            return self.metrics_df.to_dict(orient="records")
        return [
            {"Model": "XGBoost", "MAE": 0.9626, "RMSE": 1.5184, "MAPE(%)": 18.5896, "R2": 0.8687},
            {"Model": "Random Forest", "MAE": 0.9110, "RMSE": 1.5211, "MAPE(%)": 17.3294, "R2": 0.8683},
            {"Model": "Linear Regression", "MAE": 1.2744, "RMSE": 1.8669, "MAPE(%)": 27.6766, "R2": 0.8016},
            {"Model": "LSTM", "MAE": 1.5334, "RMSE": 2.2575, "MAPE(%)": 30.5871, "R2": 0.7056},
        ]

    def get_test_sequence(self, limit: int = 150) -> Dict[str, Any]:
        """Provides an interactive test-set sequence comparing actual vs model predictions."""
        if self.test_X is None or self.test_y is None:
            return {"count": 0, "series": []}

        n = min(limit, len(self.test_X))
        X_slice = self.test_X.iloc[:n]
        y_slice = self.test_y.iloc[:n].values.squeeze()

        # Generate model curves
        xgb_preds = self.xgb_model.predict(X_slice) if self.xgb_model else np.zeros(n)
        rf_preds = self.rf_model.predict(X_slice) if self.rf_model else np.zeros(n)

        lr_preds = np.zeros(n)
        if self.lr_model and self.scaler:
            X_sc = self.scaler.transform(X_slice)
            lr_preds = self.lr_model.predict(X_sc)

        series = []
        for i in range(n):
            pyr = float(X_slice["Pyranometer (Watt/m2)"].iloc[i])
            actual = float(y_slice[i])
            prev_act = float(y_slice[i - 1]) if i > 0 else actual
            ramp = round(actual - prev_act, 2)
            is_ramp = abs(ramp) >= RAMP_THRESHOLD_W

            series.append({
                "step": i,
                "actual": round(actual, 2),
                "xgboost": round(float(np.clip(xgb_preds[i], 0, None)), 2),
                "random_forest": round(float(np.clip(rf_preds[i], 0, None)), 2),
                "linear_regression": round(float(np.clip(lr_preds[i], 0, None)), 2),
                "irradiance": round(pyr, 1),
                "ramp_w": ramp,
                "is_critical_ramp": is_ramp,
                "features": {col: round(float(X_slice[col].iloc[i]), 3) for col in FEATURE_ORDER},
            })

        return {"count": n, "series": series}

    def get_preset_scenarios(self) -> List[Dict[str, Any]]:
        """Returns realistic solar & atmospheric scenario presets for quick testing."""
        return [
            {
                "id": "summer-noon-optimal",
                "name": "Summer Noon Clear Peak",
                "category": "Optimal Conditions",
                "description": "High irradiance (1015 W/m²), moderate ambient temperature, clean panel. Demonstrates maximum power output and positive SHAP drivers.",
                "features": {
                    "pyranometer": 1015.0,
                    "wind_direction": 45.0,
                    "relative_humidity": 22.0,
                    "temp1": 46.5,
                    "temp2": 45.8,
                    "wind_speed": 2.4,
                    "air_temp": 32.0,
                    "hour": 12.5,
                    "month": 6,
                    "pyranometer_lag1": 990.0,
                    "power_output_lag1": 14.8,
                    "pyranometer_lag2": 950.0,
                    "power_output_lag2": 13.9,
                    "rolling_soiling_factor": 0.0088,
                }
            },
            {
                "id": "cloud-transient-ramp",
                "name": "Cloud Transient (Severe Ramp Down)",
                "category": "Grid Ramp Event",
                "description": "Sudden thick cloud transient causes irradiance to collapse from 980 to 280 W/m². Demonstrates critical PV Ramp alert and negative waterfall explanation.",
                "features": {
                    "pyranometer": 280.0,
                    "wind_direction": 110.0,
                    "relative_humidity": 45.0,
                    "temp1": 38.0,
                    "temp2": 37.5,
                    "wind_speed": 3.8,
                    "air_temp": 29.5,
                    "hour": 13.0,
                    "month": 7,
                    "pyranometer_lag1": 980.0,
                    "power_output_lag1": 14.5,
                    "pyranometer_lag2": 965.0,
                    "power_output_lag2": 14.2,
                    "rolling_soiling_factor": 0.0085,
                }
            },
            {
                "id": "dusty-soiled-panel",
                "name": "Dust Soiled Array (Efficiency Loss)",
                "category": "Degradation",
                "description": "Panel covered with dust accumulation over prolonged dry spell. Rolling soiling baseline factor is halved (0.0042 vs 0.0085), causing output drop despite good irradiance.",
                "features": {
                    "pyranometer": 880.0,
                    "wind_direction": 200.0,
                    "relative_humidity": 30.0,
                    "temp1": 52.0,
                    "temp2": 51.5,
                    "wind_speed": 1.0,
                    "air_temp": 35.0,
                    "hour": 11.5,
                    "month": 4,
                    "pyranometer_lag1": 850.0,
                    "power_output_lag1": 6.8,
                    "pyranometer_lag2": 820.0,
                    "power_output_lag2": 6.4,
                    "rolling_soiling_factor": 0.0042,
                }
            },
            {
                "id": "high-heat-derating",
                "name": "Extreme Heat Wave (Thermal Derating)",
                "category": "Thermal Stress",
                "description": "Module temperature exceeds 68°C under hot ambient conditions (43°C). Demonstrates how large Temp_Diff induces thermal resistance and efficiency penalties.",
                "features": {
                    "pyranometer": 920.0,
                    "wind_direction": 260.0,
                    "relative_humidity": 18.0,
                    "temp1": 68.5,
                    "temp2": 67.8,
                    "wind_speed": 0.8,
                    "air_temp": 43.0,
                    "hour": 14.0,
                    "month": 5,
                    "pyranometer_lag1": 940.0,
                    "power_output_lag1": 10.2,
                    "pyranometer_lag2": 950.0,
                    "power_output_lag2": 10.5,
                    "rolling_soiling_factor": 0.0080,
                }
            },
            {
                "id": "overcast-morning",
                "name": "Overcast Dawn / Morning",
                "category": "Low Generation",
                "description": "Early morning overcast skies (8:00 AM) with diffuse low irradiance (140 W/m²) and high humidity. Stable low generation without ramp risk.",
                "features": {
                    "pyranometer": 140.0,
                    "wind_direction": 310.0,
                    "relative_humidity": 78.0,
                    "temp1": 21.0,
                    "temp2": 20.8,
                    "wind_speed": 1.2,
                    "air_temp": 19.5,
                    "hour": 8.0,
                    "month": 1,
                    "pyranometer_lag1": 95.0,
                    "power_output_lag1": 0.8,
                    "pyranometer_lag2": 50.0,
                    "power_output_lag2": 0.3,
                    "rolling_soiling_factor": 0.0085,
                }
            },
        ]
