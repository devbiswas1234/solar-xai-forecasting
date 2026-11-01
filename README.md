# Enhancing Solar Energy Forecasting Using Explainable AI (XAI)

A complete machine learning research pipeline and interactive full-stack web application for short-term solar power output forecasting, with real-time SHAP-based explainability and critical Photovoltaic (PV) ramp event detection for grid management.

---

## 🌟 Key Capabilities

- **Interactive Full-Stack Web Application:** Modern dark-glassmorphic dashboard built with FastAPI, Tailwind CSS, Chart.js, and Lucide Icons.
- **Champion Machine Learning Model:** XGBoost achieving **$R^2 = 0.8687$**, **MAE = 0.96 W**, and **RMSE = 1.52 W**.
- **Real-Time Explainable AI (XAI):** Instantaneous local SHAP attribution waterfall chart powered by in-memory `TreeExplainer`.
- **PV Ramp Event Detection:** 90th percentile critical ramp threshold (3.98 W) alerting grid operators to cloud transients with automated dispatch recommendations.
- **Leakage-Free Validation:** Strict omission of Voltage ($V$) and Current ($I$) features (preventing $P = V \times I$ shortcut leakage) and Monthly-Block Chronological Splitting (70% Train, 15% Val, 15% Test).

---

## 🚀 Quick Start

### 1. Run the Interactive Web Dashboard

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the server
python run_app.py
```

- **Web Dashboard:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive API Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 2. Run the Full ML Research Pipeline End-to-End

```bash
python run_pipeline.py
```

---

## 📁 Project Architecture

```
solar-xai-forecasting/
├── app/                                 # Full-Stack Web Application
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── main.py                      # FastAPI server & REST routes
│   │   ├── model_service.py             # Inference pipeline & SHAP explainer
│   │   └── schemas.py                   # Pydantic data models
│   └── frontend/
│       ├── index.html                   # Single-Page Dashboard
│       ├── styles.css                   # Custom styles & animations
│       └── app.js                       # Interactive charts & reactive state
├── models/                              # Serialized trained models
│   ├── xgboost_final.pkl                # Champion Model (R² = 0.8687)
│   ├── random_forest_final.pkl          # 300 Bagged Trees (R² = 0.8683)
│   ├── linear_regression_final.pkl      # Scaled OLS Baseline (R² = 0.8016)
│   ├── lstm_final.pth                   # PyTorch 2-layer LSTM (R² = 0.7056)
│   └── scaler.pkl                       # StandardScaler
├── results/                             # Output datasets, plots, and metrics
│   ├── X_train.csv / y_train.csv
│   ├── X_val.csv   / y_val.csv
│   ├── X_test.csv  / y_test.csv
│   ├── final_model_metrics.csv
│   ├── model_predictions_comparison.png
│   ├── shap_beeswarm_summary.png
│   ├── shap_global_bar.png
│   ├── shap_local_ramp_waterfall.png
│   ├── pv_ramp_events_report.csv
│   └── RESEARCH_RESULTS_SUMMARY.md
├── src/                                 # Machine Learning Modules
│   ├── __init__.py
│   ├── preprocess.py                    # Cleaning, feature engineering & split
│   ├── train_models.py                  # Model training, evaluation & serialization
│   └── xai_and_analysis.py              # SHAP analysis & ramp detection
├── merged_solar_data_2.xlsx             # 24-month raw sensor dataset
├── run_app.py                           # Web application launcher
├── run_pipeline.py                      # Master ML pipeline script
├── requirements.txt                     # Project dependencies
└── README.md
```

---

## 📊 Models Evaluated & Benchmarks

| Model | Architecture | $R^2$ Score | MAE (W) | RMSE (W) | MAPE (%) | Role |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | Histogram-based Gradient Boosted Trees | **0.8687** | **0.96 W** | **1.52 W** | 18.59% | **Champion Model** |
| **Random Forest** | 300 Bagged Decision Trees | 0.8683 | 0.91 W | 1.52 W | 17.33% | Competitive Baseline |
| **Linear Regression** | Standard Scaled OLS | 0.8016 | 1.27 W | 1.87 W | 27.68% | Linear Baseline |
| **LSTM** | 2-layer PyTorch Recurrent Neural Net | 0.7056 | 1.53 W | 2.26 W | 30.59% | Deep Sequence Model |

---

## 🔬 Explainable AI (XAI) & Physics Insights

1. **Irradiance & Autoregressive Dominance:** `Power_Output_lag1` (Mean $|SHAP| = 1.57\text{ W}$) and `Pyranometer` (Mean $|SHAP| = 1.25\text{ W}$) account for over 70% of model explanatory power.
2. **Thermal Derating:** The engineered feature `Temp_Diff = Temp1 - Air Temp` captures semiconductor bandgap resistance; high thermal differential significantly penalizes expected generation.
3. **Soiling Degradation Proxy:** A rolling median of power-to-irradiance ratio identifies dust accumulation:
   $$\text{Rolling\_Soiling\_Factor} = \text{rolling}_{16}\left(\frac{\text{Power\_Output}_{\text{lag1}}}{\text{Pyranometer}_{\text{lag1}} + 1}\right)$$
   $$\text{Expected\_Baseline\_Power} = \text{Pyranometer} \times \text{Rolling\_Soiling\_Factor}$$

---

## ⚡ PV Ramp Event Detection (Grid Management)

- **Critical Ramp Threshold:** **3.98 W** (90th percentile of test-set fluctuations).
- **Detection Accuracy:** **Precision = 71.0%**, **Recall = 51.0%**, **F1 = 0.594** on 96 critical cloud transient dropouts.
- **Grid Protocols:** Automatic trigger of Fast-Start Spinning Reserves and Battery Energy Storage System (BESS) charge/discharge dispatch.

---

## 🌐 API Overview

- `GET /` — Interactive Single-Page Web Dashboard
- `GET /api/status` — System health and active models
- `GET /api/features` — Definitions and metadata for all 19 input features
- `GET /api/presets` — Pre-configured operational scenarios
- `POST /api/predict` — Solar power output prediction & ramp delta classification
- `POST /api/explain` — Local SHAP attribution waterfall values
- `GET /api/metrics` — Comparative benchmark table
- `GET /api/test-sequence` — 150-point test time-series for actual vs. predicted replay
- `POST /api/upload-csv` — Batch inference for uploaded CSV files
- `GET /research-artifacts/{filename}` — Serves research figures and plots

---

## 📄 License

Research project — all rights reserved.
