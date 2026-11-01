# Solar Energy Forecasting & XAI Grid Intelligence System

A full-stack web application serving machine learning forecasting models, real-time Explainable AI (SHAP TreeExplainer), and critical Photovoltaic (PV) ramp event detection for solar power grid management.

---

## Quick Start

### 1. Launch the Application

From the project root:

```bash
python run_app.py
```

Or from inside `solar-xai-app`:

```bash
cd solar-xai-app
python run_app.py
```

### 2. Open the Dashboard & API Docs

- **Web Dashboard:** [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Models Evaluated & Serialized

| Model | Architecture | R² Score | MAE (W) | RMSE (W) | MAPE (%) | Role |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | Histogram-based Gradient Boosted Trees | **0.8687** | **0.96 W** | **1.52 W** | 18.59% | **Champion Model** |
| **Random Forest** | 300 Bagged Decision Trees | 0.8683 | 0.91 W | 1.52 W | 17.33% | Competitive Baseline |
| **Linear Regression** | Standard Scaled OLS | 0.8016 | 1.27 W | 1.87 W | 27.68% | Linear Baseline |
| **LSTM** | 2-layer PyTorch Recurrent Neural Net | 0.7056 | 1.53 W | 2.26 W | 30.59% | Deep Sequence Model |

---

## Key Features

1. **Live Forecasting Studio**:
   - Interactive meteorological sliders (`Pyranometer`, `Ambient & Panel Temp`, `Humidity`, `Wind Speed`, `Hour`, `Month`, `Autoregressive Lags`, `Soiling Factor`).
   - 1-click presets: *Summer Noon Peak*, *Cloud Transient (Ramp Down)*, *Dust Soiled Panel*, *Extreme Heat Wave*, *Overcast Morning*.
   - Simultaneous side-by-side inference across all 4 models.
   - Real-time local SHAP attribution waterfall chart explaining every individual prediction.
   - Live PV Ramp status indicator (Normal, Moderate, Critical) with grid dispatch guidance.

2. **Explainable AI (XAI) Hub**:
   - Global SHAP feature importance ranking (Mean |SHAP| on 961 test samples).
   - High-resolution modal gallery for research plots (SHAP Beeswarm Summary, Global Bar Chart, Worst Negative Ramp Waterfall, 150-Point Comparison).
   - Physical mechanics explanations (Irradiance inertia, thermal derating via $T_{panel} - T_{air}$, soiling degradation baseline).

3. **Model Benchmarks & Methodology**:
   - Interactive leaderboard table and comparative R² & MAE bar charts.
   - Documentation of the *Monthly-Block Chronological Split (70/15/15)* preventing inter-annual leakage.
   - Strict omission of Voltage and Current features to prevent $P = V \times I$ shortcut leakage.

4. **PV Ramp & Grid Management**:
   - Critical ramp threshold monitoring ($3.98\text{ W}$, 90th percentile).
   - Detection statistics: **71.0% Precision**, **51.0% Recall**, **0.594 F1-score**.
   - Standard Operating Protocols (SOP) for automated generation control (AGC), battery storage (BESS) dispatch, and spinning reserve scheduling.

5. **Historical Replay & Batch Tool**:
   - 150-point interactive time-series viewer with Actual vs Model predicted curves.
   - 1-click loading of any historical test sequence step directly into the simulator.
   - Batch CSV file upload and processing.

---

## API Endpoints

- `GET /` — Serves the frontend single-page application.
- `GET /api/status` — Health check and status of all 4 active models.
- `GET /api/features` — Metadata and units for all 19 input features.
- `GET /api/presets` — Operational scenario presets.
- `POST /api/predict` — Predicts solar power and analyzes ramp delta.
- `POST /api/explain` — Computes real-time SHAP waterfall values with TreeExplainer.
- `GET /api/metrics` — Comparative metrics table.
- `GET /api/ramp-report` — Summary statistics and critical event test logs.
- `GET /api/test-sequence` — Historical test set sequence for actual vs predicted replay.
- `POST /api/upload-csv` — Upload CSV for batch prediction and analysis.
- `GET /research-artifacts/{filename}` — Serves high-resolution research plots.
