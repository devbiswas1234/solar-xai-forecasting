/**
 * app.js — Interactive Client for Solar XAI Forecasting & Grid Intelligence System
 */

// Application State
const state = {
  activeTab: 'studio',
  presets: [],
  currentFeatures: {},
  debounceTimer: null,
  charts: {
    waterfall: null,
    globalBar: null,
    r2Bar: null,
    maeBar: null,
    replay: null,
  },
  testSeries: [],
  selectedTestPoint: null,
};

// Top 10 global importance ranking from research results
const GLOBAL_SHAP_IMPORTANCE = [
  { feature: 'Power_Output_lag1', label: 'Power Output (t-1)', mean_shap: 1.57 },
  { feature: 'Pyranometer (Watt/m2)', label: 'Solar Irradiance (GHI)', mean_shap: 1.25 },
  { feature: 'Expected_Baseline_Power', label: 'Expected Baseline Power', mean_shap: 0.87 },
  { feature: 'Pyranometer_lag1', label: 'Solar Irradiance (t-1)', mean_shap: 0.44 },
  { feature: 'Temp_Diff', label: 'Thermal Differential (Temp_Diff)', mean_shap: 0.38 },
  { feature: 'Hour_cos', label: 'Diurnal Solar Angle (Hour_cos)', mean_shap: 0.31 },
  { feature: 'Rolling_Soiling_Factor', label: 'Panel Soiling Factor', mean_shap: 0.28 },
  { feature: 'Temp1 (degC)', label: 'Module Temperature', mean_shap: 0.22 },
  { feature: 'Relative Humidity (%)', label: 'Relative Humidity', mean_shap: 0.16 },
  { feature: 'Wind Speed (m/s)', label: 'Wind Velocity', mean_shap: 0.12 },
];

// ─────────────────────────────────────────────────────────────────────────────
// DOM READY & INITIALIZATION
// ─────────────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  if (window.lucide) lucide.createIcons();

  setupInputListeners();
  await loadPresets();
  await loadBenchmarks();
  await loadRampEvents();
  await loadHistoricalReplay();
  initGlobalSHAPChart();

  // Initial trigger with preset #1 (Summer Noon)
  if (state.presets.length > 0) {
    applyPreset(state.presets[0].id);
  } else {
    triggerForecast();
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// TAB SWITCHING
// ─────────────────────────────────────────────────────────────────────────────
function switchTab(tabId) {
  state.activeTab = tabId;

  document.querySelectorAll('.tab-view').forEach(view => view.classList.add('hidden'));
  document.querySelectorAll('.nav-tab').forEach(btn => {
    btn.classList.remove('active-tab');
    btn.classList.add('text-slate-400');
  });

  const targetView = document.getElementById(`view-${tabId}`);
  if (targetView) targetView.classList.remove('hidden');

  const targetBtn = document.getElementById(`tab-btn-${tabId}`);
  if (targetBtn) {
    targetBtn.classList.add('active-tab');
    targetBtn.classList.remove('text-slate-400');
  }

  // Refresh charts on tab visibility change
  if (tabId === 'benchmarks') {
    if (state.charts.r2Bar) state.charts.r2Bar.resize();
    if (state.charts.maeBar) state.charts.maeBar.resize();
  } else if (tabId === 'xai') {
    if (state.charts.globalBar) state.charts.globalBar.resize();
  } else if (tabId === 'replay') {
    if (state.charts.replay) state.charts.replay.resize();
  }

  if (window.lucide) lucide.createIcons();
}

// ─────────────────────────────────────────────────────────────────────────────
// INPUT BINDINGS & SYNCHRONIZATION
// ─────────────────────────────────────────────────────────────────────────────
function setupInputListeners() {
  const inputs = [
    { id: 'input-pyranometer', display: 'val-pyranometer', format: v => `${v} W/m²` },
    { id: 'input-temp1', display: 'val-temp1', format: v => `${parseFloat(v).toFixed(1)} °C` },
    { id: 'input-air-temp', display: 'val-air-temp', format: v => `${parseFloat(v).toFixed(1)} °C` },
    { id: 'input-humidity', display: 'val-humidity', format: v => `${v} %` },
    { id: 'input-wind-speed', display: 'val-wind-speed', format: v => `${parseFloat(v).toFixed(1)} m/s` },
    { id: 'input-hour', display: 'val-hour', format: formatDecimalHour },
    { id: 'input-month', display: 'val-month', format: formatMonth },
    { id: 'input-lag1-power', display: 'val-lag1-power', format: v => `${parseFloat(v).toFixed(1)} W` },
    { id: 'input-lag1-pyr', display: 'val-lag1-pyr', format: v => `${v} W/m²` },
    { id: 'input-soiling', display: 'val-soiling', format: v => `${parseFloat(v).toFixed(4)}` },
  ];

  inputs.forEach(({ id, display, format }) => {
    const el = document.getElementById(id);
    const dispEl = document.getElementById(display);
    if (!el) return;

    el.addEventListener('input', () => {
      if (dispEl) dispEl.textContent = format(el.value);
      updateThermalDifferentialBadge();
      debouncedForecast();
    });
  });
}

function formatDecimalHour(val) {
  const h = parseFloat(val);
  const hours = Math.floor(h);
  const minutes = Math.round((h - hours) * 60);
  const minStr = minutes < 10 ? `0${minutes}` : `${minutes}`;
  const ampm = hours >= 12 ? 'PM' : 'AM';
  const displayHours = hours % 12 === 0 ? 12 : hours % 12;
  return `${displayHours}:${minStr} ${ampm} (${h.toFixed(1)}h)`;
}

function formatMonth(val) {
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const m = parseInt(val, 10);
  return `${months[m - 1]} (M=${m})`;
}

function updateThermalDifferentialBadge() {
  const t1 = parseFloat(document.getElementById('input-temp1').value || 35);
  const tAir = parseFloat(document.getElementById('input-air-temp').value || 28);
  const diff = (t1 - tAir).toFixed(1);
  const badge = document.getElementById('badge-temp-diff');
  if (badge) {
    badge.textContent = `${diff >= 0 ? '+' : ''}${diff} °C`;
    if (diff > 18) {
      badge.className = 'font-mono font-bold text-rose-400';
    } else if (diff > 10) {
      badge.className = 'font-mono font-bold text-amber-300';
    } else {
      badge.className = 'font-mono font-bold text-emerald-400';
    }
  }
}

function toggleAdvancedLags() {
  const body = document.getElementById('advanced-lags-body');
  const chevron = document.getElementById('advanced-chevron');
  if (body) {
    body.classList.toggle('hidden');
    if (chevron) {
      chevron.style.transform = body.classList.contains('hidden') ? 'rotate(0deg)' : 'rotate(180deg)';
    }
  }
}

function debouncedForecast() {
  clearTimeout(state.debounceTimer);
  state.debounceTimer = setTimeout(() => {
    triggerForecast();
  }, 120);
}

// ─────────────────────────────────────────────────────────────────────────────
// PRESETS
// ─────────────────────────────────────────────────────────────────────────────
async function loadPresets() {
  try {
    const res = await fetch('/api/presets');
    const data = await res.json();
    state.presets = data.presets || [];

    const container = document.getElementById('preset-container');
    if (!container) return;
    container.innerHTML = '';

    state.presets.forEach((p, idx) => {
      const btn = document.createElement('button');
      btn.className = `preset-pill ${idx === 0 ? 'active' : ''}`;
      btn.id = `preset-pill-${p.id}`;
      btn.innerHTML = `<span>${p.name}</span>`;
      btn.onclick = () => applyPreset(p.id);
      container.appendChild(btn);
    });
  } catch (err) {
    console.error('Failed to load presets:', err);
  }
}

function applyPreset(presetId) {
  const p = state.presets.find(item => item.id === presetId);
  if (!p) return;

  document.querySelectorAll('.preset-pill').forEach(el => el.classList.remove('active'));
  const activeBtn = document.getElementById(`preset-pill-${presetId}`);
  if (activeBtn) activeBtn.classList.add('active');

  const f = p.features;
  setInputValue('input-pyranometer', f.pyranometer, 'val-pyranometer', `${f.pyranometer} W/m²`);
  setInputValue('input-temp1', f.temp1, 'val-temp1', `${f.temp1.toFixed(1)} °C`);
  setInputValue('input-air-temp', f.air_temp, 'val-air-temp', `${f.air_temp.toFixed(1)} °C`);
  setInputValue('input-humidity', f.relative_humidity, 'val-humidity', `${f.relative_humidity} %`);
  setInputValue('input-wind-speed', f.wind_speed, 'val-wind-speed', `${f.wind_speed.toFixed(1)} m/s`);
  setInputValue('input-hour', f.hour, 'val-hour', formatDecimalHour(f.hour));
  setInputValue('input-month', f.month, 'val-month', formatMonth(f.month));
  setInputValue('input-lag1-power', f.power_output_lag1, 'val-lag1-power', `${f.power_output_lag1.toFixed(1)} W`);
  setInputValue('input-lag1-pyr', f.pyranometer_lag1, 'val-lag1-pyr', `${f.pyranometer_lag1} W/m²`);
  setInputValue('input-soiling', f.rolling_soiling_factor, 'val-soiling', `${f.rolling_soiling_factor.toFixed(4)}`);

  updateThermalDifferentialBadge();
  triggerForecast();
}

function setInputValue(id, val, dispId, dispVal) {
  const el = document.getElementById(id);
  const dispEl = document.getElementById(dispId);
  if (el) el.value = val;
  if (dispEl) dispEl.textContent = dispVal;
}

function resetToOptimal() {
  if (state.presets.length > 0) {
    applyPreset(state.presets[0].id);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// COLLECT CURRENT FORM VALUES
// ─────────────────────────────────────────────────────────────────────────────
function getFormValues() {
  const pyr = parseFloat(document.getElementById('input-pyranometer').value);
  const t1 = parseFloat(document.getElementById('input-temp1').value);
  const tAir = parseFloat(document.getElementById('input-air-temp').value);
  const hum = parseFloat(document.getElementById('input-humidity').value);
  const wind = parseFloat(document.getElementById('input-wind-speed').value);
  const hr = parseFloat(document.getElementById('input-hour').value);
  const m = parseInt(document.getElementById('input-month').value, 10);
  const lagPwr = parseFloat(document.getElementById('input-lag1-power').value);
  const lagPyr = parseFloat(document.getElementById('input-lag1-pyr').value);
  const soiling = parseFloat(document.getElementById('input-soiling').value);

  return {
    pyranometer: pyr,
    wind_direction: 180.0,
    relative_humidity: hum,
    temp1: t1,
    temp2: t1,
    wind_speed: wind,
    air_temp: tAir,
    hour: hr,
    month: m,
    pyranometer_lag1: lagPyr,
    power_output_lag1: lagPwr,
    pyranometer_lag2: lagPyr * 0.98,
    power_output_lag2: lagPwr * 0.98,
    rolling_soiling_factor: soiling,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// TRIGGER PREDICTION & XAI
// ─────────────────────────────────────────────────────────────────────────────
async function triggerForecast() {
  const features = getFormValues();

  try {
    // 1. Run multi-model forecast
    const predRes = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features, model_name: 'xgboost' }),
    });
    const predData = await predRes.json();

    // 2. Run SHAP local explanation
    const explainRes = await fetch('/api/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features }),
    });
    const explainData = await explainRes.json();

    renderPredictionHero(predData);
    renderSHAPWaterfall(explainData);
  } catch (err) {
    console.error('Forecast request failed:', err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// RENDER HERO FORECAST & MULTI-MODEL PILLS
// ─────────────────────────────────────────────────────────────────────────────
function renderPredictionHero(data) {
  const heroWatts = document.getElementById('hero-forecast-watts');
  if (heroWatts) heroWatts.textContent = data.primary_prediction_w.toFixed(2);

  // Multi-model pills
  const mm = data.multi_model_predictions || {};
  if (mm['XGBoost (Champion)'] !== undefined) {
    document.getElementById('cmp-xgboost').textContent = `${mm['XGBoost (Champion)'].toFixed(2)} W`;
  }
  if (mm['Random Forest'] !== undefined) {
    document.getElementById('cmp-rf').textContent = `${mm['Random Forest'].toFixed(2)} W`;
  }
  if (mm['Linear Regression'] !== undefined) {
    document.getElementById('cmp-lr').textContent = `${mm['Linear Regression'].toFixed(2)} W`;
  }
  if (mm['LSTM (Deep Learning)'] !== undefined) {
    document.getElementById('cmp-lstm').textContent = `${mm['LSTM (Deep Learning)'].toFixed(2)} W`;
  }

  // Ramp Status Box
  const ramp = data.ramp_analysis;
  const rampBox = document.getElementById('ramp-alert-box');
  const rampTitle = document.getElementById('ramp-alert-title');
  const rampDelta = document.getElementById('ramp-alert-delta');
  const rampDesc = document.getElementById('ramp-alert-desc');

  if (ramp && rampBox) {
    const sign = ramp.delta_w >= 0 ? '+' : '';
    rampDelta.textContent = `Δ = ${sign}${ramp.delta_w.toFixed(2)} W`;

    if (ramp.is_critical) {
      rampBox.className = 'p-4 rounded-xl border flex flex-col justify-center min-w-[240px] bg-rose-950/80 border-rose-500/60 text-rose-300 critical-ramp-glow';
      rampTitle.innerHTML = `<span class="flex items-center gap-1.5"><i data-lucide="alert-octagon" class="w-4 h-4 text-rose-400"></i> CRITICAL ${ramp.direction} ALERT</span>`;
      rampDesc.textContent = ramp.recommendation;
    } else if (ramp.alert_level === 'MODERATE') {
      rampBox.className = 'p-4 rounded-xl border flex flex-col justify-center min-w-[240px] bg-amber-950/60 border-amber-500/50 text-amber-300';
      rampTitle.innerHTML = `<span class="flex items-center gap-1.5"><i data-lucide="alert-triangle" class="w-4 h-4 text-amber-400"></i> MODERATE RAMP</span>`;
      rampDesc.textContent = ramp.recommendation;
    } else {
      rampBox.className = 'p-4 rounded-xl border flex flex-col justify-center min-w-[240px] bg-emerald-950/40 border-emerald-500/30 text-emerald-300';
      rampTitle.innerHTML = `<span class="flex items-center gap-1.5"><i data-lucide="check-circle" class="w-4 h-4 text-emerald-400"></i> STABLE RAMP (NORMAL)</span>`;
      rampDesc.textContent = ramp.recommendation;
    }

    if (window.lucide) lucide.createIcons();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// RENDER REAL-TIME LOCAL SHAP WATERFALL
// ─────────────────────────────────────────────────────────────────────────────
function renderSHAPWaterfall(data) {
  const contributions = data.contributions || [];
  const baseValue = data.base_value;
  const predictedValue = data.predicted_value;

  // Render top positive and negative driver lists
  const posList = document.getElementById('shap-top-pos-list');
  const negList = document.getElementById('shap-top-neg-list');

  if (posList) {
    posList.innerHTML = data.top_positive.length > 0
      ? data.top_positive.map(c => `
          <li class="flex justify-between items-center py-0.5 border-b border-emerald-500/10">
            <span class="text-slate-300">${c.feature_label}</span>
            <span class="font-bold text-emerald-400">+${c.shap_value.toFixed(2)} W</span>
          </li>
        `).join('')
      : '<li class="text-slate-500">None in this regime</li>';
  }

  if (negList) {
    negList.innerHTML = data.top_negative.length > 0
      ? data.top_negative.map(c => `
          <li class="flex justify-between items-center py-0.5 border-b border-rose-500/10">
            <span class="text-slate-300">${c.feature_label}</span>
            <span class="font-bold text-rose-400">${c.shap_value.toFixed(2)} W</span>
          </li>
        `).join('')
      : '<li class="text-slate-500">None in this regime</li>';
  }

  // Build Waterfall Chart data (take top 8 largest contributors)
  const topContribs = contributions.slice(0, 8);
  const labels = topContribs.map(c => c.feature_label);
  const values = topContribs.map(c => c.shap_value);
  const bgColors = values.map(v => v >= 0 ? 'rgba(16, 185, 129, 0.85)' : 'rgba(239, 68, 68, 0.85)');
  const borderColors = values.map(v => v >= 0 ? '#10B981' : '#EF4444');

  const ctx = document.getElementById('shap-waterfall-chart');
  if (!ctx) return;

  if (state.charts.waterfall) {
    state.charts.waterfall.destroy();
  }

  state.charts.waterfall = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'SHAP Contribution (W)',
        data: values,
        backgroundColor: bgColors,
        borderColor: borderColors,
        borderWidth: 1.5,
        borderRadius: 6,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0F172A',
          borderColor: '#334155',
          borderWidth: 1,
          titleFont: { size: 12, weight: 'bold' },
          bodyFont: { size: 11 },
          callbacks: {
            label: context => {
              const val = context.raw;
              const sign = val >= 0 ? '+' : '';
              return ` Impact: ${sign}${val.toFixed(3)} W (vs E[f(x)] = ${baseValue.toFixed(2)} W)`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.06)' },
          ticks: { color: '#94A3B8', font: { size: 10 } },
          title: {
            display: true,
            text: `SHAP Value Shift from Base (${baseValue.toFixed(2)} W) to Final (${predictedValue.toFixed(2)} W)`,
            color: '#64748B',
            font: { size: 10 },
          }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#E2E8F0', font: { size: 11, weight: 'bold' } }
        }
      }
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// GLOBAL SHAP IMPORTANCE CHART
// ─────────────────────────────────────────────────────────────────────────────
function initGlobalSHAPChart() {
  const ctx = document.getElementById('shap-global-bar-chart');
  if (!ctx) return;

  if (state.charts.globalBar) {
    state.charts.globalBar.destroy();
  }

  state.charts.globalBar = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: GLOBAL_SHAP_IMPORTANCE.map(d => d.label),
      datasets: [{
        label: 'Mean |SHAP| (Watts)',
        data: GLOBAL_SHAP_IMPORTANCE.map(d => d.mean_shap),
        backgroundColor: 'rgba(245, 158, 11, 0.8)',
        borderColor: '#F59E0B',
        borderWidth: 1.5,
        borderRadius: 6,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0F172A',
          borderColor: '#334155',
          borderWidth: 1,
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.06)' },
          ticks: { color: '#94A3B8' },
          title: { display: true, text: 'Mean Absolute SHAP Value (Watts Impact)', color: '#64748B' }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#E2E8F0', font: { size: 11 } }
        }
      }
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// BENCHMARKS & LEADERBOARD
// ─────────────────────────────────────────────────────────────────────────────
async function loadBenchmarks() {
  try {
    const res = await fetch('/api/metrics');
    const data = await res.json();
    const metrics = data.metrics || [];

    const tbody = document.getElementById('metrics-table-body');
    if (!tbody) return;

    tbody.innerHTML = metrics.map((m, idx) => {
      const isChamp = m.Model === 'XGBoost';
      const badge = isChamp
        ? '<span class="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">🏆 Champion</span>'
        : '<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-slate-800 text-slate-400">Baseline</span>';

      const arch = m.Model === 'XGBoost' ? 'Gradient Boosted Trees (hist)'
        : m.Model === 'Random Forest' ? '300 Bagged Trees'
        : m.Model === 'LSTM' ? '2-layer PyTorch RNN (seq=6)'
        : 'Standard Scaled OLS';

      return `
        <tr class="${isChamp ? 'bg-amber-500/5 font-semibold text-white' : 'hover:bg-slate-900/40'}">
          <td class="py-3 px-4 flex items-center gap-2">
            <span class="w-5 h-5 rounded-full bg-slate-800 text-slate-300 flex items-center justify-center text-[10px] font-bold">${idx + 1}</span>
            <span>${m.Model}</span>
          </td>
          <td class="py-3 px-4 font-bold text-amber-400">${m.R2.toFixed(4)}</td>
          <td class="py-3 px-4">${m.MAE.toFixed(3)} W</td>
          <td class="py-3 px-4">${m.RMSE.toFixed(3)} W</td>
          <td class="py-3 px-4">${m['MAPE(%)'].toFixed(2)} %</td>
          <td class="py-3 px-4 text-slate-400 font-sans">${arch}</td>
          <td class="py-3 px-4">${badge}</td>
        </tr>
      `;
    }).join('');

    renderBenchmarkCharts(metrics);
  } catch (err) {
    console.error('Failed to load metrics:', err);
  }
}

function renderBenchmarkCharts(metrics) {
  const names = metrics.map(m => m.Model);
  const r2s = metrics.map(m => m.R2);
  const maes = metrics.map(m => m.MAE);

  // 1. R2 Chart
  const r2Ctx = document.getElementById('r2-bar-chart');
  if (r2Ctx) {
    if (state.charts.r2Bar) state.charts.r2Bar.destroy();
    state.charts.r2Bar = new Chart(r2Ctx, {
      type: 'bar',
      data: {
        labels: names,
        datasets: [{
          label: 'R² Score (Higher is better)',
          data: r2s,
          backgroundColor: names.map(n => n === 'XGBoost' ? 'rgba(245, 158, 11, 0.9)' : 'rgba(56, 189, 248, 0.7)'),
          borderRadius: 6,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: { min: 0.65, max: 0.92, grid: { color: 'rgba(255, 255, 255, 0.06)' }, ticks: { color: '#94A3B8' } },
          x: { ticks: { color: '#CBD5E1' } }
        }
      }
    });
  }

  // 2. MAE Chart
  const maeCtx = document.getElementById('mae-bar-chart');
  if (maeCtx) {
    if (state.charts.maeBar) state.charts.maeBar.destroy();
    state.charts.maeBar = new Chart(maeCtx, {
      type: 'bar',
      data: {
        labels: names,
        datasets: [{
          label: 'MAE Watts (Lower is better)',
          data: maes,
          backgroundColor: names.map(n => n === 'XGBoost' ? 'rgba(16, 185, 129, 0.85)' : 'rgba(148, 163, 184, 0.6)'),
          borderRadius: 6,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: { grid: { color: 'rgba(255, 255, 255, 0.06)' }, ticks: { color: '#94A3B8' } },
          x: { ticks: { color: '#CBD5E1' } }
        }
      }
    });
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// RAMP EVENTS LOG
// ─────────────────────────────────────────────────────────────────────────────
async function loadRampEvents() {
  try {
    const res = await fetch('/api/ramp-report?limit=40');
    const data = await res.json();
    const records = data.sample_records || [];

    const tbody = document.getElementById('ramp-events-table-body');
    if (!tbody) return;

    tbody.innerHTML = records.map(r => {
      const isCrit = r.Critical_Event;
      const rampW = r.Ramp_W;
      const sign = rampW >= 0 ? '+' : '';

      return `
        <tr class="${isCrit ? 'bg-rose-950/20 text-rose-300 font-semibold' : 'hover:bg-slate-900/30 text-slate-300'}">
          <td class="py-2 px-3">#${r.Index_in_test}</td>
          <td class="py-2 px-3">${r.Power_t_minus_1.toFixed(2)} W</td>
          <td class="py-2 px-3">${r.Power_t.toFixed(2)} W</td>
          <td class="py-2 px-3 font-bold ${isCrit ? 'text-rose-400' : 'text-slate-300'}">${sign}${rampW.toFixed(2)} W</td>
          <td class="py-2 px-3">
            ${isCrit
              ? '<span class="px-1.5 py-0.5 rounded text-[10px] bg-rose-500/20 text-rose-400 border border-rose-500/30">CRITICAL</span>'
              : '<span class="text-slate-500 text-[10px]">Normal</span>'
            }
          </td>
          <td class="py-2 px-3">
            ${r.Predicted_Critical
              ? '<span class="text-emerald-400 text-[10px] font-bold">Detected ✓</span>'
              : '<span class="text-slate-500 text-[10px]">No Alarm</span>'
            }
          </td>
        </tr>
      `;
    }).join('');
  } catch (err) {
    console.error('Failed to load ramp events:', err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// HISTORICAL REPLAY TIME SERIES
// ─────────────────────────────────────────────────────────────────────────────
async function loadHistoricalReplay() {
  try {
    const res = await fetch('/api/test-sequence?limit=150');
    const data = await res.json();
    state.testSeries = data.series || [];

    const ctx = document.getElementById('historical-replay-chart');
    if (!ctx || state.testSeries.length === 0) return;

    const steps = state.testSeries.map(d => d.step);
    const actuals = state.testSeries.map(d => d.actual);
    const xgbs = state.testSeries.map(d => d.xgboost);
    const rfs = state.testSeries.map(d => d.random_forest);
    const lrs = state.testSeries.map(d => d.linear_regression);

    if (state.charts.replay) state.charts.replay.destroy();

    state.charts.replay = new Chart(ctx, {
      type: 'line',
      data: {
        labels: steps,
        datasets: [
          {
            label: 'Actual PV Output (W)',
            data: actuals,
            borderColor: '#FFFFFF',
            backgroundColor: '#FFFFFF',
            borderWidth: 2.2,
            tension: 0.15,
            pointRadius: 0,
            pointHoverRadius: 6,
          },
          {
            label: 'XGBoost (Champion)',
            data: xgbs,
            borderColor: '#38BDF8',
            borderDash: [5, 5],
            borderWidth: 1.8,
            tension: 0.15,
            pointRadius: 0,
          },
          {
            label: 'Random Forest',
            data: rfs,
            borderColor: '#10B981',
            borderDash: [3, 3],
            borderWidth: 1.5,
            tension: 0.15,
            pointRadius: 0,
          },
          {
            label: 'Linear Regression',
            data: lrs,
            borderColor: '#F43F5E',
            borderDash: [2, 2],
            borderWidth: 1.2,
            tension: 0.15,
            pointRadius: 0,
          },
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: {
            position: 'top',
            labels: { color: '#CBD5E1', boxWidth: 12, font: { size: 11 } }
          },
          tooltip: {
            backgroundColor: '#0F172A',
            borderColor: '#334155',
            borderWidth: 1,
            callbacks: {
              title: context => `Test Time Step #${context[0].label}`,
              afterBody: context => {
                const idx = context[0].dataIndex;
                const pt = state.testSeries[idx];
                return `Irradiance: ${pt.irradiance} W/m² | Ramp: ${pt.ramp_w > 0 ? '+' : ''}${pt.ramp_w} W`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94A3B8', maxTicksLimit: 15 },
            title: { display: true, text: 'Test Sample Sequence Step', color: '#64748B' }
          },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94A3B8' },
            title: { display: true, text: 'Solar Generation (Watts)', color: '#64748B' }
          }
        },
        onClick: (evt, elements) => {
          if (elements && elements.length > 0) {
            const idx = elements[0].index;
            selectTestPoint(idx);
          }
        }
      }
    });

    // Default select step 20
    if (state.testSeries.length > 20) {
      selectTestPoint(20);
    }
  } catch (err) {
    console.error('Failed to load historical replay:', err);
  }
}

function selectTestPoint(idx) {
  state.selectedTestPoint = state.testSeries[idx];
  const ind = document.getElementById('selected-point-indicator');
  if (ind && state.selectedTestPoint) {
    ind.innerHTML = `Selected: <strong>Step #${idx}</strong> (Actual: ${state.selectedTestPoint.actual} W, Irradiance: ${state.selectedTestPoint.irradiance} W/m²)`;
  }
}

function loadTestPointToSimulator() {
  if (!state.selectedTestPoint) return;
  const f = state.selectedTestPoint.features;

  setInputValue('input-pyranometer', Math.round(f['Pyranometer (Watt/m2)']), 'val-pyranometer', `${Math.round(f['Pyranometer (Watt/m2)'])} W/m²`);
  setInputValue('input-temp1', f['Temp1 (degC)'].toFixed(1), 'val-temp1', `${f['Temp1 (degC)'].toFixed(1)} °C`);
  setInputValue('input-air-temp', f['Air Temp (degC)'].toFixed(1), 'val-air-temp', `${f['Air Temp (degC)'].toFixed(1)} °C`);
  setInputValue('input-humidity', Math.round(f['Relative Humidity (%)']), 'val-humidity', `${Math.round(f['Relative Humidity (%)'])} %`);
  setInputValue('input-wind-speed', f['Wind Speed (m/s)'].toFixed(1), 'val-wind-speed', `${f['Wind Speed (m/s)'].toFixed(1)} m/s`);
  setInputValue('input-hour', f['Hour'].toFixed(2), 'val-hour', formatDecimalHour(f['Hour']));
  setInputValue('input-lag1-power', f['Power_Output_lag1'].toFixed(1), 'val-lag1-power', `${f['Power_Output_lag1'].toFixed(1)} W`);
  setInputValue('input-lag1-pyr', Math.round(f['Pyranometer_lag1']), 'val-lag1-pyr', `${Math.round(f['Pyranometer_lag1'])} W/m²`);
  setInputValue('input-soiling', f['Rolling_Soiling_Factor'].toFixed(4), 'val-soiling', `${f['Rolling_Soiling_Factor'].toFixed(4)}`);

  updateThermalDifferentialBadge();
  switchTab('studio');
  triggerForecast();
}

// ─────────────────────────────────────────────────────────────────────────────
// BATCH CSV INGESTION
// ─────────────────────────────────────────────────────────────────────────────
async function uploadBatchCSV() {
  const fileInput = document.getElementById('batch-file-input');
  if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
    alert('Please choose a CSV file first.');
    return;
  }

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append('file', file);

  const preview = document.getElementById('batch-results-preview');
  if (preview) {
    preview.classList.remove('hidden');
    preview.textContent = 'Processing CSV batch file...';
  }

  try {
    const res = await fetch('/api/upload-csv', {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();

    if (preview) {
      preview.innerHTML = `
        <strong>Successfully Processed ${data.rows_processed} rows!</strong><br>
        Preview of first ${data.results.length} predictions:<br>
        <div class="mt-2 space-y-1">
          ${data.results.slice(0, 5).map((r, i) => `
            <div>Row ${i + 1}: Forecast = <strong>${r.predicted_w.toFixed(2)} W</strong> | Ramp = ${r.ramp_w.toFixed(2)} W [${r.alert}]</div>
          `).join('')}
        </div>
      `;
    }
  } catch (err) {
    if (preview) preview.textContent = `Batch upload error: ${err.message}`;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// ARTIFACT MODAL VIEWER
// ─────────────────────────────────────────────────────────────────────────────
function openArtifactModal(imgSrc, title) {
  const modal = document.getElementById('artifact-modal');
  const modalImg = document.getElementById('modal-img');
  const modalTitle = document.getElementById('modal-title');
  if (modal && modalImg && modalTitle) {
    modalImg.src = imgSrc;
    modalTitle.textContent = title;
    modal.classList.remove('hidden');
  }
}

function closeArtifactModal() {
  const modal = document.getElementById('artifact-modal');
  if (modal) modal.classList.add('hidden');
}
