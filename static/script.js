/* ══════════════════════════════════════════════════════════════
   ChurnGuard — script.js
   Prediction form, results display, charts
══════════════════════════════════════════════════════════════ */

"use strict";

// ── Chart instances (kept for potential re-render) ────────────────────────────
const charts = {};

// ── Utility helpers ───────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

function fmt(val, decimals = 1) {
  return (val * 100).toFixed(decimals) + "%";
}

// ── 1. Load metrics & render charts on page load ──────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  loadMetrics();
  setupForm();
});

async function loadMetrics() {
  try {
    const res  = await fetch("/metrics");
    if (!res.ok) throw new Error("Failed to load metrics.");
    const data = await res.json();
    renderMetrics(data);
    renderCharts(data);
  } catch (err) {
    console.error("Metrics load error:", err);
    $("metricsGrid").innerHTML =
      '<p style="color:var(--text-secondary);font-size:0.84rem;">Could not load model metrics. Make sure the model has been trained.</p>';
  }
}

// ── 2. Metrics cards ──────────────────────────────────────────────────────────
function renderMetrics(data) {
  const items = [
    { name: "Accuracy",   value: fmt(data.accuracy) },
    { name: "Precision",  value: fmt(data.precision) },
    { name: "Recall",     value: fmt(data.recall) },
    { name: "F1 Score",   value: fmt(data.f1) },
    { name: "ROC-AUC",    value: fmt(data.roc_auc) },
  ];

  $("metricsGrid").innerHTML = items.map(item => `
    <div class="metric-card">
      <div class="metric-name">${item.name}</div>
      <div class="metric-value">${item.value}</div>
    </div>
  `).join("");
}

// ── 3. Charts ─────────────────────────────────────────────────────────────────
const CHART_COLORS = {
  indigo:  "#6366f1",
  violet:  "#7c3aed",
  green:   "#22c55e",
  amber:   "#f59e0b",
  red:     "#ef4444",
  sky:     "#38bdf8",
  gridLine: "rgba(255,255,255,0.06)",
  tickColor: "#8b92b8",
};

const baseFont = { family: "'Inter','Segoe UI',system-ui,sans-serif", size: 12 };

function defaultScales() {
  return {
    x: {
      grid:  { color: CHART_COLORS.gridLine },
      ticks: { color: CHART_COLORS.tickColor, font: baseFont },
    },
    y: {
      grid:  { color: CHART_COLORS.gridLine },
      ticks: { color: CHART_COLORS.tickColor, font: baseFont },
      beginAtZero: true,
    },
  };
}

function renderCharts(data) {
  renderChurnDist(data);
  renderContractChart(data);
  renderConfusionMatrix(data);
  renderROCCurve(data);
}

// 3a. Churn distribution (doughnut)
function renderChurnDist(data) {
  const ctx = $("churnDistChart").getContext("2d");
  const { no_churn, churn } = data.dataset;
  charts.churnDist = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["No Churn", "Churn"],
      datasets: [{
        data: [no_churn, churn],
        backgroundColor: [CHART_COLORS.green, CHART_COLORS.red],
        borderColor: "#13162b",
        borderWidth: 3,
        hoverOffset: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "62%",
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            color: CHART_COLORS.tickColor,
            font: baseFont,
            padding: 18,
            usePointStyle: true,
          },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
              const pct   = ((ctx.raw / total) * 100).toFixed(1);
              return ` ${ctx.label}: ${ctx.raw.toLocaleString()} (${pct}%)`;
            },
          },
        },
      },
    },
  });
}

// 3b. Churn by contract (horizontal bar)
function renderContractChart(data) {
  const ctx      = $("contractChart").getContext("2d");
  const records  = data.churn_by_contract;
  const labels   = records.map(r => r["Contract"]);
  const rates    = records.map(r => parseFloat((r.churn_rate * 100).toFixed(1)));

  charts.contract = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Churn Rate (%)",
        data: rates,
        backgroundColor: [
          "rgba(239,68,68,0.75)",
          "rgba(245,158,11,0.75)",
          "rgba(34,197,94,0.75)",
        ],
        borderColor: [
          CHART_COLORS.red,
          CHART_COLORS.amber,
          CHART_COLORS.green,
        ],
        borderWidth: 1.5,
        borderRadius: 5,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` Churn rate: ${ctx.raw}%`,
          },
        },
      },
      scales: {
        x: {
          grid:  { color: CHART_COLORS.gridLine },
          ticks: {
            color: CHART_COLORS.tickColor,
            font: baseFont,
            callback: (v) => v + "%",
          },
          max: 80,
        },
        y: {
          grid:  { display: false },
          ticks: { color: CHART_COLORS.tickColor, font: baseFont },
        },
      },
    },
  });
}

// 3c. Confusion Matrix (custom rendered as 2×2 grid via bar chart trick)
function renderConfusionMatrix(data) {
  const ctx = $("cmChart").getContext("2d");
  const { tn, fp, fn, tp } = data.confusion_matrix;
  const total = tn + fp + fn + tp;

  // Use a stacked bar chart to approximate a 2×2 grid
  charts.cm = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Actual: No Churn", "Actual: Churn"],
      datasets: [
        {
          label: "Predicted: No Churn",
          data: [tn, fn],
          backgroundColor: ["rgba(34,197,94,0.75)", "rgba(239,68,68,0.75)"],
          borderColor:      ["#22c55e", "#ef4444"],
          borderWidth: 1.5,
          borderRadius: 4,
        },
        {
          label: "Predicted: Churn",
          data: [fp, tp],
          backgroundColor: ["rgba(245,158,11,0.65)", "rgba(99,102,241,0.75)"],
          borderColor:      ["#f59e0b", "#6366f1"],
          borderWidth: 1.5,
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: CHART_COLORS.tickColor, font: baseFont, padding: 14, usePointStyle: true },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const pct = ((ctx.raw / total) * 100).toFixed(1);
              return ` ${ctx.dataset.label}: ${ctx.raw.toLocaleString()} (${pct}%)`;
            },
          },
        },
      },
      scales: {
        x: {
          grid:  { display: false },
          ticks: { color: CHART_COLORS.tickColor, font: baseFont },
        },
        y: {
          grid:  { color: CHART_COLORS.gridLine },
          ticks: { color: CHART_COLORS.tickColor, font: baseFont },
          beginAtZero: true,
          title: {
            display: true,
            text: "Count",
            color: CHART_COLORS.tickColor,
            font: { size: 11 },
          },
        },
      },
    },
  });
}

// 3d. ROC Curve
function renderROCCurve(data) {
  const ctx  = $("rocChart").getContext("2d");
  const fpr  = data.roc_curve.fpr;
  const tpr  = data.roc_curve.tpr;
  const auc  = data.roc_auc;

  const points = fpr.map((f, i) => ({ x: f, y: tpr[i] }));

  charts.roc = new Chart(ctx, {
    type: "line",
    data: {
      datasets: [
        {
          label: `ROC Curve (AUC = ${(auc * 100).toFixed(1)}%)`,
          data: points,
          borderColor: CHART_COLORS.indigo,
          backgroundColor: "rgba(99,102,241,0.08)",
          fill: true,
          tension: 0.15,
          pointRadius: 0,
          pointHitRadius: 8,
          borderWidth: 2.5,
        },
        {
          label: "Random Classifier",
          data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
          borderColor: "rgba(255,255,255,0.18)",
          borderDash: [5, 5],
          pointRadius: 0,
          fill: false,
          borderWidth: 1.5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: CHART_COLORS.tickColor, font: baseFont, padding: 14, usePointStyle: true },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              if (ctx.datasetIndex !== 0) return null;
              return ` FPR: ${ctx.raw.x.toFixed(3)}  TPR: ${ctx.raw.y.toFixed(3)}`;
            },
          },
        },
      },
      scales: {
        x: {
          type: "linear",
          grid:  { color: CHART_COLORS.gridLine },
          ticks: { color: CHART_COLORS.tickColor, font: baseFont },
          title: {
            display: true,
            text: "False Positive Rate",
            color: CHART_COLORS.tickColor,
            font: { size: 11 },
          },
          min: 0, max: 1,
        },
        y: {
          grid:  { color: CHART_COLORS.gridLine },
          ticks: { color: CHART_COLORS.tickColor, font: baseFont },
          title: {
            display: true,
            text: "True Positive Rate",
            color: CHART_COLORS.tickColor,
            font: { size: 11 },
          },
          min: 0, max: 1,
        },
      },
    },
  });
}

// ── 4. Form handling & prediction ─────────────────────────────────────────────
function setupForm() {
  const form = $("customerForm");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    await runPrediction();
  });
}

async function runPrediction() {
  const btn    = $("analyzeBtn");
  const btnText   = btn.querySelector(".btn-text");
  const btnLoader = $("btnLoader");
  const errorBox  = $("errorBox");

  // Loading state
  btn.disabled = true;
  btnText.style.display = "none";
  btnLoader.style.display = "inline-flex";
  errorBox.style.display = "none";

  try {
    const payload = buildPayload();
    const res     = await fetch("/predict", {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok) {
      const msg = data.details
        ? data.details.join("\n")
        : (data.error || "An unknown error occurred.");
      showError(msg);
      return;
    }

    displayResult(data, payload);
    showProfileIndicators(payload);

    // Scroll to result
    setTimeout(() => {
      $("resultSection").scrollIntoView({ behavior: "smooth", block: "start" });
    }, 60);

  } catch (err) {
    showError("Could not reach the server. Make sure app.py is running.");
    console.error(err);
  } finally {
    btn.disabled = false;
    btnText.style.display = "inline";
    btnLoader.style.display = "none";
  }
}

function buildPayload() {
  const form  = $("customerForm");
  const data  = new FormData(form);
  const obj   = {};
  for (const [key, val] of data.entries()) {
    // Numerical fields
    if (["Tenure Months", "Monthly Charges", "Total Charges"].includes(key)) {
      obj[key] = parseFloat(val);
    } else {
      obj[key] = val;
    }
  }
  return obj;
}

function displayResult(data, payload) {
  const { prediction, churn_probability, risk_level } = data;

  const pct   = (churn_probability * 100).toFixed(1);
  const badge = $("riskBadge");

  // Probability text colour based on risk
  const probEl = $("probValue");
  probEl.textContent = pct + "%";
  if (risk_level === "HIGH") {
    probEl.style.color = "var(--high-color)";
    badge.className    = "risk-badge risk-high";
  } else if (risk_level === "MEDIUM") {
    probEl.style.color = "var(--med-color)";
    badge.className    = "risk-badge risk-med";
  } else {
    probEl.style.color = "var(--low-color)";
    badge.className    = "risk-badge risk-low";
  }

  $("riskLevel").textContent    = risk_level;
  $("predictionText").textContent = prediction;

  // Meter
  const pctNum = churn_probability * 100;
  $("meterFill").style.width   = pctNum + "%";
  $("meterNeedle").style.left  = pctNum + "%";

  // Colour needle per risk
  const needleColor = risk_level === "HIGH" ? "var(--high-color)"
    : risk_level === "MEDIUM" ? "var(--med-color)"
    : "var(--low-color)";
  $("meterNeedle").style.background    = needleColor;
  $("meterNeedle").style.boxShadow     = `0 0 10px ${needleColor}`;

  $("resultSection").style.display = "block";
  $("errorBox").style.display      = "none";
}

function showError(msg) {
  const box = $("errorBox");
  box.textContent     = msg;
  box.style.display   = "block";
  $("resultSection").style.display = "block";
  $("profileSection").style.display = "none";
  $("resultSection").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── 5. Profile indicators ─────────────────────────────────────────────────────
function showProfileIndicators(payload) {
  const tags = [];

  // Contract
  if (payload["Contract"]) {
    tags.push({ icon: "📋", text: payload["Contract"] + " contract" });
  }

  // Tenure
  const t = payload["Tenure Months"];
  if (t !== undefined) {
    if (t <= 12)       tags.push({ icon: "🕐", text: `Short tenure (${t} mo)` });
    else if (t <= 24)  tags.push({ icon: "🕑", text: `Medium tenure (${t} mo)` });
    else               tags.push({ icon: "🕒", text: `Long tenure (${t} mo)` });
  }

  // Internet
  if (payload["Internet Service"] && payload["Internet Service"] !== "No") {
    tags.push({ icon: "🌐", text: payload["Internet Service"] + " internet" });
  }

  // Tech support
  if (payload["Tech Support"] === "No") {
    tags.push({ icon: "🔧", text: "No tech support" });
  } else if (payload["Tech Support"] === "Yes") {
    tags.push({ icon: "✅", text: "Has tech support" });
  }

  // Online security
  if (payload["Online Security"] === "No") {
    tags.push({ icon: "🔓", text: "No online security" });
  }

  // Senior
  if (payload["Senior Citizen"] === "Yes") {
    tags.push({ icon: "👤", text: "Senior citizen" });
  }

  // Payment method
  if (payload["Payment Method"]) {
    tags.push({ icon: "💳", text: payload["Payment Method"] });
  }

  // Monthly charges
  const mc = payload["Monthly Charges"];
  if (mc !== undefined) {
    if (mc >= 80)      tags.push({ icon: "💰", text: `High monthly charges ($${mc})` });
    else if (mc <= 30) tags.push({ icon: "💲", text: `Low monthly charges ($${mc})` });
    else               tags.push({ icon: "💲", text: `Monthly charges: $${mc}` });
  }

  // Partner / Dependents
  if (payload["Partner"] === "No" && payload["Dependents"] === "No") {
    tags.push({ icon: "🧍", text: "No partner or dependents" });
  }

  const ul = $("profileTags");
  ul.innerHTML = tags.map(t => `<li><span>${t.icon}</span>${t.text}</li>`).join("");
  $("profileSection").style.display = "block";
}
