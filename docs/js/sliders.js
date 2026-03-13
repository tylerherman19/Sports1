/**
 * sliders.js
 * Initializes all dashboard sliders with default values.
 * Handles ensemble weight locking (must sum to 100%).
 * Debounces slider changes and calls app.recalculate().
 */

// Default slider values
const SLIDER_DEFAULTS = {
  k_factor:           20,
  home_field:         65,
  form_blend:         30,    // % recent form (stored as 0-100)
  sos_weight:         1.0,
  h2h_weight:         20,    // % (stored as 0-100)
  turnover_scale:     1.0,
  rest_travel_scale:  1.0,
  time_decay_lambda:  0.05,
  weight_lr:          30,
  weight_xgb:         25,
  weight_elo:         20,
  weight_pyth:        15,
  weight_eff:         10,
};

// Current slider state (initialized from defaults)
let sliderState = { ...SLIDER_DEFAULTS };

// Debounce timer
let debounceTimer = null;
const DEBOUNCE_MS = 50;

/**
 * Initialize all sliders on page load.
 * Reads slider elements and sets up event listeners.
 */
function initSliders() {
  // Main sliders
  const mainSliders = [
    { id: "slider-k-factor",          key: "k_factor",          display: "k-factor-val",         format: v => v.toFixed(0) },
    { id: "slider-home-field",        key: "home_field",         display: "home-field-val",        format: v => v.toFixed(0) },
    { id: "slider-form-blend",        key: "form_blend",         display: "form-blend-val",        format: v => v.toFixed(0) + "%" },
    { id: "slider-sos-weight",        key: "sos_weight",         display: "sos-weight-val",        format: v => v.toFixed(1) },
    { id: "slider-h2h-weight",        key: "h2h_weight",         display: "h2h-weight-val",        format: v => v.toFixed(0) + "%" },
    { id: "slider-turnover-scale",    key: "turnover_scale",     display: "turnover-scale-val",    format: v => v.toFixed(1) },
    { id: "slider-rest-travel",       key: "rest_travel_scale",  display: "rest-travel-val",       format: v => v.toFixed(1) },
    { id: "slider-time-decay",        key: "time_decay_lambda",  display: "time-decay-val",        format: v => v.toFixed(2) },
  ];

  for (const cfg of mainSliders) {
    const el = document.getElementById(cfg.id);
    if (!el) continue;
    el.value = sliderState[cfg.key];
    updateDisplay(cfg.display, sliderState[cfg.key], cfg.format);
    el.addEventListener("input", () => {
      sliderState[cfg.key] = parseFloat(el.value);
      updateDisplay(cfg.display, sliderState[cfg.key], cfg.format);
      scheduleRecalculate();
    });
  }

  // Ensemble weight sliders (must sum to 100%)
  const weightKeys   = ["weight_lr", "weight_xgb", "weight_elo", "weight_pyth", "weight_eff"];
  const weightIds    = ["slider-weight-lr", "slider-weight-xgb", "slider-weight-elo", "slider-weight-pyth", "slider-weight-eff"];
  const weightDisplays = ["weight-lr-val", "weight-xgb-val", "weight-elo-val", "weight-pyth-val", "weight-eff-val"];

  for (let i = 0; i < weightKeys.length; i++) {
    const el = document.getElementById(weightIds[i]);
    if (!el) continue;
    el.value = sliderState[weightKeys[i]];
    updateDisplay(weightDisplays[i], sliderState[weightKeys[i]], v => v.toFixed(0) + "%");

    el.addEventListener("input", () => {
      const newVal = parseFloat(el.value);
      const oldVal = sliderState[weightKeys[i]];
      const delta = newVal - oldVal;
      sliderState[weightKeys[i]] = newVal;

      // Distribute delta proportionally among other weight sliders
      const others = weightKeys.filter((_, j) => j !== i);
      const othersTotal = others.reduce((s, k) => s + sliderState[k], 0);
      if (othersTotal > 0) {
        for (const k of others) {
          sliderState[k] = Math.max(0, sliderState[k] - delta * (sliderState[k] / othersTotal));
        }
      }

      // Normalize to exactly 100
      const total = weightKeys.reduce((s, k) => s + sliderState[k], 0);
      if (total !== 100 && total > 0) {
        for (const k of weightKeys) sliderState[k] = sliderState[k] * 100 / total;
      }

      // Update all weight slider displays
      for (let j = 0; j < weightKeys.length; j++) {
        const wEl = document.getElementById(weightIds[j]);
        if (wEl) wEl.value = sliderState[weightKeys[j]].toFixed(0);
        updateDisplay(weightDisplays[j], sliderState[weightKeys[j]], v => v.toFixed(0) + "%");
      }

      scheduleRecalculate();
    });
  }

  // Injury sliders are initialized dynamically when team data loads
  log.info("Sliders initialized");
}

/**
 * Initialize per-team injury discount sliders.
 * Called after team_ratings.json is loaded.
 * @param {Array} teams - array of team objects from team_ratings.json
 */
function initInjurySliders(teams) {
  const container = document.getElementById("injury-sliders-container");
  if (!container) return;
  container.innerHTML = "";

  for (const team of teams) {
    const abbrev = team.abbrev;
    const key = `injury_${abbrev}`;
    sliderState[key] = sliderState[key] || 0;

    const row = document.createElement("div");
    row.className = "injury-slider-row";
    row.innerHTML = `
      <img src="${teamLogo(abbrev)}" alt="${abbrev}" class="injury-team-logo">
      <span class="injury-team-name">${abbrev}</span>
      <input type="range" id="slider-injury-${abbrev}"
             min="0" max="200" step="5" value="${sliderState[key]}"
             class="slider injury-slider">
      <span id="injury-${abbrev}-val" class="slider-value">${sliderState[key]}</span>
    `;
    container.appendChild(row);

    const el = document.getElementById(`slider-injury-${abbrev}`);
    el.addEventListener("input", () => {
      sliderState[key] = parseFloat(el.value);
      document.getElementById(`injury-${abbrev}-val`).textContent = sliderState[key].toFixed(0);
      scheduleRecalculate();
    });
  }
}

/**
 * Update a display element with formatted value.
 */
function updateDisplay(elementId, value, format) {
  const el = document.getElementById(elementId);
  if (el) el.textContent = format(value);
}

/**
 * Debounced recalculate trigger.
 */
function scheduleRecalculate() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    if (typeof window.recalculate === "function") {
      window.recalculate();
    }
  }, DEBOUNCE_MS);
}

/**
 * Get current slider state as a flat object.
 * Converts form_blend and h2h_weight from 0-100 to 0-1 for model consumption.
 */
function getSliderValues() {
  return {
    ...sliderState,
    form_blend:  sliderState.form_blend / 100,
    h2h_weight:  sliderState.h2h_weight / 100,
    weight_lr:   sliderState.weight_lr,
    weight_xgb:  sliderState.weight_xgb,
    weight_elo:  sliderState.weight_elo,
    weight_pyth: sliderState.weight_pyth,
    weight_eff:  sliderState.weight_eff,
  };
}

/**
 * Reset all sliders to defaults.
 */
function resetSliders() {
  sliderState = { ...SLIDER_DEFAULTS };
  initSliders();
  scheduleRecalculate();
}

// Simple logger for slider module
const log = {
  info: (msg) => console.log(`[sliders] ${msg}`),
};
