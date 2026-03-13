/**
 * charts.js
 * Visual rendering components for the NFL dashboard.
 * Renders win probability bars, Monte Carlo distributions,
 * calibration curves, uncertainty bands, and trend arrows.
 */

// ---------------------------------------------------------------------------
// Win Probability Bar
// ---------------------------------------------------------------------------

/**
 * Render a win probability bar showing home vs away probability.
 * @param {HTMLElement} container - element to render into
 * @param {string}  homeTeam     - home team abbreviation
 * @param {string}  awayTeam     - away team abbreviation
 * @param {number}  homeProb     - home win probability [0,1]
 */
function renderProbBar(container, homeTeam, awayTeam, homeProb) {
  const homePct = (homeProb * 100).toFixed(1);
  const awayPct = ((1 - homeProb) * 100).toFixed(1);
  const homeColor = teamColor(homeTeam);
  const awayColor = teamColor(awayTeam);

  container.innerHTML = `
    <div class="prob-bar-wrapper">
      <div class="prob-bar-team prob-bar-home" style="color:${homeColor}">
        <img src="${teamLogo(homeTeam)}" class="prob-bar-logo" alt="${homeTeam}">
        <span class="prob-bar-pct">${homePct}%</span>
        <span class="prob-bar-abbrev">${homeTeam}</span>
      </div>
      <div class="prob-bar-track">
        <div class="prob-bar-fill-home" style="width:${homePct}%;background:${homeColor}"></div>
        <div class="prob-bar-fill-away" style="width:${awayPct}%;background:${awayColor}"></div>
      </div>
      <div class="prob-bar-team prob-bar-away" style="color:${awayColor}">
        <span class="prob-bar-abbrev">${awayTeam}</span>
        <span class="prob-bar-pct">${awayPct}%</span>
        <img src="${teamLogo(awayTeam)}" class="prob-bar-logo" alt="${awayTeam}">
      </div>
    </div>
  `;
}

// ---------------------------------------------------------------------------
// Edge Indicator
// ---------------------------------------------------------------------------

/**
 * Render a market edge badge (green = model sees value, red = market disagrees).
 */
function renderEdgeBadge(container, edge, kelly) {
  if (edge == null) {
    container.innerHTML = `<span class="edge-badge edge-neutral">No odds</span>`;
    return;
  }
  const edgePct = (edge * 100).toFixed(1);
  const sign = edge >= 0 ? "+" : "";
  const cls = edge >= 0.02 ? "edge-positive" : edge <= -0.02 ? "edge-negative" : "edge-neutral";
  const kellyStr = kelly != null ? ` · Kelly: ${(kelly * 100).toFixed(1)}%` : "";
  container.innerHTML = `
    <span class="edge-badge ${cls}" title="Edge = Model prob minus Market implied prob">
      ${sign}${edgePct}% edge${kellyStr}
    </span>
  `;
}

// ---------------------------------------------------------------------------
// Monte Carlo Distribution
// ---------------------------------------------------------------------------

/**
 * Render Monte Carlo margin histogram as a simple SVG bar chart.
 * @param {HTMLElement} container
 * @param {Object} mcResult - from models.js monteCarlo()
 */
function renderMonteCarlo(container, mcResult) {
  const margin = mcResult.expected_margin.toFixed(1);
  const dir = mcResult.expected_margin >= 0 ? "Home" : "Away";
  const absMargin = Math.abs(mcResult.expected_margin).toFixed(1);
  const ci_low  = mcResult.ci_low.toFixed(1);
  const ci_high = mcResult.ci_high.toFixed(1);

  container.innerHTML = `
    <div class="mc-summary">
      <div class="mc-stat">
        <span class="mc-label">Expected Margin</span>
        <span class="mc-value">${dir} by ${absMargin} pts</span>
      </div>
      <div class="mc-stat">
        <span class="mc-label">80% CI</span>
        <span class="mc-value">${ci_low} to ${ci_high}</span>
      </div>
      <div class="mc-margins">
        <span class="mc-margin-tag">Win by 7+: ${(mcResult.prob_win_by_7 * 100).toFixed(0)}%</span>
        <span class="mc-margin-tag">Win by 14+: ${(mcResult.prob_win_by_14 * 100).toFixed(0)}%</span>
        <span class="mc-margin-tag">Win by 21+: ${(mcResult.prob_win_by_21 * 100).toFixed(0)}%</span>
      </div>
    </div>
  `;
}

// ---------------------------------------------------------------------------
// Calibration Curve
// ---------------------------------------------------------------------------

/**
 * Render calibration curve as an SVG chart.
 * @param {HTMLElement} container
 * @param {Array} buckets - from model_performance.json
 */
function renderCalibrationCurve(container, buckets) {
  const W = 280, H = 200;
  const PAD = 35;

  // Build SVG
  let points = "";
  let dots = "";
  let labels = "";

  for (const b of buckets) {
    if (b.actual_rate == null || b.n === 0) continue;
    const x = PAD + (b.predicted_avg) * (W - PAD * 2);
    const y = H - PAD - (b.actual_rate) * (H - PAD * 2);
    points += `${x},${y} `;

    const dotClass = Math.abs(b.predicted_avg - b.actual_rate) < 0.05 ? "cal-dot-good" : "cal-dot-bad";
    dots += `<circle cx="${x}" cy="${y}" r="5" class="${dotClass}" title="${b.bucket}: predicted ${(b.predicted_avg*100).toFixed(0)}% actual ${(b.actual_rate*100).toFixed(0)}% (n=${b.n})"/>`;
    labels += `<text x="${x}" y="${y - 8}" class="cal-label">${b.bucket}</text>`;
  }

  // Diagonal reference line (perfect calibration)
  const x0 = PAD + 0.5 * (W - PAD * 2);
  const y0 = H - PAD - 0.5 * (H - PAD * 2);
  const x1 = PAD + 1.0 * (W - PAD * 2);
  const y1 = H - PAD - 1.0 * (H - PAD * 2);

  container.innerHTML = `
    <svg width="${W}" height="${H}" class="cal-chart">
      <line x1="${x0}" y1="${y0}" x2="${x1}" y2="${y1}" class="cal-reference"/>
      ${points ? `<polyline points="${points.trim()}" class="cal-line" fill="none"/>` : ""}
      ${dots}
      ${labels}
      <text x="${PAD}" y="${H - 5}" class="cal-axis-label">Predicted Probability</text>
      <text x="5" y="${H/2}" class="cal-axis-label" transform="rotate(-90, 5, ${H/2})">Actual Win Rate</text>
    </svg>
    <div class="cal-legend">
      <span class="cal-dot-good-legend">● Within 5%</span>
      <span class="cal-dot-bad-legend">● &gt;5% off</span>
    </div>
  `;
}

// ---------------------------------------------------------------------------
// ELO Uncertainty Band
// ---------------------------------------------------------------------------

/**
 * Render an ELO value with uncertainty band as a visual range.
 */
function renderUncertaintyBand(container, mu, sigma) {
  const lo = (mu - sigma).toFixed(0);
  const hi = (mu + sigma).toFixed(0);
  const center = mu.toFixed(0);

  // Width of band: proportional to sigma (max ~150 pts)
  const totalRange = 300;
  const bandPct = Math.min(100, (sigma * 2 / totalRange) * 100);
  const offsetPct = 50 - bandPct / 2;

  container.innerHTML = `
    <div class="elo-band-wrapper" title="ELO: ${lo} – ${hi}">
      <span class="elo-value">${center}</span>
      <div class="elo-band-track">
        <div class="elo-band-fill" style="left:${offsetPct}%;width:${bandPct}%"></div>
      </div>
      <span class="elo-band-range">±${Math.round(sigma)}</span>
    </div>
  `;
}

// ---------------------------------------------------------------------------
// Trend Arrow
// ---------------------------------------------------------------------------

/**
 * Render a trend arrow based on recent ELO direction.
 */
function renderTrendArrow(container, trend) {
  const arrows = { up: "▲", down: "▼", neutral: "▶" };
  const classes = { up: "trend-up", down: "trend-down", neutral: "trend-neutral" };
  const arrow = arrows[trend] || "▶";
  const cls = classes[trend] || "trend-neutral";
  container.innerHTML = `<span class="${cls}">${arrow}</span>`;
}

// ---------------------------------------------------------------------------
// Pythagorean Flag
// ---------------------------------------------------------------------------

/**
 * Render overperforming / underperforming badge.
 */
function renderPythFlag(container, flag) {
  if (!flag) {
    container.innerHTML = `<span class="pyth-on-track">On track</span>`;
    return;
  }
  const cls = flag === "overperforming" ? "pyth-over" : "pyth-under";
  const label = flag === "overperforming" ? "↓ May regress" : "↑ May improve";
  container.innerHTML = `<span class="pyth-flag ${cls}">${label}</span>`;
}

// ---------------------------------------------------------------------------
// System breakdown table (for Matchup Predictor)
// ---------------------------------------------------------------------------

/**
 * Render a detailed system-by-system probability breakdown.
 */
function renderSystemBreakdown(container, prediction) {
  const rows = [
    { label: "Logistic Regression",     prob: prediction.probs.lr,   system: "System 1" },
    { label: "XGBoost",                  prob: prediction.probs.xgb,  system: "System 9" },
    { label: "ELO Rating",               prob: prediction.probs.elo,  system: "System 2" },
    { label: "Pythagorean",              prob: prediction.probs.pyth, system: "System 4" },
    { label: "Efficiency",               prob: prediction.probs.eff,  system: "System 5" },
    { label: "ENSEMBLE (Final)",         prob: prediction.final_prob,  system: "System 12", highlight: true },
  ];

  const rowsHtml = rows.map(r => `
    <tr class="${r.highlight ? 'ensemble-row' : ''}">
      <td>${r.system}</td>
      <td>${r.label}</td>
      <td class="prob-cell">
        <div class="mini-prob-bar">
          <div class="mini-prob-fill" style="width:${(r.prob*100).toFixed(1)}%"></div>
        </div>
        ${(r.prob * 100).toFixed(1)}%
      </td>
    </tr>
  `).join("");

  container.innerHTML = `
    <table class="system-breakdown-table">
      <thead><tr><th>System</th><th>Model</th><th>Home Win %</th></tr></thead>
      <tbody>${rowsHtml}</tbody>
    </table>
  `;
}
