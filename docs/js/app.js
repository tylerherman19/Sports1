/**
 * app.js
 * Main dashboard application.
 * Loads all JSON data, renders all 5 sections, and handles recalculation
 * triggered by slider changes.
 */

// ---------------------------------------------------------------------------
// Global data store
// ---------------------------------------------------------------------------
let DATA = {
  games:       null,   // from games.json
  ratings:     null,   // from team_ratings.json (also keyed by abbrev)
  ratingsMap:  {},     // team abbrev -> rating object
  weights:     null,   // from model_weights.json
  performance: null,   // from model_performance.json
  odds:        null,   // from odds.json
};

// ---------------------------------------------------------------------------
// Bootstrap: load all data then render
// ---------------------------------------------------------------------------

async function init() {
  showLoadingState("Loading NFL prediction data...");

  try {
    const base = getDataBasePath();
    const [games, ratings, weights, performance, odds] = await Promise.all([
      fetchJSON(`${base}/games.json`),
      fetchJSON(`${base}/team_ratings.json`),
      fetchJSON(`${base}/model_weights.json`),
      fetchJSON(`${base}/model_performance.json`),
      fetchJSON(`${base}/odds.json`),
    ]);

    DATA.games       = games;
    DATA.ratings     = ratings;
    DATA.weights     = weights;
    DATA.performance = performance;
    DATA.odds        = odds;

    // Build fast lookup map
    if (ratings && ratings.teams) {
      for (const t of ratings.teams) {
        DATA.ratingsMap[t.abbrev] = t;
      }
    }

    hideLoadingState();
    updateLastUpdated(games && games.updated);

    // Initialize sliders (must come before first render)
    initSliders();
    if (ratings && ratings.teams) {
      initInjurySliders(ratings.teams);
    }

    // Render all sections
    renderAll();

    // Set up matchup predictor dropdowns
    setupMatchupPredictor();

    console.log("[app] Dashboard initialized successfully");
  } catch (err) {
    showError(`Failed to load data: ${err.message}. The pipeline may not have run yet — check GitHub Actions.`);
    console.error("[app] Init error:", err);
  }
}

// ---------------------------------------------------------------------------
// Full render (called on load and after slider changes)
// ---------------------------------------------------------------------------

function renderAll() {
  const sliders = getSliderValues();
  renderGamesSection(sliders);
  renderLeaderboard(sliders);
  renderModelPerformance();
}

// Exposed globally for sliders.js debounce
window.recalculate = renderAll;

// ---------------------------------------------------------------------------
// Section 1: This Week's Games
// ---------------------------------------------------------------------------

function renderGamesSection(sliders) {
  const container = document.getElementById("games-container");
  if (!container || !DATA.games) return;

  if (!DATA.games.games || DATA.games.games.length === 0) {
    container.innerHTML = `<p class="no-data">No games scheduled for this week. Check back when the season is active.</p>`;
    return;
  }

  const lrWeights = DATA.weights && DATA.weights.logistic_regression;
  container.innerHTML = "";

  for (const game of DATA.games.games) {
    const pred = computeGamePrediction(game, sliders, DATA.ratingsMap, lrWeights);
    const card = buildGameCard(game, pred, sliders);
    container.appendChild(card);
  }
}

function buildGameCard(game, pred, sliders) {
  const card = document.createElement("div");
  card.className = "game-card";

  const homeTeam = game.home_team;
  const awayTeam = game.away_team;
  const homeName = game.home_team_name || homeTeam;
  const awayName = game.away_team_name || awayTeam;
  const kickoff  = game.kickoff ? new Date(game.kickoff).toLocaleString("en-US", { weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }) : "";

  card.innerHTML = `
    <div class="game-card-header">
      <div class="game-teams">
        <div class="team-block away-block">
          <img src="${teamLogo(awayTeam)}" alt="${awayTeam}" class="team-logo">
          <div class="team-info">
            <span class="team-abbrev">${awayTeam}</span>
            <span class="team-record">${getRecord(awayTeam)}</span>
          </div>
        </div>
        <div class="game-meta">
          <span class="at-symbol">@</span>
          <span class="kickoff-time">${kickoff}</span>
          <span class="venue">${game.venue || ""}</span>
        </div>
        <div class="team-block home-block">
          <div class="team-info">
            <span class="team-abbrev">${homeTeam}</span>
            <span class="team-record">${getRecord(homeTeam)}</span>
          </div>
          <img src="${teamLogo(homeTeam)}" alt="${homeTeam}" class="team-logo">
        </div>
      </div>
    </div>
    <div class="game-card-body">
      <div class="prob-bar-container" id="prob-bar-${game.game_id}"></div>
      <div class="edge-container" id="edge-${game.game_id}"></div>
      <div class="elo-row">
        <span class="elo-label">ELO</span>
        <span class="elo-team">${awayTeam}: ${getElo(awayTeam)}</span>
        <span class="elo-team">${homeTeam}: ${getElo(homeTeam)}</span>
      </div>
      <div class="rest-travel-row">
        <span class="rt-item">Rest days: ${awayTeam} ${game.adjustments && game.adjustments.away_rest_days != null ? game.adjustments.away_rest_days : "?"} / ${homeTeam} ${game.adjustments && game.adjustments.home_rest_days != null ? game.adjustments.home_rest_days : "?"}</span>
        <span class="rt-item">Travel: ${(game.adjustments && game.adjustments.away_travel_dist != null) ? Math.round(game.adjustments.away_travel_dist) + " mi" : "—"}</span>
      </div>
      <div class="mc-container" id="mc-${game.game_id}"></div>
    </div>
  `;

  // Render visual components
  const probBarEl = card.querySelector(`#prob-bar-${game.game_id}`);
  if (probBarEl) renderProbBar(probBarEl, homeTeam, awayTeam, pred.final_prob);

  const edgeEl = card.querySelector(`#edge-${game.game_id}`);
  if (edgeEl) renderEdgeBadge(edgeEl, pred.edge, pred.kelly);

  const mcEl = card.querySelector(`#mc-${game.game_id}`);
  if (mcEl) renderMonteCarlo(mcEl, pred.monte_carlo);

  return card;
}

function getRecord(abbrev) {
  const r = DATA.ratingsMap[abbrev];
  return r ? r.record : "";
}

function getElo(abbrev) {
  const r = DATA.ratingsMap[abbrev];
  return r ? Math.round(r.elo.mu) : "—";
}

// ---------------------------------------------------------------------------
// Section 3: Matchup Predictor
// ---------------------------------------------------------------------------

function setupMatchupPredictor() {
  const homeSelect = document.getElementById("predictor-home");
  const awaySelect = document.getElementById("predictor-away");
  if (!homeSelect || !awaySelect) return;

  // Populate dropdowns with all 32 teams
  const teams = Object.keys(NFL_TEAMS).sort();
  for (const abbrev of teams) {
    const nameStr = teamName(abbrev);
    homeSelect.innerHTML += `<option value="${abbrev}">${nameStr}</option>`;
    awaySelect.innerHTML += `<option value="${abbrev}">${nameStr}</option>`;
  }

  // Defaults
  homeSelect.value = "KC";
  awaySelect.value = "BUF";

  homeSelect.addEventListener("change", updateMatchupPredictor);
  awaySelect.addEventListener("change", updateMatchupPredictor);

  updateMatchupPredictor();
}

function updateMatchupPredictor() {
  const homeTeam = document.getElementById("predictor-home").value;
  const awayTeam = document.getElementById("predictor-away").value;
  if (!homeTeam || !awayTeam || homeTeam === awayTeam) return;

  const sliders = getSliderValues();
  const lrWeights = DATA.weights && DATA.weights.logistic_regression;

  // Build synthetic game object for predictor
  const homeRating = DATA.ratingsMap[homeTeam] || {};
  const awayRating = DATA.ratingsMap[awayTeam] || {};

  const homeElo = (homeRating.elo && homeRating.elo.mu) || 1500;
  const awayElo = (awayRating.elo && awayRating.elo.mu) || 1500;

  const syntheticGame = {
    game_id: `predictor_${homeTeam}_${awayTeam}`,
    home_team: homeTeam,
    away_team: awayTeam,
    home_team_name: teamName(homeTeam),
    away_team_name: teamName(awayTeam),
    adjustments: {
      home_rest_days: 7,
      away_rest_days: 7,
      home_rest_adj: 0,
      away_rest_adj: 0,
      home_travel_pen: 0,
      away_travel_pen: teamDistance(awayTeam, homeTeam) / 1000 * -3,
      away_travel_dist: teamDistance(awayTeam, homeTeam),
      home_to_adj: 0,
      away_to_adj: 0,
      h2h_adj: 0,
    },
    features: {
      elo_diff: homeElo - awayElo,
      offensive_rating_diff: (homeRating.efficiency && homeRating.efficiency.off_efficiency || 1.0) - (awayRating.efficiency && awayRating.efficiency.off_efficiency || 1.0),
      defensive_rating_diff: (homeRating.efficiency && homeRating.efficiency.def_efficiency || 1.0) - (awayRating.efficiency && awayRating.efficiency.def_efficiency || 1.0),
      home_field: 1.0,
      rest_days_diff: 0,
      pythagorean_diff: (homeRating.pyth && homeRating.pyth.expectation || 0.5) - (awayRating.pyth && awayRating.pyth.expectation || 0.5),
      net_efficiency_diff: (homeRating.efficiency && homeRating.efficiency.net_efficiency || 0) - (awayRating.efficiency && awayRating.efficiency.net_efficiency || 0),
      turnover_diff_adjusted: 0,
      travel_distance_diff: teamDistance(awayTeam, homeTeam),
      last5_win_rate_diff: (homeRating.last5_win_rate || 0.5) - (awayRating.last5_win_rate || 0.5),
    },
    base_probs: {
      pyth_prob: calcPythProb(homeRating, awayRating),
      eff_prob: calcEffProb(homeRating, awayRating),
      xgb_prob: null,  // no pre-computed XGB for custom predictor
    },
    market: null,
  };

  const pred = computeGamePrediction(syntheticGame, sliders, DATA.ratingsMap, lrWeights);

  // Render results
  const resultEl = document.getElementById("predictor-result");
  if (!resultEl) return;

  resultEl.innerHTML = `
    <div class="predictor-prob-bar" id="pred-prob-bar"></div>
    <div class="predictor-systems" id="pred-systems"></div>
    <div class="predictor-mc" id="pred-mc"></div>
    <div class="predictor-elo-breakdown">
      <h4>ELO Breakdown</h4>
      <div class="elo-detail-row">
        <span>${homeTeam} base ELO: ${homeElo.toFixed(0)}</span>
        <span>HFA bonus: +${sliders.home_field}</span>
        <span>H2H adj: ${pred.elo.components.h2h_adj.toFixed(1)}</span>
      </div>
      <div class="elo-detail-row">
        <span>${awayTeam} base ELO: ${awayElo.toFixed(0)}</span>
        <span>Travel penalty: ${pred.elo.components.away_travel.toFixed(1)}</span>
      </div>
      <div class="elo-detail-row final-elo">
        <strong>Final ELO diff: ${pred.elo.eloDiff.toFixed(1)} (favors ${pred.elo.eloDiff >= 0 ? homeTeam : awayTeam})</strong>
      </div>
    </div>
    <div class="predictor-edge">
      <div id="pred-edge"></div>
    </div>
    <div class="predictor-pyth-flag">
      Home team Pythagorean: <span id="pred-pyth-home"></span>
      Away team Pythagorean: <span id="pred-pyth-away"></span>
    </div>
    <div class="predictor-uncertainty">
      <strong>Bayesian uncertainty:</strong>
      ${homeTeam}: ELO ${homeElo.toFixed(0)} ± ${(homeRating.elo && homeRating.elo.sigma || 75).toFixed(0)} |
      ${awayTeam}: ELO ${awayElo.toFixed(0)} ± ${(awayRating.elo && awayRating.elo.sigma || 75).toFixed(0)}
    </div>
    <div class="predictor-explanation">
      <h4>What This Means</h4>
      <p>${generateExplanation(pred, syntheticGame, DATA.ratingsMap)}</p>
    </div>
  `;

  renderProbBar(document.getElementById("pred-prob-bar"), homeTeam, awayTeam, pred.final_prob);
  renderSystemBreakdown(document.getElementById("pred-systems"), pred);
  renderMonteCarlo(document.getElementById("pred-mc"), pred.monte_carlo);

  const edgeEl = document.getElementById("pred-edge");
  if (edgeEl) renderEdgeBadge(edgeEl, pred.edge, pred.kelly);

  const pythHomeEl = document.getElementById("pred-pyth-home");
  if (pythHomeEl) renderPythFlag(pythHomeEl, homeRating.pyth && homeRating.pyth.flag);
  const pythAwayEl = document.getElementById("pred-pyth-away");
  if (pythAwayEl) renderPythFlag(pythAwayEl, awayRating.pyth && awayRating.pyth.flag);
}

function calcPythProb(homeRating, awayRating) {
  const ph = homeRating.pyth && homeRating.pyth.expectation || 0.5;
  const pa = awayRating.pyth && awayRating.pyth.expectation || 0.5;
  const total = ph + pa;
  return total > 0 ? ph / total : 0.5;
}

function calcEffProb(homeRating, awayRating) {
  const eh = homeRating.efficiency && homeRating.efficiency.elo_equiv || 1500;
  const ea = awayRating.efficiency && awayRating.efficiency.elo_equiv || 1500;
  return eloToProb(eh - ea);
}

// ---------------------------------------------------------------------------
// Section 4: ELO Leaderboard
// ---------------------------------------------------------------------------

let leaderboardSortKey = "elo";
let leaderboardSortAsc = false;

function renderLeaderboard(sliders) {
  const container = document.getElementById("leaderboard-container");
  if (!container || !DATA.ratings) return;

  const teams = [...(DATA.ratings.teams || [])];

  // Sort
  teams.sort((a, b) => {
    let va, vb;
    switch (leaderboardSortKey) {
      case "elo":      va = a.elo.mu; vb = b.elo.mu; break;
      case "pyth":     va = a.pyth && a.pyth.expectation || 0; vb = b.pyth && b.pyth.expectation || 0; break;
      case "eff":      va = a.efficiency && a.efficiency.net_efficiency || 0; vb = b.efficiency && b.efficiency.net_efficiency || 0; break;
      case "playoff":  va = a.playoff_prob || 0; vb = b.playoff_prob || 0; break;
      case "record":   va = a.wins || 0; vb = b.wins || 0; break;
      default:         va = a.elo.mu; vb = b.elo.mu;
    }
    return leaderboardSortAsc ? va - vb : vb - va;
  });

  // Rank
  teams.forEach((t, i) => { t._rank = i + 1; });

  let html = `
    <table class="leaderboard-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Team</th>
          <th class="sortable" data-sort="record">Record</th>
          <th class="sortable" data-sort="elo">ELO ± σ</th>
          <th class="sortable" data-sort="pyth">Pythagorean</th>
          <th class="sortable" data-sort="eff">Net Efficiency</th>
          <th class="sortable" data-sort="playoff">Playoff %</th>
          <th>Trend</th>
          <th>Flag</th>
        </tr>
      </thead>
      <tbody>
  `;

  for (const team of teams) {
    const abbrev = team.abbrev;
    const elo = team.elo;
    const pyth = team.pyth || {};
    const eff = team.efficiency || {};
    const flag = pyth.flag;
    const trend = team.elo_trend || "neutral";
    const playoffPct = ((team.playoff_prob || 0) * 100).toFixed(0);
    const netEff = eff.net_efficiency != null ? eff.net_efficiency.toFixed(3) : "—";

    html += `
      <tr class="lb-row" style="border-left: 3px solid ${teamColor(abbrev)}">
        <td class="rank-cell">${team._rank}</td>
        <td class="team-cell">
          <img src="${teamLogo(abbrev)}" alt="${abbrev}" class="lb-logo">
          <span class="lb-name">${abbrev}</span>
        </td>
        <td>${team.record || "—"}</td>
        <td>
          <div class="elo-band-container" id="lb-elo-${abbrev}"></div>
        </td>
        <td>
          <span class="${flag === 'overperforming' ? 'pyth-over' : flag === 'underperforming' ? 'pyth-under' : ''}">
            ${pyth.expectation != null ? (pyth.expectation * 100).toFixed(1) + "%" : "—"}
          </span>
        </td>
        <td>${netEff}</td>
        <td class="${parseFloat(playoffPct) >= 60 ? 'playoff-high' : parseFloat(playoffPct) >= 30 ? 'playoff-mid' : 'playoff-low'}">${playoffPct}%</td>
        <td><span id="lb-trend-${abbrev}"></span></td>
        <td><span id="lb-flag-${abbrev}"></span></td>
      </tr>
    `;
  }

  html += `</tbody></table>`;
  container.innerHTML = html;

  // Render visual components after setting innerHTML
  for (const team of teams) {
    const abbrev = team.abbrev;
    const eloEl = document.getElementById(`lb-elo-${abbrev}`);
    if (eloEl) renderUncertaintyBand(eloEl, team.elo.mu, team.elo.sigma);

    const trendEl = document.getElementById(`lb-trend-${abbrev}`);
    if (trendEl) renderTrendArrow(trendEl, team.elo_trend || "neutral");

    const flagEl = document.getElementById(`lb-flag-${abbrev}`);
    if (flagEl) renderPythFlag(flagEl, team.pyth && team.pyth.flag);
  }

  // Sortable column headers
  container.querySelectorAll(".sortable").forEach(th => {
    th.addEventListener("click", () => {
      const key = th.dataset.sort;
      if (leaderboardSortKey === key) {
        leaderboardSortAsc = !leaderboardSortAsc;
      } else {
        leaderboardSortKey = key;
        leaderboardSortAsc = false;
      }
      renderLeaderboard(getSliderValues());
    });
    th.style.cursor = "pointer";
    if (th.dataset.sort === leaderboardSortKey) {
      th.textContent += leaderboardSortAsc ? " ↑" : " ↓";
    }
  });
}

// ---------------------------------------------------------------------------
// Section 5: Model Performance
// ---------------------------------------------------------------------------

function renderModelPerformance() {
  const container = document.getElementById("performance-container");
  if (!container || !DATA.performance) return;

  const p = DATA.performance;
  const ll   = p.log_loss   != null ? p.log_loss.toFixed(4)   : "N/A";
  const bs   = p.brier_score != null ? p.brier_score.toFixed(4) : "N/A";
  const auc  = p.auc         != null ? p.auc.toFixed(4)         : "N/A";
  const n    = p.n_test_games || "?";
  const date = p.computed_on || "unknown";

  container.innerHTML = `
    <div class="performance-metrics">
      <div class="metric-card">
        <div class="metric-label">Log Loss</div>
        <div class="metric-value">${ll}</div>
        <div class="metric-desc">Probability calibration (lower = better, random = 0.693)</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Brier Score</div>
        <div class="metric-value">${bs}</div>
        <div class="metric-desc">Forecast accuracy (lower = better, max = 0.25)</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">AUC</div>
        <div class="metric-value">${auc}</div>
        <div class="metric-desc">Discrimination (higher = better, random = 0.5)</div>
      </div>
      <div class="metric-card metric-secondary">
        <div class="metric-label">Test Games</div>
        <div class="metric-value">${n}</div>
        <div class="metric-desc">Games evaluated (last 2 seasons)</div>
      </div>
    </div>
    <div class="calibration-section">
      <h4>Calibration Curve</h4>
      <p class="cal-desc">How often does the model's confidence translate to actual results? The closer each point is to the diagonal, the better-calibrated the model.</p>
      <div id="cal-chart-container"></div>
      ${buildCalibrationBucketsTable(p.calibration || [])}
    </div>
    <div class="perf-footer">Computed on ${date} using FiveThirtyEight historical data as ground truth.</div>
  `;

  const calContainer = document.getElementById("cal-chart-container");
  if (calContainer && p.calibration) {
    renderCalibrationCurve(calContainer, p.calibration);
  }
}

function buildCalibrationBucketsTable(buckets) {
  if (!buckets || buckets.length === 0) return "";
  let rows = buckets.map(b => {
    const actualStr = b.actual_rate != null ? (b.actual_rate * 100).toFixed(1) + "%" : "N/A";
    const predStr   = b.predicted_avg != null ? (b.predicted_avg * 100).toFixed(1) + "%" : "N/A";
    const diff      = b.actual_rate != null ? Math.abs(b.predicted_avg - b.actual_rate) : null;
    const cls       = diff != null && diff < 0.05 ? "cal-good" : "cal-off";
    return `<tr class="${cls}">
      <td>${b.bucket}</td>
      <td>${predStr}</td>
      <td>${actualStr}</td>
      <td>${b.n}</td>
    </tr>`;
  }).join("");
  return `
    <table class="cal-table">
      <thead><tr><th>Bucket</th><th>Model Confidence</th><th>Actual Win Rate</th><th>N Games</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

function getDataBasePath() {
  // When running on GitHub Pages, data files are relative
  // When running locally, still relative
  return "data";
}

async function fetchJSON(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${url}`);
  return resp.json();
}

function showLoadingState(msg) {
  const el = document.getElementById("loading-state");
  if (el) { el.style.display = "block"; el.textContent = msg; }
}

function hideLoadingState() {
  const el = document.getElementById("loading-state");
  if (el) el.style.display = "none";
}

function showError(msg) {
  const el = document.getElementById("error-state");
  if (el) { el.style.display = "block"; el.textContent = msg; }
  hideLoadingState();
}

function updateLastUpdated(isoStr) {
  const el = document.getElementById("last-updated");
  if (!el || !isoStr) return;
  const d = new Date(isoStr);
  el.textContent = `Data updated: ${d.toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short" })}`;
}

// Start the app on DOM ready
document.addEventListener("DOMContentLoaded", init);
