/**
 * models.js
 * All real-time prediction math running in the browser.
 * Called whenever slider values change to instantly recalculate all predictions.
 *
 * Systems implemented here:
 * - System 1: Logistic Regression inference (dot product + sigmoid)
 * - System 2: ELO adjustments (K-factor, HFA, rest, travel, H2H, injuries)
 * - System 3: Bayesian uncertainty bands
 * - System 4: Pythagorean probability
 * - System 5: Efficiency-based probability
 * - System 10: Monte Carlo simulation
 * - System 12: Ensemble blend
 * - Market edge + Kelly Criterion
 */

// ---------------------------------------------------------------------------
// Core math utilities
// ---------------------------------------------------------------------------

function sigmoid(x) {
  return 1.0 / (1.0 + Math.exp(-x));
}

function eloToProb(eloDiff) {
  return 1.0 / (1.0 + Math.pow(10, -eloDiff / 400.0));
}

function probToEloDiff(prob) {
  prob = Math.max(0.001, Math.min(0.999, prob));
  return -400.0 * Math.log10(1.0 / prob - 1.0);
}

// ---------------------------------------------------------------------------
// System 1: Logistic Regression inference
// ---------------------------------------------------------------------------

/**
 * Compute win probability using pre-trained LR weights.
 * @param {Object} features - feature values keyed by name
 * @param {Object} weights  - {intercept, coefficients: {feature: weight}}
 * @returns {number} win probability [0,1]
 */
function logisticProb(features, weights) {
  if (!weights || !weights.coefficients) return 0.5;
  let linear = weights.intercept || 0;
  for (const [fname, coef] of Object.entries(weights.coefficients)) {
    linear += coef * (features[fname] || 0);
  }
  return sigmoid(linear);
}

// ---------------------------------------------------------------------------
// System 2: ELO with real-time slider adjustments
// ---------------------------------------------------------------------------

/**
 * Apply all ELO adjustments to a game based on current slider values.
 * Returns adjusted ELO diff and win probability.
 *
 * @param {Object} game    - game data from games.json
 * @param {Object} sliders - current slider values
 * @param {Object} ratings - team_ratings.json data keyed by abbrev
 * @returns {Object} {eloDiff, eloProb, components}
 */
function applyEloAdjustments(game, sliders, ratings) {
  const home = game.home_team;
  const away = game.away_team;
  const homeRating = ratings[home] || { elo: { mu: 1500, sigma: 75 } };
  const awayRating = ratings[away] || { elo: { mu: 1500, sigma: 75 } };

  // Base ELOs from Bayesian posteriors
  let homeElo = homeRating.elo.mu;
  let awayElo = awayRating.elo.mu;

  // Injury discounts
  const homeInjury = parseFloat(sliders[`injury_${home}`] || 0);
  const awayInjury = parseFloat(sliders[`injury_${away}`] || 0);
  homeElo -= homeInjury;
  awayElo -= awayInjury;

  // Recent form blend
  const formBlend = parseFloat(sliders.form_blend || 0.30);
  const homeFormElo = homeRating.recent_form_elo || homeElo;
  const awayFormElo = awayRating.recent_form_elo || awayElo;
  homeElo = homeElo * (1 - formBlend) + homeFormElo * formBlend;
  awayElo = awayElo * (1 - formBlend) + awayFormElo * formBlend;

  // H2H adjustment (stored in game.adjustments)
  const h2hWeight = parseFloat(sliders.h2h_weight || 0.20);
  const h2hAdj = (game.adjustments && game.adjustments.h2h_adj)
    ? game.adjustments.h2h_adj * (h2hWeight / 0.20)
    : 0;
  homeElo += h2hAdj;
  awayElo -= h2hAdj;

  // Rest & travel adjustments (scaled by slider)
  const restTravelScale = parseFloat(sliders.rest_travel_scale || 1.0);
  const adjs = game.adjustments || {};
  const homeRestAdj = (adjs.home_rest_adj || 0) * restTravelScale;
  const awayRestAdj = (adjs.away_rest_adj || 0) * restTravelScale;
  const homeTravelPen = (adjs.home_travel_pen || 0) * restTravelScale;
  const awayTravelPen = (adjs.away_travel_pen || 0) * restTravelScale;
  homeElo += homeRestAdj + homeTravelPen;
  awayElo += awayRestAdj + awayTravelPen;

  // Turnover adjustment
  const toScale = parseFloat(sliders.turnover_scale || 1.0);
  const homeToAdj = (adjs.home_to_adj || 0) * toScale;
  const awayToAdj = (adjs.away_to_adj || 0) * toScale;
  homeElo += homeToAdj;
  awayElo += awayToAdj;

  // Home field advantage
  const hfa = parseFloat(sliders.home_field || 65);
  const eloDiff = (homeElo + hfa) - awayElo;
  const winProb = eloToProb(eloDiff);

  return {
    eloDiff,
    eloProb: winProb,
    homeElo,
    awayElo,
    components: {
      base_home: homeRating.elo.mu,
      base_away: awayRating.elo.mu,
      hfa,
      h2h_adj: h2hAdj,
      home_rest: homeRestAdj,
      away_rest: awayRestAdj,
      home_travel: homeTravelPen,
      away_travel: awayTravelPen,
      home_injury: homeInjury,
      away_injury: awayInjury,
    }
  };
}

// ---------------------------------------------------------------------------
// System 10: Monte Carlo Simulation
// ---------------------------------------------------------------------------

/**
 * Run N Monte Carlo simulations for one matchup.
 * Draws team strengths from Bayesian Normal(mu, sigma) posteriors.
 *
 * @param {number} homeMu    - home team ELO posterior mean
 * @param {number} homeSigma - home team ELO posterior std dev
 * @param {number} awayMu    - away team ELO posterior mean
 * @param {number} awaySigma - away team ELO posterior std dev
 * @param {number} hfa       - home field advantage in ELO points
 * @param {number} n         - number of simulations (default 10000)
 * @returns {Object} win probability, expected margin, margin distribution
 */
function monteCarlo(homeMu, homeSigma, awayMu, awaySigma, hfa = 65, n = 10000) {
  let homeWins = 0;
  let totalMargin = 0;
  let margin7plus = 0, margin14plus = 0, margin21plus = 0;
  const margins = [];

  for (let i = 0; i < n; i++) {
    // Sample from posteriors using Box-Muller transform
    const homeStrength = homeMu + homeSigma * boxMuller();
    const awayStrength = awayMu + awaySigma * boxMuller();
    const eloDiff = (homeStrength + hfa) - awayStrength;
    const prob = eloToProb(eloDiff);

    // Simulate game outcome
    const homeWin = Math.random() < prob ? 1 : 0;
    homeWins += homeWin;

    // Simulate point margin (rough approximation: eloDiff / 25 ≈ expected pts)
    const expectedMarginPts = eloDiff / 25;
    const noise = gaussianNoise(0, 10); // game-to-game variance
    const simulatedMargin = expectedMarginPts + noise;
    const actualMargin = homeWin ? Math.abs(simulatedMargin) : -Math.abs(simulatedMargin);
    totalMargin += actualMargin;
    margins.push(actualMargin);

    if (Math.abs(simulatedMargin) >= 7)  margin7plus++;
    if (Math.abs(simulatedMargin) >= 14) margin14plus++;
    if (Math.abs(simulatedMargin) >= 21) margin21plus++;
  }

  margins.sort((a, b) => a - b);
  const ci_low  = margins[Math.floor(n * 0.10)];
  const ci_high = margins[Math.floor(n * 0.90)];

  return {
    win_prob: homeWins / n,
    expected_margin: totalMargin / n,
    ci_low,
    ci_high,
    prob_win_by_7:  margin7plus / n,
    prob_win_by_14: margin14plus / n,
    prob_win_by_21: margin21plus / n,
    n_sims: n,
  };
}

function boxMuller() {
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function gaussianNoise(mean, std) {
  return mean + std * boxMuller();
}

// ---------------------------------------------------------------------------
// System 12: Ensemble blend
// ---------------------------------------------------------------------------

/**
 * Blend all probability signals using current slider weights.
 * Weights are normalized to sum to 1.
 *
 * @param {Object} probs   - {lr, xgb, elo, pyth, eff}
 * @param {Object} sliders - current slider values
 * @returns {number} blended win probability
 */
function ensembleProb(probs, sliders) {
  const w = {
    lr:   parseFloat(sliders.weight_lr   || 30) / 100,
    xgb:  parseFloat(sliders.weight_xgb  || 25) / 100,
    elo:  parseFloat(sliders.weight_elo  || 20) / 100,
    pyth: parseFloat(sliders.weight_pyth || 15) / 100,
    eff:  parseFloat(sliders.weight_eff  || 10) / 100,
  };
  const total = w.lr + w.xgb + w.elo + w.pyth + w.eff;
  if (total === 0) return 0.5;

  const blended = (
    (probs.lr   || 0.5) * w.lr   +
    (probs.xgb  || 0.5) * w.xgb  +
    (probs.elo  || 0.5) * w.elo  +
    (probs.pyth || 0.5) * w.pyth +
    (probs.eff  || 0.5) * w.eff
  ) / total;

  return Math.max(0.001, Math.min(0.999, blended));
}

// ---------------------------------------------------------------------------
// Market edge & Kelly Criterion
// ---------------------------------------------------------------------------

/**
 * Model edge = model probability - market implied probability.
 * Positive = model sees value over market.
 */
function computeEdge(modelProb, marketImpliedProb) {
  if (marketImpliedProb == null) return null;
  return modelProb - marketImpliedProb;
}

/**
 * Kelly Criterion bet sizing (for reference only — not investment advice).
 * f* = (b*p - q) / b
 * where b = decimal payout, p = model prob, q = 1-p
 */
function kellyFraction(modelProb, americanOdds, maxKelly = 0.25) {
  if (americanOdds == null) return null;
  const b = americanOdds >= 0
    ? americanOdds / 100.0
    : 100.0 / Math.abs(americanOdds);
  const p = Math.max(0.001, Math.min(0.999, modelProb));
  const q = 1 - p;
  const kelly = (b * p - q) / b;
  return Math.max(0, Math.min(maxKelly, kelly));
}

// ---------------------------------------------------------------------------
// Compute full prediction for one game (called on every slider change)
// ---------------------------------------------------------------------------

/**
 * Compute all prediction outputs for a single game with current slider values.
 * This is the main function called by app.js on every update.
 *
 * @param {Object} game    - from games.json
 * @param {Object} sliders - current slider state
 * @param {Object} ratings - team_ratings keyed by abbrev
 * @param {Object} lrWeights - from model_weights.json
 * @returns {Object} full prediction breakdown
 */
function computeGamePrediction(game, sliders, ratingsMap, lrWeights) {
  const home = game.home_team;
  const away = game.away_team;

  // System 2: ELO with adjustments
  const eloResult = applyEloAdjustments(game, sliders, ratingsMap);

  // System 1: LR with current features (features baked in from pipeline)
  const lrProb = logisticProb(game.features || {}, lrWeights);

  // XGBoost (pre-computed, not recalculated in browser)
  const xgbProb = (game.base_probs && game.base_probs.xgb_prob) || lrProb;

  // System 4: Pythagorean
  const pythProb = (game.base_probs && game.base_probs.pyth_prob) || 0.5;

  // System 5: Efficiency
  const effProb = (game.base_probs && game.base_probs.eff_prob) || 0.5;

  // System 12: Ensemble blend
  const probs = {
    lr: lrProb,
    xgb: xgbProb,
    elo: eloResult.eloProb,
    pyth: pythProb,
    eff: effProb,
  };
  const finalProb = ensembleProb(probs, sliders);

  // System 10: Monte Carlo
  const homeRating = ratingsMap[home] || { elo: { mu: 1500, sigma: 75 } };
  const awayRating = ratingsMap[away] || { elo: { mu: 1500, sigma: 75 } };
  const hfa = parseFloat(sliders.home_field || 65);
  const mc = monteCarlo(
    homeRating.elo.mu, homeRating.elo.sigma,
    awayRating.elo.mu, awayRating.elo.sigma,
    hfa, 10000
  );

  // Market edge
  const marketImplied = game.market && game.market.home_implied;
  const edge = computeEdge(finalProb, marketImplied);
  const kelly = kellyFraction(finalProb, game.market && game.market.home_american);

  return {
    home_team: home,
    away_team: away,
    final_prob: finalProb,
    probs,
    elo: eloResult,
    monte_carlo: mc,
    edge,
    kelly,
    market: game.market || {},
  };
}

/**
 * Generate a 5-sentence plain-English matchup explanation.
 */
function generateExplanation(prediction, game, ratingsMap) {
  const home = game.home_team_name || game.home_team;
  const away = game.away_team_name || game.away_team;
  const prob = (prediction.final_prob * 100).toFixed(1);
  const winner = prediction.final_prob >= 0.5 ? home : away;
  const winProb = prediction.final_prob >= 0.5
    ? (prediction.final_prob * 100).toFixed(1)
    : ((1 - prediction.final_prob) * 100).toFixed(1);

  const eloDiff = Math.abs(prediction.elo.eloDiff).toFixed(0);
  const eloFavor = prediction.elo.eloDiff >= 0 ? home : away;

  const mc = prediction.monte_carlo;
  const margin = Math.abs(mc.expected_margin).toFixed(1);
  const marginDir = mc.expected_margin >= 0 ? home : away;

  const edgeStr = prediction.edge != null
    ? (prediction.edge >= 0.02
        ? `The model sees ${(prediction.edge * 100).toFixed(1)}% edge over the market line.`
        : prediction.edge <= -0.02
          ? `The market currently disagrees, pricing ${away} ${(Math.abs(prediction.edge) * 100).toFixed(1)}% higher than the model.`
          : "The model and market are closely aligned on this game.")
    : "No market odds are available for this game.";

  const uncertainty = ratingsMap[game.home_team]
    ? `Confidence in this prediction is ${ratingsMap[game.home_team].elo.sigma < 30 ? "high" : ratingsMap[game.home_team].elo.sigma < 50 ? "moderate" : "lower"} given the sample size of games played so far this season.`
    : "Confidence reflects current season data.";

  return [
    `The model gives ${winner} a ${winProb}% chance to win this matchup.`,
    `ELO ratings favor ${eloFavor} by ${eloDiff} points after accounting for home field advantage, rest, and travel.`,
    `Monte Carlo simulation of 10,000 games projects ${marginDir} to win by an expected margin of ${margin} points (80% confidence interval: ${Math.abs(mc.ci_low).toFixed(1)} to ${Math.abs(mc.ci_high).toFixed(1)} points).`,
    edgeStr,
    uncertainty,
  ].join(" ");
}
