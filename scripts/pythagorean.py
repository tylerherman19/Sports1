"""
pythagorean.py
System 4: Pythagorean Win Expectation.
Uses NFL-optimized exponent of 2.37.
Flags teams where actual win rate deviates from Pythagorean by > 0.10.
"""

import logging
from typing import Dict, Optional

log = logging.getLogger(__name__)

PYTH_EXPONENT = 2.37
REGRESSION_THRESHOLD = 0.10
DEFAULT_ELO = 1500.0


def pythagorean_expectation(points_for: float, points_against: float, exp: float = PYTH_EXPONENT) -> float:
    """
    Pythagorean expectation: PF^exp / (PF^exp + PA^exp)
    Returns expected win probability (0 to 1).
    """
    if points_for <= 0 or points_against <= 0:
        return 0.5
    pf_exp = points_for ** exp
    pa_exp = points_against ** exp
    return pf_exp / (pf_exp + pa_exp)


def pyth_to_elo(pyth: float) -> float:
    """Convert Pythagorean expectation to ELO equivalent."""
    return DEFAULT_ELO + (pyth - 0.5) * 400.0


def flag_regression_candidate(
    actual_win_rate: float,
    pyth_win_rate: float,
    threshold: float = REGRESSION_THRESHOLD,
) -> Optional[str]:
    """
    Returns:
    - 'overperforming'  if actual wins > Pyth by threshold (likely to regress down)
    - 'underperforming' if actual wins < Pyth by threshold (likely to improve)
    - None if within expected range
    """
    diff = actual_win_rate - pyth_win_rate
    if diff > threshold:
        return "overperforming"
    elif diff < -threshold:
        return "underperforming"
    return None


def compute_team_pythagorean(
    standings: Dict[str, dict],
) -> Dict[str, dict]:
    """
    Compute Pythagorean expectation for all teams from standings data.
    standings: dict of team_abbrev -> {wins, losses, ties, points_for, points_against, games_played}
    Returns dict of team_abbrev -> {expectation, elo_equiv, actual_win_rate, flag}
    """
    log.info("Computing Pythagorean win expectations...")
    results = {}
    for team, s in standings.items():
        pf = float(s.get("points_for", 0))
        pa = float(s.get("points_against", 0))
        gp = int(s.get("games_played", 1)) or 1
        wins = float(s.get("wins", 0))
        losses = float(s.get("losses", 0))
        ties = float(s.get("ties", 0))

        pyth = pythagorean_expectation(pf, pa)
        actual_win_rate = (wins + 0.5 * ties) / gp if gp > 0 else 0.5
        elo_equiv = pyth_to_elo(pyth)
        flag = flag_regression_candidate(actual_win_rate, pyth)

        results[team] = {
            "expectation": round(pyth, 4),
            "elo_equiv": round(elo_equiv, 1),
            "actual_win_rate": round(actual_win_rate, 4),
            "points_for": pf,
            "points_against": pa,
            "games_played": gp,
            "flag": flag,
        }

    overperforming = sum(1 for v in results.values() if v["flag"] == "overperforming")
    underperforming = sum(1 for v in results.values() if v["flag"] == "underperforming")
    log.info(f"Pythagorean: {overperforming} overperforming, {underperforming} underperforming")
    return results
