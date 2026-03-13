"""
efficiency.py
System 5: Offensive and Defensive Efficiency.
Uses yards per play (YPP) adjusted for strength of schedule (SOS).
"""

import logging
from typing import Dict

import numpy as np

log = logging.getLogger(__name__)

LEAGUE_AVG_YPP = 5.5  # NFL average yards per play
DEFAULT_ELO = 1500.0
EFF_TO_ELO_SCALE = 150.0


def calculate_sos_multiplier(avg_opponent_elo: float) -> float:
    """
    SOS multiplier = average opponent ELO / 1500.
    Teams playing stronger opponents get credit for their efficiency.
    """
    return avg_opponent_elo / DEFAULT_ELO


def calculate_efficiency(
    team_ypp_offense: float,
    team_ypp_allowed: float,
    league_avg_ypp: float = LEAGUE_AVG_YPP,
    avg_opponent_elo: float = DEFAULT_ELO,
) -> dict:
    """
    Calculate offensive, defensive, and net efficiency for a team.

    Off_efficiency = (Team_YPP_offense / League_avg_YPP) * SOS_multiplier
    Def_efficiency = (League_avg_YPP_allowed / Team_YPP_allowed) * SOS_multiplier
    Net_efficiency = Off_efficiency - Def_efficiency
    """
    sos = calculate_sos_multiplier(avg_opponent_elo)
    off = (team_ypp_offense / league_avg_ypp) * sos
    def_ = (league_avg_ypp / max(team_ypp_allowed, 0.1)) * sos
    net = off - def_
    return {
        "off_efficiency": round(off, 4),
        "def_efficiency": round(def_, 4),
        "net_efficiency": round(net, 4),
        "sos_multiplier": round(sos, 4),
    }


def efficiency_to_elo(net_efficiency: float) -> float:
    """Eff_ELO = 1500 + Net_efficiency * 150"""
    return DEFAULT_ELO + net_efficiency * EFF_TO_ELO_SCALE


def compute_team_efficiencies(
    pfr_stats: Dict[str, dict],
    team_elos: Dict[str, float],
    season_games: list,
) -> Dict[str, dict]:
    """
    Compute efficiency metrics for all teams.

    pfr_stats: dict of team_abbrev -> {ypp_offense, ypp_allowed, ...}
    team_elos: current ELO ratings
    season_games: list of completed games (to calculate avg opponent ELO)

    Returns dict of team_abbrev -> efficiency metrics + ELO equiv
    """
    log.info("Computing offensive/defensive efficiencies...")

    # Calculate average opponent ELO for each team (SOS)
    opponent_elos: Dict[str, list] = {team: [] for team in team_elos}
    for game in season_games:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        if home in opponent_elos and away in team_elos:
            opponent_elos[home].append(team_elos.get(away, DEFAULT_ELO))
        if away in opponent_elos and home in team_elos:
            opponent_elos[away].append(team_elos.get(home, DEFAULT_ELO))

    # League averages from PFR data
    all_ypp_off = [s.get("ypp_offense", LEAGUE_AVG_YPP) for s in pfr_stats.values()]
    all_ypp_def = [s.get("ypp_allowed", LEAGUE_AVG_YPP) for s in pfr_stats.values()]
    league_avg = (np.mean(all_ypp_off) + np.mean(all_ypp_def)) / 2

    results = {}
    for team, stats in pfr_stats.items():
        ypp_off = float(stats.get("ypp_offense", LEAGUE_AVG_YPP))
        ypp_def = float(stats.get("ypp_allowed", LEAGUE_AVG_YPP))
        opp_elos = opponent_elos.get(team, [DEFAULT_ELO])
        avg_opp_elo = float(np.mean(opp_elos)) if opp_elos else DEFAULT_ELO

        eff = calculate_efficiency(ypp_off, ypp_def, league_avg_ypp=league_avg, avg_opponent_elo=avg_opp_elo)
        eff_elo = efficiency_to_elo(eff["net_efficiency"])

        results[team] = {
            **eff,
            "ypp_offense": round(ypp_off, 2),
            "ypp_allowed": round(ypp_def, 2),
            "avg_opp_elo": round(avg_opp_elo, 1),
            "elo_equiv": round(eff_elo, 1),
        }

    log.info(f"Efficiency computed for {len(results)} teams")
    return results
