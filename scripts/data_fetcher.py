"""
data_fetcher.py
Fetches all external data: FiveThirtyEight historical CSV, ESPN unofficial API,
The Odds API market lines, and Pro Football Reference efficiency stats.
"""

import os
import time
import math
import json
import logging
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FTE_CSV_URL = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nfl-elo/nfl_elo.csv"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
ODDS_BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
PFR_BASE = "https://www.pro-football-reference.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NFLDashboard/1.0)"
}


# ---------------------------------------------------------------------------
# FiveThirtyEight historical data
# ---------------------------------------------------------------------------

def fetch_fte_historical() -> pd.DataFrame:
    """Download the FiveThirtyEight NFL ELO CSV and return a cleaned DataFrame."""
    log.info("Fetching FiveThirtyEight historical NFL ELO data...")
    df = pd.read_csv(FTE_CSV_URL)
    df["date"] = pd.to_datetime(df["date"])
    # Keep only completed games with known results
    completed = df[df["score1"].notna() & df["score2"].notna()].copy()
    completed["season"] = completed["season"].astype(int)
    completed["home_win"] = (completed["score1"] > completed["score2"]).astype(int)
    completed["point_diff"] = completed["score1"] - completed["score2"]
    log.info(f"FTE data: {len(completed)} completed games across {completed['season'].nunique()} seasons")
    return completed


# ---------------------------------------------------------------------------
# ESPN unofficial API
# ---------------------------------------------------------------------------

def _espn_get(path: str, params: dict = None) -> dict:
    url = f"{ESPN_BASE}/{path}"
    r = requests.get(url, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_espn_scoreboard() -> list:
    """Return list of this week's games from ESPN scoreboard."""
    log.info("Fetching ESPN scoreboard...")
    data = _espn_get("scoreboard")
    games = []
    for event in data.get("events", []):
        comp = event["competitions"][0]
        home = next(t for t in comp["competitors"] if t["homeAway"] == "home")
        away = next(t for t in comp["competitors"] if t["homeAway"] == "away")
        status = comp["status"]["type"]["name"]  # STATUS_SCHEDULED, STATUS_FINAL, etc.
        game = {
            "game_id": event["id"],
            "name": event["name"],
            "kickoff": event["date"],
            "status": status,
            "home_team": home["team"]["abbreviation"],
            "home_team_id": home["team"]["id"],
            "home_team_name": home["team"]["displayName"],
            "home_score": int(home.get("score", 0) or 0),
            "away_team": away["team"]["abbreviation"],
            "away_team_id": away["team"]["id"],
            "away_team_name": away["team"]["displayName"],
            "away_score": int(away.get("score", 0) or 0),
            "venue": comp.get("venue", {}).get("fullName", ""),
            "neutral_site": comp.get("neutralSite", False),
        }
        games.append(game)
    log.info(f"ESPN scoreboard: {len(games)} games found")
    return games


def fetch_espn_standings() -> dict:
    """Return dict of team_abbrev -> {wins, losses, ties, points_for, points_against}."""
    log.info("Fetching ESPN standings...")
    data = _espn_get("standings")
    standings = {}
    for conf in data.get("children", []):
        for div in conf.get("children", []):
            for entry in div.get("standings", {}).get("entries", []):
                team = entry["team"]["abbreviation"]
                stats = {s["name"]: s["value"] for s in entry.get("stats", [])}
                standings[team] = {
                    "wins": int(stats.get("wins", 0)),
                    "losses": int(stats.get("losses", 0)),
                    "ties": int(stats.get("ties", 0)),
                    "points_for": float(stats.get("pointsFor", 0)),
                    "points_against": float(stats.get("pointsAgainst", 0)),
                    "games_played": int(stats.get("gamesPlayed", 0)),
                    "win_pct": float(stats.get("winPercent", 0)),
                }
    log.info(f"ESPN standings: {len(standings)} teams")
    return standings


def fetch_espn_injuries() -> dict:
    """Return dict of team_abbrev -> list of injured players."""
    log.info("Fetching ESPN injuries...")
    try:
        data = _espn_get("injuries")
        injuries = {}
        for item in data.get("items", []):
            team = item.get("team", {}).get("abbreviation", "")
            players = []
            for inj in item.get("injuries", []):
                players.append({
                    "name": inj.get("athlete", {}).get("displayName", ""),
                    "position": inj.get("athlete", {}).get("position", {}).get("abbreviation", ""),
                    "status": inj.get("status", ""),
                })
            if team:
                injuries[team] = players
        log.info(f"Injuries: {sum(len(v) for v in injuries.values())} across {len(injuries)} teams")
        return injuries
    except Exception as e:
        log.warning(f"Could not fetch injuries: {e}")
        return {}


def fetch_espn_team_schedule(team_id: str) -> list:
    """Return completed games for a team to calculate rest days."""
    try:
        data = _espn_get(f"teams/{team_id}/schedule")
        games = []
        for event in data.get("events", []):
            comp = event["competitions"][0]
            status = comp["status"]["type"]["name"]
            if status == "STATUS_FINAL":
                games.append({
                    "date": event["date"],
                    "opponent": "",
                    "result": comp["status"]["type"]["description"],
                })
        return sorted(games, key=lambda x: x["date"])
    except Exception as e:
        log.warning(f"Could not fetch schedule for team {team_id}: {e}")
        return []


def calculate_rest_days(team_id: str) -> int:
    """Calculate days since last game for a team."""
    schedule = fetch_espn_team_schedule(team_id)
    if not schedule:
        return 7  # default
    last_game_date = datetime.fromisoformat(schedule[-1]["date"].replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    return max(0, (now - last_game_date).days)


# ---------------------------------------------------------------------------
# The Odds API
# ---------------------------------------------------------------------------

def fetch_odds(api_key: str) -> list:
    """Fetch current NFL game odds and return list of market data."""
    if not api_key:
        log.warning("No ODDS_API_KEY set — skipping odds fetch")
        return []
    log.info("Fetching odds from The Odds API...")
    try:
        r = requests.get(
            f"{ODDS_BASE}/odds",
            params={
                "apiKey": api_key,
                "regions": "us",
                "markets": "h2h",
                "oddsFormat": "american",
            },
            headers=HEADERS,
            timeout=15,
        )
        r.raise_for_status()
        raw = r.json()
        games = []
        for game in raw:
            home_name = game["home_team"]
            away_name = game["away_team"]
            best_home_odds = None
            best_away_odds = None
            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market["key"] != "h2h":
                        continue
                    for outcome in market["outcomes"]:
                        odds = outcome["price"]
                        if outcome["name"] == home_name:
                            if best_home_odds is None or abs(odds) < abs(best_home_odds):
                                best_home_odds = odds
                        elif outcome["name"] == away_name:
                            if best_away_odds is None or abs(odds) < abs(best_away_odds):
                                best_away_odds = odds
            if best_home_odds is not None and best_away_odds is not None:
                home_implied = american_to_implied(best_home_odds)
                away_implied = american_to_implied(best_away_odds)
                total = home_implied + away_implied
                games.append({
                    "home_team_full": home_name,
                    "away_team_full": away_name,
                    "home_american": best_home_odds,
                    "away_american": best_away_odds,
                    "home_implied_raw": home_implied,
                    "away_implied_raw": away_implied,
                    "home_implied": home_implied / total,  # vig removed
                    "away_implied": away_implied / total,
                    "commence_time": game["commence_time"],
                })
        log.info(f"Odds: {len(games)} games with market data")
        return games
    except Exception as e:
        log.warning(f"Could not fetch odds: {e}")
        return []


def american_to_implied(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


# ---------------------------------------------------------------------------
# Pro Football Reference scraper
# ---------------------------------------------------------------------------

def scrape_pfr_efficiency() -> dict:
    """
    Scrape team offensive/defensive YPP and turnover data from PFR.
    Returns dict of team_abbrev -> stats.
    Uses 3-second delays between requests.
    """
    log.info("Scraping Pro Football Reference for efficiency data...")
    stats = {}

    # PFR team abbreviation mapping (PFR abbrev -> ESPN abbrev)
    pfr_to_espn = {
        "crd": "ARI", "atl": "ATL", "rav": "BAL", "buf": "BUF", "car": "CAR",
        "chi": "CHI", "cin": "CIN", "cle": "CLE", "dal": "DAL", "den": "DEN",
        "det": "DET", "gnb": "GB",  "htx": "HOU", "clt": "IND", "jax": "JAX",
        "kan": "KC",  "rai": "LV",  "sdg": "LAC", "ram": "LAR", "mia": "MIA",
        "min": "MIN", "nwe": "NE",  "nor": "NO",  "nyg": "NYG", "nyj": "NYJ",
        "phi": "PHI", "pit": "PIT", "sea": "SEA", "sfo": "SF",  "tam": "TB",
        "oti": "TEN", "was": "WAS",
    }

    try:
        # Offensive stats
        time.sleep(3)
        url = f"{PFR_BASE}/leagues/NFL/2024/passing.htm"
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        # Also get team rushing and total offense
        time.sleep(3)
        url = f"{PFR_BASE}/leagues/NFL/2024/opp.htm"
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")

        # Parse the "Team Defense" table
        table = soup.find("table", id="team_stats")
        if table is None:
            table = soup.find("table")
        if table is not None:
            rows = table.find("tbody").find_all("tr", class_=lambda c: c != "thead")
            for row in rows:
                cells = row.find_all(["th", "td"])
                if not cells:
                    continue
                team_cell = row.find("td", {"data-stat": "team"})
                if team_cell is None:
                    continue
                pfr_abbrev = team_cell.get_text(strip=True).lower().replace("*", "").replace("+", "")
                espn_abbrev = pfr_to_espn.get(pfr_abbrev, pfr_abbrev.upper())

                def get_stat(data_stat):
                    cell = row.find("td", {"data-stat": data_stat})
                    if cell is None:
                        return 0.0
                    try:
                        return float(cell.get_text(strip=True).replace(",", "") or 0)
                    except ValueError:
                        return 0.0

                stats[espn_abbrev] = {
                    "ypp_allowed": get_stat("yards_per_play"),
                    "turnovers_forced": get_stat("turnovers"),
                    "games": get_stat("g") or 17,
                }

        # Offensive YPP
        time.sleep(3)
        url = f"{PFR_BASE}/leagues/NFL/2024/offense.htm"
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        table = soup.find("table", id="team_stats")
        if table is not None:
            rows = table.find("tbody").find_all("tr", class_=lambda c: c != "thead")
            for row in rows:
                team_cell = row.find("td", {"data-stat": "team"})
                if team_cell is None:
                    continue
                pfr_abbrev = team_cell.get_text(strip=True).lower().replace("*", "").replace("+", "")
                espn_abbrev = pfr_to_espn.get(pfr_abbrev, pfr_abbrev.upper())

                def get_stat(data_stat):
                    cell = row.find("td", {"data-stat": data_stat})
                    if cell is None:
                        return 0.0
                    try:
                        return float(cell.get_text(strip=True).replace(",", "") or 0)
                    except ValueError:
                        return 0.0

                ypp = get_stat("yards_per_play")
                turnovers = get_stat("turnovers")
                if espn_abbrev not in stats:
                    stats[espn_abbrev] = {"ypp_allowed": 5.5, "turnovers_forced": 20, "games": 17}
                stats[espn_abbrev]["ypp_offense"] = ypp
                stats[espn_abbrev]["turnovers_committed"] = turnovers

        log.info(f"PFR efficiency: {len(stats)} teams scraped")
    except Exception as e:
        log.warning(f"PFR scrape failed: {e}. Using estimated values.")
        # Fill with league-average fallbacks
        from teams_config import NFL_TEAMS
        for abbrev in NFL_TEAMS:
            if abbrev not in stats:
                stats[abbrev] = {
                    "ypp_offense": 5.5,
                    "ypp_allowed": 5.5,
                    "turnovers_committed": 20,
                    "turnovers_forced": 20,
                    "games": 17,
                }

    # Ensure all teams have both offense and defense
    for abbrev, s in stats.items():
        if "ypp_offense" not in s:
            s["ypp_offense"] = 5.5
        if "ypp_allowed" not in s:
            s["ypp_allowed"] = 5.5
        if "turnovers_committed" not in s:
            s["turnovers_committed"] = 20
        if "turnovers_forced" not in s:
            s["turnovers_forced"] = 20

    return stats
