"""
teams_config.py
Static configuration for all 32 NFL teams.
Used by Python pipeline for abbreviation mapping and fallback data.
"""

NFL_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB",  "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV",  "MIA", "MIN", "NE",  "NO",  "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF",  "TB",  "TEN", "WAS",
]

# ESPN team ID mapping (used for schedule/injury fetches)
ESPN_TEAM_IDS = {
    "ARI": "22", "ATL": "1",  "BAL": "33", "BUF": "2",  "CAR": "29",
    "CHI": "3",  "CIN": "4",  "CLE": "5",  "DAL": "6",  "DEN": "7",
    "DET": "8",  "GB":  "9",  "HOU": "34", "IND": "11", "JAX": "30",
    "KC":  "12", "LAC": "24", "LAR": "14", "LV":  "13", "MIA": "15",
    "MIN": "16", "NE":  "17", "NO":  "18", "NYG": "19", "NYJ": "20",
    "PHI": "21", "PIT": "23", "SEA": "26", "SF":  "25", "TB":  "27",
    "TEN": "10", "WAS": "28",
}

# FiveThirtyEight abbreviation -> ESPN abbreviation
FTE_TO_ESPN = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BUF": "BUF",
    "CAR": "CAR", "CHI": "CHI", "CIN": "CIN", "CLE": "CLE",
    "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GB":  "GB",
    "HOU": "HOU", "IND": "IND", "JAC": "JAX", "JAX": "JAX",
    "KC":  "KC",  "LAC": "LAC", "LA":  "LAR", "LAR": "LAR",
    "LV":  "LV",  "OAK": "LV",  "MIA": "MIA", "MIN": "MIN",
    "NE":  "NE",  "NO":  "NO",  "NYG": "NYG", "NYJ": "NYJ",
    "PHI": "PHI", "PIT": "PIT", "SD":  "LAC", "SEA": "SEA",
    "SF":  "SF",  "STL": "LAR", "TB":  "TB",  "TEN": "TEN",
    "WAS": "WAS", "WSH": "WAS",
}
