/**
 * teams.js
 * Static configuration for all 32 NFL teams.
 * Includes: primary color, city coordinates, ESPN ID, full name.
 * Used for team color theming, travel distance calculation, and logo URLs.
 */

const NFL_TEAMS = {
  ARI: { name: "Arizona Cardinals",       color: "#97233F", lat: 33.5277, lon: -112.2626, espnId: "22" },
  ATL: { name: "Atlanta Falcons",          color: "#A71930", lat: 33.7550, lon: -84.4010,  espnId: "1"  },
  BAL: { name: "Baltimore Ravens",         color: "#241773", lat: 39.2781, lon: -76.6227,  espnId: "33" },
  BUF: { name: "Buffalo Bills",            color: "#00338D", lat: 42.7738, lon: -78.7870,  espnId: "2"  },
  CAR: { name: "Carolina Panthers",        color: "#0085CA", lat: 35.2258, lon: -80.8531,  espnId: "29" },
  CHI: { name: "Chicago Bears",            color: "#0B162A", lat: 41.8623, lon: -87.6167,  espnId: "3"  },
  CIN: { name: "Cincinnati Bengals",       color: "#FB4F14", lat: 39.0954, lon: -84.5160,  espnId: "4"  },
  CLE: { name: "Cleveland Browns",         color: "#311D00", lat: 41.5061, lon: -81.6995,  espnId: "5"  },
  DAL: { name: "Dallas Cowboys",           color: "#003594", lat: 32.7473, lon: -97.0945,  espnId: "6"  },
  DEN: { name: "Denver Broncos",           color: "#FB4F14", lat: 39.7439, lon: -105.0201, espnId: "7"  },
  DET: { name: "Detroit Lions",            color: "#0076B6", lat: 42.3400, lon: -83.0456,  espnId: "8"  },
  GB:  { name: "Green Bay Packers",        color: "#203731", lat: 44.5013, lon: -88.0622,  espnId: "9"  },
  HOU: { name: "Houston Texans",           color: "#03202F", lat: 29.6847, lon: -95.4107,  espnId: "34" },
  IND: { name: "Indianapolis Colts",       color: "#002C5F", lat: 39.7601, lon: -86.1639,  espnId: "11" },
  JAX: { name: "Jacksonville Jaguars",     color: "#006778", lat: 30.3239, lon: -81.6373,  espnId: "30" },
  KC:  { name: "Kansas City Chiefs",       color: "#E31837", lat: 39.0489, lon: -94.4839,  espnId: "12" },
  LAC: { name: "Los Angeles Chargers",     color: "#0080C6", lat: 33.8644, lon: -118.2611, espnId: "24" },
  LAR: { name: "Los Angeles Rams",         color: "#003594", lat: 33.9534, lon: -118.3393, espnId: "14" },
  LV:  { name: "Las Vegas Raiders",        color: "#000000", lat: 36.0909, lon: -115.1833, espnId: "13" },
  MIA: { name: "Miami Dolphins",           color: "#008E97", lat: 25.9580, lon: -80.2389,  espnId: "15" },
  MIN: { name: "Minnesota Vikings",        color: "#4F2683", lat: 44.9736, lon: -93.2575,  espnId: "16" },
  NE:  { name: "New England Patriots",     color: "#002244", lat: 42.0909, lon: -71.2643,  espnId: "17" },
  NO:  { name: "New Orleans Saints",       color: "#D3BC8D", lat: 29.9511, lon: -90.0812,  espnId: "18" },
  NYG: { name: "New York Giants",          color: "#0B2265", lat: 40.8135, lon: -74.0745,  espnId: "19" },
  NYJ: { name: "New York Jets",            color: "#125740", lat: 40.8135, lon: -74.0745,  espnId: "20" },
  PHI: { name: "Philadelphia Eagles",      color: "#004C54", lat: 39.9007, lon: -75.1675,  espnId: "21" },
  PIT: { name: "Pittsburgh Steelers",      color: "#FFB612", lat: 40.4468, lon: -80.0158,  espnId: "23" },
  SEA: { name: "Seattle Seahawks",         color: "#002244", lat: 47.5952, lon: -122.3316, espnId: "26" },
  SF:  { name: "San Francisco 49ers",      color: "#AA0000", lat: 37.4033, lon: -121.9694, espnId: "25" },
  TB:  { name: "Tampa Bay Buccaneers",     color: "#D50A0A", lat: 27.9759, lon: -82.5033,  espnId: "27" },
  TEN: { name: "Tennessee Titans",         color: "#0C2340", lat: 36.1664, lon: -86.7713,  espnId: "10" },
  WAS: { name: "Washington Commanders",    color: "#5A1414", lat: 38.9076, lon: -76.8645,  espnId: "28" },
};

/**
 * Calculate distance in miles between two teams' cities using Haversine formula.
 */
function teamDistance(abbrev1, abbrev2) {
  const t1 = NFL_TEAMS[abbrev1];
  const t2 = NFL_TEAMS[abbrev2];
  if (!t1 || !t2) return 0;
  const R = 3958.8;
  const lat1 = t1.lat * Math.PI / 180;
  const lat2 = t2.lat * Math.PI / 180;
  const dLat = (t2.lat - t1.lat) * Math.PI / 180;
  const dLon = (t2.lon - t1.lon) * Math.PI / 180;
  const a = Math.sin(dLat/2)**2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon/2)**2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}

/**
 * Get ESPN logo URL for a team abbreviation.
 */
function teamLogo(abbrev) {
  return `https://a.espncdn.com/i/teamlogos/nfl/500/${abbrev.toLowerCase()}.png`;
}

/**
 * Get primary color for a team (with fallback).
 */
function teamColor(abbrev) {
  return (NFL_TEAMS[abbrev] && NFL_TEAMS[abbrev].color) || "#4a9eff";
}

/**
 * Get full team name.
 */
function teamName(abbrev) {
  return (NFL_TEAMS[abbrev] && NFL_TEAMS[abbrev].name) || abbrev;
}
