"""Every tunable number in the model lives here."""

# --- data ---
DATA_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
FIRST_SEASON = 2003            # earliest season loaded (ratings need a little history)
TEAM_RENAMES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

# --- stage 1: team ratings (weighted ridge regression on points) ---
HALF_LIFE_DAYS = 150           # a game ~5 months old counts half as much as last week's (tuned 2012-2026)
OFFSEASON_DECAY = 0.80         # extra weight multiplier for each season boundary crossed
LOOKBACK_SEASONS = 2           # seasons of history used besides the current one
RIDGE_LAMBDA = 4.0             # shrinkage of team ratings toward league average

# --- stage 2: situational adjustments (fit walk-forward from history) ---
STAGE2_MIN_SEASON = 2008       # first season whose games train stage 2 / the market blend
QB_LOOKBACK_GAMES = 6          # "usual starter" = most common QB over this many prior games

# --- probability model ---
KEY_NUMBER_SMOOTHING = 2.5     # kernel width used to find key numbers (3, 7, 10...)

# --- betting ---
DEFAULT_ODDS = -110
MIN_EV = 0.03                  # flag bets with at least +3% expected value
KELLY_FRACTION = 0.25
MAX_STAKE_PCT = 0.03
BANKROLL = 1000.0

# --- props ---
PROP_CV = {"pass_yds": 0.33, "rush_yds": 0.78, "rec_yds": 0.79}   # sd / mean of projection misses, measured by props-backtest 2025-26
PROP_NB_SIZE = {"receptions": 7.5}  # receptions vary more than a Poisson count: negative binomial with this size
PROP_COUNT = {"receptions", "pass_tds", "anytime_td"}
PROP_SCRIPT_WEIGHT = 0.5       # how much a team's projected points move its players' projections

# --- time zones (hours behind Eastern) for travel effects ---
TEAM_TZ = {
    "ARI": 3, "ATL": 0, "BAL": 0, "BUF": 0, "CAR": 0, "CHI": 1, "CIN": 0, "CLE": 0,
    "DAL": 1, "DEN": 2, "DET": 0, "GB": 1, "HOU": 1, "IND": 0, "JAX": 0, "KC": 1,
    "LA": 3, "LAC": 3, "LV": 3, "MIA": 0, "MIN": 1, "NE": 0, "NO": 1, "NYG": 0,
    "NYJ": 0, "PHI": 0, "PIT": 0, "SEA": 3, "SF": 3, "TB": 0, "TEN": 1, "WAS": 0,
}

# --- play-by-play efficiency (EPA) ---
EPA_WP_MIN = 0.05              # plays outside this win-probability window are garbage time
EPA_WP_MAX = 0.95
EPA_RIDGE_LAMBDA = 4.0         # shrinkage of EPA ratings toward average
USE_AGG = False                # coach 4th-down aggressiveness in spread/total: tested Oct 22, no gain vs the market (off)
USE_EPA = True                 # include EPA/play and success rate in the model
