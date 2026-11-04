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
QB_RETURN_LOOKBACK_GAMES = 17  # totals/props: not a change if he was the most common starter over this many (returning starter)

# --- probability model ---
KEY_NUMBER_SMOOTHING = 2.5     # kernel width used to find key numbers (3, 7, 10...)

# --- betting ---
DEFAULT_ODDS = -110
MIN_EV = 0.03                  # flag bets with at least +3% expected value
# Edges above this are leans, not bets, for player props and moneylines (Nov 3, 2026): a huge edge usually means
# the model is missing news (a role change, a returning teammate), not that the market is that wrong. Real closing
# lines (DraftKings + FanDuel, current model): prop bets 2025 -7.2% ROI (155) / 2026 +38.2% (139) -> capped
# +1.3% (142) / +38.8% (133); 2025 TD bets over 15% went 0-13. Moneylines walk-forward: over 15% lost in both
# 2012-19 (-6%, 21) and 2020-26 (-43%, 18). Spreads/totals mixed, no cap. Live Weeks 7-8: 8%+ edges -45% ROI (28).
MAX_BET_EDGE = 0.15
MAX_EDGE_MARKETS = {"moneyline", "prop"}
# Game totals: leans only (Nov 3, 2026). Walk-forward 2012-26 at closing lines, total bets at 3%+ edge lost in
# 11 of 15 seasons: 2012-16 -1.3% (775), 2017-21 -5.2% (533), 2022-26 -7.1% (215). A higher bar didn't fix it
# (8%+: 2012-16 +2.4%, 2017-21 -5.7%, 2022-26 -56% on 31). Shown and tracked as leans; revisit if the total model changes.
LEAN_ONLY_MARKETS = {"total"}
# Longshot moneylines (+300 or longer) must also clear the bar at DraftKings' own price; an edge that exists only at
# FanDuel's longer price is a lean (Alex's call, Nov 3, 2026, after Week 9's NYJ +420 at FD / +380 at DK). Evidence was
# thin either way: 2012-26 big-dog MLs where the model agrees with the spread paid (+44% on 60, +20% on 44), live
# +300 MLs 0-2 (leans 0-3), FanDuel-only ML edges 2025-26 too few to read. Reviewed every Tuesday.
LONGSHOT_ML_ODDS = 300
# Moneyline favorites are leans only (Nov 3, 2026, after Week 9's DEN -142, which only matched the 2.5-pt spread).
# Walk-forward 2012-26, favorite ML bets at 3%+: 2012-19 -0.6% (239), 2020-26 -3.9% (131); -150 to -200 -10.6% (57).
# A favorite's moneyline is mostly the spread at a worse price; take the spread when the model likes the side.
ML_FAVORITES_LEAN_ONLY = True
KELLY_FRACTION = 0.25
MAX_STAKE_PCT = 0.03
BANKROLL = 1000.0
# Suggested bet size (Oct 28, 2026): 1 unit on every Best bet; at plus money longer than +150 the size shrinks so a
# win pays about 1.5 units. Bigger edges did not win more often in testing, so size does not scale with edge.
UNIT_LONGSHOT_ODDS = 150
UNIT_LONGSHOT_WIN = 1.5

# --- props ---
PROP_CV = {"pass_yds": 0.33, "rush_yds": 0.78, "rec_yds": 0.79}   # sd / mean of projection misses, measured by props-backtest 2025-26
# Rushing and receiving yards are right-skewed (57% of games land below the player's average), so a normal curve
# centred on the average overrates Overs against DraftKings lines, which sit near the median. Gamma shape k fit on
# 2025 player-games: rushing k=2.5 also wins on 2026 (log loss 0.6556 vs 0.6667 normal); receiving k=2.0 wins on 2025
# (0.661 vs 0.673), ties on 2026, and wins on Week 7's real DraftKings lines (0.689 vs 0.694). Passing yards stay normal.
PROP_GAMMA_SHAPE = {"rush_yds": 2.5, "rec_yds": 2.0}
PROP_NB_SIZE = {"receptions": 7.5}  # receptions vary more than a Poisson count: negative binomial with this size
PROP_COUNT = {"receptions", "pass_tds", "anytime_td"}
PROP_SCRIPT_WEIGHT = 0.5       # how much a team's projected points move its players' projections (default)
# Per stat (Oct 28, 2026), tested on real DraftKings closing lines, 2025 wk1-6 (2,151 lines) and 2026 wk1-7
# (2,628), log loss of the live 25/75 blend. Factor = 1 + w x (team's market-implied points / its season ppg - 1).
#   pass TDs   w 0.5 -> 1.0: 0.6731 -> 0.6721 (2025), 0.6508 -> 0.6500 (2026)  better in both
#   receptions w 0.5 -> 0  : 0.6875 -> 0.6877 (2025), 0.6871 -> 0.6844 (2026)  better overall
#   passing/receiving/rushing yds: mixed between seasons for every w and baseline tried (season ppg, ppg shrunk
#   toward last season, league average), so they stay at 0.5.
PROP_SCRIPT_WEIGHT_BY_STAT = {"pass_tds": 1.0, "receptions": 0.0}
# Tested and OFF (Oct 28): pulling receivers' receiving yds and the QB's passing yds toward each other when they
# disagree (QB proj / sum of receivers' proj vs its median 1.25). Receiving yds +0.0003 both seasons (noise-sized),
# passing yds and receptions better in one season, worse in the other.

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
