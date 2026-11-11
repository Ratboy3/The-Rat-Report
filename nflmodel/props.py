"""Player prop pricer: your projection + game script from the team model -> P(over) and EV."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_d, nbinom, norm, poisson

from . import config as C
from .betting import ev, implied, kelly_stake


def price_props(props: pd.DataFrame, week_proj: pd.DataFrame, team_ppg: dict[str, float]) -> pd.DataFrame:
    """props columns: player, stat, team, projection, line, over_odds, under_odds
    stat: pass_yds | rush_yds | rec_yds | receptions | pass_tds | anytime_td
    For anytime_td, projection is your probability of scoring (0-1)."""
    team_pts = {}
    for g in week_proj.itertuples():
        team_pts[g.home_team] = g.proj_home_pts
        team_pts[g.away_team] = g.proj_away_pts

    rows = []
    for p in props.itertuples():
        factor = 1.0
        if p.team in team_pts and team_ppg.get(p.team):
            factor = 1 + C.PROP_SCRIPT_WEIGHT * (team_pts[p.team] / team_ppg[p.team] - 1)
        if p.stat == "anytime_td":
            lam = -math.log(1 - min(0.95, max(0.01, p.projection))) * factor
            adj = 1 - math.exp(-lam)
            over = (adj, 0.0, 1 - adj)
        elif p.stat in C.PROP_CV:
            adj = p.projection * factor
            sd = max(1.0, adj * C.PROP_CV[p.stat])
            o = 1 - norm.cdf(p.line, adj, sd)
            over = (o, 0.0, 1 - o)
        elif p.stat in C.PROP_COUNT:
            adj = max(0.01, p.projection * factor)
            k = math.floor(p.line)
            if float(p.line).is_integer():
                push = poisson.pmf(k, adj)
                over = (1 - poisson.cdf(k, adj), push, poisson.cdf(k - 1, adj))
            else:
                over = (1 - poisson.cdf(k, adj), 0.0, poisson.cdf(k, adj))
        else:
            raise ValueError(f"unknown stat '{p.stat}' for {p.player}")
        under = (over[2], over[1], over[0])
        e_o, e_u = ev(over, p.over_odds), ev(under, p.under_odds)
        best, probs, odds = ("Over", over, p.over_odds) if e_o >= e_u else ("Under", under, p.under_odds)
        rows.append(dict(player=p.player, stat=p.stat, team=p.team, projection=p.projection,
                         script_factor=round(factor, 3), adj_projection=round(adj, 2), line=p.line,
                         p_over=round(over[0], 3), best_side=best, ev=round(max(e_o, e_u), 4),
                         stake=round(kelly_stake(probs, odds), 2),
                         flag=max(e_o, e_u) >= C.MIN_EV and p.stat not in LEAN_ONLY_STATS))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ automatic props
def _script_factor(team: str, team_pts: dict, team_ppg: dict, stat: str | None = None) -> float:
    w = C.PROP_SCRIPT_WEIGHT_BY_STAT.get(stat, C.PROP_SCRIPT_WEIGHT)
    if team in team_pts and team_ppg.get(team):
        return 1 + w * (team_pts[team] / team_ppg[team] - 1)
    return 1.0


def _dist(stat: str, adj: float) -> dict:
    """Distribution the page's line checker uses: normal (mean, sd), poisson (lam) or td (p)."""
    if stat == "anytime_td":
        return dict(dist="td", a=adj, b=0.0)
    if stat in C.PROP_GAMMA_SHAPE:
        return dict(dist="gamma", a=max(0.5, adj), b=C.PROP_GAMMA_SHAPE[stat])
    if stat in C.PROP_CV:
        return dict(dist="normal", a=adj, b=max(1.0, adj * C.PROP_CV[stat]))
    if stat in C.PROP_NB_SIZE:
        return dict(dist="nbinom", a=adj, b=C.PROP_NB_SIZE[stat])
    return dict(dist="poisson", a=adj, b=0.0)


def _cdf(d: dict, k: int) -> float:
    if k < 0:
        return 0.0
    if d["dist"] == "nbinom":
        r = d["b"]
        return float(nbinom.cdf(k, r, r / (r + max(1e-6, d["a"]))))
    return float(poisson.cdf(k, d["a"]))


def _probs(d: dict, line: float) -> tuple[float, float, float]:
    """(over, push, under) at a line."""
    if d["dist"] == "td":
        return (d["a"], 0.0, 1 - d["a"])
    if d["dist"] == "normal":
        o = 1 - norm.cdf(line, d["a"], d["b"])
        return (o, 0.0, 1 - o)
    if d["dist"] == "gamma":
        o = float(gamma_d.sf(line, d["b"], scale=d["a"] / d["b"]))
        return (o, 0.0, 1 - o)
    k = math.floor(line)
    if float(line).is_integer():
        return (1 - _cdf(d, k), _cdf(d, k) - _cdf(d, k - 1), _cdf(d, k - 1))
    return (1 - _cdf(d, k), 0.0, _cdf(d, k))


def _fair_line(d: dict) -> float | None:
    if d["dist"] == "td":
        return None
    if d["dist"] == "normal":
        return round(d["a"] * 2) / 2 if round(d["a"] * 2) % 2 == 1 else math.floor(d["a"]) + 0.5
    if d["dist"] == "gamma":            # skewed: the 50/50 line is the median, below the average
        return math.floor(float(gamma_d.median(d["b"], scale=d["a"] / d["b"]))) + 0.5
    best = min((abs(_probs(d, k + 0.5)[0] - 0.5), k + 0.5) for k in range(0, 15))
    return best[1]


ALIASES = {"zonovan knight": "bam knight", "hollywood brown": "marquise brown"}


def _key(name) -> str:
    """Match player names across sources: no accents, punctuation or Jr./Sr./II-style suffixes."""
    import re
    import unicodedata
    n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"[^a-z ]", "", n.replace("-", " "))
    n = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", n.strip())
    n = re.sub(r"\s+", " ", n)
    return ALIASES.get(n, n)


PROP_MODEL_WEIGHT = 0.25  # yards/receptions/pass TDs: 25% model + 75% the DraftKings line (the market knows roles and injuries)
# Per stat (Nov 3, 2026), real DraftKings closing lines with replayed projections, 2025 wk5-18 (4,995 lines) and
# 2026 wk3-8 (2,641), fit on one season and checked on the other: receiving and rushing yards are more accurate at
# 40% model in both seasons (rec 0.6916 -> 0.6911 / 0.6910 -> 0.6902; rush 0.6915 -> 0.6913 / 0.6919 -> 0.6917).
# Pass TDs and receptions were mixed between seasons (stay 25%); passing yards were best at 0% (leans only).
PROP_MODEL_WEIGHT_BY_STAT = {"rec_yds": 0.40, "rush_yds": 0.40}


def model_weight(stat: str) -> float:
    return PROP_MODEL_WEIGHT_BY_STAT.get(stat, PROP_MODEL_WEIGHT)


def _no_vig_over(oo, uo) -> float | None:
    def imp(o):
        return 100 / (o + 100) if o > 0 else -o / (-o + 100)
    if oo is None or uo is None or pd.isna(oo) or pd.isna(uo):
        return None
    a, b = imp(oo), imp(uo)
    return a / (a + b)


def _market_mean(stat: str, line: float, p_over: float | None) -> float:
    """The mean the DraftKings line and prices imply."""
    p = 0.5 if p_over is None else min(0.9, max(0.1, p_over))
    if stat in C.PROP_GAMMA_SHAPE:        # P(X > line) = p with X = mean * Gamma(k, 1/k)
        k = C.PROP_GAMMA_SHAPE[stat]
        return line / max(0.05, float(gamma_d.isf(p, k, scale=1 / k)))
    if stat in C.PROP_CV:
        cv = C.PROP_CV[stat]
        z = norm.ppf(p)                   # P(X > line) = p with X ~ N(m, cv*m)  ->  m = line / (1 - z*cv)
        return line / max(0.3, 1 - z * cv)
    lo, hi = 0.01, 30.0                   # count stats: find the mean with P(X > line) = p
    for _ in range(60):
        mid = (lo + hi) / 2
        if 1 - _cdf(_dist(stat, mid), math.floor(line)) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _trust(base: float, model: float, market: float) -> float:
    """Model weight, shrunk when the model and market disagree a lot (usually a role change or injury the
    season averages haven't caught up with): 1.2x apart keeps ~80% of the weight, 2x apart ~25%, 3x ~12%."""
    r = math.log(max(1e-6, model) / max(1e-6, market))
    return base / (1 + (r / 0.4) ** 2)


TD_MODEL_WEIGHT = 0.2      # touchdown chance = 20% model + 80% betting market, when market prices exist
# Was 0.4 until Nov 10, 2026. Real DraftKings closing TD lines with replayed projections (current code, x0.92 market),
# blend log loss by model weight 0 / 0.1 / 0.2 / 0.3 / 0.4:
#   2025 wk1-18 (3,250 lines): 0.54435 / 0.54428 / 0.54428 / 0.54435 / 0.54449
#   2026 wk1-7  (1,310)      : 0.53715 / 0.53702 / 0.53697 / 0.53699 / 0.53708
#   2026 wk8-9  (373, live)  : 0.54425 / 0.54437 / 0.54456 / 0.54481 / 0.54512
# Fit on 2025 (0.1-0.2) beats 0.4 on 2026; fit on 2026 (0.2) beats 0.4 on 2025. Week 9 live board: 0.5364 -> 0.5355.
# The model alone runs high at long odds (+601-1000: 12.9% vs 10.7% scored; +1001 up: 11.9% vs 6.2%), so less weight
# mostly trims longshot TD "edges". Bets at closing DK/FD prices: 2025 -1.3% (173) -> -4.6% (106), 2026 wk1-7 +52% (82)
# -> +26% (35), wk8-9 -100% (12) -> +492% (6): noisy, fewer bets. Not changed: odds-dependent margin (fit one season,
# worse on the other: 0.53886 -> 0.53937, 0.54449 -> 0.54567).
TD_BOOK_MARGIN = 0.92      # sportsbook "Yes" prices carry the book's margin; scale their implied chance down


# Longshot TD bets need a same-day DraftKings price from The Odds API. Week 7: all 3 TD bets at +600 or longer
# (Cody White +1100, Marquise Brown +1000, Emari Demercado +1800) came from another source or an older pull;
# Brown and Demercado never appeared in a DraftKings API pull, and at DraftKings' API price White's edge was +1%
# (a lean, not a bet). With the rule: 0 bets instead of 0-2 and a void. 13 of 14 longshot TD bets/leans that
# week were priced off-API. Below +600 the rule doesn't apply. (Oct 27, 2026)
LONGSHOT_TD_ODDS = 600

# TD bets need our model alone to clear the price's break-even (Nov 10, 2026, Alex). The 80%-market blend can flag a
# TD the model itself doesn't like, purely because one book's price beats the market. Weeks 8-9 logged TD bets:
# model alone above break-even 13-44 +11.4u CLV +3.8% (57); below 0-16 -16.0u CLV -0.2% (16). Leans: +4.1% vs +0.7%
# CLV. Below the break-even it stays a lean. Watched in the Tuesday review (small sample).
TD_MODEL_MUST_CLEAR = True

# Books Alex can bet at. DraftKings stays the market anchor (its own price tested more accurate than the other
# books' consensus); each bet is priced at whichever of these books pays more for it (line shopping, Oct 27, 2026).
BET_BOOKS = {"draftkings": "DraftKings", "fanduel": "FanDuel"}

# Stats shown as leans only, never flagged as bets, until they beat the market on real lines. Passing yards
# (Oct 27, 2026): worse than DraftKings' no-vig close on 2026 weeks 3-7 (log loss 0.6979 vs 0.6934, 137 lines),
# on Week 7's lines (0.7061 vs 0.6935), and passing-yds bets went 9 at -15.7% at the close. Every other prop stat
# beat DraftKings on those lines. Re-enable once out-of-sample real-line tests (e.g. fit on 2025 historical props,
# checked on 2026) show passing yards beating the market. The weekly review checks it every Tuesday.
# Receptions (Nov 10, 2026): leans only. The Nov 3 rule (50+ receptions bets with average CLV below zero) triggered
# after Week 9: 66 bets, CLV -0.53%, beat the close 26%. Closing-line test (DK + FD close, current model, line-shop and
# sharp-book filters, 3-15% edge): receptions bets 2025 -15.0% ROI (103), 2026 wk1-7 +20.1% (30), wk8-9 -16.4% (16);
# -12.1u on 149. Every receptions bet at the close was a FanDuel price (none clear 3% at DraftKings' own line).
# Live record: Week 8 6-17 (-10.5u), Week 9 20-21 (+1.9u). Turn back on only if leans beat the close on 100+.
LEAN_ONLY_STATS = {"pass_yds", "receptions"}
# FanDuel line-shop props (FanDuel's line differs from DraftKings'): a bet only if the same side is also worth at
# least this much at DraftKings' own line and price; otherwise a lean (Nov 3, 2026). Tested on real closing lines
# (DraftKings + FanDuel, 2025 wk1-6 and 2026 wk1-7, current model): line-shop bets 2025 -7.4% ROI (212) / 2026
# +3.0% (167) -> with the rule -1.6% (72) / +10.7% (50); thresholds 0, 1.5% and 3% all better in both seasons.
# Live Week 8: line-shop prop bets went 32-36 (-7.3u) and FanDuel's line moved further from DraftKings' 42% of the time.
LINE_SHOP_MIN_DK_EV = 0.015
# Sharp-book check on props (Nov 3, 2026). Pinnacle + BetOnline consensus at our line vs DraftKings' own no-vig chance,
# real closing lines 2025 wk5-18 / 2026 wk3-8, current model: bets where those books liked our side MORE than DraftKings
# lost (-5.9% ROI on 235 / -9.7% on 97); where they liked it less, won (+5.2% on 313 / +12.6% on 150). DraftKings has
# been the sharper prop book (consensus-test), so when the others are friendlier than DraftKings, DraftKings is usually
# right. Such bets are leans. Not applied to TDs (one-sided prices) or when neither book has the prop.
SHARP_PROP_BOOKS = ("pinnacle", "betonlineag")
SHARP_PROP_FILTER = True
SNAP_DROP_LEAN = True     # Overs on players whose last-game snap share fell below half their usual are leans
BET_BOOK_SOURCES = tuple(BET_BOOKS.values())


def _side_prices(d: dict, stat: str, r) -> tuple | None:
    """(ev, side, probs, odds, line, over_odds, under_odds) for the better side of one book's line."""
    oo = r.over_odds if pd.notna(r.over_odds) else None
    uo = r.under_odds if pd.notna(r.under_odds) else None
    if oo is None and uo is None:
        return None
    if stat != "anytime_td" and pd.isna(r.line):
        return None
    line = 0.5 if stat == "anytime_td" else float(r.line)
    over = _probs(d, line)
    under = (over[2], over[1], over[0])
    e_o = ev(over, oo) if oo is not None else -1.0
    e_u = ev(under, uo) if uo is not None else -1.0
    if e_o >= e_u:
        return e_o, "Over", over, oo, line, oo, uo
    return e_u, "Under", under, uo, line, oo, uo


def _same_day_api(b) -> bool:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    src, asof = str(getattr(b, "source", "")), str(getattr(b, "as_of", ""))
    today = datetime.now(ZoneInfo("America/Chicago")).strftime("%a %b %-d")
    return "Odds API" in src and src.startswith(BET_BOOK_SOURCES) and asof.split(",")[0].strip() == today


def _td_market(player: str, book_row, market: dict) -> float | None:
    """Market chance to score: prediction-market price and/or sportsbook price, averaged."""
    vals = []
    if _key(player) in market:
        vals.append(market[_key(player)])
    if book_row is not None and pd.notna(getattr(book_row, "over_odds", np.nan)):
        vals.append(min(0.97, (1 / (1 + (book_row.over_odds / 100 if book_row.over_odds > 0
                                         else 100 / abs(book_row.over_odds)))) * TD_BOOK_MARGIN))
    return float(np.mean(vals)) if vals else None


def _parse_asof(asof, ref):
    """'Wed Nov 4, 9:11 AM CT' (no year) -> Timestamp. ref is a reference date (or a year): the year that puts the
    date closest to ref is used, so 'Mon Dec 28' read in January is last December (Nov 4, 2026 audit)."""
    import re
    m = re.match(r"\w+ (\w+ \d+), (.*?) CT", str(asof).strip())
    if not m:
        return pd.NaT
    ref = pd.Timestamp(year=int(ref), month=7, day=1) if isinstance(ref, (int, np.integer)) else pd.Timestamp(ref)
    best = pd.NaT
    for y in (ref.year - 1, ref.year, ref.year + 1):
        t = pd.to_datetime(f"{m.group(1)} {y} {m.group(2)}", errors="coerce")
        if pd.notna(t) and (pd.isna(best) or abs(t - ref) < abs(best - ref)):
            best = t
    return best


# Market-move alarm (Nov 4, 2026). How far DraftKings' price for our side has moved since it first posted this week, as
# change in DraftKings' own chance for our side at our line (TDs: Yes price x 0.92). Week 8 logged props (bets + leans,
# 6+ h of history): DK moved 3+ points against us before we logged -> -16% ROI (62), CLV -0.5%; 6+ points -> -22% (37);
# everything else +0.3% (206). One week only, so it FLAGS (shown in "Why this bet", checked first by the research role
# check, tracked in the weekly review) and does not hold yet. MOVE_HOLD switches it to a hold once it repeats.
MOVE_ALARM = -0.06
MOVE_HOLD = False


def add_market_moves(props: pd.DataFrame | None, history: str, week_proj: pd.DataFrame) -> pd.DataFrame | None:
    """Adds mkt_move (DK's chance for our side now minus when first posted this week) and holds/flags big moves."""
    from pathlib import Path
    if props is None or not len(props) or not Path(history).exists():
        return props
    try:
        h = pd.read_csv(history, comment="#", usecols=["game", "player", "stat", "book", "line", "over_odds", "under_odds",
                                                        "fetched_at"])
    except Exception:
        return props
    h = h[h.book == "draftkings"].copy()
    h["t"] = pd.to_datetime(h.fetched_at, errors="coerce")
    h = h[h.t >= week_start(week_proj.gameday)]
    h["k"] = h.player.map(_key)
    first = h.sort_values("t").groupby(["game", "k", "stat"]).first()
    props = props.copy()
    props["mkt_move"] = np.nan

    def side_p(stat, line, oo, uo, at, side):
        if stat == "anytime_td":
            if pd.isna(oo):
                return np.nan
            d = (1 + oo / 100) if oo > 0 else (1 + 100 / abs(oo))
            p = min(0.97, TD_BOOK_MARGIN / d)
            return p if side == "Over" else 1 - p
        if pd.isna(oo) or pd.isna(uo) or pd.isna(line):
            return np.nan
        pr = _probs(_dist(stat, _market_mean(stat, float(line), _no_vig_over(oo, uo))), at)
        return pr[0] if side == "Over" else pr[2]

    for i, r in props[props.best_side.isin(["Over", "Under"]) & props.line.notna()].iterrows():
        k = (r.game, _key(r.player), r.stat)
        if k not in first.index or pd.isna(r.get("dk_line", np.nan)):
            continue
        a = first.loc[k]
        # DraftKings' current price for this prop (the DK row is what dk_line came from)
        cur = h[(h.game == r.game) & (h.k == _key(r.player)) & (h.stat == r.stat)].sort_values("t").iloc[-1]
        p0 = side_p(r.stat, a.line, a.over_odds, a.under_odds, r.line, r.best_side)
        p1 = side_p(r.stat, cur.line, cur.over_odds, cur.under_odds, r.line, r.best_side)
        if pd.notna(p0) and pd.notna(p1):
            props.at[i, "mkt_move"] = round(float(p1 - p0), 4)
            if MOVE_HOLD and p1 - p0 <= MOVE_ALARM and bool(r.flag):
                props.at[i, "flag"] = False
    return props


def week_start(gamedays) -> pd.Timestamp:
    """00:00 on the Tuesday on or before the week's first game: an NFL week's prices start after last week's Monday game."""
    first = pd.to_datetime(pd.Series(gamedays)).min().normalize()
    return first - pd.Timedelta(days=(first.weekday() - 1) % 7)


def _this_week_lines(lines: pd.DataFrame, week_proj: pd.DataFrame, books: pd.DataFrame | None) -> pd.DataFrame:
    """Only this week's prices (Nov 4, 2026 fix). prop_lines.csv is keyed by player and stat with no game, and the
    fetcher kept a player's old line when DraftKings didn't list him in a new pull, so last week's line was priced as
    this week's: 16 of 34 Week 9 prop bets (and both Emanuel Wilson bets) used Week 8 lines. Now a DraftKings Odds API
    row counts only if DraftKings lists that player and stat for this week's game in the latest all-books pull, and any
    other row only if it was found after this week's slate opened (the Tuesday before the first game)."""
    keep = pd.Series(True, index=lines.index)
    api = lines.source.astype(str).str.contains("Odds API")
    if books is not None and len(books) and "game" in books:
        dk = books[books.book == "draftkings"]
        listed = set(zip(dk.player.map(_key), dk.stat.astype(str).str.strip()))
        keep &= ~api | pd.Series([(_key(p), str(s_).strip()) in listed for p, s_ in zip(lines.player, lines.stat)],
                                 index=lines.index)
    if "gameday" in week_proj and len(week_proj):
        start = week_start(week_proj.gameday)
        t = lines.as_of.map(lambda a: _parse_asof(a, start))
        keep &= t.isna() | (t >= start)
    dropped = int((~keep).sum())
    if dropped:
        print(f"  props: ignored {dropped} prop price(s) from before this week (book hasn't posted them this week)")
    return lines[keep]


def price_auto(players: pd.DataFrame, week_proj: pd.DataFrame, team_ppg: dict,
               lines: pd.DataFrame | None, td_market: pd.DataFrame | None = None,
               books: pd.DataFrame | None = None) -> pd.DataFrame:
    """players: output of player_proj.project. lines: player, stat, line, over_odds, under_odds, source, as_of."""
    if players is None or players.empty:
        return pd.DataFrame()
    team_pts = {}
    for g in week_proj.itertuples():
        team_pts[g.home_team] = g.proj_home_pts
        team_pts[g.away_team] = g.proj_away_pts
    if books is not None and len(books) and "game" in books:
        # only this week's games: prop_books.csv can still hold last week's pull (Oct 27 fix: Week 7 FanDuel
        # lines were being matched by player name to Week 8 props)
        wk_games = {f"{g.away_team} @ {g.home_team}" for g in week_proj.itertuples()}
        books = books[books.game.isin(wk_games)]
    # Every price is matched by GAME + player + stat (Nov 9, 2026: same-name players on different teams, e.g.
    # MIN WR Justin Jefferson and CLE LB Justin Jefferson, must never share a price, a grade or a status).
    book = {}
    if books is not None and len(books):
        # DraftKings from the all-books pull, which records the game (same pull that writes prop_lines.csv)
        for r in books[books.book == "draftkings"].itertuples():
            book[(r.game, _key(r.player), str(r.stat).strip())] = pd.Series(dict(
                player=r.player, stat=r.stat, line=r.line, over_odds=r.over_odds, under_odds=r.under_odds,
                source="DraftKings (The Odds API)", as_of=r.as_of))
    if lines is not None and len(lines):
        lines = _this_week_lines(lines, week_proj, books)
        # rows without a game (article prices, or no all-books file): only for names unique among this week's players
        names = players.drop_duplicates(["player", "team"]).player.map(_key).value_counts()
        for r in lines.itertuples():
            k_, st_ = _key(r.player), str(r.stat).strip()
            if "Odds API" in str(r.source) and books is not None and len(books):
                continue                                   # already taken from the all-books pull, with its game
            if names.get(k_, 0) == 1:
                book[(None, k_, st_)] = r
    # FanDuel's prices from the all-books pull, as rows shaped like prop_lines (source names the book)
    alt = {}
    if books is not None and len(books):
        fb = books[books.book == "fanduel"]
        for r in fb.itertuples():
            alt[(r.game, _key(r.player), str(r.stat).strip())] = pd.Series(dict(
                player=r.player, stat=r.stat, line=r.line, over_odds=r.over_odds, under_odds=r.under_odds,
                source="FanDuel (The Odds API)", as_of=r.as_of))
    # the sharper offshore books' view of each prop (Pinnacle, BetOnline): mean implied by their own line and prices
    sharp = {}
    if books is not None and len(books):
        sb = books[books.book.isin(SHARP_PROP_BOOKS) & books.line.notna() & books.over_odds.notna() & books.under_odds.notna()
                   & (books.stat != "anytime_td")]
        for r in sb.itertuples():
            sharp.setdefault((r.game, _key(r.player), str(r.stat).strip()), []).append(
                _market_mean(r.stat, float(r.line), _no_vig_over(r.over_odds, r.under_odds)))
    bk_get = lambda d, p_: d.get((p_.game, _key(p_.player), p_.stat), d.get((None, _key(p_.player), p_.stat)))
    market = {}
    if td_market is not None and len(td_market) and "as_of" in td_market and "gameday" in week_proj and len(week_proj):
        # only this week's TD market prices (Nov 4, 2026 audit: a finished Monday game's prices blended into next week)
        st_ = week_start(week_proj.gameday)
        t_ = td_market.as_of.map(lambda a: _parse_asof(a, st_))
        td_market = td_market[t_.notna() & (t_ >= st_)]
    if td_market is not None and len(td_market):
        # td_market.csv has no team: a name shared by two players this week is ambiguous, so neither gets it
        dup = set(players.drop_duplicates(["player", "team"]).player.map(_key).value_counts().loc[lambda x: x > 1].index)
        market = {_key(r.player): float(r.prob) for r in td_market.itertuples() if pd.notna(r.prob) and _key(r.player) not in dup}
    from . import consensus as K
    cons = {}
    if books is not None and len(books):
        for gm, gb in books.groupby("game"):
            for (k_, st_), v_ in K.table(gb).items():
                cons[(gm, k_, st_)] = v_
    anchor_cons = K.use_consensus()   # other books' consensus as the market anchor, once it has tested better
    rows = []
    for p in players.itertuples():
        f = _script_factor(p.team, team_pts, team_ppg, p.stat)
        c = cons.get((p.game, _key(p.player), p.stat))
        model_p, market_p = None, None
        if p.stat == "anytime_td":
            f = 1.0                     # already built from the team's projected points
            model_p = min(0.95, max(0.01, p.projection))
            b_td = bk_get(book, p)
            market_p = _td_market(p.player, b_td if b_td is not None else bk_get(alt, p), market)
            if anchor_cons and c is not None and pd.notna(c["all_td"]):
                market_p = c["all_td"] if market_p is None else (market_p + c["all_td"]) / 2
            if market_p is None:
                adj = model_p
            else:
                w = _trust(TD_MODEL_WEIGHT, model_p, market_p)
                adj = w * model_p + (1 - w) * market_p
        else:
            adj = p.projection * f
            b0 = bk_get(book, p)
            if b0 is None:
                b0 = bk_get(alt, p)                         # no DraftKings line: FanDuel's is the market
            if b0 is not None and pd.notna(b0.line):
                market_p = round(_market_mean(p.stat, float(b0.line), _no_vig_over(b0.over_odds, b0.under_odds)), 3)
                if anchor_cons and c is not None and pd.notna(c["all_mean"]):
                    market_p = round(c["all_mean"], 3)
                model_p = round(adj, 3)
                w = _trust(model_weight(p.stat), adj, market_p)
                adj = w * adj + (1 - w) * market_p
        d = _dist(p.stat, adj)
        row = dict(game=p.game, team=p.team, opp=p.opp, player=p.player, position=p.position, stat=p.stat,
                   injury=p.injury, script_factor=round(f, 3), adj_projection=round(adj, 3),
                   fair_line=_fair_line(d), dist=d["dist"], a=round(d["a"], 4), b=round(d["b"], 4),
                   model_prob=model_p, market_prob=market_p,
                   line=np.nan, over_odds=np.nan, under_odds=np.nan, source="", as_of="",
                   p_over=np.nan, best_side="", ev=np.nan, stake=0.0, flag=False,
                   n_books=int(c["n"]) if c else 0, cons_p_over=np.nan, cons_ev=np.nan,
                   book="", dk_line=np.nan, dk_ev=np.nan)
        row.update({k: v for k, v in p._asdict().items() if k.startswith("why_")})   # "Why this bet" factors
        row.update(why_team_pts=team_pts.get(p.team, np.nan), why_team_ppg=team_ppg.get(p.team, np.nan))
        # price the bet at each book Alex uses and keep the better one
        cands = []
        for name, src in (("DraftKings", book), ("FanDuel", alt)):
            r_ = bk_get(src, p)
            pr_ = _side_prices(d, p.stat, r_) if r_ is not None else None
            if pr_ is not None:
                cands.append((pr_, name, r_))
        if cands:
            dk = [x for x in cands if x[1] == "DraftKings"]
            if dk:
                row.update(dk_line=dk[0][0][4], dk_ev=round(dk[0][0][0], 4))
            (e_best, best, probs, odds, line, oo, uo), bname, b = max(cands, key=lambda x: x[0][0])
            over = _probs(d, line)
            row.update(line=line, over_odds=oo if oo is not None else np.nan, under_odds=uo if uo is not None else np.nan,
                       source=str(b.source), as_of=str(b.as_of), p_over=round(over[0], 3), best_side=best,
                       ev=round(e_best, 4), stake=round(kelly_stake(probs, odds), 2),
                       flag=e_best >= C.MIN_EV and e_best <= C.MAX_BET_EDGE and p.stat not in LEAN_ONLY_STATS, book=bname)
            if (p.stat == "anytime_td" and best == "Over" and odds is not None and odds > LONGSHOT_TD_ODDS
                    and not _same_day_api(b)):
                row.update(flag=False)       # stays a lean until a same-day Odds API DraftKings price confirms it
            if (TD_MODEL_MUST_CLEAR and p.stat == "anytime_td" and best == "Over" and oo is not None
                    and pd.notna(oo) and model_p < implied(oo)):
                row.update(flag=False, why_td_model_short=1.0)   # model alone below break-even: a price, not a read
            # Bets need a live sportsbook price from The Odds API; prices copied from articles stay leans (Nov 3,
            # 2026). Weeks 7-8 logged props: Odds API-priced bets CLV +2.1% (149), article-priced -10.4% (7), and
            # leans +1.1% (373) vs -9.8% (9); article prices beat the close 14-22% of the time vs 33-39%.
            if "Odds API" not in str(b.source):
                row.update(flag=False)
            # played only special teams in his team's last game (player_proj._st_only, Nov 4, 2026): 7 for 7 under
            # projection in 2025-26, so never a bet that week
            if row.get("why_st_only") == 1.0:
                row.update(flag=False)
            # role hold from the research runs (role_holds.csv, Nov 4, 2026): the role shrank in a way stats don't show
            if isinstance(row.get("why_hold"), str) and row.get("why_hold"):
                row.update(flag=False)
            # DraftKings (the anchor book) must list the prop this week: a book pulling a player's props is usually
            # news (inactive, demoted, late scratch), so a FanDuel-only price stays a lean (Nov 4, 2026)
            if bname == "FanDuel" and not dk:
                row.update(flag=False)
            # snap share last game under half his usual (player_proj._snap_drop, Nov 4, 2026): Overs lost -26% / -22%
            sd_ = row.get("why_snap_drop")
            if SNAP_DROP_LEAN and best == "Over" and sd_ is not None and pd.notna(sd_):
                row.update(flag=False)
            # line shopping: a FanDuel bet at a different line than DraftKings' must also be worth something at
            # DraftKings' own line and price (same side), or it stays a lean (LINE_SHOP_MIN_DK_EV, Nov 3, 2026)
            if bname == "FanDuel" and dk and p.stat != "anytime_td":
                r_dk = dk[0][2]
                if pd.notna(r_dk.line) and float(r_dk.line) != line:
                    o_dk = _probs(d, float(r_dk.line))
                    dk_side = (ev(o_dk, r_dk.over_odds) if best == "Over" and pd.notna(r_dk.over_odds) else
                               ev((o_dk[2], o_dk[1], o_dk[0]), r_dk.under_odds) if best == "Under" and pd.notna(r_dk.under_odds)
                               else -1.0)
                    row.update(dk_ev=round(dk_side, 4))
                    if dk_side < LINE_SHOP_MIN_DK_EV:
                        row.update(flag=False)
            # DraftKings is the sharper book on props: when Pinnacle/BetOnline like our side MORE than DraftKings does,
            # DraftKings has usually been right, so the bet is a lean (SHARP_PROP_FILTER, Nov 3, 2026)
            sm_ = sharp.get((p.game, _key(p.player), p.stat))
            if sm_ and p.stat != "anytime_td" and dk and pd.notna(dk[0][2].line):
                r_dk = dk[0][2]
                dk_mean = _market_mean(p.stat, float(r_dk.line), _no_vig_over(r_dk.over_odds, r_dk.under_odds))
                sp, dp = _probs(_dist(p.stat, float(np.mean(sm_))), line), _probs(_dist(p.stat, dk_mean), line)
                s_side, d_side = (sp[0], dp[0]) if best == "Over" else (sp[2], dp[2])
                row.update(sharp_side_p=round(s_side, 3), dk_side_p=round(d_side, 3))
                if SHARP_PROP_FILTER and s_side > d_side:
                    row.update(flag=False)
            # the other books' view of DraftKings' line: their consensus chance and the edge it implies at DK's price
            if c is not None:
                cp = (c["others_td"] if p.stat == "anytime_td" else
                      (K.p_over_at(p.stat, c["others_mean"], line) if pd.notna(c["others_mean"]) else np.nan))
                if pd.notna(cp):
                    cp = float(cp)
                    side_p, side_odds = (cp, oo) if best == "Over" else (1 - cp, uo)
                    row.update(cons_p_over=round(cp, 3),
                               cons_ev=round(ev((side_p, 0.0, 1 - side_p), side_odds), 4) if side_odds is not None else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)
