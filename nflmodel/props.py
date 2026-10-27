"""Player prop pricer: your projection + game script from the team model -> P(over) and EV."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_d, nbinom, norm, poisson

from . import config as C
from .betting import ev, kelly_stake


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
                         flag=max(e_o, e_u) >= C.MIN_EV))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ automatic props
def _script_factor(team: str, team_pts: dict, team_ppg: dict) -> float:
    if team in team_pts and team_ppg.get(team):
        return 1 + C.PROP_SCRIPT_WEIGHT * (team_pts[team] / team_ppg[team] - 1)
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


TD_MODEL_WEIGHT = 0.4      # touchdown chance = 40% model + 60% betting market, when market prices exist
TD_BOOK_MARGIN = 0.92      # sportsbook "Yes" prices carry the book's margin; scale their implied chance down


# Longshot TD bets need a same-day DraftKings price from The Odds API. Week 7: all 3 TD bets at +600 or longer
# (Cody White +1100, Marquise Brown +1000, Emari Demercado +1800) came from another source or an older pull;
# Brown and Demercado never appeared in a DraftKings API pull, and at DraftKings' API price White's edge was +1%
# (a lean, not a bet). With the rule: 0 bets instead of 0-2 and a void. 13 of 14 longshot TD bets/leans that
# week were priced off-API. Below +600 the rule doesn't apply. (Oct 27, 2026)
LONGSHOT_TD_ODDS = 600


def _same_day_api(b) -> bool:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    src, asof = str(getattr(b, "source", "")), str(getattr(b, "as_of", ""))
    today = datetime.now(ZoneInfo("America/Chicago")).strftime("%a %b %-d")
    return "Odds API" in src and src.startswith("DraftKings") and asof.split(",")[0].strip() == today


def _td_market(player: str, book_row, market: dict) -> float | None:
    """Market chance to score: prediction-market price and/or sportsbook price, averaged."""
    vals = []
    if _key(player) in market:
        vals.append(market[_key(player)])
    if book_row is not None and pd.notna(getattr(book_row, "over_odds", np.nan)):
        vals.append(min(0.97, (1 / (1 + (book_row.over_odds / 100 if book_row.over_odds > 0
                                         else 100 / abs(book_row.over_odds)))) * TD_BOOK_MARGIN))
    return float(np.mean(vals)) if vals else None


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
    book = {}
    if lines is not None and len(lines):
        for r in lines.itertuples():
            book[(_key(r.player), str(r.stat).strip())] = r
    market = {}
    if td_market is not None and len(td_market):
        market = {_key(r.player): float(r.prob) for r in td_market.itertuples() if pd.notna(r.prob)}
    from . import consensus as K
    cons = K.table(books)
    anchor_cons = K.use_consensus()   # other books' consensus as the market anchor, once it has tested better
    rows = []
    for p in players.itertuples():
        f = _script_factor(p.team, team_pts, team_ppg)
        c = cons.get((_key(p.player), p.stat))
        model_p, market_p = None, None
        if p.stat == "anytime_td":
            f = 1.0                     # already built from the team's projected points
            model_p = min(0.95, max(0.01, p.projection))
            market_p = _td_market(p.player, book.get((_key(p.player), p.stat)), market)
            if anchor_cons and c is not None and pd.notna(c["all_td"]):
                market_p = c["all_td"] if market_p is None else (market_p + c["all_td"]) / 2
            if market_p is None:
                adj = model_p
            else:
                w = _trust(TD_MODEL_WEIGHT, model_p, market_p)
                adj = w * model_p + (1 - w) * market_p
        else:
            adj = p.projection * f
            b0 = book.get((_key(p.player), p.stat))
            if b0 is not None and pd.notna(b0.line):
                market_p = round(_market_mean(p.stat, float(b0.line), _no_vig_over(b0.over_odds, b0.under_odds)), 3)
                if anchor_cons and c is not None and pd.notna(c["all_mean"]):
                    market_p = round(c["all_mean"], 3)
                model_p = round(adj, 3)
                w = _trust(PROP_MODEL_WEIGHT, adj, market_p)
                adj = w * adj + (1 - w) * market_p
        d = _dist(p.stat, adj)
        row = dict(game=p.game, team=p.team, opp=p.opp, player=p.player, position=p.position, stat=p.stat,
                   injury=p.injury, script_factor=round(f, 3), adj_projection=round(adj, 3),
                   fair_line=_fair_line(d), dist=d["dist"], a=round(d["a"], 4), b=round(d["b"], 4),
                   model_prob=model_p, market_prob=market_p,
                   line=np.nan, over_odds=np.nan, under_odds=np.nan, source="", as_of="",
                   p_over=np.nan, best_side="", ev=np.nan, stake=0.0, flag=False,
                   n_books=int(c["n"]) if c else 0, cons_p_over=np.nan, cons_ev=np.nan)
        b = book.get((_key(p.player), p.stat))
        if b is not None and (pd.notna(b.over_odds) or pd.notna(b.under_odds)):
            line = 0.5 if p.stat == "anytime_td" else float(b.line)
            over = _probs(d, line)
            under = (over[2], over[1], over[0])
            oo = b.over_odds if pd.notna(b.over_odds) else None
            uo = b.under_odds if pd.notna(b.under_odds) else None
            e_o = ev(over, oo) if oo is not None else -1.0
            e_u = ev(under, uo) if uo is not None else -1.0
            best, probs, odds = ("Over", over, oo) if e_o >= e_u else ("Under", under, uo)
            row.update(line=line, over_odds=oo if oo is not None else np.nan, under_odds=uo if uo is not None else np.nan,
                       source=str(b.source), as_of=str(b.as_of), p_over=round(over[0], 3), best_side=best,
                       ev=round(max(e_o, e_u), 4), stake=round(kelly_stake(probs, odds), 2),
                       flag=max(e_o, e_u) >= C.MIN_EV)
            if (p.stat == "anytime_td" and best == "Over" and odds is not None and odds > LONGSHOT_TD_ODDS
                    and not _same_day_api(b)):
                row.update(flag=False)       # stays a lean until a same-day Odds API DraftKings price confirms it
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
