"""Other sportsbooks vs. DraftKings on player props.

DraftKings sets many prop lines on its own; when the other books (FanDuel, BetMGM, Caesars, BetRivers,
Bovada, ...) price the same player differently, the consensus is usually closer to the truth. For each
DraftKings prop this module turns every other book's line and prices into an implied average (no-vig),
takes the median, and prices DraftKings' line off it.

Tested before it changes any number: `python run.py consensus-test` replays every archived pre-kickoff
pull (prop_books_history.csv) against results and compares DraftKings' own no-vig chance with the other
books' consensus. The verdict goes to consensus_test.csv; once there are enough graded props and the
consensus is more accurate, it becomes the market anchor in prop pricing (USE_CONSENSUS).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "prop_books_history.csv"
TEST_FILE = ROOT / "consensus_test.csv"
DK = "draftkings"
MIN_BOOKS = 2          # other books needed for a consensus
MIN_TEST_N = 300       # graded props needed before the verdict can switch the anchor
TD_MARGIN = 0.92


def _imp(o):
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


def use_consensus() -> bool:
    try:
        t = pd.read_csv(TEST_FILE)
        return bool(str(t.use_consensus.iloc[-1]).strip().lower() == "true")
    except Exception:
        return False


def book_means(books: pd.DataFrame) -> pd.DataFrame:
    """One row per book-player-stat: the average (or TD chance) that book's line and prices imply."""
    from . import props as P
    b = books.copy()
    out = []
    for r in b.itertuples():
        if r.stat == "anytime_td":
            if pd.notna(r.over_odds):
                out.append((r.player, r.stat, r.book, np.nan, min(0.97, _imp(float(r.over_odds)) * TD_MARGIN)))
            continue
        if pd.isna(r.line) or pd.isna(r.over_odds) or pd.isna(r.under_odds):
            continue
        p = P._no_vig_over(float(r.over_odds), float(r.under_odds))
        out.append((r.player, r.stat, r.book, P._market_mean(r.stat, float(r.line), p), np.nan))
    return pd.DataFrame(out, columns=["player", "stat", "book", "mean", "p_td"])


def table(books: pd.DataFrame | None) -> dict:
    """(player key, stat) -> {others_mean, all_mean, others_td, all_td, n}: n = other books priced."""
    from . import props as P
    if books is None or books.empty:
        return {}
    m = book_means(books)
    out = {}
    for (pl, st), g in m.groupby(["player", "stat"]):
        o = g[g.book != DK]
        if len(o) < MIN_BOOKS:
            continue
        out[(P._key(pl), st)] = dict(others_mean=float(o["mean"].median()) if o["mean"].notna().any() else np.nan,
                                     all_mean=float(g["mean"].median()) if g["mean"].notna().any() else np.nan,
                                     others_td=float(o.p_td.median()) if o.p_td.notna().any() else np.nan,
                                     all_td=float(g.p_td.median()) if g.p_td.notna().any() else np.nan,
                                     n=int(len(o)))
    return out


def p_over_at(stat: str, mean: float, line: float) -> float:
    from . import props as P
    return float(P._probs(P._dist(stat, mean), line)[0]) / max(1e-9, 1 - float(P._probs(P._dist(stat, mean), line)[1]))


# ------------------------------------------------------------------ test
def _week_of(games: pd.DataFrame, game: str, kickoff) -> tuple[int, int] | None:
    away, home = [t.strip() for t in game.split("@")]
    ts = pd.to_datetime(kickoff, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    day = ts.tz_convert("America/New_York").strftime("%Y-%m-%d")
    g = games[(games.away_team == away) & (games.home_team == home) & (games.gameday == day)]
    return (int(g.iloc[0].season), int(g.iloc[0].week)) if len(g) else None


def run_test(games: pd.DataFrame) -> pd.DataFrame | None:
    """DraftKings' own no-vig chance vs. the other books' consensus, at DraftKings' line, on graded props."""
    from . import player_proj as PP
    from . import props as P
    if not HISTORY.exists():
        print("No prop_books_history.csv yet")
        return None
    h = pd.read_csv(HISTORY)
    h["ko"] = pd.to_datetime(h.kickoff, utc=True, errors="coerce")
    h["t"] = pd.to_datetime(h.fetched_at).dt.tz_localize("America/Chicago").dt.tz_convert("UTC")
    h = h[h.t < h.ko]
    # last pre-kickoff pull for each game
    last = h.groupby("game").apply(lambda g: g[g.t == g.t.max()], include_groups=False).reset_index(level=0)
    stats_cache, rows = {}, []
    for (game, ko), grp in last.groupby(["game", "kickoff"]):
        sw = _week_of(games, game, ko)
        if sw is None:
            continue
        season, week = sw
        if season not in stats_cache:
            s = PP._load("stats", season, False)
            stats_cache[season] = s[s.season_type == "REG"] if s is not None else None
        s = stats_cache[season]
        if s is None:
            continue
        s = s[s.week == week]
        if s.empty:
            continue                      # not graded yet
        cons = table(grp)
        dk = grp[grp.book == DK]
        for r in dk.itertuples():
            c = cons.get((P._key(r.player), r.stat))
            a = s[s.player_display_name == r.player]
            if c is None or a.empty:
                continue
            a = a.iloc[0]
            if r.stat == "anytime_td":
                if pd.isna(r.over_odds) or pd.isna(c["others_td"]):
                    continue
                y = float((a.rushing_tds or 0) + (a.receiving_tds or 0) > 0)
                rows.append(dict(game=game, season=season, week=week, player=r.player, stat=r.stat, line=0.5,
                                 p_dk=min(0.97, _imp(float(r.over_odds)) * TD_MARGIN), p_cons=c["others_td"], y=y, n_books=c["n"]))
                continue
            if pd.isna(r.line) or pd.isna(r.over_odds) or pd.isna(r.under_odds) or pd.isna(c["others_mean"]):
                continue
            col = PP.STATS[r.stat][0]
            val = float(a[col]) if not isinstance(col, tuple) else float(sum(a[x] for x in col))
            if val == float(r.line):
                continue                  # push
            rows.append(dict(game=game, season=season, week=week, player=r.player, stat=r.stat, line=float(r.line),
                             p_dk=P._no_vig_over(float(r.over_odds), float(r.under_odds)),
                             p_cons=p_over_at(r.stat, c["others_mean"], float(r.line)), y=float(val > float(r.line)),
                             over_odds=r.over_odds, under_odds=r.under_odds, n_books=c["n"]))
    if not rows:
        print("No graded props in the archive yet")
        return None
    d = pd.DataFrame(rows)
    return d


def summarize(d: pd.DataFrame) -> dict:
    from .betting import decimal
    out = {}
    for name, part in (("yards/receptions", d[d.stat != "anytime_td"]), ("anytime TD", d[d.stat == "anytime_td"]), ("all", d)):
        if part.empty:
            continue
        pdk, pc, y = part.p_dk.clip(0.02, 0.98), part.p_cons.clip(0.02, 0.98), part.y
        ll = lambda p: float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        out[name] = dict(n=len(part), brier_dk=float(((pdk - y) ** 2).mean()), brier_cons=float(((pc - y) ** 2).mean()),
                         ll_dk=ll(pdk), ll_cons=ll(pc))
    # the bet itself: take DraftKings' price when the other books disagree by 4+ points
    g = d[(d.stat != "anytime_td") & ((d.p_cons - d.p_dk).abs() >= 0.04)]
    if len(g):
        side_over = g.p_cons > g.p_dk
        odds = np.where(side_over, g.over_odds, g.under_odds).astype(float)
        win = np.where(side_over, g.y == 1, g.y == 0)
        pnl = np.where(win, [decimal(o) - 1 for o in odds], -1.0)
        out["disagree_bets"] = dict(n=len(g), wins=int(win.sum()), units=float(pnl.sum()))
    return out


def verdict(d: pd.DataFrame | None, write: bool = True) -> dict:
    if d is None or d.empty:
        res = dict(n=0, use_consensus=False, note="no graded props yet")
    else:
        s = summarize(d)
        a = s.get("all", {})
        better = a and a["ll_cons"] < a["ll_dk"] and a["brier_cons"] < a["brier_dk"]
        res = dict(n=a.get("n", 0), ll_dk=round(a.get("ll_dk", np.nan), 4), ll_cons=round(a.get("ll_cons", np.nan), 4),
                   brier_dk=round(a.get("brier_dk", np.nan), 4), brier_cons=round(a.get("brier_cons", np.nan), 4),
                   disagree_n=s.get("disagree_bets", {}).get("n", 0), disagree_units=round(s.get("disagree_bets", {}).get("units", 0.0), 2),
                   use_consensus=bool(better and a.get("n", 0) >= MIN_TEST_N))
    if write:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        res = {"tested_at": datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M"), **res}
        t = pd.DataFrame([res])
        t.to_csv(TEST_FILE, mode="a", header=not TEST_FILE.exists(), index=False)
    return res
