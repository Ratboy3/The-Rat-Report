"""Market data from The Odds API files: DraftKings game lines, line movement, and best price across books.

  game_lines.csv     every US book's spread / total / moneyline (fetch-games)
  prop_books.csv     every US book's player props (fetch-props)
  lines_history.csv  the market line for each game at every hourly run (built here), for line movement
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .betting import decimal

CT = ZoneInfo("America/Chicago")
BOOK_NAMES = {"draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "williamhill_us": "Caesars",
              "espnbet": "ESPN BET", "fanatics": "Fanatics", "betrivers": "BetRivers", "ballybet": "Bally Bet",
              "hardrockbet": "Hard Rock", "betonlineag": "BetOnline", "bovada": "Bovada", "lowvig": "LowVig",
              "mybookieag": "MyBookie", "betus": "BetUS", "fliff": "Fliff", "novig": "Novig", "prophetx": "ProphetX"}
MAX_AGE_H = 48     # DraftKings game lines older than this aren't used


def _parse(stamp: str) -> datetime | None:
    try:
        return datetime.strptime(f"{datetime.now(CT).year} {stamp.replace(' CT', '')}", "%Y %a %b %d, %I:%M %p").replace(tzinfo=CT)
    except Exception:
        return None


def _read(path) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p, comment="#") if p.exists() else None


def apply_dk_game_lines(df: pd.DataFrame, path) -> tuple[pd.DataFrame, dict]:
    """Use DraftKings' own spread/total/moneyline (from the API) for upcoming games when fetched recently."""
    gl = _read(path)
    used = {}
    if gl is None or gl.empty:
        return df, used
    dk = gl[gl.book == "draftkings"]
    now = datetime.now(CT)
    df = df.copy()
    key = df.away_team + " @ " + df.home_team
    for game, g in dk.groupby("game"):
        m = (key == game) & ~df.played
        if not m.any():
            continue
        ts = _parse(str(g.as_of.iloc[0]))
        if ts is None or now - ts > timedelta(hours=MAX_AGE_H):
            continue
        get = lambda mk, sd: g[(g.market == mk) & (g.side == sd)]
        hs, as_, ov, un, hm, am = (get("spread", "home"), get("spread", "away"), get("total", "over"),
                                   get("total", "under"), get("moneyline", "home"), get("moneyline", "away"))
        if len(hs) and len(as_):
            df.loc[m, ["spread_line", "mkt_margin"]] = -float(hs.line.iloc[0])
            df.loc[m, "home_spread_odds"], df.loc[m, "away_spread_odds"] = float(hs.odds.iloc[0]), float(as_.odds.iloc[0])
        if len(ov) and len(un):
            df.loc[m, ["total_line", "mkt_total"]] = float(ov.line.iloc[0])
            df.loc[m, "over_odds"], df.loc[m, "under_odds"] = float(ov.odds.iloc[0]), float(un.odds.iloc[0])
        if len(hm) and len(am):
            df.loc[m, "home_moneyline"], df.loc[m, "away_moneyline"] = float(hm.odds.iloc[0]), float(am.odds.iloc[0])
        used[game] = str(g.as_of.max())
    return df, used


def record_history(path, proj: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Append each game's current market line when it changes; return the full history."""
    now = now or datetime.now(CT)
    p = Path(path)
    cols = ["at", "game_id", "spread_line", "total_line", "home_moneyline", "away_moneyline"]
    h = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=cols)
    new = []
    for g in proj.itertuples():
        cur = (g.spread_line, g.total_line, g.home_moneyline, g.away_moneyline)
        last = h[h.game_id == g.game_id].tail(1)
        if len(last) and tuple(last[["spread_line", "total_line", "home_moneyline", "away_moneyline"]].iloc[0]) == cur:
            continue
        new.append(dict(at=now.strftime("%a %b %-d, %-I:%M %p CT"), game_id=g.game_id, spread_line=cur[0],
                        total_line=cur[1], home_moneyline=cur[2], away_moneyline=cur[3]))
    if new:
        h = pd.concat([h, pd.DataFrame(new)], ignore_index=True)
        h.to_csv(p, index=False)
    return h


def movement(history: pd.DataFrame | None, game_id: str) -> dict | None:
    """First recorded line vs. now."""
    if history is None or history.empty:
        return None
    h = history[history.game_id == game_id]
    if len(h) < 2:
        return None
    a, b = h.iloc[0], h.iloc[-1]
    return dict(since=a["at"], spread=(a.spread_line, b.spread_line), total=(a.total_line, b.total_line))


def best_game_prices(path) -> dict:
    """(game, market, side) -> list of (book, line, odds), best first (better line, then better price)."""
    gl = _read(path)
    out = {}
    if gl is None or gl.empty:
        return out
    gl = gl[gl.book != "pinnacle"]                           # sharp reference only: not a US book you can bet at
    for (game, mk, side), g in gl.groupby(["game", "market", "side"]):
        g = g.dropna(subset=["odds"]).copy()
        if mk == "spread":
            g["k"] = g.line.astype(float)                    # more points for the side = better
        elif mk == "total":
            g["k"] = -g.line.astype(float) if side == "over" else g.line.astype(float)
        else:
            g["k"] = 0.0
        g["dec"] = g.odds.map(decimal)
        g = g.sort_values(["k", "dec"], ascending=False)
        out[(game, mk, side)] = [(BOOK_NAMES.get(b, b), l, o) for b, l, o in zip(g.book, g.line, g.odds)]
    return out


def book_game_lines(path, book: str = "fanduel") -> dict:
    """(game, market, side) -> (line, odds) at one book, from the all-books game-lines pull."""
    gl = _read(path)
    if gl is None or gl.empty:
        return {}
    gl = gl[(gl.book == book) & gl.odds.notna()]
    return {(r.game, r.market, r.side): (r.line, r.odds) for r in gl.itertuples()}


def best_prop_prices(path) -> dict:
    """(player, stat, side, line) -> (book, odds) with the best price at that exact line."""
    pb = _read(path)
    out = {}
    if pb is None or pb.empty:
        return out
    for side, col in (("Over", "over_odds"), ("Under", "under_odds")):
        x = pb.dropna(subset=[col]).copy()
        x["dec"] = x[col].map(decimal)
        x = x.sort_values("dec", ascending=False).drop_duplicates(["player", "stat", "line"])
        for r in x.itertuples():
            out[(r.player, r.stat, side, float(r.line))] = (BOOK_NAMES.get(r.book, r.book), getattr(r, col))
    return out


def timing_test(history_path, games: pd.DataFrame, out_path="timing_test.csv") -> dict:
    """Do line moves keep going? For each finished game: the move from the first line we saw to 24 hours before
    kickoff, then the move from there to the close. If early moves tend to continue, a lean or bet the line is
    moving toward is worth taking early; if they reverse, waiting gets better prices."""
    h = _read(history_path)
    res = dict(games=0)
    if h is not None and len(h):
        h = h.copy()
        g = games.set_index("game_id")
        h = h[h.game_id.isin(g.index)]
        h["t"] = [_parse(s) for s in h["at"]]
        h = h.dropna(subset=["t"])
        rows = []
        for gid, grp in h.groupby("game_id"):
            row = g.loc[gid]
            if pd.isna(row.get("result")):
                continue
            ko = datetime.strptime(f"{row.gameday} {row.gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo("America/New_York"))
            grp = grp[grp.t < ko].sort_values("t")
            mid = grp[grp.t <= ko - timedelta(hours=24)]
            if len(grp) < 2 or mid.empty or grp.t.iloc[0] == mid.t.iloc[-1]:
                continue
            for col in ("spread_line", "total_line"):
                a, b, c = grp[col].iloc[0], mid[col].iloc[-1], grp[col].iloc[-1]
                if pd.notna(a) and pd.notna(b) and pd.notna(c):
                    rows.append(dict(game_id=gid, market=col, early=b - a, late=c - b))
        d = pd.DataFrame(rows)
        if len(d):
            moved = d[d.early != 0]
            cont = float(((moved.late * moved.early) > 0).mean()) if len(moved) else np.nan
            rev = float(((moved.late * moved.early) < 0).mean()) if len(moved) else np.nan
            res = dict(games=int(d.game_id.nunique()), early_moves=int(len(moved)),
                       continued=round(cont, 3) if len(moved) else None, reversed=round(rev, 3) if len(moved) else None,
                       corr=round(float(np.corrcoef(d.early, d.late)[0, 1]), 3) if len(d) > 2 and d.early.std() > 0 and d.late.std() > 0 else None)
    res = {"tested_at": datetime.now(CT).strftime("%Y-%m-%d %H:%M"), **res}
    # rewrite (not append) so a row with new columns never lands under an older, shorter header
    old = pd.read_csv(out_path, on_bad_lines="skip") if Path(out_path).exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([res])], ignore_index=True).to_csv(out_path, index=False)
    return res


HISTORY_MIRROR = "https://raw.githubusercontent.com/Ratboy3/The-Rat-Report/main/prop_books_history.csv"


def frozen_prop_lines(proj: pd.DataFrame, now: datetime | None = None, path="prop_books_history.csv") -> tuple[pd.DataFrame, set]:
    """DraftKings prop lines as they stood at the last Odds API pull before each started game's kickoff, so a game's
    card keeps its pre-game props (frozen) after kickoff instead of going blank. Returns (lines, started game keys)."""
    now = now or datetime.now(CT)
    et = ZoneInfo("America/New_York")
    started, ko = set(), {}
    for g in proj.itertuples():
        k = datetime.strptime(f"{g.gameday} {g.gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=et)
        if k <= now:
            started.add(f"{g.away_team} @ {g.home_team}")
            ko[f"{g.away_team} @ {g.home_team}"] = k
    empty = pd.DataFrame(columns=["player", "stat", "line", "over_odds", "under_odds", "source", "as_of"])
    if not started:
        return empty, started
    p = Path(path)
    try:
        h = pd.read_csv(p if p.exists() else HISTORY_MIRROR)
    except Exception as e:
        print(f"  frozen props skipped ({e})")
        return empty, started
    h = h[(h.book == "draftkings") & h.game.isin(started)].copy()
    if h.empty:
        return empty, started
    h["t"] = pd.to_datetime(h.fetched_at).dt.tz_localize(CT)
    h["ko"] = pd.to_datetime(h.kickoff, utc=True, errors="coerce")
    out = []
    for game, grp in h.groupby("game"):
        k = pd.Timestamp(ko[game])
        grp = grp[(grp.t < k) & ((grp.ko - k).abs() < pd.Timedelta(hours=6))]
        if grp.empty:
            continue
        out.append(grp[grp.t == grp.t.max()])
    if not out:
        return empty, started
    f = pd.concat(out).assign(source="DraftKings (The Odds API, last pre-game price)")
    return f[["player", "stat", "line", "over_odds", "under_odds", "source", "as_of"]].reset_index(drop=True), started
