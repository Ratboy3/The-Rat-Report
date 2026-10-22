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
