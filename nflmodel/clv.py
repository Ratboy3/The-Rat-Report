"""Closing line value (CLV) tracker.

Every bet the report flags is logged once, at the price it had when it was first flagged. While the game
hasn't kicked off, each run records the latest DraftKings/market price for the same bet; the last one
before kickoff is the closing price. After the game, the bet is graded.

CLV = how much better our bet was than making the same bet at the closing line and price (0 = same as the close),
measured with the no-vig closing market. value_at_close = our bet's expected profit judged by that closing market. Beating the close
consistently is the earliest reliable sign a model has a real edge; win/loss records take years.

Leans (positive edge below the betting bar) are logged and graded the same way with tier "lean", under their
own id, so they never mix into the bet record; a lean that later becomes a bet gets a separate bet entry.

Log file: bets_log.csv (kept with the other inputs so every hourly run adds to it).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import config as C
from .betting import decimal, ev

ET = ZoneInfo("America/New_York")
CT = ZoneInfo("America/Chicago")
COLS = ["bet_id", "logged_at", "season", "week", "game_id", "game", "kickoff", "market", "side", "pick", "player",
        "stat", "line", "odds", "model_prob", "ev", "close_line", "close_odds", "close_odds2", "close_at", "clv", "value_at_close",
        "result", "pnl", "tier", "cons_prob", "hours_before", "book"]


def _kickoff(g) -> datetime:
    return datetime.strptime(f"{g.gameday} {g.gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)


def _stamp(dt: datetime) -> str:
    return dt.astimezone(CT).strftime("%a %b %-d, %-I:%M %p CT")


def _no_vig(a, b):
    ia, ib = 1 / decimal(a), 1 / decimal(b)
    return ia / (ia + ib)


def _game_close_probs(row, g, sm):
    """(win, push, lose) for the logged bet, priced off the current market line."""
    if row.market == "spread":
        if pd.isna(g.spread_line):
            return None
        # lines are stored home-favored (nflverse convention); the closing line is the market's expected margin
        return sm.spread_probs(g.spread_line, row.line)[row.side]
    if row.market == "total":
        if pd.isna(g.total_line):
            return None
        return sm.total_probs(g.total_line, row.line)[row.side]
    if row.market == "moneyline":
        if pd.isna(g.home_moneyline) or pd.isna(g.away_moneyline):
            return None
        ph = _no_vig(g.home_moneyline, g.away_moneyline)
        p = ph if row.side == "home" else 1 - ph
        return (p, 0.0, 1 - p)
    return None


def _prop_close_probs(row, pr):
    from . import props as P
    if row.stat == "anytime_td":
        if pd.isna(pr.over_odds):
            return None
        p = min(0.97, (1 / decimal(pr.over_odds)) * P.TD_BOOK_MARGIN)
        return (p, 0.0, 1 - p) if row.side == "Over" else (1 - p, 0.0, p)
    if pd.isna(pr.line):
        return None
    mean = P._market_mean(row.stat, float(pr.line), P._no_vig_over(pr.over_odds, pr.under_odds))
    over = P._probs(P._dist(row.stat, mean), float(row.line))
    return over if row.side == "Over" else (over[2], over[1], over[0])


def update(path: str | Path, season: int, week: int, proj: pd.DataFrame, game_opts: list, sm,
           props: pd.DataFrame | None, games: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    path = Path(path)
    now = now or datetime.now(CT)
    log = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLS)
    for c in COLS:
        if c not in log:
            log[c] = np.nan
    log["tier"] = log["tier"].fillna("bet")
    log = log[COLS].astype({c: "object" for c in ("player", "stat", "close_at", "result", "tier", "book")})
    hrs = lambda g: round((_kickoff(g) - now).total_seconds() / 3600, 1)
    miss = log.hours_before.isna()
    if miss.any():   # rows logged before this column existed: from the logged-at stamp
        la = pd.to_datetime(log.loc[miss, "season"].astype(str) + " " + log.loc[miss, "logged_at"].astype(str).str.replace(" CT", ""),
                            format="%Y %a %b %d, %I:%M %p", errors="coerce").dt.tz_localize(CT)
        ko = pd.to_datetime(log.loc[miss, "kickoff"], utc=True, errors="coerce")
        log.loc[miss, "hours_before"] = ((ko - la.dt.tz_convert("UTC")).dt.total_seconds() / 3600).round(1)
    known = set(log.bet_id)
    gid = {f"{g.away_team} @ {g.home_team}": g for _, g in proj.iterrows()}
    started = {k: _kickoff(g) <= now for k, g in gid.items()}
    new = []

    # 1. log newly flagged bets (first price seen)
    for item in game_opts:
        g = item["g"]
        key = f"{g.away_team} @ {g.home_team}"
        if started.get(key):
            continue
        for o in item["opts"]:
            if o["ev"] <= 0:
                continue
            tier = "bet" if o["ev"] >= C.MIN_EV else "lean"
            line = o.get("line") if o.get("line") is not None else (
                g.spread_line if o["market"] == "spread" else (g.total_line if o["market"] == "total" else np.nan))
            bid = f"{g.game_id}|{o['market']}|{o['side']}||" + ("" if tier == "bet" else "|lean")
            if bid in known:
                continue
            known.add(bid)
            new.append(dict(bet_id=bid, logged_at=_stamp(now), season=season, week=week, game_id=g.game_id, game=key,
                            kickoff=_kickoff(g).isoformat(), market=o["market"], side=o["side"], pick=o["pick"],
                            player="", stat="", line=line, odds=o["odds"], model_prob=round(o["win_prob"], 4),
                            ev=round(o["ev"], 4), tier=tier, hours_before=hrs(g), book=o.get("book", "DraftKings")))
    if props is not None and len(props):
        for r in props[props.flag.astype(bool) | (props.ev > 0)].itertuples():
            g = gid.get(r.game)
            if g is None or started.get(r.game) or pd.isna(r.line) or pd.isna(r.ev):
                continue
            tier = "bet" if bool(r.flag) else "lean"
            bid = f"{g.game_id}|prop|{r.best_side}|{r.player}|{r.stat}" + ("" if tier == "bet" else "|lean")
            if bid in known:
                continue
            known.add(bid)
            odds = r.over_odds if r.best_side == "Over" else r.under_odds
            what = "anytime TD" if r.stat == "anytime_td" else f"{r.best_side} {r.line:g} {r.stat}"
            new.append(dict(bet_id=bid, logged_at=_stamp(now), season=season, week=week, game_id=g.game_id, game=r.game,
                            kickoff=_kickoff(g).isoformat(), market="prop", side=r.best_side, pick=f"{r.player} {what}",
                            player=r.player, stat=r.stat, line=r.line, odds=odds,
                            model_prob=round(float(r.p_over if r.best_side == "Over" else 1 - r.p_over), 4), ev=round(r.ev, 4),
                            tier=tier, hours_before=hrs(g), book=(getattr(r, "book", "") or "DraftKings"),
                            cons_prob=(round(float(r.cons_p_over if r.best_side == "Over" else 1 - r.cons_p_over), 4)
                                       if pd.notna(getattr(r, "cons_p_over", np.nan)) else np.nan)))
    if new:
        log = pd.concat([log, pd.DataFrame(new)], ignore_index=True)

    # 2. latest pre-kickoff price for every open bet = closing price once the game starts
    pidx = {}
    if props is not None and len(props):
        pidx = {(r.player, r.stat): r for r in props.itertuples() if pd.notna(r.line)}
    for i, row in log.iterrows():
        g = gid.get(row.game)
        if g is None or started.get(row.game):
            continue
        probs, cl, co, co2 = None, np.nan, np.nan, np.nan
        if row.market == "prop":
            pr = pidx.get((row.player, row.stat))
            if pr is not None:
                probs = _prop_close_probs(row, pr)
                cl, co, co2 = pr.line, pr.over_odds, pr.under_odds
        else:
            probs = _game_close_probs(row, g, sm)
            if row.market == "spread":
                cl, co, co2 = g.spread_line, g.home_spread_odds, g.away_spread_odds
            elif row.market == "total":
                cl, co, co2 = g.total_line, g.over_odds, g.under_odds
            else:
                co, co2 = g.home_moneyline, g.away_moneyline
        close_side_odds = None
        if row.market == "prop":
            close_side_odds = co if row.side == "Over" else co2
        elif row.market == "spread":
            close_side_odds = co if row.side == "home" else co2
        elif row.market == "total":
            close_side_odds = co if row.side == "over" else co2
        else:
            close_side_odds = co if row.side == "home" else co2
        if probs is not None and pd.notna(close_side_odds):
            # price beaten = value of our bet minus value of the same bet made at the close
            shadow = row.copy()
            shadow["line"] = cl if pd.notna(cl) else row.line
            cp = (_prop_close_probs(shadow, pr) if row.market == "prop" else _game_close_probs(shadow, g, sm))
            ours, close_ev = ev(probs, float(row.odds)), ev(cp, float(close_side_odds))
            log.at[i, "clv"] = round(ours - close_ev, 4)
            log.at[i, "value_at_close"] = round(ours, 4)
            log.at[i, "close_line"], log.at[i, "close_odds"], log.at[i, "close_odds2"] = cl, co, co2
            log.at[i, "close_at"] = _stamp(now)

    # 3. grade finished games
    stats = None
    for i, row in log[log.result.isna() | (log.result == "")].iterrows():
        gg = games[games.game_id == row.game_id]
        if gg.empty or pd.isna(gg.iloc[0].result):
            continue
        g = gg.iloc[0]
        if row.market == "prop":
            if stats is None:
                try:
                    from . import player_proj as PP
                    stats = PP._load("stats", int(row.season), True)
                except Exception:
                    stats = pd.DataFrame()
            s = stats[(stats.week == row.week) & (stats.player_display_name == row.player)] if len(stats) else stats
            if s.empty:
                continue
            col = {"pass_yds": "passing_yards", "pass_tds": "passing_tds", "rush_yds": "rushing_yards",
                   "rec_yds": "receiving_yards", "receptions": "receptions"}.get(row.stat)
            val = float(s.iloc[0][["rushing_tds", "receiving_tds"]].fillna(0).sum()) if row.stat == "anytime_td" else float(s.iloc[0][col])
            diff = (val - float(row.line)) if row.side == "Over" else (float(row.line) - val)
        elif row.market == "spread":
            diff = (g.result - row.line) if row.side == "home" else (row.line - g.result)
        elif row.market == "total":
            diff = (g.total - row.line) if row.side == "over" else (row.line - g.total)
        else:
            diff = g.result if row.side == "home" else -g.result
        log.at[i, "result"] = "push" if diff == 0 else ("win" if diff > 0 else "loss")
        log.at[i, "pnl"] = 0.0 if diff == 0 else (round(decimal(float(row.odds)) - 1, 4) if diff > 0 else -1.0)

    log.to_csv(path, index=False)
    return log


def summary(log: pd.DataFrame | None) -> dict | None:
    if log is None or log.empty:
        return None
    out = {}
    now = datetime.now(CT)
    closed = pd.to_datetime(log.kickoff, utc=True, errors="coerce") <= pd.Timestamp(now)
    log = log.assign(closed=closed.values)
    tier = log["tier"].fillna("bet") if "tier" in log else pd.Series("bet", index=log.index)
    bets, leans = log[tier == "bet"], log[tier == "lean"]
    for name, part in (("All bets", bets), ("Game bets", bets[bets.market != "prop"]), ("Prop bets", bets[bets.market == "prop"]),
                       ("Leans: games", leans[leans.market != "prop"]), ("Leans: props", leans[leans.market == "prop"])):
        c = part[part.clv.notna() & part.closed]
        s = part[part.result.isin(["win", "loss", "push"])]
        out[name] = dict(bets=len(part), with_close=len(c), clv=float(c.clv.mean()) if len(c) else np.nan,
                         beat=float((c.clv > 0).mean()) if len(c) else np.nan, settled=len(s),
                         wins=int((s.result == "win").sum()), losses=int((s.result == "loss").sum()),
                         units=float(s.pnl.astype(float).sum()) if len(s) else 0.0)
    return out


def timing(log: pd.DataFrame | None) -> pd.DataFrame | None:
    """Average CLV by how long before kickoff a bet was logged: tells us whether to bet early or wait."""
    if log is None or log.empty or "hours_before" not in log:
        return None
    lg = log[log.clv.notna() & log.hours_before.notna()].copy()
    if "tier" not in lg:
        lg["tier"] = "bet"
    lg["tier"] = lg.tier.fillna("bet")
    lg = lg[pd.to_datetime(lg.kickoff, utc=True, errors="coerce") <= pd.Timestamp.now(tz="UTC")]
    if lg.empty:
        return None
    lg["when"] = pd.cut(lg.hours_before.astype(float), [-1, 12, 48, 1e4], labels=["Under 12 h", "12-48 h", "Over 48 h"])
    return lg.groupby(["when", "tier"], observed=True).agg(n=("clv", "size"), clv=("clv", "mean"),
                                                           beat=("clv", lambda x: float((x > 0).mean()))).reset_index()
