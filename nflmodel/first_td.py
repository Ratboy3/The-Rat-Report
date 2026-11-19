"""First touchdown scorer model (Nov 19, 2026; Alex asked for it as its own model and tab).

P(player scores the game's first TD) = P(his team scores the first offensive TD) x his share of his team's TDs.

Team part: competing scoring rates. With each team's projected TDs (the anytime-TD model's team TDs), the chance the
home team scores the first offensive touchdown is h^k / (h^k + a^k), times the share of games whose first TD is an
offensive one. From 2016-2026 regular-season play-by-play (2,789 games): 4.7% of first TDs were defense/special teams
and 0.65% of games had no TD. Favorites score first more often than their share of points: k fit on 2016-20 = 1.40
(test 2021-25 log loss 0.6721 at k=1 -> 0.6693), fit on 2021-25 = 1.57 (test 2016-20 0.6757 -> 0.6744); 2026 best
near 1.5. So k = 1.5.

Player part: this season's TD share from the anytime-TD model (recent games weighted), capped so a team's listed
players sum to at most 97%. Tested on 2025 wk5-18 / 2026 wk3-10 replays (326 games): k 1.0 -> 1.5 log loss
0.21836 -> 0.21787 / 0.19944 -> 0.19916; a position tilt from history (RB x1.05, WR x0.96) helped 2025, hurt 2026:
off. Using anytime-TD chances instead of shares was worse in both (0.2197 / 0.2016).

Not tested against past first-TD prices yet (needs the 2025 historical odds pull), so everything here is a LEAN:
shown and tracked, never a Best bet (FIRST_TD_BETS = False).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .betting import decimal, fair_american, implied, suggested_units

ROOT = Path(__file__).resolve().parent.parent
BOOKS = ROOT / "first_td_books.csv"
LOG = ROOT / "first_td_log.csv"
K = 1.5
NON_OFFENSE = 0.047
NO_TD = 0.0065
CAP = 0.97
MIN_TD_GAMES = 3          # players with fewer games of data are left out (one fluky TD makes a huge share)
PER_TEAM = 4              # shown per team on the tab: covers ~78% of a team's first-TD chance (top 3: ~66%)
VALUE_EDGE = 0.03
FIRST_TD_BETS = False
CT = ZoneInfo("America/Chicago")
LOG_COLS = ["key", "logged_at", "season", "week", "game", "kickoff", "team", "player", "position", "p", "fair",
            "pteam", "market", "book", "odds", "edge", "rank", "value", "result", "pnl"]


def _key(n) -> str:
    from .props import _key as k
    return k(n)


def compute(auto_props: pd.DataFrame, books_path: Path | str = BOOKS) -> pd.DataFrame:
    if auto_props is None or auto_props.empty:
        return pd.DataFrame()
    t = auto_props[auto_props.stat == "anytime_td"].copy()
    t = t.drop(columns=[c for c in ("book", "odds", "market", "kickoff", "edge", "fair", "p", "size", "rank", "value", "key")
                        if c in t.columns])
    need = {"why_share", "why_team_tds", "team", "game"}
    if t.empty or not need <= set(t.columns):
        return pd.DataFrame()
    t = t[t.why_team_tds.notna() & t.why_share.notna()]
    if "why_td_games" in t:
        t = t[t.why_td_games.fillna(0) >= MIN_TD_GAMES]
    lam = t.groupby(["game", "team"]).why_team_tds.first().reset_index()
    lam["opp"] = lam.groupby("game").why_team_tds.transform("sum") - lam.why_team_tds
    lam["pteam"] = lam.why_team_tds ** K / (lam.why_team_tds ** K + lam.opp ** K) * (1 - NON_OFFENSE - NO_TD)
    t = t.merge(lam[["game", "team", "pteam"]], on=["game", "team"])
    s = t.why_share.clip(lower=0)
    tot = s.groupby([t.game, t.team]).transform("sum")
    t["share"] = np.where(tot > CAP, s / tot * CAP, s)
    t["p"] = t.pteam * t.share
    t["key"] = t.player.map(_key)
    bp = Path(books_path)
    b = pd.read_csv(bp) if bp.exists() else pd.DataFrame(columns=["game", "kickoff", "player", "book", "odds"])
    b = b[b.game.isin(set(t.game))].copy()
    b["key"] = b.player.map(_key)
    dk = b[b.book == "draftkings"].copy()
    if len(dk):
        dk["imp"] = dk.odds.map(implied)
        dk["market"] = dk.imp / dk.groupby("game").imp.transform("sum") * (1 - NON_OFFENSE - NO_TD)
    else:
        dk["market"] = np.nan
    two = b[b.book.isin(["draftkings", "fanduel"])]
    best = two.sort_values("odds", ascending=False).drop_duplicates(["game", "key"])
    t = (t.merge(dk[["game", "key", "market"]].drop_duplicates(["game", "key"]), on=["game", "key"], how="left")
           .merge(best[["game", "key", "odds", "book", "kickoff"]], on=["game", "key"], how="left"))
    priced = set(two.game)
    before = t.groupby(["game", "team"]).share.transform("sum")
    t = t[~t.game.isin(priced) | t.odds.notna()].copy()   # DraftKings and FanDuel both took him down: usually news (out)
    # his touchdowns go to the teammates who are still listed (Kenneth Walker III pulled -> Emmett Johnson moves up)
    after = t.groupby(["game", "team"]).share.transform("sum")
    scale = (before.loc[t.index] / after).where(after > 0, 1.0)
    t["share"] = t.share * scale
    tot2 = t.groupby(["game", "team"]).share.transform("sum")
    t["share"] = np.where(tot2 > CAP, t.share / tot2 * CAP, t.share)
    t["p"] = t.pteam * t.share
    t["fair"] = t.p.map(fair_american)
    t["edge"] = [p * decimal(o) - 1 if pd.notna(o) else np.nan for p, o in zip(t.p, t.odds)]
    t["size"] = t.odds.map(lambda o: suggested_units(o) if pd.notna(o) else np.nan)
    t["book"] = t.book.map({"draftkings": "DK", "fanduel": "FD"})
    t["rank"] = t.groupby(["game", "team"]).p.rank(ascending=False, method="first").astype(int)
    t["value"] = t.edge >= VALUE_EDGE
    cols = ["game", "kickoff", "team", "player", "position", "p", "fair", "pteam", "market", "book", "odds", "edge",
            "size", "rank", "value"]
    return t[cols].sort_values(["game", "team", "rank"]).reset_index(drop=True)


def log(ft: pd.DataFrame, season: int, week: int, path: Path | str = LOG) -> pd.DataFrame:
    """Keep the first price we showed for each player-game (top picks and values), like bets_log.csv."""
    p_ = Path(path)
    old = pd.read_csv(p_) if p_.exists() else pd.DataFrame(columns=LOG_COLS)
    if ft is None or ft.empty:
        return old
    now = datetime.now(CT)
    x = ft[(ft["rank"] <= PER_TEAM) | ft.value].copy()
    x = x[x.odds.notna()]
    ko = pd.to_datetime(x.kickoff, utc=True, errors="coerce")
    x = x[ko > pd.Timestamp.now(tz="UTC")]                    # only before kickoff
    x["key"] = [f"{season}|{week}|{g}|{_key(pl)}" for g, pl in zip(x.game, x.player)]
    x = x[~x.key.isin(set(old.key))]
    if x.empty:
        return old
    x["logged_at"] = now.strftime("%a %b %-d, %-I:%M %p CT")
    x["season"], x["week"] = season, week
    new = x.assign(result=np.nan, pnl=np.nan)[LOG_COLS]
    out = pd.concat([old, new], ignore_index=True)
    out.to_csv(p_, index=False)
    return out


def _abbr(n: str) -> str:
    p = str(n).replace(".", " ").split()
    return f"{p[0][0]}.{p[-1]}" if len(p) > 1 else str(n)


def grade(season: int, path: Path | str = LOG, pbp=None) -> pd.DataFrame:
    """Grade logged rows once the game is in nflverse play-by-play: win if he scored the game's first TD."""
    p_ = Path(path)
    if not p_.exists():
        return pd.DataFrame(columns=LOG_COLS)
    lg = pd.read_csv(p_)
    todo = lg.result.isna() & (pd.to_datetime(lg.kickoff, utc=True, errors="coerce")
                               < pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=5))
    if not todo.any():
        return lg
    try:
        if pbp is None:
            url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
            pbp = pd.read_csv(url, low_memory=False, usecols=["game_id", "season", "week", "home_team", "away_team",
                                                              "play_id", "touchdown", "td_team", "td_player_name", "posteam"])
    except Exception as e:
        print(f"  first TD grading skipped ({e})")
        return lg
    td = pbp[(pbp.touchdown == 1) & pbp.td_team.notna()].sort_values(["game_id", "play_id"]).groupby("game_id").head(1)
    td = td.assign(game=td.away_team + " @ " + td.home_team)
    first = {(int(r.week), r.game): (r.td_team, str(r.td_player_name)) for r in td.itertuples()}
    played = set(zip(pbp.week.astype(int), pbp.away_team + " @ " + pbp.home_team))
    ina = set()
    ip = ROOT / "inactives.csv"
    if ip.exists():
        ia = pd.read_csv(ip, comment="#")
        ina = {(int(r.season), int(r.week), _key(r.player), r.team) for r in ia.itertuples()}
    for i in lg[todo].index:
        r = lg.loc[i]
        k = (int(r.week), r.game)
        if (int(r.season), int(r.week), _key(r.player), r.team) in ina:
            lg.loc[i, ["result", "pnl"]] = ["void", 0.0]
            continue
        if k not in played:
            continue
        team, who = first.get(k, (None, ""))
        win = team == r.team and who == _abbr(r.player)
        lg.loc[i, ["result", "pnl"]] = ["win", decimal(float(r.odds)) - 1] if win else ["loss", -1.0]
    lg.to_csv(p_, index=False)
    return lg


def record(lg: pd.DataFrame, season: int) -> list[dict]:
    """Track record by group: values (edge 3%+), the model's #1 pick per team, everything logged."""
    if lg is None or lg.empty:
        return []
    g = lg[(lg.season == season) & lg.result.isin(["win", "loss"])].copy()
    out = []
    for name, x in (("Values (edge 3%+)", g[g.value.astype(str).str.lower() == "true"]),
                    ("Model's top pick per team", g[g["rank"] == 1]),
                    ("Every player listed", g)):
        if len(x):
            sz = x.odds.map(lambda o: suggested_units(o))
            out.append(dict(group=name, n=len(x), wins=int((x.result == "win").sum()),
                            units=round(float(x.pnl.sum()), 1), sized=round(float((x.pnl * sz).sum()), 1),
                            hit=round(float((x.result == "win").mean()), 3), exp=round(float(x.p.mean()), 3)))
    return out
