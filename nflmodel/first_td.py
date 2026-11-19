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
# First-seen columns (the price we first showed, like bets_log.csv), then close_* = the last board before kickoff
# (updated on every run until kickoff), then the grade. Every priced player is logged, not just the ones shown, so the
# backtest has the whole board (Alex, Nov 19, 2026: "make sure we are tracking these numbers").
LOG_COLS = ["key", "logged_at", "season", "week", "game", "kickoff", "team", "player", "position", "p", "fair",
            "pteam", "market", "gap", "book", "odds", "edge", "rank", "rank_gap", "value",
            "close_at", "close_p", "close_market", "close_gap", "close_book", "close_odds", "close_edge", "close_value",
            "close_rank_gap", "clv", "scorer", "result", "pnl"]
CLOSE_MAP = {"p": "close_p", "market": "close_market", "gap": "close_gap", "book": "close_book", "odds": "close_odds",
             "edge": "close_edge", "value": "close_value", "rank_gap": "close_rank_gap"}


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
    t["gap"] = t.p - t.market                              # our chance minus the book's (no-vig), in probability
    t["value"] = t.edge > 0                                  # the price pays more than our fair odds
    t["rank_gap"] = t.gap.fillna(-1).groupby(t.game).rank(ascending=False, method="first").astype(int)   # order shown on the tab
    cols = ["game", "kickoff", "team", "player", "position", "p", "fair", "pteam", "market", "gap", "book", "odds", "edge",
            "size", "rank", "rank_gap", "value"]
    return t[cols].sort_values(["game", "team", "rank"]).reset_index(drop=True)


def _load(path: Path | str = LOG) -> pd.DataFrame:
    p_ = Path(path)
    lg = pd.read_csv(p_) if p_.exists() else pd.DataFrame(columns=LOG_COLS)
    for c in LOG_COLS:
        if c not in lg:
            lg[c] = np.nan
    if lg.gap.isna().any():                                   # rows logged before Nov 19 evening had no gap column
        lg["gap"] = lg.gap.fillna(lg.p - lg.market)
    for c in ("book", "close_book", "close_at", "scorer", "result", "logged_at", "value", "close_value"):
        lg[c] = lg[c].astype(object)
    return lg[LOG_COLS]


def log(ft: pd.DataFrame, season: int, week: int, path: Path | str = LOG) -> pd.DataFrame:
    """Every priced player before kickoff: the first price we showed is kept (p, odds, ...), and the close_* columns
    are overwritten on each run until kickoff, so they end as the last pre-game board (closing line value + backtest)."""
    p_ = Path(path)
    old = _load(p_)
    if ft is None or ft.empty:
        return old
    now = datetime.now(CT)
    x = ft[ft.odds.notna()].copy()
    ko = pd.to_datetime(x.kickoff, utc=True, errors="coerce")
    x = x[ko > pd.Timestamp.now(tz="UTC")]                    # only before kickoff
    if x.empty:
        return old
    x["key"] = [f"{season}|{week}|{g}|{_key(pl)}" for g, pl in zip(x.game, x.player)]
    stamp = now.strftime("%a %b %-d, %-I:%M %p CT")
    # the latest pre-game board for players already logged
    x = x.drop_duplicates("key")
    cur = x.set_index("key")
    hit = old.key.isin(cur.index)
    if hit.any():
        for src, dst in CLOSE_MAP.items():
            old[dst] = old[dst].astype(object)
            old.loc[hit, dst] = old.loc[hit, "key"].map(cur[src]).astype(object)
        old.loc[hit, "close_at"] = stamp
    new = x[~x.key.isin(set(old.key))].copy()
    if len(new):
        new["logged_at"] = stamp
        new["season"], new["week"] = season, week
        new["close_at"] = stamp
        for src, dst in CLOSE_MAP.items():
            new[dst] = new[src]
        new = new.assign(clv=np.nan, scorer=np.nan, result=np.nan, pnl=np.nan)[LOG_COLS]
        old = pd.concat([old, new], ignore_index=True) if len(old) else new
    # closing line value: how much better the first price was than the last pre-game price (same book or not)
    both = old.odds.notna() & old.close_odds.notna()
    old.loc[both, "clv"] = [decimal(float(a)) / decimal(float(b)) - 1 for a, b in zip(old.odds[both], old.close_odds[both])]
    old.to_csv(p_, index=False)
    return old


def _abbr(n: str) -> str:
    p = str(n).replace(".", " ").split()
    return f"{p[0][0]}.{p[-1]}" if len(p) > 1 else str(n)


def _name_key(s) -> str:
    return " ".join(str(s).lower().replace(".", " ").replace("'", "").replace("-", " ").split())


def espn_first_tds(days_back: int = 10, timeout: int = 20) -> dict:
    """{(away, home): (team, scoring-play text)} for the first touchdown of each finished game, from ESPN's scoring
    plays, so the First TD tab grades the night of the game (nflverse play-by-play can take a day). Never raises."""
    import json
    import urllib.request
    from .scores import ESPN_TO_NFLVERSE
    hdr = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    end = datetime.now(CT) + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=days_back + 1)
    out = {}
    try:
        url = ("https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
               f"?dates={start:%Y%m%d}-{end:%Y%m%d}&limit=100")
        with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=timeout) as r:
            evs = json.load(r).get("events", [])
    except Exception as e:
        print(f"  ESPN first TDs unavailable ({e})")
        return out
    for ev in evs:
        try:
            comp = ev["competitions"][0]
            if not comp.get("status", ev.get("status", {})).get("type", {}).get("completed"):
                continue
            side = {c["homeAway"]: ESPN_TO_NFLVERSE.get(c["team"]["abbreviation"], c["team"]["abbreviation"])
                    for c in comp["competitors"]}
            ids = {c["team"]["id"]: ESPN_TO_NFLVERSE.get(c["team"]["abbreviation"], c["team"]["abbreviation"])
                   for c in comp["competitors"]}
            su = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={ev['id']}"
            with urllib.request.urlopen(urllib.request.Request(su, headers=hdr), timeout=timeout) as r:
                plays = json.load(r).get("scoringPlays", [])
            tds = [p for p in plays if "touchdown" in str(p.get("type", {}).get("text", "")).lower()
                   or str(p.get("scoringType", {}).get("abbreviation", "")).upper() == "TD"]
            if not tds:
                out[(side["away"], side["home"])] = (None, "No touchdown")
                continue
            f = tds[0]
            out[(side["away"], side["home"])] = (ids.get(str(f.get("team", {}).get("id")), None), str(f.get("text", "")))
        except Exception:
            continue
    return out


def _scored_espn(player: str, text: str) -> bool:
    """ESPN scoring text starts with the scorer: 'Derrick Henry 5 Yd Run (...)', 'Zay Flowers 12 Yd pass from ...'."""
    t, n = _name_key(text), _name_key(player)
    return bool(n) and (t == n or t.startswith(n + " "))


def grade(season: int, path: Path | str = LOG, pbp=None, espn=None) -> pd.DataFrame:
    """Grade logged rows once the game is over: win if he scored the game's first TD. ESPN's scoring plays first
    (same night), nflverse play-by-play as the fallback."""
    p_ = Path(path)
    if not p_.exists():
        return pd.DataFrame(columns=LOG_COLS)
    lg = _load(p_)
    todo = lg.result.isna() & (pd.to_datetime(lg.kickoff, utc=True, errors="coerce")
                               < pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=3))
    if not todo.any():
        return lg
    ina = set()
    ip = ROOT / "inactives.csv"
    if ip.exists():
        ia = pd.read_csv(ip, comment="#")
        ina = {(int(r.season), int(r.week), _key(r.player), r.team) for r in ia.itertuples()}
    espn = espn_first_tds() if espn is None else espn
    first, played = {}, set()
    try:
        if pbp is None:
            url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
            pbp = pd.read_csv(url, low_memory=False, usecols=["game_id", "season", "week", "home_team", "away_team",
                                                              "play_id", "touchdown", "td_team", "td_player_name", "posteam"])
        td = pbp[(pbp.touchdown == 1) & pbp.td_team.notna()].sort_values(["game_id", "play_id"]).groupby("game_id").head(1)
        td = td.assign(game=td.away_team + " @ " + td.home_team)
        first = {(int(r.week), r.game): (r.td_team, str(r.td_player_name)) for r in td.itertuples()}
        played = set(zip(pbp.week.astype(int), pbp.away_team + " @ " + pbp.home_team))
    except Exception as e:
        print(f"  first TD play-by-play unavailable ({e})")
    for i in lg[todo].index:
        r = lg.loc[i]
        k = (int(r.week), r.game)
        aw, hm = str(r.game).split(" @ ")
        if (int(r.season), int(r.week), _key(r.player), r.team) in ina:
            lg.loc[i, ["result", "pnl"]] = ["void", 0.0]
            continue
        if (aw, hm) in espn:
            team, text = espn[(aw, hm)]
            win = team == r.team and _scored_espn(r.player, text)
            who = text.split(" (")[0]
            lg.loc[i, "scorer"] = f"{who} ({team})" if team else who
        elif k in played:
            team, who = first.get(k, (None, ""))
            win = team == r.team and who == _abbr(r.player)
            lg.loc[i, "scorer"] = f"{who} ({team})" if team else "No touchdown"
        else:
            continue
        odds = float(r.odds)
        lg.loc[i, ["result", "pnl"]] = ["win", decimal(odds) - 1] if win else ["loss", -1.0]
    lg.to_csv(p_, index=False)
    return lg


def frozen(lg: pd.DataFrame, game: str, season: int, week: int) -> pd.DataFrame:
    """The last pre-game board for a game that has kicked off, from the log (same columns as compute())."""
    if lg is None or lg.empty:
        return pd.DataFrame()
    d = lg[(lg.season == season) & (lg.week == week) & (lg.game == game)].copy()
    if d.empty:
        return d
    for src, dst in CLOSE_MAP.items():
        d[src] = d[dst].where(d[dst].notna(), d[src])
    d["value"] = d.value.astype(str).str.lower() == "true"
    d["fair"] = d.p.map(fair_american)
    d["size"] = d.odds.map(lambda o: suggested_units(o) if pd.notna(o) else np.nan)
    return d


def record(lg: pd.DataFrame, season: int) -> list[dict]:
    """Track record by group, at the first price we showed: values (✓), the top 4 per game by gap over the book (what
    the tab lists first), the model's #1 pick per team, everything logged. Beat close = first price better than the
    last pre-game price."""
    if lg is None or lg.empty:
        return []
    g = lg[(lg.season == season) & lg.result.isin(["win", "loss"])].copy()
    if g.empty:
        return []
    rg = g.close_rank_gap.where(g.close_rank_gap.notna(), g.rank_gap)
    if rg.isna().any():                                       # rows from before rank_gap was logged
        alt = g.gap.fillna(-1).groupby([g.season, g.week, g.game]).rank(ascending=False, method="first")
        rg = rg.fillna(alt)
    out = []
    for name, x in (("Values (price beats our fair odds)", g[g.value.astype(str).str.lower() == "true"]),
                    ("Top 4 per game by gap over the book", g[rg <= PER_TEAM]),
                    ("Model's top pick per team", g[g["rank"] == 1]),
                    ("Every player logged", g)):
        if len(x):
            sz = x.odds.map(lambda o: suggested_units(o))
            clv = x.clv.dropna()
            out.append(dict(group=name, n=len(x), wins=int((x.result == "win").sum()),
                            units=round(float(x.pnl.sum()), 1), sized=round(float((x.pnl * sz).sum()), 1),
                            hit=round(float((x.result == "win").mean()), 3), exp=round(float(x.p.mean()), 3),
                            beat_close=round(float((clv > 0).mean()), 3) if len(clv) else None))
    return out
