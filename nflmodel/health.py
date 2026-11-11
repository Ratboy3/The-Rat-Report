"""Data health and integrity checks (Nov 4, 2026).

Alex: "It just doesn't feel like everything is up to date with the website." In one day three bugs showed old data as
current (last week's prop lines, last week's game-day inactives, a special-teams-only game read as a missed game).
This module is the guard rail, run on every report build:

1. sources(): how old every input is (prop prices, game lines, injury reports, last week's stats and snap counts, news,
   weather, rosters), per game where it matters, shown on the site as "Data health".
2. hold_stale(): a bet that depends on an input that should have updated by now and hasn't becomes a lean, with the
   reason. A stale source never produces a bet.
3. integrity(): hard checks on every bet about to be shown (price is this week's and matches the latest pull for that
   game and book, player not ruled out / held / special-teams-only, no "out" status carried from last week's roster,
   one row per bet in the tracker). A violation is held as a lean too, listed on the page, and makes the GitHub
   rebuild finish red (run.py integrity-check) so it gets noticed.
Writes output/health.json.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

CT = ZoneInfo("America/Chicago")
ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

PROP_MAX_H = 48          # full prop pulls run Tue-Sat 9 AM / 4 PM / 9 PM and Sun 7 AM: Sat 9 PM -> Mon 4 PM is ~43 h
PROP_NEAR_MAX_H = 2.5    # within 3 h of kickoff props are pulled every hour
LINES_MAX_H = 3          # game lines pulled every hour 6 AM-11 PM CT
LINES_NIGHT_MAX_H = 9
INJ_DUE_H = 72           # a team's first injury report of the week is due by ~3 days before kickoff
STATS_DUE_H = 40         # last week's player stats and snap counts are due ~1.5 days after that game
NEWS_MAX_H = 6           # research update every 2 h, 7:50 AM-9:50 PM


def _parse_asof(s, year=None):
    """Timestamp string without a year -> aware datetime, year chosen closest to now (Dec -> Jan safe)."""
    from .props import _parse_asof as p_
    t = p_(s, pd.Timestamp(datetime.now(CT).replace(tzinfo=None)))
    return None if pd.isna(t) else t.to_pydatetime().replace(tzinfo=CT)


def _checked(path: str | Path):
    p = Path(path)
    if not p.exists():
        return None
    first = p.open().readline()
    return _parse_asof(first.split(":", 1)[1], datetime.now(CT).year) if first.startswith("# checked:") else None


def _kick(g) -> datetime:
    return datetime.strptime(f"{g.gameday} {g.gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=ET).astimezone(CT)


def _age(t, now) -> float | None:
    return None if t is None else round((now - t).total_seconds() / 3600, 1)


def _fmt(t) -> str:
    return t.strftime("%a %b %-d, %-I:%M %p CT") if t is not None else "never"


def sources(season: int, week: int, proj: pd.DataFrame, prop_books: str = "prop_books.csv",
            game_lines: str = "game_lines.csv", news: str = "news.csv", weather: str = "weather.csv",
            now: datetime | None = None) -> dict:
    """{'rows': [source rows for the panel], 'stale': {game: {'game': [reasons], 'props': [reasons], 'team': {...}}}}"""
    now = now or datetime.now(CT)
    rows, stale = [], {}
    games = [g for g in proj.itertuples() if _kick(g) > now]
    key = lambda g: f"{g.away_team} @ {g.home_team}"

    def add(name, t, limit, scope, detail="", ok=None):
        a = _age(t, now)
        good = ok if ok is not None else (a is not None and a <= limit)
        rows.append(dict(source=name, updated=_fmt(t), age_h=a, limit_h=limit, ok=bool(good), scope=scope, detail=detail))
        return good

    # prop prices: per game, DraftKings' latest price for that game in the all-books pull
    pb = pd.read_csv(prop_books, comment="#") if Path(prop_books).exists() else pd.DataFrame()
    late = []
    for g in games:
        k = key(g)
        d = pb[(pb.get("game") == k) & (pb.get("book") == "draftkings")] if len(pb) else pb
        t = max((x for x in (_parse_asof(a, now.year) for a in d.as_of.unique()) if x is not None), default=None) if len(d) else None
        near = (_kick(g) - now) <= timedelta(hours=3)
        lim = PROP_NEAR_MAX_H if near else PROP_MAX_H
        if t is None or _age(t, now) > lim:
            late.append(f"{k} ({_fmt(t)})")
            stale.setdefault(k, {}).setdefault("props", []).append(
                "no DraftKings prop prices for this game yet" if t is None else f"DraftKings prop prices from {_fmt(t)}")
    add("Prop prices (DraftKings, FanDuel)", _checked(prop_books), PROP_MAX_H, "props",
        ("behind for " + "; ".join(late)) if late else "every upcoming game current", ok=not late)

    # game lines: the file must be fresh AND every upcoming game must be in it at DraftKings (a game missing from the
    # pull has no current line at any book: Nov 4, 2026 audit)
    t = _checked(game_lines)
    lim = LINES_MAX_H if 6 <= now.hour <= 23 else LINES_NIGHT_MAX_H
    gl = pd.read_csv(game_lines, comment="#") if Path(game_lines).exists() else pd.DataFrame()
    have_dk = set(gl[gl.book == "draftkings"].game) if len(gl) else set()
    no_line = [key(g) for g in games if key(g) not in have_dk]
    fresh = _age(t, now) is not None and _age(t, now) <= lim
    add("Game lines (all books)", t, lim, "game bets",
        ("no DraftKings line in the latest pull for " + ", ".join(no_line)) if no_line else "every upcoming game listed",
        ok=fresh and not no_line)
    for g in games:
        if not fresh:
            stale.setdefault(key(g), {}).setdefault("game", []).append(f"game lines last pulled {_fmt(t)}")
        elif key(g) in no_line:
            stale.setdefault(key(g), {}).setdefault("game", []).append("no DraftKings game line in the latest pull")

    # nflverse: injury reports this week, last week's stats and snaps
    from .player_proj import _load
    inj = _load("injuries", season, False)
    stats = _load("stats", season, False)
    from .injuries import _load_snaps
    snaps = _load_snaps(season, False)
    roster = _load("roster", season, False)
    inj_w = inj[(inj.season == season) & (inj.week == week)] if inj is not None and len(inj) else pd.DataFrame()
    have_inj = set(inj_w.team) if len(inj_w) else set()
    missing_inj, due_later = [], []
    for g in games:
        for team in (g.away_team, g.home_team):
            if team in have_inj:
                continue
            if (_kick(g) - now) <= timedelta(hours=INJ_DUE_H):
                missing_inj.append(team)
                s = stale.setdefault(key(g), {})
                s.setdefault("game", []).append(f"{team} injury report not in the data yet")
                s.setdefault("team", {}).setdefault(team, []).append(f"{team} injury report not in the data yet")
            else:
                due_later.append(team)
    add("Injury reports (this week)", None, INJ_DUE_H, "game bets, props",
        (f"posted for {len(have_inj)} teams" + (f"; overdue: {', '.join(sorted(set(missing_inj)))}" if missing_inj else "")
         + (f"; not due yet: {len(set(due_later))} teams" if due_later else "")), ok=not missing_inj)

    sched = proj  # this week's games only; last games come from stats/snaps weeks
    lag_stats, lag_snaps = [], []
    all_games = _load_games_cached()
    for g in games:
        for team in (g.away_team, g.home_team):
            last = _last_game(all_games, team, season, week)
            if last is None:
                continue
            lw, lt = last
            if (now - lt) < timedelta(hours=STATS_DUE_H):
                continue
            if stats is not None and not len(stats[(stats.week == lw) & (stats.team == team)]):
                lag_stats.append(team)
                stale.setdefault(key(g), {}).setdefault("team", {}).setdefault(team, []).append(
                    f"{team}'s Week {lw} player stats not posted yet")
            if snaps is not None and not len(snaps[(snaps.week == lw) & (snaps.team == team)]):
                lag_snaps.append(team)
                stale.setdefault(key(g), {}).setdefault("team", {}).setdefault(team, []).append(
                    f"{team}'s Week {lw} snap counts not posted yet")
    add("Last week's player stats", None, STATS_DUE_H, "props",
        "posted for every team" if not lag_stats else "missing: " + ", ".join(sorted(set(lag_stats))), ok=not lag_stats)
    add("Last week's snap counts", None, STATS_DUE_H, "props",
        "posted for every team" if not lag_snaps else "missing: " + ", ".join(sorted(set(lag_snaps))), ok=not lag_snaps)

    rw = int(roster.week.max()) if roster is not None and len(roster) else None
    add("Rosters", None, 0, "info", (f"Week {rw} roster" + (" (this week's not posted yet; last week's game-day "
                                       "inactives are not treated as out)" if rw is not None and rw < week else "")), ok=True)
    nt = _checked(news)
    add("News check (QBs, injuries, role checks)", nt, NEWS_MAX_H if 7 <= now.hour <= 22 else 12, "notes")
    add("Weather forecasts", _checked(weather), 12, "outdoor games")
    return dict(rows=rows, stale=stale, now=_fmt(now))


_GAMES = {}


def _load_games_cached():
    if "g" not in _GAMES:
        try:
            from .data import load_games
            _GAMES["g"] = load_games(refresh=False)
        except Exception:
            _GAMES["g"] = None
    return _GAMES["g"]


def _last_game(games, team, season, week):
    if games is None:
        return None
    g = games[(games.season == season) & (games.week < week) & ((games.home_team == team) | (games.away_team == team))
              & games.result.notna()]
    if g.empty:
        return None
    x = g.sort_values("week").iloc[-1]
    try:
        return int(x.week), _kick(x)
    except Exception:
        return int(x.week), datetime.now(CT) - timedelta(days=3)


def hold_stale(games_out: list, props: pd.DataFrame | None, health: dict) -> int:
    """Turn bets that depend on a stale input into leans. Returns how many were held."""
    n, st = 0, health.get("stale", {})
    for item in games_out:
        g = item["g"]
        s = st.get(f"{g.away_team} @ {g.home_team}", {})
        why = s.get("game", [])
        if not why:
            continue
        for o in item["opts"]:
            if o.get("bet"):
                o["bet"], o["held"] = False, "data check: " + "; ".join(why)
                n += 1
    if props is not None and len(props) and "flag" in props:
        if "why_stale" not in props:
            props["why_stale"] = ""
        for i, r in props[props.flag.astype(bool)].iterrows():
            s = st.get(r.game, {})
            why = s.get("props", []) + s.get("team", {}).get(r.team, [])
            if why:
                props.at[i, "flag"] = False
                props.at[i, "why_stale"] = "data check: " + "; ".join(why)
                n += 1
    return n


def integrity(season: int, week: int, games_out: list, props: pd.DataFrame | None, injuries: dict | None,
              prop_books: str = "prop_books.csv", bets_log: str = "bets_log.csv", now: datetime | None = None) -> list:
    """Hard checks on what's about to be shown. Each violation is held as a lean. Returns the list of violations."""
    now = now or datetime.now(CT)
    v = []
    start = None
    ks = [_kick(item["g"]) for item in games_out]
    if ks:
        from .props import week_start
        start = week_start([k.date() for k in ks]).to_pydatetime().replace(tzinfo=CT)
    pb = pd.read_csv(prop_books, comment="#") if Path(prop_books).exists() else pd.DataFrame()
    holds = set()
    for fn in ("inactives.csv", "role_holds.csv"):
        p = ROOT / fn
        if p.exists():
            d = pd.read_csv(p, comment="#")
            d = d[(d.season == season) & (d.week == week)]
            holds |= set(zip(d.player, d.team))
    if props is not None and len(props) and "flag" in props:
        if "why_stale" not in props:
            props["why_stale"] = ""
        bk = {"DraftKings": "draftkings", "FanDuel": "fanduel"}
        for i, r in props[props.flag.astype(bool)].iterrows():
            bad = []
            t = _parse_asof(r.as_of, now.year)
            if start is not None and (t is None or t < start):
                bad.append(f"price from {r.as_of}, before this week")
            if "Odds API" not in str(r.source):
                bad.append("price not from a live sportsbook pull")
            if len(pb) and r.book in bk:
                from .props import _key
                m = pb[(pb.game == r.game) & (pb.book == bk[r.book]) & (pb.stat == r.stat)]
                m = m[m.player.map(_key) == _key(r.player)]
                if m.empty:
                    bad.append(f"{r.book} doesn't list this prop for {r.game} in the latest pull")
                else:
                    side_odds = r.over_odds if r.best_side == "Over" else r.under_odds
                    col = "over_odds" if r.best_side == "Over" else "under_odds"
                    if not ((m.line == r.line) & (m[col] == side_odds)).any():
                        bad.append(f"{r.book} price doesn't match the latest pull for {r.game}")
            if r.book == "FanDuel" and pd.isna(r.get("dk_line", np.nan)):
                bad.append("DraftKings doesn't list this prop")
            from .props import _key as _k
            if str(r.team) not in str(r.game).split(" @ "):
                bad.append(f"player's team {r.team} isn't in {r.game}")
            if (r.player, r.team) in holds or (_k(r.player), r.team) in {(_k(a), b) for a, b in holds}:
                bad.append("player ruled out or on a role hold")
            if r.get("why_st_only", np.nan) == 1.0:
                bad.append("played only special teams last game")
            if r.stat == "anytime_td" and pd.notna(r.get("model_prob", np.nan)) and pd.notna(r.over_odds):
                from .betting import implied as _imp
                from .props import TD_MODEL_MUST_CLEAR as _tdm
                if _tdm and r.model_prob < _imp(r.over_odds):
                    bad.append("TD bet where our model alone is below the break-even")
            if bad:
                props.at[i, "flag"] = False
                props.at[i, "why_stale"] = "data check: " + "; ".join(bad)
                v.append(dict(kind="prop", bet=f"{r.player} {r.best_side} {r.line:g} {r.stat} ({r.game})", problems=bad))
    # injury labels: nothing "Inactive" from an earlier week's roster
    try:
        from .player_proj import _load
        ro = _load("roster", season, False)
        if ro is not None and int(ro.week.max()) < week and injuries:
            for team, d in injuries.items():
                for (name, pos, pts, note) in d.get("players", []):
                    if note == "Inactive":
                        v.append(dict(kind="injury", bet=f"{name} ({team})",
                                      problems=["listed inactive from last week's roster"]))
    except Exception:
        pass
    # game bets: not started, priced from a game line pull within 48 h
    for item in games_out:
        g = item["g"]
        if _kick(g) <= now:
            continue
        for o in item["opts"]:
            if o.get("bet") and o.get("odds") is None:
                o["bet"], o["held"] = False, "data check: no price"
                v.append(dict(kind="game", bet=f"{o.get('pick')} ({g.away_team} @ {g.home_team})", problems=["no price"]))
    # bet tracker: one row per bet
    p = Path(bets_log)
    if p.exists():
        L = pd.read_csv(p)
        d = L[L.bet_id.duplicated()]
        if len(d):
            v.append(dict(kind="log", bet=f"{len(d)} duplicate bet ids", problems=list(d.bet_id.head(5))))
    return v


def write(health: dict, violations: list, held: int, season: int, week: int) -> dict:
    OUT.mkdir(exist_ok=True)
    out = dict(season=season, week=week, checked=health.get("now"), sources=health.get("rows", []),
               violations=violations, held_stale=held)
    (OUT / "health.json").write_text(json.dumps(out, indent=1, default=str))
    return out
