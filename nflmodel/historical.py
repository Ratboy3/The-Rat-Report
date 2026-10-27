"""One-off historical pull (paid Odds API plan): closing game lines at DraftKings, FanDuel, the other US books and
Pinnacle for past weeks, to test DraftKings vs Pinnacle without waiting weeks for live data (Oct 27, 2026).

For every distinct kickoff time in the chosen weeks, one snapshot of all games ~5 minutes before kickoff; only the
games kicking off at that time are kept (their closing lines). Runs in GitHub Actions, where the API key lives:

  python run.py fetch-history --weeks 2025:1-18,2026:1-7 --max-credits 4500 [--test]

Writes data/hist_game_lines.csv (appends; snapshots already fetched are skipped) and data/hist_pull_log.csv
(one row per call: snapshot time, games kept, credits used by the call, credits left).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from . import odds_api as O

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "hist_game_lines.csv"
LOG = ROOT / "data" / "hist_pull_log.csv"
ET = ZoneInfo("America/New_York")
URL = "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/odds"
KEEP_FLOOR = 8000      # never let this pull take the month's credits below this (the live schedule needs ~7,500)


def parse_weeks(spec: str) -> list[tuple[int, int]]:
    out = []
    for part in spec.split(","):
        s, wk = part.split(":")
        a, b = (wk.split("-") + [wk])[:2]
        out += [(int(s), w) for w in range(int(a), int(b) + 1)]
    return out


def snapshots(games: pd.DataFrame, weeks: list[tuple[int, int]]) -> list[datetime]:
    g = games[games.set_index(["season", "week"]).index.isin(weeks) & games.gametime.notna()]
    ko = {datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M").replace(tzinfo=ET) for d, t in zip(g.gameday, g.gametime)}
    return sorted(k - timedelta(minutes=5) for k in ko)


def run(games: pd.DataFrame, spec: str, max_credits: int = 4500, test: bool = False) -> None:
    key = O.api_key()
    snaps = snapshots(games, parse_weeks(spec))
    done = set(pd.read_csv(LOG).snapshot) if LOG.exists() else set()
    todo = [s for s in snaps if s.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") not in done]
    print(f"{len(snaps)} kickoff snapshots, {len(todo)} still to fetch")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if not todo:
        pd.DataFrame([dict(snapshot="ERROR none", data_time="", games=0, rows=0, cost=0, left=-1,
                           error=f"no snapshots found for {spec} ({len(games)} games loaded)")]).to_csv(
            LOG.with_name("hist_pull_errors.csv"), mode="a", header=not LOG.with_name("hist_pull_errors.csv").exists(), index=False)
    spent = 0
    for s in todo[:1] if test else todo:
        iso = s.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        r = requests.get(URL, timeout=60, params={"apiKey": key, "bookmakers": ",".join(O.GAME_BOOKS),
                                                   "markets": "h2h,spreads,totals", "oddsFormat": "american", "date": iso})
        if r.status_code != 200:
            msg = r.text[:300].replace(key, "***")
            print(f"  {iso}: HTTP {r.status_code} {msg}")
            pd.DataFrame([dict(snapshot=f"ERROR {iso}", data_time="", games=0, rows=0, cost=0, left=-1, error=f"HTTP {r.status_code}: {msg}")]
                         ).to_csv(LOG.with_name("hist_pull_errors.csv"), mode="a", header=not LOG.with_name("hist_pull_errors.csv").exists(), index=False)
            break
        cost = int(float(r.headers.get("x-requests-last", 0) or 0))
        left = int(float(r.headers.get("x-requests-remaining", 0) or 0))
        spent += cost
        body = r.json()
        rows = []
        for e in body.get("data", []):
            ko = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
            if not (s < ko <= s + timedelta(minutes=20)):
                continue                       # only the games kicking off at this time: their closing line
            home, away = O.TEAM_ABBR.get(e["home_team"]), O.TEAM_ABBR.get(e["away_team"])
            for bm in e.get("bookmakers", []):
                for m in bm.get("markets", []):
                    for o in m.get("outcomes", []):
                        name = o.get("name", "")
                        side = ("home" if O.TEAM_ABBR.get(name) == home else "away") if m["key"] != "totals" else name.lower()
                        rows.append(dict(snapshot=iso, kickoff=e["commence_time"], game=f"{away} @ {home}", book=bm["key"],
                                         market={"h2h": "moneyline", "spreads": "spread", "totals": "total"}[m["key"]],
                                         side=side, line=o.get("point"), odds=o.get("price")))
        if rows:
            pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        pd.DataFrame([dict(snapshot=iso, data_time=body.get("timestamp"), games=len({r["game"] for r in rows}),
                           rows=len(rows), cost=cost, left=left)]).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
        print(f"  {iso}: {len({r['game'] for r in rows})} games, cost {cost}, {left} left")
        if spent + cost > max_credits or left - cost < KEEP_FLOOR:
            print(f"Stopping: spent {spent} (cap {max_credits}), {left} left (floor {KEEP_FLOOR})")
            break
    print(f"Done: {spent} credits spent")


# ------------------------------------------------------------------ historical player props
PROPS_OUT = ROOT / "data" / "hist_prop_lines.csv"
PROPS_LOG = ROOT / "data" / "hist_props_log.csv"
EVENTS_URL = "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/events"
EVENT_ODDS_URL = "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/events/{eid}/odds"


def run_props(games: pd.DataFrame, spec: str, max_credits: int = 6000) -> None:
    """Closing player props (the 6 markets the model prices, 10 named books incl. DraftKings, FanDuel, Pinnacle)
    ~5 minutes before each kickoff. Costs ~1 credit per snapshot to list events plus ~60 per game (10 x 6 markets).
    Stops at max_credits or before the month's balance drops under KEEP_FLOOR. Resumable: games already fetched
    (hist_props_log.csv) are skipped."""
    key = O.api_key()
    snaps = snapshots(games, parse_weeks(spec))
    done = set(pd.read_csv(PROPS_LOG).event_id) if PROPS_LOG.exists() else set()
    PROPS_LOG.parent.mkdir(parents=True, exist_ok=True)
    spent, left = 0, None
    for s in snaps:
        iso = s.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        r = requests.get(EVENTS_URL, timeout=60, params={"apiKey": key, "date": iso})
        if r.status_code != 200:
            print(f"  events {iso}: HTTP {r.status_code}")
            break
        spent += int(float(r.headers.get("x-requests-last", 0) or 0))
        evs = [e for e in r.json().get("data", [])
               if s < datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) <= s + timedelta(minutes=20)
               and e["id"] not in done]
        for e in evs:
            if left is not None and (spent + 60 > max_credits or left - 60 < KEEP_FLOOR):
                print(f"Stopping: spent {spent} (cap {max_credits}), {left} left (floor {KEEP_FLOOR})")
                print(f"Done: {spent} credits spent")
                return
            q = requests.get(EVENT_ODDS_URL.format(eid=e["id"]), timeout=60, params={
                "apiKey": key, "bookmakers": ",".join(O.PROP_BOOKS), "markets": ",".join(O.MARKETS),
                "oddsFormat": "american", "date": iso})
            if q.status_code != 200:
                print(f"  {e['id']}: HTTP {q.status_code} {q.text[:150].replace(key, '***')}")
                continue
            cost = int(float(q.headers.get("x-requests-last", 0) or 0))
            left = int(float(q.headers.get("x-requests-remaining", 0) or 0))
            spent += cost
            ev = q.json().get("data", {})
            game = f"{O.TEAM_ABBR.get(e['away_team'], e['away_team'])} @ {O.TEAM_ABBR.get(e['home_team'], e['home_team'])}"
            rows: dict = {}
            for bm in ev.get("bookmakers", []):
                for m in bm.get("markets", []):
                    stat = O.MARKETS.get(m["key"])
                    if not stat:
                        continue
                    for o in m.get("outcomes", []):
                        player = (o.get("description") or "").strip()
                        if not player:
                            continue
                        row = rows.setdefault((bm["key"], player, stat), dict(
                            snapshot=iso, kickoff=e["commence_time"], game=game, player=player, stat=stat, book=bm["key"],
                            line=0.5 if stat == "anytime_td" else o.get("point"), over_odds=None, under_odds=None))
                        side = o.get("name", "").lower()
                        if side in ("over", "yes"):
                            row["over_odds"] = o.get("price")
                            if stat != "anytime_td":
                                row["line"] = o.get("point")
                        elif side in ("under", "no"):
                            row["under_odds"] = o.get("price")
            if rows:
                pd.DataFrame(list(rows.values())).to_csv(PROPS_OUT, mode="a", header=not PROPS_OUT.exists(), index=False)
            pd.DataFrame([dict(event_id=e["id"], snapshot=iso, game=game, rows=len(rows), cost=cost, left=left)]).to_csv(
                PROPS_LOG, mode="a", header=not PROPS_LOG.exists(), index=False)
            print(f"  {game} {iso}: {len(rows)} lines, cost {cost}, {left} left", flush=True)
    print(f"Done: {spent} credits spent")
