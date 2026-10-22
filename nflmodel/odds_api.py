"""Live DraftKings player prop lines from The Odds API (https://the-odds-api.com).

  python run.py fetch-props            # rewrites prop_lines.csv for this week's unplayed games

Needs an API key in the ODDS_API_KEY environment variable (or a one-line .odds_api_key file,
which is git-ignored). Never commit the key.

Cost: player props are priced per game: [markets requested] x [regions] credits per game.
With the six markets below and one region that is 6 credits a game, about 80-90 for a full
Sunday slate. Listing the week's games is free. Each run prints the credits left.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
BOOK = "draftkings"
SOURCE = "DraftKings (The Odds API)"
CT = ZoneInfo("America/Chicago")
COLS = ["player", "stat", "line", "over_odds", "under_odds", "source", "as_of"]

# The Odds API market key -> the stat names nflmodel/props.py prices
MARKETS = {
    "player_pass_yds": "pass_yds",
    "player_pass_tds": "pass_tds",
    "player_rush_yds": "rush_yds",
    "player_reception_yds": "rec_yds",
    "player_receptions": "receptions",
    "player_anytime_td": "anytime_td",
}


def api_key() -> str:
    key = os.environ.get("ODDS_API_KEY", "").strip()
    f = Path(__file__).resolve().parent.parent / ".odds_api_key"
    if not key and f.exists():
        key = f.read_text().strip()
    if not key:
        raise SystemExit("No Odds API key: set ODDS_API_KEY (or put it in .odds_api_key, which git ignores).")
    return key


def _ct(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(CT)


def _stamp(dt: datetime) -> str:
    return dt.strftime("%a %b %-d, %-I:%M %p CT")


def week_events(key: str, now: datetime) -> list[dict]:
    """Games that haven't kicked off yet and start before 6 AM CT next Tuesday (the end of this NFL week)."""
    days = (1 - now.weekday()) % 7 or 7          # Tuesday = 1
    end = (now + timedelta(days=days)).replace(hour=6, minute=0, second=0, microsecond=0)
    r = requests.get(f"{BASE}/events", params={"apiKey": key}, timeout=30)
    r.raise_for_status()
    return [e for e in r.json() if now < _ct(e["commence_time"]) < end]


def event_lines(key: str, event: dict) -> tuple[list[dict], str | None]:
    r = requests.get(f"{BASE}/events/{event['id']}/odds", timeout=30, params={
        "apiKey": key, "bookmakers": BOOK, "markets": ",".join(MARKETS), "oddsFormat": "american"})
    r.raise_for_status()
    left = r.headers.get("x-requests-remaining")
    rows: dict[tuple[str, str], dict] = {}
    for bm in r.json().get("bookmakers", []):
        if bm.get("key") != BOOK:
            continue
        for m in bm.get("markets", []):
            stat = MARKETS.get(m["key"])
            if not stat:
                continue
            as_of = _stamp(_ct(m.get("last_update") or bm["last_update"]))
            for o in m.get("outcomes", []):
                player = (o.get("description") or "").strip()
                if not player:
                    continue
                side = o.get("name", "").lower()
                row = rows.setdefault((player, stat), dict(player=player, stat=stat, line=0.5 if stat == "anytime_td" else o.get("point"),
                                                          over_odds=None, under_odds=None, source=SOURCE, as_of=as_of))
                if side in ("over", "yes"):
                    row["over_odds"] = o.get("price")
                    if stat != "anytime_td":
                        row["line"] = o.get("point")
                elif side in ("under", "no"):
                    row["under_odds"] = o.get("price")
    return list(rows.values()), left


def fetch(path: str = "prop_lines.csv", hours: float | None = None) -> None:
    """hours: only fetch games kicking off within this many hours (saves credits); None = rest of the week."""
    key = api_key()
    now = datetime.now(CT)
    events = week_events(key, now)
    if hours is not None:
        events = [e for e in events if _ct(e["commence_time"]) < now + timedelta(hours=hours)]
    new, left = [], None
    for e in events:
        rows, left = event_lines(key, e)
        new += rows
        print(f"  {e['away_team']} @ {e['home_team']}: {len(rows)} lines")
    fresh = pd.DataFrame(new, columns=COLS)

    # Games already under way are skipped (their in-game odds would mislead the model), so keep
    # the last pre-game API lines for those players. Older second-hand rows are dropped.
    out = Path(path)
    if out.exists():
        old = pd.read_csv(out, comment="#")
        old = old[old.source.astype(str).eq(SOURCE)]
        seen = set(zip(fresh.player, fresh.stat))
        keep = pd.Series([(p, s) not in seen for p, s in zip(old.player, old.stat)], index=old.index, dtype=bool)
        old = old[keep]
        fresh = pd.concat([fresh, old[COLS]], ignore_index=True)

    fresh = fresh.dropna(subset=["over_odds", "under_odds"], how="all").sort_values(["stat", "player"])
    with open(out, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"# checked: {_stamp(now)}\n")
        fresh.to_csv(fh, index=False)
    print(f"Wrote {len(fresh)} DraftKings lines from {len(events)} games to {out.name}"
          + (f"  (Odds API credits left: {left})" if left is not None else ""))
