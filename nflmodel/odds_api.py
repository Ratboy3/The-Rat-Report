"""Live odds from The Odds API (https://the-odds-api.com): DraftKings plus every other US sportsbook.

  python run.py fetch-props [--hours 12]   # player props -> prop_lines.csv (DraftKings) + prop_books.csv (all books)
  python run.py fetch-games                # spreads, totals, moneylines -> game_lines.csv (all books)

Needs an API key in the ODDS_API_KEY environment variable (or a one-line .odds_api_key file,
which is git-ignored). Never commit the key.

Cost (free plan = 500 credits a month): credits = markets x regions. Asking for the whole "us" region
returns every US book for the same price as DraftKings alone.
  props: 6 markets x 1 region = 6 credits per game fetched
  games: 3 markets x 1 region = 3 credits per call, all games at once
Listing the week's games is free. Each run prints the credits left.
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
REGION = "us"
CT = ZoneInfo("America/Chicago")
COLS = ["player", "stat", "line", "over_odds", "under_odds", "source", "as_of"]
BOOK_COLS = ["game", "player", "stat", "book", "line", "over_odds", "under_odds", "as_of"]
GAME_COLS = ["game", "kickoff", "book", "market", "side", "line", "odds", "as_of"]

# The Odds API market key -> the stat names nflmodel/props.py prices
MARKETS = {
    "player_pass_yds": "pass_yds",
    "player_pass_tds": "pass_tds",
    "player_rush_yds": "rush_yds",
    "player_reception_yds": "rec_yds",
    "player_receptions": "receptions",
    "player_anytime_td": "anytime_td",
}
TEAM_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
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


def _game(e: dict) -> str:
    return f"{TEAM_ABBR.get(e['away_team'], e['away_team'])} @ {TEAM_ABBR.get(e['home_team'], e['home_team'])}"


def week_events(key: str, now: datetime) -> list[dict]:
    """Games that haven't kicked off yet and start before 6 AM CT next Tuesday (the end of this NFL week)."""
    days = (1 - now.weekday()) % 7 or 7          # Tuesday = 1
    end = (now + timedelta(days=days)).replace(hour=6, minute=0, second=0, microsecond=0)
    r = requests.get(f"{BASE}/events", params={"apiKey": key}, timeout=30)
    r.raise_for_status()
    return [e for e in r.json() if now < _ct(e["commence_time"]) < end]


def event_lines(key: str, event: dict) -> tuple[list[dict], str | None]:
    """Every US book's prices for one game: rows with book, player, stat, line, over/under odds."""
    r = requests.get(f"{BASE}/events/{event['id']}/odds", timeout=30, params={
        "apiKey": key, "regions": REGION, "markets": ",".join(MARKETS), "oddsFormat": "american"})
    r.raise_for_status()
    left = r.headers.get("x-requests-remaining")
    rows: dict[tuple[str, str, str], dict] = {}
    for bm in r.json().get("bookmakers", []):
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
                row = rows.setdefault((bm["key"], player, stat), dict(
                    game=_game(event), player=player, stat=stat, book=bm["key"],
                    line=0.5 if stat == "anytime_td" else o.get("point"), over_odds=None, under_odds=None, as_of=as_of))
                if side in ("over", "yes"):
                    row["over_odds"] = o.get("price")
                    if stat != "anytime_td":
                        row["line"] = o.get("point")
                elif side in ("under", "no"):
                    row["under_odds"] = o.get("price")
    return list(rows.values()), left


def _write(path: Path, df: pd.DataFrame, now: datetime) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"# checked: {_stamp(now)}\n")
        df.to_csv(fh, index=False)


def fetch(path: str = "prop_lines.csv", hours: float | None = None, books_path: str = "prop_books.csv") -> None:
    """hours: only fetch games kicking off within this many hours (saves credits); None = rest of the week."""
    key = api_key()
    now = datetime.now(CT)
    events = week_events(key, now)
    if hours is not None:
        events = [e for e in events if _ct(e["commence_time"]) < now + timedelta(hours=hours)]
    allrows, left = [], None
    for e in events:
        rows, left = event_lines(key, e)
        allrows += rows
        print(f"  {_game(e)}: {sum(1 for r in rows if r['book'] == BOOK)} DraftKings lines, "
              f"{len({r['book'] for r in rows})} books")
    books = pd.DataFrame(allrows, columns=BOOK_COLS)
    fresh = books[books.book == BOOK].assign(source=SOURCE)[COLS]

    # Games already under way are skipped (their in-game odds would mislead the model), so keep
    # the last pre-game API lines for those players. Older second-hand rows are dropped.
    out = Path(path)
    if out.exists():
        old = pd.read_csv(out, comment="#")
        old = old[old.source.astype(str).eq(SOURCE)]
        seen = set(zip(fresh.player, fresh.stat))
        keep = pd.Series([(p, s) not in seen for p, s in zip(old.player, old.stat)], index=old.index, dtype=bool)
        fresh = pd.concat([fresh, old[keep][COLS]], ignore_index=True)
    bout = Path(books_path)
    if bout.exists():
        oldb = pd.read_csv(bout, comment="#")
        fetched = set(books.game)
        books = pd.concat([books, oldb[~oldb.game.isin(fetched)]], ignore_index=True)

    fresh = fresh.dropna(subset=["over_odds", "under_odds"], how="all").sort_values(["stat", "player"])
    _write(out, fresh, now)
    _write(bout, books.dropna(subset=["over_odds", "under_odds"], how="all").sort_values(["game", "player", "stat", "book"]), now)
    print(f"Wrote {len(fresh)} DraftKings lines and {len(books)} lines from all books ({len(events)} games)"
          + (f"  (Odds API credits left: {left})" if left is not None else ""))


def fetch_games(path: str = "game_lines.csv") -> None:
    """Spreads, totals and moneylines for every upcoming game at every US book: one call, 3 credits."""
    key = api_key()
    now = datetime.now(CT)
    r = requests.get(f"{BASE}/odds", timeout=30, params={
        "apiKey": key, "regions": REGION, "markets": "h2h,spreads,totals", "oddsFormat": "american"})
    r.raise_for_status()
    rows = []
    for e in r.json():
        if _ct(e["commence_time"]) <= now:
            continue
        home, away = TEAM_ABBR.get(e["home_team"]), TEAM_ABBR.get(e["away_team"])
        for bm in e.get("bookmakers", []):
            for m in bm.get("markets", []):
                as_of = _stamp(_ct(m.get("last_update") or bm["last_update"]))
                for o in m.get("outcomes", []):
                    name = o.get("name", "")
                    side = ("home" if TEAM_ABBR.get(name) == home else "away") if m["key"] != "totals" else name.lower()
                    rows.append(dict(game=_game(e), kickoff=_stamp(_ct(e["commence_time"])), book=bm["key"],
                                     market={"h2h": "moneyline", "spreads": "spread", "totals": "total"}[m["key"]],
                                     side=side, line=o.get("point"), odds=o.get("price"), as_of=as_of))
    df = pd.DataFrame(rows, columns=GAME_COLS)
    out = Path(path)
    if out.exists():   # keep the last pre-game lines for games that have started
        old = pd.read_csv(out, comment="#")
        df = pd.concat([df, old[~old.game.isin(set(df.game))]], ignore_index=True)
    _write(out, df, now)
    print(f"Wrote {len(df)} game lines from {df.book.nunique()} books"
          + (f"  (Odds API credits left: {r.headers.get('x-requests-remaining')})" if r.headers.get("x-requests-remaining") else ""))
