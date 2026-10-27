"""Live odds from The Odds API (https://the-odds-api.com): DraftKings plus every other US sportsbook.

  python run.py fetch-props [--hours 12]   # player props -> prop_lines.csv (DraftKings) + prop_books.csv (all books)
  python run.py fetch-games                # spreads, totals, moneylines -> game_lines.csv (all books)

Needs an API key in the ODDS_API_KEY environment variable (or a one-line .odds_api_key file,
which is git-ignored). Never commit the key.

Cost (free plan = 500 credits a month): credits = markets x regions. Asking for the whole "us" region
returns every US book for the same price as DraftKings alone.
  props: 6 markets x 1 region = 6 credits per game fetched
  games: 3 markets x 1 region = 3 credits per call, all games at once. The call names its books instead of a
         region (the 9 US books seen so far + Pinnacle): up to 10 named books cost the same as one region, so
         Pinnacle (the sharp offshore book whose lines move first) comes in for no extra credits. It is archived
         for the sharp-book test and never shown as a place to bet.
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
# Props: named books instead of the "us" region (up to 10 named books cost the same as one region, 6 credits per game),
# which adds Pinnacle as a sharp reference plus Caesars, ESPN Bet and Fanatics where the API carries them (Oct 27, 2026).
PROP_BOOKS = ["draftkings", "fanduel", "betmgm", "betrivers", "bovada", "betonlineag", "williamhill_us", "espnbet",
              "fanatics", "pinnacle"]
GAME_BOOKS = ["draftkings", "fanduel", "betmgm", "betrivers", "bovada", "betonlineag", "lowvig", "mybookieag",
              "betus", "pinnacle"]   # 10 books = 1 region-equivalent: still 3 credits
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
        "apiKey": key, "bookmakers": ",".join(PROP_BOOKS), "markets": ",".join(MARKETS), "oddsFormat": "american"})
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


def _archive(path: str, df: pd.DataFrame, now: datetime) -> None:
    """Append every pull to a history file (prop_books.csv / game_lines.csv only keep the latest), so the
    sharp-book and bet-timing tests have the full record."""
    if df is None or df.empty:
        return
    out = Path(path)
    df = df.assign(fetched_at=now.astimezone(CT).strftime("%Y-%m-%d %H:%M"))
    df.to_csv(out, mode="a", header=not out.exists(), index=False)


def fetch(path: str = "prop_lines.csv", hours: float | None = None, books_path: str = "prop_books.csv") -> int:
    """hours: only fetch games kicking off within this many hours (saves credits); None = rest of the week."""
    key = api_key()
    now = datetime.now(CT)
    events = week_events(key, now)
    if hours is not None:
        events = [e for e in events if _ct(e["commence_time"]) < now + timedelta(hours=hours)]
    if not events:
        print(f"No games {'in the next %g hours' % hours if hours is not None else 'left this week'}; no props fetched (0 credits)")
        return 0
    allrows, left = [], None
    for e in events:
        rows, left = event_lines(key, e)
        for r in rows:
            r["kickoff"] = e["commence_time"]
        allrows += rows
        print(f"  {_game(e)}: {sum(1 for r in rows if r['book'] == BOOK)} DraftKings lines, "
              f"{len({r['book'] for r in rows})} books")
    _archive("prop_books_history.csv", pd.DataFrame(allrows, columns=BOOK_COLS + ["kickoff"]), now)
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
    return len(events)


def fetch_early(hours: float = 3.5) -> None:
    """Sunday-morning pull for games abroad (London, Paris, etc. kick off around 8:30 AM CT, before the 11 AM pull).
    Listing events is free, so on weeks with no early game this spends nothing."""
    key = api_key()
    now = datetime.now(CT)
    soon = [e for e in week_events(key, now) if _ct(e["commence_time"]) < now + timedelta(hours=hours)]
    if not soon:
        print(f"No games in the next {hours:g} hours; nothing fetched (0 credits)")
        return
    fetch(hours=hours)
    fetch_games()


def fetch_games(path: str = "game_lines.csv") -> None:
    """Spreads, totals and moneylines for every upcoming game at every US book: one call, 3 credits."""
    key = api_key()
    now = datetime.now(CT)
    r = requests.get(f"{BASE}/odds", timeout=30, params={
        "apiKey": key, "bookmakers": ",".join(GAME_BOOKS), "markets": "h2h,spreads,totals", "oddsFormat": "american"})
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
    _archive("game_lines_history.csv", df, now)
    out = Path(path)
    if out.exists():   # keep the last pre-game lines for games that have started
        old = pd.read_csv(out, comment="#")
        df = pd.concat([df, old[~old.game.isin(set(df.game))]], ignore_index=True)
    _write(out, df, now)
    print(f"Wrote {len(df)} game lines from {df.book.nunique()} books"
          + (f"  (Odds API credits left: {r.headers.get('x-requests-remaining')})" if r.headers.get("x-requests-remaining") else ""))


# ------------------------------------------------------------------ paid-plan schedule (20K credits/month, Oct 27, 2026)
PROP_FULL_HOURS = (9, 16, 21)   # Wed-Sat full-week prop pulls (CT): morning, after the ~3-4 PM injury reports, evening
LINES_FROM, LINES_TO = 6, 23    # game lines every hour in this window (CT), 3 credits each
NEAR_KICKOFF_H = 3              # every hour: props for games kicking off within this many hours
CREDIT_FLOOR = 1500             # below this many credits left, keep only the near-kickoff pulls


def credits_left() -> int | None:
    """Credits remaining, from a free call (listing events costs nothing)."""
    r = requests.get(f"{BASE}/events", params={"apiKey": api_key()}, timeout=30)
    r.raise_for_status()
    v = r.headers.get("x-requests-remaining")
    return int(float(v)) if v is not None else None


def fetch_plan(now: datetime | None = None) -> None:
    """Runs every hour from GitHub Actions and decides what to pull (expected ~7,500 credits/month):
      game lines (all books + Pinnacle) every hour 6 AM-11 PM CT                    ~340/week
      props for every game this week, Wed-Sat at 9 AM, 4 PM and 9 PM CT             ~1,000/week
      props for the whole day's games, Sunday 7 AM CT                               ~85/week
      props for games kicking off within 3 hours, every hour (inactives, late moves) ~300/week"""
    now = now or datetime.now(CT)
    wd, hr = now.weekday(), now.hour          # Mon = 0 ... Sun = 6
    left = credits_left()
    print(f"{_stamp(now)}: {left} credits left")
    low = left is not None and left < CREDIT_FLOOR
    if LINES_FROM <= hr <= LINES_TO:
        fetch_games()
    if not low and wd in (2, 3, 4, 5) and hr in PROP_FULL_HOURS:
        fetch()
    elif not low and wd == 6 and hr == 7:
        fetch(hours=14)
    elif LINES_FROM <= hr <= LINES_TO:
        fetch(hours=NEAR_KICKOFF_H)
