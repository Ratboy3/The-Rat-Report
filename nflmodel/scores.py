"""Final scores straight from ESPN's public scoreboard (Nov 3, 2026).

nflverse's games.csv can take hours to post scores, and on Nov 1-3 the claude.ai runs that wrote "Final:" news
notes all stopped (usage limit), so no Week 8 game graded or moved to the bottom until Tuesday. The website
rebuild (GitHub Actions) now fills any missing score for a finished game from ESPN, so finals and grading no
longer depend on a Claude run. Never raises: if ESPN can't be reached, the games keep the nflverse values.
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timedelta, timezone

import pandas as pd

URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={start}-{end}&limit=100"
ESPN_TO_NFLVERSE = {"WSH": "WAS", "LAR": "LA", "JAC": "JAX"}


def espn_finals(days_back: int = 10, timeout: int = 20) -> dict:
    """{(away, home): (away_score, home_score)} for completed NFL games in the last `days_back` days."""
    end = datetime.now(timezone.utc) + timedelta(days=1)
    start = end - timedelta(days=days_back + 1)
    urls = [URL.format(start=start.strftime("%Y%m%d"), end=end.strftime("%Y%m%d")),
            "https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={}-{}&limit=100".format(
                start.strftime("%Y%m%d"), end.strftime("%Y%m%d")),
            "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"]
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
               "Accept": "application/json", "Referer": "https://www.espn.com/nfl/scoreboard"}
    data, err = None, None
    for url in urls:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
                data = json.load(r)
            break
        except Exception as e:
            err = e
    if data is None:
        print(f"  ESPN scores unavailable ({err})")
        return {}
    out = {}
    for ev in data.get("events", []):
        try:
            comp = ev["competitions"][0]
            if not comp.get("status", ev.get("status", {})).get("type", {}).get("completed"):
                continue
            side = {}
            for c in comp["competitors"]:
                ab = c["team"]["abbreviation"]
                side[c["homeAway"]] = (ESPN_TO_NFLVERSE.get(ab, ab), int(float(c["score"])))
            day = str(ev.get("date", ""))[:10]          # ISO kickoff (UTC); matched to the schedule within a day
            out[(side["away"][0], side["home"][0], day)] = (side["away"][1], side["home"][1])
        except Exception:
            continue
    return out


def fill_finals(df: pd.DataFrame, finals: dict | None = None) -> pd.DataFrame:
    """Fill home_score / away_score / result / total for finished games nflverse hasn't scored yet."""
    if "result" not in df:
        return df
    todo = df.result.isna() & (pd.to_datetime(df.gameday) <= pd.Timestamp.now() + pd.Timedelta(days=1))
    if not todo.any():
        return df
    finals = espn_finals() if finals is None else finals
    if not finals:
        return df
    df = df.copy()
    n = 0
    for i in df.index[todo]:
        # match on teams AND date (Nov 4, 2026 audit: matching on teams alone gave a Wild Card rematch the Week 18 score)
        gd = pd.Timestamp(df.at[i, "gameday"])
        hit = [v for (aw, hm, day), v in finals.items() if aw == df.at[i, "away_team"] and hm == df.at[i, "home_team"]
               and day and abs((pd.Timestamp(day) - gd).days) <= 1]
        if hit:
            a, h = hit[0]
            df.at[i, "away_score"], df.at[i, "home_score"] = a, h
            df.at[i, "result"], df.at[i, "total"] = h - a, h + a
            n += 1
    if n:
        print(f"  ESPN: filled {n} final score(s) nflverse hasn't posted yet")
    return df
