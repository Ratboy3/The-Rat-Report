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
    url = URL.format(start=start.strftime("%Y%m%d"), end=end.strftime("%Y%m%d"))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "rat-report-site-bot"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
    except Exception as e:
        print(f"  ESPN scores unavailable ({e})")
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
            out[(side["away"][0], side["home"][0])] = (side["away"][1], side["home"][1])
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
        k = (df.at[i, "away_team"], df.at[i, "home_team"])
        if k in finals:
            a, h = finals[k]
            df.at[i, "away_score"], df.at[i, "home_score"] = a, h
            df.at[i, "result"], df.at[i, "total"] = h - a, h + a
            n += 1
    if n:
        print(f"  ESPN: filled {n} final score(s) nflverse hasn't posted yet")
    return df
