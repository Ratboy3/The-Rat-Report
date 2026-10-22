"""Game-time weather forecasts for outdoor stadiums (Open-Meteo, free, no key).

  python run.py fetch-weather        # -> weather.csv: wind (mph), temperature (F), precipitation at kickoff

Runs in GitHub Actions (this session's network can't reach weather services). The model's totals already
learned from history how wind above 10 mph and sub-freezing cold affect scoring; this supplies the forecast.
Domes and retractable roofs are skipped.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
ET, CT = ZoneInfo("America/New_York"), ZoneInfo("America/Chicago")
# home stadium coordinates (open-air stadiums; domes are skipped using the schedule's roof field)
STADIUM = {
    "BAL": (39.278, -76.623), "BUF": (42.774, -78.787), "CAR": (35.226, -80.853), "CHI": (41.862, -87.617),
    "CIN": (39.095, -84.516), "CLE": (41.506, -81.700), "DEN": (39.744, -105.020), "GB": (44.501, -88.062),
    "JAX": (30.324, -81.637), "KC": (39.049, -94.484), "LAC": (33.953, -118.339), "LA": (33.953, -118.339),
    "MIA": (25.958, -80.239), "NE": (42.091, -71.264), "NYG": (40.813, -74.074), "NYJ": (40.813, -74.074),
    "PHI": (39.901, -75.168), "PIT": (40.447, -80.016), "SEA": (47.595, -122.332), "SF": (37.403, -121.970),
    "TB": (27.976, -82.503), "TEN": (36.166, -86.771), "WAS": (38.908, -76.864),
}
COLS = ["game_id", "game", "kickoff", "wind_mph", "temp_f", "precip_mm", "fetched_at"]


def fetch(path: str = "weather.csv", days: int = 8) -> None:
    now = datetime.now(CT)
    g = pd.read_csv(GAMES_URL)
    g = g[g.result.isna() & g.gameday.notna() & g.gametime.notna()].copy()
    g["ko"] = [datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M").replace(tzinfo=ET) for d, t in zip(g.gameday, g.gametime)]
    g = g[(g.ko > now) & (g.ko < now + timedelta(days=days)) & (g.roof == "outdoors") & g.home_team.isin(STADIUM)]
    rows = []
    for r in g.itertuples():
        lat, lon = STADIUM[r.home_team]
        resp = requests.get("https://api.open-meteo.com/v1/forecast", timeout=30, params={
            "latitude": lat, "longitude": lon, "hourly": "wind_speed_10m,temperature_2m,precipitation",
            "wind_speed_unit": "mph", "temperature_unit": "fahrenheit", "timezone": "UTC", "forecast_days": 16})
        resp.raise_for_status()
        h = resp.json()["hourly"]
        times = [datetime.fromisoformat(t).replace(tzinfo=ZoneInfo("UTC")) for t in h["time"]]
        # average over the first ~3 hours of the game
        idx = [i for i, t in enumerate(times) if r.ko <= t.astimezone(ET) + timedelta(minutes=30) < r.ko + timedelta(hours=3)]
        if not idx:
            continue
        mean = lambda k: round(sum(h[k][i] for i in idx) / len(idx), 1)
        rows.append(dict(game_id=r.game_id, game=f"{r.away_team} @ {r.home_team}",
                         kickoff=r.ko.astimezone(CT).strftime("%a %b %-d, %-I:%M %p CT"),
                         wind_mph=mean("wind_speed_10m"), temp_f=mean("temperature_2m"),
                         precip_mm=round(sum(h["precipitation"][i] for i in idx), 1),
                         fetched_at=now.strftime("%a %b %-d, %-I:%M %p CT")))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"# checked: {now.strftime('%a %b %-d, %-I:%M %p CT')}\n")
        pd.DataFrame(rows, columns=COLS).to_csv(fh, index=False)
    print(f"Wrote forecasts for {len(rows)} outdoor games to {path}")


def apply(df: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Put forecast wind/temp/precip into upcoming games (outdoor only), the same way overrides do."""
    path = Path(path)
    if not path.exists():
        return df
    w = pd.read_csv(path, comment="#")
    if w.empty:
        return df
    df = df.copy()
    for r in w.itertuples():
        m = (df.game_id == r.game_id) & (df.dome == 0)
        if not m.any():
            continue
        df.loc[m, "wind"] = r.wind_mph
        df.loc[m, "temp"] = r.temp_f
        df.loc[m, "wind_excess"] = max(0.0, r.wind_mph - 10)
        df.loc[m, "cold"] = int(r.temp_f < 32)
        df.loc[m, "precip"] = int(r.precip_mm >= 1.0)
    return df
