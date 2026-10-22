"""NFL betting model.

  python run.py picks                      # this week's board (auto-detects the next unplayed week)
  python run.py picks --season 2026 --week 7 --overrides overrides.csv --props props.csv
  python run.py backtest --start 2012      # walk-forward test against closing lines
  python run.py ratings                    # current power ratings
  python run.py report --props props.csv   # everything above as one readable HTML page
  python run.py build-epa                  # rebuild the play-by-play EPA file from scratch
  add --refresh to any command to re-download the latest data from nflverse
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nflmodel import config as C
from nflmodel import epa as epa_mod
from nflmodel.betting import fair_american, game_options, grade
from nflmodel.data import apply_overrides, load_games
from nflmodel.model import fit_season_model, key_number_multipliers
from nflmodel.props import price_props
from nflmodel.ratings import attach_epa, fit_play_ratings, fit_ratings, walk_forward

OUT = Path(__file__).resolve().parent / "output"
WF_CACHE = OUT / "walk_forward.pkl"


def get_walk_forward(games: pd.DataFrame, refresh: bool) -> pd.DataFrame:
    """Rating predictions for every game since 2006 using only earlier data (cached)."""
    OUT.mkdir(exist_ok=True)
    if refresh:
        epa_mod.refresh(int(games[games.played].season.max()))
    epa = epa_mod.load()
    if WF_CACHE.exists() and not refresh:
        wf, n_epa = pd.read_pickle(WF_CACHE)
        if len(wf) == len(games) and wf.played.sum() == games.played.sum() and n_epa == len(epa):
            # carry over any freshly applied overrides / features
            for c in games.columns:
                wf[c] = games[c].values
            return wf
    print("Fitting walk-forward ratings (one-time, ~1-2 min)...")
    wf = walk_forward(games, start_season=C.STAGE2_MIN_SEASON - 2, epa=epa)
    pd.to_pickle((wf, len(epa)), WF_CACHE)
    return wf


def current_play_ratings(games: pd.DataFrame, cutoff, season: int) -> dict:
    tg = attach_epa(epa_mod.load(), games)
    return {m: fit_play_ratings(tg, m, cutoff, season) for m in ("epa", "sr")}


def next_week(games: pd.DataFrame) -> tuple[int, int]:
    up = games[~games.played].sort_values("date")
    r = up.iloc[0]
    return int(r.season), int(r.week)


def _prepare(games: pd.DataFrame, a) -> pd.DataFrame:
    """Forecast weather, DraftKings' own game lines (The Odds API), then manual overrides on top."""
    from nflmodel import market, weather
    games = weather.apply(games, getattr(a, "weather", None) or "weather.csv")
    games, used = market.apply_dk_game_lines(games, getattr(a, "game_lines", None) or "game_lines.csv")
    a._dk_used = used
    return apply_overrides(games, a.overrides)


# ------------------------------------------------------------------ picks
def cmd_picks(a):
    games = load_games(refresh=a.refresh)
    season, week = (a.season, a.week) if a.week else next_week(games)
    games = _prepare(games, a)
    wf = get_walk_forward(games, a.refresh)
    sm = fit_season_model(wf, season, key_number_multipliers(games[games.season < season]))
    wk = wf[(wf.season == season) & (wf.week == week)]
    if wk.empty:
        raise SystemExit(f"No games found for {season} week {week}")
    from nflmodel import injuries
    wk, _ = injuries.attach(wk, season, week, a.refresh)
    proj = sm.predict(wk)

    print(f"\n{season} Week {week}  |  model weight vs market: spread {sm.w_margin:.0%}, total {sm.w_total:.0%}"
          f"  |  margin SD {sm.sd_margin:.1f}, total SD {sm.sd_total:.1f}\n")
    board, bets = [], []
    for _, g in proj.iterrows():
        opts = game_options(g, sm)
        ph = sm.ml_probs(g.proj_margin)["home"]
        board.append(dict(
            game=f"{g.away_team} @ {g.home_team}", kickoff=f"{g.gameday} {g.gametime}",
            away_qb=g.away_qb_name, home_qb=g.home_qb_name,
            mkt_spread=f"{g.home_team} {-g.spread_line:+.1f}" if pd.notna(g.spread_line) else "",
            model_spread=f"{g.home_team} {-g.model_margin:+.1f}",
            proj_spread=f"{g.home_team} {-g.proj_margin:+.1f}",
            mkt_total=g.total_line, model_total=round(g.model_total, 1), proj_total=round(g.proj_total, 1),
            proj_score=f"{g.away_team} {g.proj_away_pts:.1f} - {g.home_team} {g.proj_home_pts:.1f}",
            home_win=round(ph[0] + ph[1] / 2, 3), fair_home_ml=round(fair_american(ph[0] + ph[1] / 2)),
        ))
        for o in opts:
            if o["ev"] >= C.MIN_EV:
                bets.append(dict(game=f"{g.away_team} @ {g.home_team}", market=o["market"], pick=o["pick"],
                                 odds=int(o["odds"]), win_prob=round(o["win_prob"], 3),
                                 ev=round(o["ev"], 4), stake=round(o["stake"], 2)))
    board = pd.DataFrame(board)
    bets = pd.DataFrame(bets).sort_values("ev", ascending=False) if bets else pd.DataFrame()
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(board.drop(columns=["kickoff"]).to_string(index=False))
        print(f"\nBets at or above {C.MIN_EV:.0%} EV (stake = {C.KELLY_FRACTION:.0%} Kelly on ${C.BANKROLL:,.0f}):")
        print(bets.to_string(index=False) if len(bets) else "  none this week")
    OUT.mkdir(exist_ok=True)
    board.to_csv(OUT / f"board_{season}_wk{week}.csv", index=False)
    bets.to_csv(OUT / f"bets_{season}_wk{week}.csv", index=False)

    if a.props:
        played = games[games.played & (games.season == season)]
        ppg = {}
        for t in set(played.home_team) | set(played.away_team):
            pts = pd.concat([played.loc[played.home_team == t, "home_score"], played.loc[played.away_team == t, "away_score"]])
            ppg[t] = pts.mean()
        pr = price_props(pd.read_csv(a.props, comment="#"), proj, ppg)
        print("\nProps:")
        print(pr.to_string(index=False))
        pr.to_csv(OUT / f"props_{season}_wk{week}.csv", index=False)
    print(f"\nSaved to {OUT}/")


# ------------------------------------------------------------------ backtest
def summarize(df: pd.DataFrame) -> pd.Series:
    dec = df[df.pnl != 0]
    return pd.Series(dict(bets=len(df), win_pct=round((dec.pnl > 0).mean(), 3) if len(dec) else np.nan,
                          units=round(df.pnl.sum(), 1), roi=round(df.pnl.mean(), 4) if len(df) else np.nan))


def run_backtest(games: pd.DataFrame, wf: pd.DataFrame, start: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, records = [], []
    for season in sorted(wf.season.unique()):
        if season < start:
            continue
        sm = fit_season_model(wf, season, key_number_multipliers(games[games.season < season]))
        sg = wf[(wf.season == season) & wf.played & wf.spread_line.notna()]
        if sg.empty:
            continue
        proj = sm.predict(sg)
        for _, g in proj.iterrows():
            for o in game_options(g, sm):
                records.append(dict(season=season, market=o["market"], ev=o["ev"], pnl=grade(o, g),
                                    win_prob=o["win_prob"], side=o["side"]))
        rows.append(dict(
            season=season, games=len(proj), w_spread=round(sm.w_margin, 2), w_total=round(sm.w_total, 2),
            mae_market=round((proj.result - proj.spread_line).abs().mean(), 2),
            mae_model=round((proj.result - proj.model_margin).abs().mean(), 2),
            mae_blend=round((proj.result - proj.proj_margin).abs().mean(), 2),
            tot_mae_market=round((proj.total - proj.total_line).abs().mean(), 2),
            tot_mae_blend=round((proj.total - proj.proj_total).abs().mean(), 2),
        ))
    rec = pd.DataFrame(records)
    rec["ev_bucket"] = pd.cut(rec.ev, [-1, 0, 0.02, 0.04, 0.07, 1], labels=["<0", "0-2%", "2-4%", "4-7%", "7%+"])
    return pd.DataFrame(rows), rec


def cmd_backtest(a):
    if a.no_epa:
        C.USE_EPA = False
        print("EPA and success rate turned OFF for this run")
    games = load_games(refresh=a.refresh)
    wf = get_walk_forward(games, a.refresh)
    acc, rec = run_backtest(games, wf, a.start)
    bets = rec[rec.ev >= C.MIN_EV]
    print("\nAccuracy by season (mean absolute error, points; lower is better)")
    print(acc.to_string(index=False))
    print(f"\nBets flagged at >= {C.MIN_EV:.0%} EV, graded at closing odds, by market")
    print(bets.groupby("market").apply(summarize, include_groups=False).to_string())
    print("\nBy season (all markets)")
    print(bets.groupby("season").apply(summarize, include_groups=False).to_string())
    print("\nCalibration: realized ROI by predicted EV bucket (spread + total)")
    st = rec[rec.market != "moneyline"]
    print(st.groupby("ev_bucket", observed=True).apply(summarize, include_groups=False).to_string())
    OUT.mkdir(exist_ok=True)
    acc.to_csv(OUT / "backtest_accuracy.csv", index=False)
    bets.to_csv(OUT / "backtest_bets.csv", index=False)
    print(f"\nSaved to {OUT}/")


# ------------------------------------------------------------------ report
def cmd_report(a):
    from nflmodel.report import render
    games = load_games(refresh=a.refresh)
    season, week = (a.season, a.week) if a.week else next_week(games)
    games = _prepare(games, a)
    wf = get_walk_forward(games, a.refresh)
    sm = fit_season_model(wf, season, key_number_multipliers(games[games.season < season]))
    from nflmodel import injuries
    wk, inj = injuries.attach(wf[(wf.season == season) & (wf.week == week)], season, week, a.refresh)
    proj = sm.predict(wk)
    if proj.empty:
        raise SystemExit(f"No games found for {season} week {week}")
    games_out = []
    for _, g in proj.sort_values(["date", "gametime"]).iterrows():
        opts = game_options(g, sm)
        ph = sm.ml_probs(g.proj_margin)["home"]
        games_out.append(dict(g=g, opts=opts, home_win=ph[0] + ph[1] / 2))
    cutoff = games[(games.season == season) & (games.week == week)].date.min()
    ratings = fit_ratings(games, cutoff, season)
    play = current_play_ratings(games, cutoff, season)
    acc, rec = run_backtest(games, wf, a.start)
    props = None
    if a.props:
        played = games[games.played & (games.season == season)]
        ppg = {t: pd.concat([played.loc[played.home_team == t, "home_score"],
                             played.loc[played.away_team == t, "away_score"]]).mean()
               for t in set(played.home_team) | set(played.away_team)}
        props = price_props(pd.read_csv(a.props, comment="#"), proj, ppg)
    from nflmodel import player_proj
    from nflmodel.props import price_auto
    played = games[games.played & (games.season == season)]
    team_ppg = {t: pd.concat([played.loc[played.home_team == t, "home_score"],
                              played.loc[played.away_team == t, "away_score"]]).mean()
                for t in set(played.home_team) | set(played.away_team)}
    auto_props, lines_checked = None, None
    try:
        players = player_proj.project(season, week, proj, refresh=a.refresh, games=games)
        plines = None
        if a.prop_lines and Path(a.prop_lines).exists():
            plines = pd.read_csv(a.prop_lines, comment="#")
            with open(a.prop_lines) as fh:
                first = fh.readline()
            if first.startswith("# checked:"):
                lines_checked = first.split(":", 1)[1].strip()
        tdm = pd.read_csv(a.td_market, comment="#") if a.td_market and Path(a.td_market).exists() else None
        auto_props = price_auto(players, proj, team_ppg, plines, tdm)
        auto_props.to_csv(OUT / f"props_{season}_wk{week}.csv", index=False)
    except Exception as e:  # props are an extra; never block the main report
        print(f"Props skipped: {e}")
    from nflmodel import market, pbp_extra
    _, tm = pbp_extra.load()
    mkt = dict(history=market.record_history(a.lines_history, proj), best_games=market.best_game_prices(a.game_lines),
               best_props=market.best_prop_prices(a.prop_books), dk_used=getattr(a, "_dk_used", {}),
               tendencies=pbp_extra.tendencies(tm, season, week).set_index("team") if tm is not None else None)
    prop_bt = pd.read_csv("prop_backtest.csv") if Path("prop_backtest.csv").exists() else None
    clv_log = None
    if a.bets_log:
        try:
            from nflmodel import clv
            clv_log = clv.update(a.bets_log, season, week, proj, games_out, sm, auto_props, games)
        except Exception as e:
            print(f"Bet log skipped: {e}")
    news, checked = None, None
    if a.news and Path(a.news).exists():
        news = pd.read_csv(a.news, comment="#")
        with open(a.news) as fh:
            first = fh.readline()
        if first.startswith("# checked:"):
            checked = first.split(":", 1)[1].strip()
    html = render(season=season, week=week, sm=sm, games=games_out, ratings=ratings, play=play,
                  acc=acc, rec=rec, props=props, summarize=summarize,
                  data_through=games[games.played].date.max(), news=news, checked=checked,
                  auto_props=auto_props, lines_checked=lines_checked, injuries=inj, clv_log=clv_log,
                  prop_bt=prop_bt, market_info=mkt)
    OUT.mkdir(exist_ok=True)
    path = OUT / f"report_{season}_wk{week}.html"
    path.write_text(html, encoding="utf-8")
    print(f"Wrote {path}  (open it in any browser)")


def cmd_props_backtest(a):
    """Replay past weeks of player projections; save a per-stat summary the report shows."""
    import warnings
    warnings.filterwarnings("ignore")
    from nflmodel import props_backtest as B
    games = load_games(refresh=False)
    s, w = next_week(games)
    rows = B.run(games, B.default_weeks(games, (s, w)))
    sc = B.score(rows)
    base = B.baseline(rows, games)
    sc = sc.merge(base, on="stat", how="left")
    sc.to_csv(a.out, index=False)
    print(sc.round(3).to_string(index=False))


# ------------------------------------------------------------------ news timestamp
def cmd_stamp_news(a):
    """Write the current Central time into news.csv's "# checked:" line (never type it by hand)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    path = Path(a.news)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True) if path.exists() else []
    lines = [l for l in lines if not l.startswith("# checked:")]
    now = datetime.now(ZoneInfo("America/Chicago")).strftime("%a %b %-d, %-I:%M %p CT")
    path.write_text(f"# checked: {now}\n" + "".join(lines), encoding="utf-8")
    print(f"{path.name} stamped {now}")


# ------------------------------------------------------------------ ratings
def cmd_ratings(a):
    games = load_games(refresh=a.refresh)
    season, week = next_week(games)
    cutoff = games[(games.season == season) & (games.week == week)].date.min()
    rt = fit_ratings(games, cutoff, season)
    t = rt.table()
    play = current_play_ratings(games, cutoff, season)
    t["epa_off"] = t.team.map(play["epa"].off)
    t["epa_def"] = t.team.map(play["epa"].deff)
    t["sr_off"] = t.team.map(play["sr"].off) * 100
    t["sr_def"] = t.team.map(play["sr"].deff) * 100
    t = t.round({"off": 2, "def": 2, "net": 2, "epa_off": 3, "epa_def": 3, "sr_off": 1, "sr_def": 1})
    t.index += 1
    print(f"\nRatings entering {season} Week {week} (points vs average; league avg {rt.mu:.1f} ppg, HFA {rt.hfa:.2f})")
    print("epa_* = EPA per play vs average, sr_* = success rate vs average (pct points); defense: + is better")
    print(t.to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pk = sub.add_parser("picks")
    pk.add_argument("--season", type=int)
    pk.add_argument("--week", type=int)
    pk.add_argument("--overrides", default="overrides.csv")
    pk.add_argument("--props", default=None)
    bt = sub.add_parser("backtest")
    bt.add_argument("--start", type=int, default=2012)
    bt.add_argument("--no-epa", action="store_true", help="points-only model, for comparison")
    sub.add_parser("ratings")
    sub.add_parser("build-epa", help="rebuild data/epa_team_games.csv from nflverse play-by-play")
    rp = sub.add_parser("report")
    rp.add_argument("--season", type=int)
    rp.add_argument("--week", type=int)
    rp.add_argument("--overrides", default="overrides.csv")
    rp.add_argument("--props", default=None)
    rp.add_argument("--news", default="news.csv", help="per-game QB/injury/line notes shown on the page")
    rp.add_argument("--td-market", default="td_market.csv", help="market touchdown chances: player,prob,source,as_of")
    rp.add_argument("--bets-log", default="bets_log.csv", help="closing-line tracker log (every flagged bet)")
    rp.add_argument("--prop-lines", default="prop_lines.csv", help="DraftKings prop lines: player,stat,line,over_odds,under_odds,source,as_of")
    rp.add_argument("--start", type=int, default=2012)
    for x in (pk, rp):
        x.add_argument("--weather", default="weather.csv", help="game-time forecasts (fetch-weather)")
        x.add_argument("--game-lines", default="game_lines.csv", help="all books' game lines (fetch-games)")
    rp.add_argument("--prop-books", default="prop_books.csv", help="all books' prop lines (fetch-props)")
    rp.add_argument("--lines-history", default="lines_history.csv", help="market line at every run, for line movement")
    fp = sub.add_parser("fetch-props", help="live player props from The Odds API: DraftKings -> prop_lines.csv, all books -> prop_books.csv")
    fp.add_argument("--prop-lines", default="prop_lines.csv")
    fp.add_argument("--hours", type=float, default=None, help="only games kicking off within this many hours (saves API credits)")
    sub.add_parser("fetch-games", help="spreads/totals/moneylines at every US book -> game_lines.csv (3 credits)")
    sub.add_parser("fetch-weather", help="kickoff forecasts for outdoor games -> weather.csv (free)")
    sub.add_parser("build-extras", help="red-zone usage and team tendencies from play-by-play")
    pb = sub.add_parser("props-backtest", help="replay past weeks of player projections against results")
    pb.add_argument("--out", default="prop_backtest.csv")
    sn = sub.add_parser("stamp-news", help="stamp news.csv (or --news FILE) with the current Central time")
    sn.add_argument("--news", default="news.csv")
    for s in (pk, bt, rp, sub.choices["ratings"], sub.choices["build-epa"]):
        s.add_argument("--refresh", action="store_true", help="re-download data from nflverse")
    a = p.parse_args()
    {"picks": cmd_picks, "backtest": cmd_backtest, "ratings": cmd_ratings, "report": cmd_report,
     "stamp-news": cmd_stamp_news, "props-backtest": cmd_props_backtest,
     "fetch-props": lambda a: __import__("nflmodel.odds_api", fromlist=["fetch"]).fetch(a.prop_lines, a.hours),
     "fetch-games": lambda a: __import__("nflmodel.odds_api", fromlist=["fetch_games"]).fetch_games(),
     "fetch-weather": lambda a: __import__("nflmodel.weather", fromlist=["fetch"]).fetch(),
     "build-extras": lambda a: __import__("nflmodel.pbp_extra", fromlist=["build"]).build(
         [load_games(False).season.max() - 1, load_games(False).season.max()]),
     "build-epa": lambda a: epa_mod.build_from_web()}[a.cmd](a)


if __name__ == "__main__":
    main()
