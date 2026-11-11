"""'Why this bet': the reasons behind each flagged bet, in sections (the number, form, usage, matchup, game,
price), built from the same numbers the model used. Nothing here changes a projection; it only explains one.
Built Oct 28, 2026; deeper stats version the same day (game logs, hit rates vs the line, usage, opponent ranks,
team rating ranks, injuries).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import props as P
from .betting import implied
from .model import _design, margin_features, total_features

STAT_WORD = {"pass_yds": "passing yards", "pass_tds": "passing TDs", "rush_yds": "rushing yards",
             "rec_yds": "receiving yards", "receptions": "receptions", "anytime_td": "touchdown"}
STAT_SHORT = {"pass_yds": "pass yds", "pass_tds": "pass TDs", "rush_yds": "rush yds", "rec_yds": "rec yds",
              "receptions": "catches"}
POS_WORD = {"QB": "QBs", "RB": "RBs", "WR": "WRs", "TE": "TEs"}


def _pts(x: float) -> str:
    return f"{abs(x):.1f} pt" + ("" if abs(round(x, 1)) == 1.0 else "s")


def _odds(o) -> str:
    return f"{int(o):+d}" if pd.notna(o) else "—"


def _ord(n: int) -> str:
    n = int(n)
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _rank_word(rank: int, n: int, most: str, fewest: str) -> str:
    """'3rd most of 32' / '2nd fewest of 32'."""
    if rank <= n / 2:
        return f"{_ord(rank)} {most} of {n}"
    return f"{_ord(n - rank + 1)} {fewest} of {n}"


def _line(team: str, margin: float) -> str:
    if abs(margin) < 0.05:
        return "pick'em"
    return f"{team} by {abs(margin):.1f}"


def _num(x, nd=1) -> str:
    return f"{x:.{nd}f}".rstrip("0").rstrip(".") if nd else f"{x:.0f}"


# ------------------------------------------------------------------ team context for game bets
def team_context(ratings, play, injuries=None, qbs=None, qb_status=None) -> dict:
    """team -> ranks and numbers shown in game explanations (built once per page)."""
    ctx = {}
    try:
        t = ratings.table()
        n = len(t)
        t["net_rk"] = t.net.rank(ascending=False, method="min")
        t["off_rk"] = t["off"].rank(ascending=False, method="min")
        t["def_rk"] = t["def"].rank(ascending=False, method="min")
        for r in t.itertuples():
            ctx[r.team] = dict(n=n, net=r.net, net_rk=int(r.net_rk), off=r.off, off_rk=int(r.off_rk),
                               dfn=r[3], def_rk=int(r.def_rk))
    except Exception:
        pass
    try:
        eo, ed = pd.Series(play["epa"].off), pd.Series(play["epa"].deff)
        eo_rk, ed_rk = eo.rank(ascending=False, method="min"), ed.rank(ascending=False, method="min")
        for team in eo.index:
            ctx.setdefault(team, {}).update(epa_off=float(eo[team]), epa_off_rk=int(eo_rk[team]),
                                            epa_def=float(ed.get(team, np.nan)), epa_def_rk=int(ed_rk.get(team, 0)),
                                            epa_n=len(eo))
    except Exception:
        pass
    for team, v in (injuries or {}).items():
        if v and v.get("players"):
            ctx.setdefault(team, {})["inj"] = [(nm, p, x, st) for nm, p, x, st in v["players"] if abs(x) >= 0.1][:3]
    for team, q in (qbs or {}).items():
        if q:
            ctx.setdefault(team, {})["qb"] = q
    for team, q in (qb_status or {}).items():
        if q:
            ctx.setdefault(team, {})["qb_status"] = q
    return ctx


def _team_line(t: str, c: dict) -> str | None:
    if not c or "net_rk" not in c:
        return None
    s = (f"{t} rates {c['net']:+.1f} pts vs an average team ({_ord(c['net_rk'])} of {c['n']}; "
         f"offense {_ord(c['off_rk'])}, defense {_ord(c['def_rk'])})")
    if "epa_off_rk" in c:
        s += (f", EPA/play offense {c['epa_off']:+.2f} ({_ord(c['epa_off_rk'])}), "
              f"defense {_ord(c['epa_def_rk'])}")
    return s


def _inj_line(home, away, ctx) -> str | None:
    bits = []
    for t in (away, home):
        c = ctx.get(t, {})
        q, qs = c.get("qb"), c.get("qb_status")
        if qs and qs["kind"] != "starter":
            bits.append(f"{t}: {qs['text']}" + (f" ({q['pts']:+.1f} pts on the spread vs {q['usual']})" if q else ""))
        elif q:
            bits.append(f"{t} QB {q['new']} in for {q['usual']} ({q['pts']:+.1f} pts on the spread)")
        for nm, p, x, st in c.get("inj", [])[:2]:
            bits.append(f"{t} {nm} {p} {x:+.1f}" + (f" ({st})" if st else ""))
    return "; ".join(bits) if bits else None


# ------------------------------------------------------------------ game bets
def _margin_parts(g: pd.Series, sm) -> dict:
    """Contributions (home-team points) to the pure model margin, grouped for people."""
    d = _design(pd.DataFrame([g]))
    if "home_qb_pts" in d:
        dm = d.copy()
        for side in ("home", "away"):
            dm.loc[d[f"{side}_qb_pts"].notna(), f"{side}_qb_change"] = 0
    else:
        dm = d
    feats = margin_features()
    x = dm[feats].iloc[0].astype(float).fillna(0.0).to_dict()
    b = dict(zip(["home"] + feats, sm.b_margin))
    c = {f: b[f] * x[f] for f in feats}
    inj = (-(g.get("home_inj_off", 0) or 0) - (g.get("home_inj_def", 0) or 0)
           + (g.get("away_inj_off", 0) or 0) + (g.get("away_inj_def", 0) or 0))
    qb = 0.0
    if pd.notna(g.get("home_qb_pts")) or pd.notna(g.get("away_qb_pts")):
        qb = float((g.get("home_qb_pts") or 0) - (g.get("away_qb_pts") or 0))
    qb += c.get("home_qb_change", 0) + c.get("away_qb_change", 0)
    return {
        "strength": c.get("r_margin", 0) + c.get("div_x_margin", 0) + c.get("epa_margin", 0) + c.get("sr_margin", 0),
        "home": 0.0 if g.get("neutral") else b["home"],
        "rest": c.get("home_bye", 0) + c.get("away_bye", 0) + c.get("rest_diff", 0),
        "travel": c.get("away_west_early", 0) + c.get("long_trip", 0),
        "qb": qb,
        "injuries": float(np.nan_to_num(inj)),
        "coach": c.get("agg_diff", 0),
        "manual": float((g.get("home_adj", 0) or 0) - (g.get("away_adj", 0) or 0)),
    }


def _total_parts(g: pd.Series, sm) -> dict:
    d = _design(pd.DataFrame([g]))
    feats = total_features()
    x = d[feats].iloc[0].astype(float).fillna(0.0).to_dict()
    b = dict(zip(["const"] + feats, sm.b_total))
    c = {f: b[f] * x[f] for f in feats}
    inj = (-(g.get("home_inj_off", 0) or 0) - (g.get("away_inj_off", 0) or 0)
           + (g.get("home_inj_def", 0) or 0) + (g.get("away_inj_def", 0) or 0))
    return {
        "weather": c.get("wind_excess", 0) + c.get("cold", 0) + (-2.0 * float(g.get("precip", 0) or 0)),
        "dome": c.get("dome", 0),
        "qb": c.get("qb_changes", 0),
        "injuries": float(np.nan_to_num(inj)),
        "coach": c.get("agg_sum", 0),
    }


def game_why(g: pd.Series, o: dict, sm, ctx: dict | None = None) -> list[tuple[str, str]]:
    """Sections explaining one game bet (spread, total or moneyline)."""
    ctx = ctx or {}
    home, away = g.home_team, g.away_team
    out = []
    need = implied(o["odds"]) if pd.notna(o.get("odds")) else None
    if o["market"] in ("spread", "moneyline"):
        bet_team = home if o["side"] == "home" else away
        sign = 1 if o["side"] == "home" else -1
        fav = lambda m: _line(home if m >= 0 else away, m)
        out.append(("The number", f"Our model alone makes it {fav(g.model_margin)}; the market has {fav(g.spread_line)}. "
                                  f"Blended (mostly market) the projection is {fav(g.proj_margin)}, "
                                  f"{abs(g.proj_margin - g.spread_line):.1f} pts toward {bet_team} vs the line."))
        lines = [x for x in (_team_line(away, ctx.get(away)), _team_line(home, ctx.get(home))) if x]
        if lines:
            out.append(("Matchup", ". ".join(lines) + "."))
        parts = _margin_parts(g, sm)
        words = {
            "strength": lambda v: f"team strength {_pts(v)} to {home if v > 0 else away}",
            "home": lambda v: f"home field {_pts(v)} to {home}",
            "rest": lambda v: f"rest {_pts(v)} to {home if v > 0 else away}"
                              + (f" ({home} {int(g.home_rest)} days, {away} {int(g.away_rest)})" if pd.notna(g.get("home_rest")) else ""),
            "travel": lambda v: f"travel/body clock {_pts(v)} against {away}",
            "qb": lambda v: f"QB situation {_pts(v)} to {home if v > 0 else away}",
            "injuries": lambda v: f"injuries {_pts(v)} to {home if v > 0 else away}",
            "coach": lambda v: f"4th-down aggressiveness {_pts(v)} to {home if v > 0 else away}",
            "manual": lambda v: f"news adjustment {_pts(v)} to {home if v > 0 else away}",
        }
        ranked = sorted(((k, v) for k, v in parts.items() if abs(v) >= 0.2), key=lambda kv: -abs(kv[1]))
        if ranked:
            out.append(("How the model gets there", "; ".join(words[k](v) for k, v in ranked[:5]) + "."))
        inj = _inj_line(home, away, ctx)
        if inj:
            out.append(("Injuries and QBs", inj + "."))
        if o["market"] == "moneyline" and need is not None:
            price = (f"At {_odds(o['odds'])}, {bet_team} needs to win {need:.1%} of the time to break even; "
                     f"we have {o['win_prob']:.1%}.")
            if o["odds"] >= 200:
                price += (" Long-odds underdog: a small difference in win chance makes a big edge, so this is one "
                          "of the moneylines the Tuesday review is watching.")
            out.append(("Price", price))
        elif need is not None:
            line = -g.spread_line if o["side"] == "home" else g.spread_line
            out.append(("Price", f"{bet_team} {line:+g} covers {o['win_prob']:.1%} of the time in our numbers; "
                                 f"{_odds(o['odds'])} needs {need:.1%}."))
    else:
        side = "Over" if o["side"] == "over" else "Under"
        out.append(("The number", f"Our model alone projects {g.model_total:.1f} points; the market total is {g.total_line:g}. "
                                  f"Blended (mostly market) the projection is {g.proj_total:.1f}."))
        lines = []
        for t, opp in ((away, home), (home, away)):
            c, co = ctx.get(t, {}), ctx.get(opp, {})
            if "off_rk" in c and "def_rk" in co:
                lines.append(f"{t} offense {_ord(c['off_rk'])} vs {opp} defense {_ord(co['def_rk'])}")
        if lines:
            out.append(("Matchup", "; ".join(lines) + f" (points ratings, of {ctx[away]['n']})."))
        parts = _total_parts(g, sm)
        bits = []
        if pd.notna(g.get("r_total")):
            bits.append(f"the two teams' ratings add up to {g.r_total:.1f} points")
        if isinstance(g.get("wx"), str):
            bits.append(f"weather {g.wx}" + (f" ({parts['weather']:+.1f})" if abs(parts["weather"]) >= 0.2 else ", no effect"))
        for k, lab in (("injuries", "injuries"), ("qb", "backup QB starting"), ("coach", "4th-down aggressiveness"), ("dome", "dome")):
            if abs(parts[k]) >= 0.2:
                bits.append(f"{lab} {parts[k]:+.1f}")
        if bits:
            out.append(("How the model gets there", "; ".join(bits) + "."))
        inj = _inj_line(home, away, ctx)
        if inj:
            out.append(("Injuries and QBs", inj + "."))
        if need is not None:
            out.append(("Price", f"The {side} hits {o['win_prob']:.1%} of the time in our numbers; {_odds(o['odds'])} needs {need:.1%}."))
    return out


# ------------------------------------------------------------------ props
def _log_vals(s) -> list[tuple[float, str]]:
    if not isinstance(s, str) or not s:
        return []
    out = []
    for part in s.split("|"):
        v, _, o = part.partition(":")
        try:
            out.append((float(v), o))
        except ValueError:
            pass
    return out


def prop_why(r) -> list[tuple[str, str]]:
    """Sections explaining one flagged prop (a row of the props table from props.price_auto)."""
    g = lambda k, d=np.nan: getattr(r, k, d)
    stat, side = r.stat, r.best_side
    book = "FanDuel" if str(g("book", "")) == "FanDuel" else "DraftKings"
    price = r.over_odds if side == "Over" else r.under_odds
    win = r.p_over if side == "Over" else 1 - r.p_over
    out = []
    if stat == "anytime_td":
        tds, gp = g("why_td_scored"), g("why_td_games")
        if pd.notna(gp) and gp > 0:
            s = f"Scored in {int(tds)} of {int(gp)} games this season"
            if pd.notna(g("why_rush_td")):
                s += f" ({int(g('why_rush_td'))} rushing, {int(g('why_rec_td'))} receiving TDs)"
            out.append(("This season", s + "."))
        if pd.notna(g("why_rz_pg")):
            out.append(("Usage", f"{g('why_rz_pg'):.1f} red-zone touches per game, {g('why_gl_pg'):.1f} inside the 10 "
                                  f"(carries + targets)."))
        share, ttd = g("why_share"), g("why_team_tds")
        if pd.notna(share) and pd.notna(ttd):
            tp = g("why_team_pts")
            s = f"{r.team} projects {ttd:.1f} touchdowns" + (f" ({tp:.1f} points)" if pd.notna(tp) else "")
            s += f"; he gets about {share:.0%} of the team's rushing/receiving TDs, so the model alone gives him {g('model_prob'):.0%}."
            out.append(("The number", s))
        mp = g("market_prob")
        if pd.notna(mp):
            s = (f"The market says {mp:.0%} after the book's margin; blended {P.TD_MODEL_WEIGHT:.0%} model / {1 - P.TD_MODEL_WEIGHT:.0%} market = {r.adj_projection:.1%}. "
                 f"At {book} {_odds(price)} the break-even is {implied(price):.1%}.")
            if pd.notna(price) and price >= 600:
                s += " Long-odds touchdown: small chance differences make big edges, so the Tuesday review is watching these."
            out.append(("Price", s))
        cp = g("cons_p_over")
        if pd.notna(cp):
            out.append(("Other books", f"{cp:.0%} to score across the other books "
                                       f"({'above' if cp > implied(price) else 'below'} the {implied(price):.1%} break-even)."))
        return out

    word, short = STAT_WORD.get(stat, stat), STAT_SHORT.get(stat, stat)
    line = float(r.line)
    log = _log_vals(g("why_log"))
    if log:
        # Oct 28 (Alex): no "hit in X of 5" count. Five games is a tiny sample the model doesn't use; show the game
        # log for context and the weighted average the projection actually starts from.
        last = log[:5]
        vals = [v for v, _ in last]
        s = "Last " + (f"{len(last)}: " if len(last) > 1 else "game: ") + ", ".join(f"{_num(v)} vs {o}" for v, o in last)
        s += f" (avg {np.mean(vals):.1f})"
        gp = g("why_gp")
        if pd.notna(gp) and gp > len(last):
            s += f"; season avg {g('why_season_avg'):.1f} over {int(gp)} games"
        if pd.notna(gp) and gp < 4 and pd.notna(g("why_prev_avg")) and g("why_prev_gp", 0) >= 4:
            s += f"; only {int(gp)} game{'s' if gp != 1 else ''} this season, {g('why_prev_avg'):.1f} avg over {int(g('why_prev_gp'))} last season"
        base = g("why_base")
        if pd.notna(base):
            how = ("his average with recent games weighted most (last season counts some), blended with what his "
                   "targets and air yards usually produce, then pulled toward a typical player in his role"
                   if stat in ("rec_yds", "receptions") and r.position in ("WR", "TE", "RB") else
                   "his average with recent games weighted most (last season counts some), pulled toward a typical "
                   "player in his role")
            s += f". The model starts from {base:.1f} {short}: {how}"
        out.append(("Recent games", s + "."))
    use = []
    if stat in ("rec_yds", "receptions") and pd.notna(g("why_tgt_pg")):
        use.append(f"{g('why_tgt_pg'):.1f} targets per game")
        if pd.notna(g("why_tshare")):
            use.append(f"{g('why_tshare'):.0%} target share")
        if stat == "rec_yds" and pd.notna(g("why_ay_pg")):
            use.append(f"{g('why_ay_pg'):.0f} air yards per game")
        if stat == "receptions" and pd.notna(g("why_catch")):
            use.append(f"{g('why_catch'):.0%} catch rate")
    elif stat == "rush_yds" and pd.notna(g("why_car_pg")):
        use.append(f"{g('why_car_pg'):.1f} carries per game")
        if pd.notna(g("why_ypc")):
            use.append(f"{g('why_ypc'):.1f} yards per carry")
    elif stat in ("pass_yds", "pass_tds") and pd.notna(g("why_att_pg")):
        use.append(f"{g('why_att_pg'):.1f} attempts per game")
        if pd.notna(g("why_ypa")):
            use.append(f"{g('why_ypa'):.1f} yards per attempt")
    if use:
        out.append(("Usage", ", ".join(use) + " this season."))
    if pd.notna(g("why_opp_pg")):
        fo = g("why_f_opp")
        grp = "QBs" if r.position == "QB" else f"opposing {POS_WORD.get(r.position, r.position)} combined"
        s = (f"{r.opp} allows {g('why_opp_pg'):.1f} {short} per game to {grp} over its last 8 games "
             f"({_rank_word(int(g('why_opp_rank')), int(g('why_opp_n')), 'most', 'fewest')})")
        if pd.notna(fo) and abs(fo - 1) >= 0.01:
            s += f", worth {fo - 1:+.0%} to the projection"
        out.append(("Matchup", s + "."))
    game = []
    tp, ppg = g("why_team_pts"), g("why_team_ppg")
    sf = g("script_factor")
    if pd.notna(tp) and pd.notna(ppg) and stat != "receptions":   # team total isn't used for receptions (Oct 28 test)
        game.append(f"{r.team} projected for {tp:.1f} points vs its {ppg:.1f} average"
                    + (f" ({sf - 1:+.0%})" if pd.notna(sf) and abs(sf - 1) >= 0.01 else ""))
    fv = g("why_f_vol")
    if pd.notna(fv) and abs(fv - 1) >= 0.01:
        game.append(f"expected plays and pass/run mix {fv - 1:+.0%}")
    fb = g("why_f_bqb")
    if pd.notna(fb) and abs(fb - 1) >= 0.01:
        game.append(f"backup QB starting {fb - 1:+.0%}")
    fn = g("why_f_next")
    if pd.notna(fn) and abs(fn - 1) >= 0.01:
        game.append(f"top teammate at his position out {fn - 1:+.0%}")
    fd_ = g("why_f_depth")
    if pd.notna(fd_) and abs(fd_ - 1) >= 0.01:
        game.append(f"moved down the depth chart in the last week {fd_ - 1:+.0%}")
    if g("why_st_only") == 1.0:
        game.append("played only special teams in his team's last game (no offensive snaps), so this is a lean, "
                    "not a bet: his recent numbers came in a bigger role")
    mv_ = g("mkt_move")
    if pd.notna(mv_) and mv_ <= -0.06:
        game.append(f"heads up: DraftKings has moved {abs(mv_) * 100:.0f} points of win chance against this side since "
                    "it first posted this week; that has often meant news, so check before betting")
    st_ = g("why_stale", "")
    if isinstance(st_, str) and st_:
        game.append(f"{st_}, so this is a lean until the data updates")
    hold_ = g("why_hold", "")
    if isinstance(hold_, str) and hold_:
        game.append(f"role check: {hold_}, so this is a lean, not a bet")
    sd_ = g("why_snap_drop")
    if pd.notna(sd_) and g("why_st_only") != 1.0:
        game.append(f"his snap share last game was {sd_:.0%} of his usual, so an Over is a lean, not a bet")
    if pd.notna(g("why_f_back")):
        game.append("heads up: the regular at his position missed the last game and is back this week, "
                    "so his recent numbers may overstate his role")
    if str(g("injury", "") or ""):
        game.append(f"injury report: {g('injury')}")
    if game:
        out.append(("Game", "; ".join(game) + "."))
    mp, mo = g("market_prob"), g("model_prob")
    if pd.notna(mp) and pd.notna(mo):
        out.append(("The number", f"Model alone {mo:.1f}, the line implies {mp:.1f}; blended (mostly market) {r.adj_projection:.1f}."))
    s = f"{side} {line:g} hits {win:.1%} of the time; {book} {_odds(price)} needs {implied(price):.1%}."
    dkl = g("dk_line")
    if book == "FanDuel" and pd.notna(dkl) and abs(float(dkl) - line) >= 0.5:
        better = (side == "Over" and line < dkl) or (side == "Under" and line > dkl)
        s += f" DraftKings has {dkl:g}" + (f", so FanDuel's {line:g} is the better number for the {side}." if better else ".")
    out.append(("Price", s))
    cp = g("cons_p_over")
    if pd.notna(cp):
        cs = cp if side == "Over" else 1 - cp
        out.append(("Other books", f"{cs:.0%} for the {side} across the other books "
                                   f"({'above' if cs > implied(price) else 'below'} the {implied(price):.1%} break-even)."))
    return out


# ------------------------------------------------------------------ v2 notes (Nov 10, 2026; behind report.WHY_V2)
# Alex: "precise reasons, not generic." Each note says: what the bet is, which of the model's factors moved the
# number and by how much (in yards / TDs, biggest first), what has to happen for it to win, what in the data points
# the other way, and whether it's a model edge or only a price at one book. Lines that would be true of every bet
# (blend weights, how the baseline is built) are left out.
_V_SIDE = {"Over": "over", "Under": "under"}


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _qb_text(r, ctx) -> str | None:
    c = (ctx or {}).get(r.team, {})
    qs, q = c.get("qb_status"), c.get("qb")
    if q and q.get("new") and q.get("usual"):
        return f"{q['new']} starting for {q['usual']}"
    if qs and qs.get("kind") not in (None, "starter"):
        return qs.get("text")
    return None


def _factors(r, ctx) -> list[tuple[str, float, str]]:
    """(key, factor, words) for each multiplier that moved this projection by 1% or more."""
    g = lambda k, d=np.nan: getattr(r, k, d)
    short = STAT_SHORT.get(r.stat, r.stat)
    grp = "QBs" if r.position == "QB" else POS_WORD.get(r.position, r.position)
    out = []
    fo = g("why_f_opp")
    if pd.notna(fo) and pd.notna(g("why_opp_pg")):
        rk = _rank_word(int(g("why_opp_rank")), int(g("why_opp_n")), "most", "fewest")
        out.append(("opp", fo, f"{r.opp} allows {g('why_opp_pg'):.1f} {short} a game to {grp} ({rk})"))
    sf, tp, ppg = g("script_factor"), g("why_team_pts"), g("why_team_ppg")
    if pd.notna(sf) and pd.notna(tp) and pd.notna(ppg):
        out.append(("script", sf, f"{r.team} projected for {tp:.1f} points vs its {ppg:.1f} average"))
    fv = g("why_f_vol")
    if pd.notna(fv):
        kind = "runs" if r.stat == "rush_yds" else "passes"
        out.append(("vol", fv, f"{r.team} expected to call {'more' if fv > 1 else 'fewer'} {kind} than usual"))
    fb = g("why_f_bqb")
    if pd.notna(fb):
        qt = _qb_text(r, ctx)
        out.append(("bqb", fb, f"QB change: {qt}" if qt else "backup QB starting"))
    fn = g("why_f_next")
    if pd.notna(fn):
        out.append(("next", fn, "top teammate at his position is out, more work for him"))
    fd_ = g("why_f_depth")
    if pd.notna(fd_):
        out.append(("depth", fd_, "moved down the depth chart this week"))
    return [x for x in out if abs(x[1] - 1) >= 0.01]


def _hits(log, line, side):
    """(games on our side of the line, games, lowest, highest) from the recent log."""
    vals = [v for v, _ in log[:6]]
    if not vals:
        return None
    k = sum((v > line) if side == "Over" else (v < line) for v in vals)
    return k, len(vals), min(vals), max(vals)


def _dk_price(r, side, win):
    """DraftKings price implied by dk_ev when DraftKings has the same line (else None)."""
    dke, dkl = getattr(r, "dk_ev", np.nan), getattr(r, "dk_line", np.nan)
    if pd.isna(dke) or pd.isna(dkl) or abs(float(dkl) - float(r.line)) > 0.01 or not win:
        return None
    dec = (1 + dke) / win
    if dec <= 1.01:
        return None
    return round((dec - 1) * 100) if dec >= 2 else round(-100 / (dec - 1))


def prop_brief(r, ctx=None) -> list[tuple[str, str]]:
    g = lambda k, d=np.nan: getattr(r, k, d)
    stat, side = r.stat, r.best_side
    book = "FanDuel" if str(g("book", "")) == "FanDuel" else "DraftKings"
    price = r.over_odds if side == "Over" else r.under_odds
    win = r.p_over if side == "Over" else 1 - r.p_over
    need = implied(price)
    mo, mp = g("model_prob"), g("market_prob")
    out, against, watch = [], [], []
    is_td = stat == "anytime_td"
    line = float(r.line)

    # 1. the bet in one line
    if is_td:
        out.append(("The bet", f"{r.player} to score at {book} {_odds(price)}: we have {win:.1%}, the price needs {need:.1%} "
                               f"(edge {r.ev:+.1%}). Model alone {mo:.1%}, market {mp:.1%}."))
    else:
        unit, d0 = STAT_SHORT.get(stat, stat), (2 if stat == "pass_tds" else 1)
        out.append(("The bet", f"{side} {line:g} {STAT_WORD.get(stat, stat)} at {book} {_odds(price)}. Our model has him at "
                               f"{mo:.{d0}f} {unit}, the market {mp:.{d0}f}; blended that's {r.adj_projection:.{d0}f}, which goes "
                               f"{_V_SIDE[side]} {line:g} {win:.1%} of the time vs {need:.1%} needed (edge {r.ev:+.1%})."))

    # 2. what moved the number / 3. what has to be true
    true_ = []
    if is_td:
        share, ttd, tp, ppg = g("why_share"), g("why_team_tds"), g("why_team_pts"), g("why_team_ppg")
        bits = []
        if pd.notna(share) and pd.notna(ttd):
            bits.append(f"{r.team} projects {ttd:.1f} TDs and he gets about {share:.0%} of them")
            if pd.notna(tp) and pd.notna(ppg) and abs(tp / ppg - 1) >= 0.05:
                usual = 1 - np.exp(-ttd * ppg / tp * share)
                bits.append(f"at {r.team}'s usual {ppg:.1f} points he'd be {usual:.0%}; the {tp:.1f} projected moves it to {mo:.0%}")
        tds, gp = g("why_td_scored"), g("why_td_games")
        if pd.notna(gp) and gp > 0:
            how = []
            if pd.notna(g("why_rush_td")) and g("why_rush_td") > 0:
                how.append(f"{int(g('why_rush_td'))} rushing")
            if pd.notna(g("why_rec_td")) and g("why_rec_td") > 0:
                how.append(f"{int(g('why_rec_td'))} receiving")
            bits.append(f"scored in {int(tds)} of {int(gp)} games" + (f" ({', '.join(how)})" if how else ""))
        if pd.notna(g("why_gl_pg")):
            bits.append(f"{g('why_rz_pg'):.2f} red-zone and {g('why_gl_pg'):.2f} inside-the-10 touches a game")
        if bits:
            out.append(("Why the model likes him", "; ".join(bits) + "."))
        if pd.notna(mp) and pd.notna(mo):
            if mo > mp:
                true_.append(f"his scoring role has to be real: the model's {mo:.0%} is above the market's {mp:.0%}, "
                             f"and the blend only gets to {win:.1%} because touchdowns lean mostly on the market")
            else:
                against.append(f"our model alone is lower than the market ({mo:.0%} vs {mp:.0%}), so the edge is only "
                               f"this book's price, not a read on him")
        if pd.notna(g("why_gl_pg")) and g("why_gl_pg") < 0.15:
            against.append(f"he barely gets touches near the goal line ({g('why_gl_pg'):.2f} a game inside the 10)")
        if pd.notna(gp) and gp >= 4 and tds == 0:
            against.append(f"no touchdowns in {int(gp)} games this season")
    else:
        base = g("why_base")
        fs = _factors(r, ctx)
        if pd.notna(base) and pd.notna(mo):
            tot = float(mo) - float(base)
            logs = [np.log(f) for _, f, _ in fs]
            sl = sum(logs)
            contrib = [(k, (tot * lg / sl) if abs(sl) > 1e-9 else 0.0, w) for (k, f, w), lg in zip(fs, logs)]
            contrib = [c for c in contrib if abs(c[1]) >= 0.3 or (stat == "pass_tds" and abs(c[1]) >= 0.02)]
            contrib.sort(key=lambda c: -abs(c[1]))
            nd = 2 if stat == "pass_tds" else 1
            s = f"Starts at {base:.{nd}f} (his recent-weighted average)"
            if contrib:
                s += "; then " + "; ".join(f"{w} ({v:+.{nd}f})" for _, v, w in contrib)
            s += f"; = {mo:.{nd}f}."
            out.append(("How the model gets there", s))
            # what has to be true: the biggest factors pointing our way, as conditions
            sign = -1 if side == "Under" else 1
            fmap = {k: f for k, f, _ in fs}
            mine = [c for c in contrib if np.sign(c[1]) == sign and abs(fmap[c[0]] - 1) >= 0.05][:2]
            tp, ppg = g("why_team_pts"), g("why_team_ppg")
            for k, v, _ in mine:
                if k == "script":
                    true_.append(f"{r.team} has to score like a {tp:.0f}-point team" + (f" (they average {ppg:.0f})" if side == "Under" else ""))
                elif k == "opp":
                    true_.append(f"{r.opp} has to keep doing what it's done against {('QBs' if r.position == 'QB' else POS_WORD.get(r.position))} "
                                 f"({g('why_opp_pg'):.1f} a game)")
                elif k == "vol":
                    true_.append(f"{r.team} has to {'lean on the run' if (stat == 'rush_yds') == (side == 'Over') else 'throw more and run less'} as projected")
                elif k == "bqb":
                    qt = _qb_text(r, ctx)
                    true_.append(f"the QB change has to hurt him" + (f" ({qt})" if qt else ""))
                elif k == "next":
                    true_.append("he has to soak up the missing teammate's work")
            if (side == "Under" and base < line) or (side == "Over" and base > line):
                true_.append(f"he just has to play to his normal: his baseline {base:.{nd}f} is already {_V_SIDE[side]} {line:g}")
            # model far from the market
            if pd.notna(mp) and mp and abs(mo - mp) / abs(mp) >= 0.25:
                against.append(f"the market's {mp:.{nd}f} is far from our {mo:.{nd}f}; gaps that big are often news we "
                               f"don't have, which is why the blend sits much closer to the market")
        h = _hits(_log_vals(g("why_log")), line, side)
        if h:
            k, n, lo, hi = h
            rng = f" (range {_num(lo)}–{_num(hi)})" if n > 1 else ""
            if k * 2 < n:
                opp_ = "topped" if side == "Under" else "stayed under"
                against.append(f"he's {opp_} {line:g} in {n - k} of his last {n}{rng}")
            elif k * 2 == n:
                true_.append(f"recent form is a coin flip: {_V_SIDE[side]} {line:g} in {k} of his last {n}{rng}")
            else:
                true_.append(f"recent form helps: {_V_SIDE[side]} {line:g} in {k} of his last {n}{rng}")
        l3, sa = g("why_l3"), g("why_season_avg")
        if pd.notna(l3) and pd.notna(sa) and sa > 0 and abs(l3 / sa - 1) >= 0.25 and g("why_gp", 0) >= 5:
            up = l3 > sa
            s = f"trending {'up' if up else 'down'}: {_num(l3)} a game over his last 3 vs {_num(sa)} for the season"
            (true_ if up == (side == "Over") else against).append(s)
        if pd.notna(g("why_gp")) and g("why_gp") < 4:
            against.append(f"only {int(g('why_gp'))} games of data this season")
    if true_:
        out.append(("What has to happen", "; ".join(t[0].upper() + t[1:] if i == 0 else t for i, t in enumerate(true_)) + "."))

    # 4. against it, from other books
    cp = g("cons_p_over")
    if pd.notna(cp):
        cs = cp if side == "Over" else 1 - cp
        if cs < need:
            nb = g("n_books")
            against.append(f"the other books{f' ({int(nb)})' if pd.notna(nb) else ''} put the {side.lower() if not is_td else 'TD'} at "
                           f"{cs:.1%}, under the {need:.1%} this price needs")
    sp = g("sharp_side_p")
    if pd.notna(sp) and sp < need:
        against.append(f"the sharpest book has it at {sp:.1%}, also under {need:.1%}")
    mv_ = g("mkt_move")
    if pd.notna(mv_) and mv_ <= -0.03:
        against.append(f"DraftKings has moved {abs(mv_) * 100:.0f} points of win chance against this side this week")
    out.append(("Against it", (against[0][0].upper() + against[0][1:] + ("; " + "; ".join(against[1:]) if len(against) > 1 else "") + ".")
                if against else "Nothing in the data we have points the other way."))

    # 5. news / watch
    if g("why_st_only") == 1.0:
        watch.append("played only special teams last game; lean, not a bet")
    hold_ = g("why_hold", "")
    if isinstance(hold_, str) and hold_:
        watch.append(f"role check: {hold_}; lean, not a bet")
    sd_ = g("why_snap_drop")
    if pd.notna(sd_) and g("why_st_only") != 1.0:
        watch.append(f"snap share last game was {sd_:.0%} of his usual")
    if pd.notna(g("why_f_back")):
        watch.append("the regular at his position is back this week, so recent numbers may overstate his role")
    st_ = g("why_stale", "")
    if isinstance(st_, str) and st_:
        watch.append(st_)
    if str(g("injury", "") or "") not in ("", "nan"):
        watch.append(f"injury report: {g('injury')}")
    if watch:
        out.append(("Watch", _cap("; ".join(watch)) + "."))

    # 6. what kind of bet this is
    dke, dkl = g("dk_ev"), g("dk_line")
    if book == "FanDuel" and pd.notna(dke):
        dkp = _dk_price(r, side, win)
        better = pd.notna(dkl) and ((side == "Over" and line < dkl) or (side == "Under" and line > dkl))
        if dke <= 0:
            at = f" ({_odds(dkp)})" if dkp is not None else (f" at {dkl:g}" if pd.notna(dkl) else "")
            out.append(("Type", f"Price bet. At DraftKings{at} it's {dke:+.1%}, no edge; it only works at FanDuel "
                                f"{_odds(price)}{'' if is_td else f' on {line:g}'}. Skip it at a worse price."))
        elif better:
            out.append(("Type", f"Model edge plus a better number: FanDuel {line:g} vs DraftKings {dkl:g}. Still {dke:+.1%} "
                                f"at DraftKings."))
        else:
            out.append(("Type", f"Model edge: also {dke:+.1%} at DraftKings."))
    elif book == "DraftKings":
        out.append(("Type", "Model edge at DraftKings."))
    return out


def game_brief(g: pd.Series, o: dict, sm, ctx: dict | None = None) -> list[tuple[str, str]]:
    """v2 game note: drops the generic ratings line, adds what has to happen and what points the other way."""
    base = [x for x in game_why(g, o, sm, ctx) if x[0] != "Matchup"]
    home, away = g.home_team, g.away_team
    true_, against = [], []
    if o["market"] in ("spread", "moneyline"):
        team = home if o["side"] == "home" else away
        gap = abs(g.model_margin - g.spread_line)
        if o["market"] == "spread":
            line = -g.spread_line if o["side"] == "home" else g.spread_line
            true_.append(f"{team} {line:+g}: the game has to land closer to our {_line(home if g.proj_margin >= 0 else away, g.proj_margin)} "
                         f"than the market's {_line(home if g.spread_line >= 0 else away, g.spread_line)}")
        else:
            true_.append(f"{team} has to win outright")
        if gap >= 3:
            against.append(f"our model alone is {gap:.1f} pts off the market; gaps that size often mean news the model doesn't have")
    else:
        side = "Over" if o["side"] == "over" else "Under"
        true_.append(f"the teams have to combine for {'more' if side == 'Over' else 'fewer'} than {g.total_line:g}; we project {g.proj_total:.1f}")
        gap = abs(g.model_total - g.total_line)
        if gap >= 4:
            against.append(f"our model alone is {gap:.1f} pts off the market total; gaps that size often mean news the model doesn't have")
    base.insert(min(len(base), 2), ("What has to happen", _cap("; ".join(true_)) + "."))
    base.append(("Against it", (_cap("; ".join(against)) + ".") if against else "Nothing in the data we have points the other way."))
    return base
