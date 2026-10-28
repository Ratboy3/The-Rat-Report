"""'Why this bet': the reasons behind each flagged bet, in sections (the number, form, usage, matchup, game,
price), built from the same numbers the model used. Nothing here changes a projection; it only explains one.
Built Oct 28, 2026; deeper stats version the same day (game logs, hit rates vs the line, usage, opponent ranks,
team rating ranks, injuries).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

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
def team_context(ratings, play, injuries=None, qbs=None) -> dict:
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
        q = c.get("qb")
        if q:
            bits.append(f"{t} QB {q['new']} in for {q['usual']} ({q['pts']:+.1f} pts)")
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
        for k, lab in (("injuries", "injuries"), ("qb", "QB change"), ("coach", "4th-down aggressiveness"), ("dome", "dome")):
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
            s = (f"The market says {mp:.0%} after the book's margin; blended 40% model / 60% market = {r.adj_projection:.1%}. "
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
