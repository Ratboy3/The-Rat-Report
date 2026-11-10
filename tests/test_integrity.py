"""Regression tests for the stale-data bugs found Nov 4, 2026. Run on every website rebuild:
    python tests/test_integrity.py
Each test builds a small fake case of a bug that really happened and checks the fix still catches it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

from nflmodel import props as P
from nflmodel import health as H

FAILS = []


def check(name, ok):
    print(("ok   " if ok else "FAIL ") + name)
    if not ok:
        FAILS.append(name)


# 1. last week's DraftKings line must not price this week's game (Emanuel Wilson, Week 9)
week_proj = pd.DataFrame([dict(away_team="ARI", home_team="SEA", gameday="2026-11-08")])
# a Sunday-only week must keep Tuesday-Saturday prices (week starts the Tuesday before)
lines = pd.DataFrame([
    dict(player="Emanuel Wilson", stat="rush_yds", line=19.5, over_odds=-110, under_odds=-114,
         source="DraftKings (The Odds API)", as_of="Mon Nov 2, 5:52 PM CT"),          # Week 8 price, not listed now
    dict(player="Kenneth Walker III", stat="rush_yds", line=60.5, over_odds=-110, under_odds=-110,
         source="DraftKings (The Odds API)", as_of="Wed Nov 4, 9:11 AM CT"),
    dict(player="Old Article", stat="rec_yds", line=30.5, over_odds=-110, under_odds=-110,
         source="SI (DraftKings odds)", as_of="Fri Oct 30, 9:00 AM CT")])
books = pd.DataFrame([dict(game="ARI @ SEA", player="Kenneth Walker III", stat="rush_yds", book="draftkings", line=60.5,
                           over_odds=-110, under_odds=-110, as_of="Wed Nov 4, 9:11 AM CT")])
kept = P._this_week_lines(lines, week_proj, books)
check("last week's DraftKings line is dropped", "Emanuel Wilson" not in set(kept.player))
check("this week's DraftKings line is kept", "Kenneth Walker III" in set(kept.player))
check("article price from before this week is dropped", "Old Article" not in set(kept.player))

# 2. dates without a year: Dec -> Jan rollover
t = P._parse_asof("Wed Dec 30, 9:00 AM CT", 2026)
check("as_of parses", t is not None and t.month == 12 and t.day == 30)

# 3. rosters: each team's own latest week (bye teams kept), last week's game-day INA not out, one team per player
from nflmodel.player_proj import current_roster
ro = pd.DataFrame([
    dict(week=8, team="KC", gsis_id="d", full_name="Mansoor Delane", status="INA"),     # inactive for Week 8 only
    dict(week=8, team="KC", gsis_id="s", full_name="L'Jarius Sneed", status="RES"),     # injured reserve: stays out
    dict(week=7, team="HOU", gsis_id="h", full_name="Bye Team Player", status="ACT"),   # HOU on bye in Week 8
    dict(week=6, team="HOU", gsis_id="w", full_name="Jared Wayne", status="ACT"),       # moved to MIA in Week 7
    dict(week=8, team="MIA", gsis_id="w", full_name="Jared Wayne", status="ACT")])
cr = current_roster(ro, 9)
st = dict(zip(cr.gsis_id, cr.status)); tm = dict(zip(cr.gsis_id, cr.team))
check("last week's game-day inactive counts as active", st.get("d") == "ACT")
check("injured reserve still counts as out", st.get("s") == "RES")
check("team on bye last week keeps its roster", st.get("h") == "ACT")
check("player who changed teams is only on his new team", tm.get("w") == "MIA")

# 3b. playoff rematch can't take the regular-season score
from nflmodel.scores import fill_finals
df = pd.DataFrame([dict(away_team="BAL", home_team="CIN", gameday="2023-01-08", result=None, home_score=None,
                        away_score=None, total=None),
                   dict(away_team="BAL", home_team="CIN", gameday="2023-01-15", result=None, home_score=None,
                        away_score=None, total=None)])
f = fill_finals(df, {("BAL", "CIN", "2023-01-08"): (16, 27)})
check("final score goes to the right date's game only", f.result.notna().tolist() == [True, False])

# 3c. year rollover
check("'Mon Dec 28' read in January is last December", P._parse_asof("Mon Dec 28, 1:00 PM CT", pd.Timestamp("2027-01-12")).year == 2026)

# 4. the integrity check catches a bet priced from a line no book lists for that game, and a held player
props = pd.DataFrame([
    dict(game="ARI @ SEA", team="SEA", player="Emanuel Wilson", stat="rush_yds", line=19.5, best_side="Over",
         over_odds=-110, under_odds=-114, source="DraftKings (The Odds API)", as_of="Mon Nov 2, 5:52 PM CT",
         book="DraftKings", dk_line=19.5, flag=True),
    dict(game="ARI @ SEA", team="SEA", player="Kenneth Walker III", stat="rush_yds", line=60.5, best_side="Over",
         over_odds=-110, under_odds=-110, source="DraftKings (The Odds API)", as_of="Wed Nov 4, 9:11 AM CT",
         book="DraftKings", dk_line=60.5, flag=True)])
tmp = Path("/tmp/_rr_test_books.csv")
books.to_csv(tmp, index=False)


class G:  # a game row
    away_team, home_team, gameday, gametime = "ARI", "SEA", "2026-11-08", "16:25"


v = H.integrity(2026, 9, [dict(g=G(), opts=[])], props, {}, prop_books=str(tmp), bets_log="/nonexistent")
check("integrity check flags the stale Wilson price", any("Emanuel Wilson" in x["bet"] for x in v))
check("integrity check passes the current price", not any("Kenneth Walker" in x["bet"] for x in v))
check("flagged stale prop is turned into a lean", not bool(props.loc[0, "flag"]) and bool(props.loc[1, "flag"]))

# 5. FanDuel-only price (DraftKings pulled the prop) can't be a bet
props2 = pd.DataFrame([dict(game="ARI @ SEA", team="SEA", player="Kenneth Walker III", stat="rush_yds", line=60.5,
                            best_side="Over", over_odds=-110, under_odds=-110, source="FanDuel (The Odds API)",
                            as_of="Wed Nov 4, 9:11 AM CT", book="FanDuel", dk_line=float("nan"), flag=True)])
v2 = H.integrity(2026, 9, [dict(g=G(), opts=[])], props2, {}, prop_books=str(tmp), bets_log="/nonexistent")
check("FanDuel-only prop is held", not bool(props2.loc[0, "flag"]))

# 6. a game the books took down (or that came back empty) loses its old prices; games not asked for keep theirs
src = (Path(__file__).resolve().parent.parent / "nflmodel" / "odds_api.py").read_text()
check("prop pull drops old prices for every game it asked for", "fetched = {_game(e) for e in events}" in src)
check("game-line pull keeps old lines only for games that have started", "old = old[ko.notna() & (ko <= ref)" in src)

# 7. snap-based checks use the stats' last game; grading from snaps alone needs that game's stats posted
pp = (Path(__file__).resolve().parent.parent / "nflmodel" / "player_proj.py").read_text()
check("snap checks follow the stats' last game", "team_last = stats[stats.season == season]" in pp)
cl = (Path(__file__).resolve().parent.parent / "nflmodel" / "clv.py").read_text()
check("no grading from snap counts before the game's stats post", "don't grade from snaps" in cl)

check("inactive voids match the player's team, not just his name", "for t in str(row.game).split(\" @ \")" in cl)

# 8. same-name players on different teams never share a price, a grade or a status (Nov 9, 2026: CLE LB and MIN WR
#    Justin Jefferson)
pr = (Path(__file__).resolve().parent.parent / "nflmodel" / "props.py").read_text()
check("prop prices are matched by game + player + stat", "book[(r.game, _key(r.player)" in pr and "alt[(r.game, _key(r.player)" in pr
      and "r_ = bk_get(src, p)" in pr and "cons[(gm, k_, st_)]" in pr and "sharp.setdefault((r.game," in pr)
check("TD market prices skip names shared by two players this week", "_key(r.player) not in dup" in pr)
check("grading matches the player on one of the game's teams", "stats[tcol].isin(gteams)" in cl and "sw.team.isin(teams)" in cl)
props3 = pd.DataFrame([dict(game="ARI @ SEA", team="CLE", player="Justin Jefferson", stat="rec_yds", line=60.5,
                            best_side="Under", over_odds=-110, under_odds=-110, source="DraftKings (The Odds API)",
                            as_of="Wed Nov 4, 9:11 AM CT", book="DraftKings", dk_line=60.5, flag=True)])
H.integrity(2026, 9, [dict(g=G(), opts=[])], props3, {}, prop_books=str(tmp), bets_log="/nonexistent")
check("a bet on a player whose team isn't in that game is held", not bool(props3.loc[0, "flag"]))

print(f"{len(FAILS)} failed" if FAILS else "all passed")
sys.exit(1 if FAILS else 0)
