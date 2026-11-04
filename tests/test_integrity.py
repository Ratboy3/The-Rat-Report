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

# 3. last week's game-day inactive (INA) must not count as out this week before the new roster posts
src = Path(__file__).resolve().parent.parent / "nflmodel" / "injuries.py"
code = src.read_text()
check("injuries.py treats last week's INA as active", 'if rw.week.max() < week:' in code and '"ACT" if s_ == "INA"' in code)
pp = (Path(__file__).resolve().parent.parent / "nflmodel" / "player_proj.py").read_text()
check("player_proj.py keeps last week's INA players in projections", 'ok = ("ACT", "INA") if rw.week.max() < week' in pp)
qb = (Path(__file__).resolve().parent.parent / "nflmodel" / "qb.py").read_text()
check("qb.py ignores last week's INA", 'replace("INA", "ACT")' in qb)

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

print(f"{len(FAILS)} failed" if FAILS else "all passed")
sys.exit(1 if FAILS else 0)
