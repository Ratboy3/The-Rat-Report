# The Rat Report

Weekly NFL projections and bets from nflverse data. The live report is `index.html`, published with GitHub Pages and rebuilt every hour.

```
python run.py build-epa                      # one time: condense play-by-play into data/
python run.py fetch-props                    # live DraftKings player props -> prop_lines.csv
python run.py report --refresh               # this week's report -> output/report_<season>_wk<week>.html
```

`overrides.csv` corrects starting QBs (and DraftKings lines when confirmed) for upcoming games; `news.csv` is the per-game news check shown on the report.

`prop_lines.csv` comes from The Odds API (DraftKings book). Put your key in the `ODDS_API_KEY` environment variable or a git-ignored `.odds_api_key` file. Each refresh costs about 6 credits per game still to be played (about 80-90 for a full Sunday slate).
