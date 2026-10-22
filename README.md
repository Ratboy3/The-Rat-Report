# The Rat Report

Weekly NFL projections and bets from nflverse data. The live report is `index.html`, published with GitHub Pages and rebuilt every hour.

```
python run.py build-epa                      # one time: condense play-by-play into data/
python run.py report --refresh               # this week's report -> output/report_<season>_wk<week>.html
```

`overrides.csv` corrects starting QBs (and DraftKings lines when confirmed) for upcoming games; `news.csv` is the per-game news check shown on the report.
