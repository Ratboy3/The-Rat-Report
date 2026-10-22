"""Turn output/report_<season>_wk<week>.html into the page published as the Gridline artifact.

  python make_page.py 2026 7 /path/to/gridline-week7.html

The artifact host adds its own <html>/<head>/<body> wrapper, so this keeps only the
title, meta, font link and styles from the head, plus the body content.
"""
import sys
from pathlib import Path

season, week, dest = sys.argv[1], sys.argv[2], sys.argv[3]
src = Path(__file__).resolve().parent / "output" / f"report_{season}_wk{week}.html"
s = src.read_text(encoding="utf-8")
head = s[s.index("<title>"):s.index("</head>")]
body = s[s.index("<body>") + len("<body>"):s.rindex("</body>")]
Path(dest).write_text(head + "\n" + body, encoding="utf-8")
print(f"Wrote {dest}")
