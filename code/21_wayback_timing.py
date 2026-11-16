import gzip
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TGT, OUT = ROOT / "data/raw/wayback_targets.csv", ROOT / "data/raw/wayback_timing.jsonl"
UA = "Mozilla/5.0 (compatible; academic study of game store disclosures)"
AI_H2 = "AI Generated Content Disclosure"
LEAD = "The developers describe how their game uses AI Generated Content like this:"
N_RANDOM, WINDOW = 150, 30


def targets():
    g = pd.read_csv(ROOT / "data/processed/games.csv", low_memory=False)
    ra = pd.read_csv(ROOT / "data/processed/review_ai.csv")
    g = g[(g.has_page == 1) & (g.release_date >= "2024-01-01") & (g.release_date <= "2026-06-30") & (g.review_count >= 10)]
    g["dev_key"] = g.developer.fillna("unknown_" + g.appid.astype(str)).str.lower().str.strip()
    t = g.merge(ra, on="appid")
    t = t[t.n_reviews_text >= 5]
    mg = t[t.groupby("dev_key").appid.transform("size") >= 2]
    sw = mg.groupby("dev_key").disclosed.agg(["min", "max"])
    sw = sw[(sw["min"] == 0) & (sw["max"] == 1)].index
    a = mg[mg.dev_key.isin(sw) & (mg.disclosed == 1)].assign(group="fe")
    rest = t[(t.disclosed == 1) & ~t.appid.isin(a.appid)].sample(N_RANDOM, random_state=20261031).assign(group="random")
    pd.concat([a, rest])[["appid", "release_date", "group"]].to_csv(TGT, index=False)
    print(len(a), "fe targets,", len(rest), "random")


def get(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=60) as r:
                b = r.read()
                if r.headers.get("Content-Encoding") == "gzip" or b[:2] == b"\x1f\x8b":
                    b = gzip.decompress(b)
                time.sleep(1.0)
                return b.decode("utf-8", "replace")
        except Exception as e:
            if getattr(e, "code", None) == 404:
                return None
            time.sleep(5 * (k + 1))
    return None


def snapshots(appid):
    q = urllib.parse.urlencode({"url": f"store.steampowered.com/app/{appid}*", "output": "json",
                                "fl": "timestamp,original", "filter": "statuscode:200", "collapse": "timestamp:8"})
    txt = get("https://web.archive.org/cdx/search/cdx?" + q)
    if not txt:
        return []
    rows = [r for r in json.loads(txt)[1:] if f"/app/{appid}/" in r[1] or r[1].rstrip("/").endswith(f"/app/{appid}")]
    return sorted((datetime.strptime(ts[:8], "%Y%m%d"), ts, u) for ts, u in rows)


def read(ts, url):
    h = get(f"https://web.archive.org/web/{ts}id_/{url}")
    if not h or "apphub_AppName" not in h or 'lang="en"' not in h[:3000]:
        return None
    return AI_H2 in h


def first_valid(snaps, tries=3):
    for d, ts, u in snaps[:tries]:
        v = read(ts, u)
        if v is not None:
            return d, v
    return None, None


def classify(appid, release):
    rel = datetime.strptime(release, "%Y-%m-%d")
    cut = rel + timedelta(days=WINDOW)
    snaps = snapshots(appid)
    row = {"appid": appid, "n_snapshots": len(snaps)}
    d, v = first_valid([s for s in snaps if s[0] <= cut][::-1])
    if d is not None and v:
        return {**row, "status": "early", "date": d.strftime("%Y-%m-%d")}
    if d is not None and d >= rel:
        return {**row, "status": "late", "date": d.strftime("%Y-%m-%d")}
    after = [s for s in snaps if s[0] > max(rel, d or rel)]
    d2, v2 = first_valid(after)
    if d2 is None:
        return {**row, "status": "unknown" if d is None else "uncertain", "date": None}
    if not v2:
        return {**row, "status": "late", "date": d2.strftime("%Y-%m-%d")}
    return {**row, "status": "early" if d2 <= cut else ("uncertain" if d is not None else "unknown"),
            "date": d2.strftime("%Y-%m-%d")}


def display(n_games=40, per_period=10):
    g = pd.read_csv(ROOT / "data/processed/games.csv", low_memory=False)
    lv = g[(g.disclosed == 1) & ((g.c_live_system == 1) | (g.c_live_player == 1))]
    rows = []
    for appid in lv.sort_values("review_count", ascending=False).appid.head(n_games):
        snaps = snapshots(int(appid))
        for pre in (True, False):
            for d, ts, u in [s for s in snaps if (s[0].year < 2026) == pre and s[0].year >= 2024][::-1][:2]:
                h = get(f"https://web.archive.org/web/{ts}id_/{u}")
                if not h or AI_H2 not in h or 'lang="en"' not in h[:3000]:
                    continue
                k = h.index(AI_H2)
                e = h.find("<h2", k + len(AI_H2))
                sec = h[k:e if e > 0 else k + 6000]
                txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", sec))
                rows.append({"appid": int(appid), "date": d.strftime("%Y-%m-%d"), "lead_in": int(LEAD in txt),
                             "headings": len(re.findall(r"<h[1-6]", sec)), "guardrail_label": int("guardrail" in txt.lower())})
                break
        if sum(r["date"] < "2026" for r in rows) >= per_period and sum(r["date"] >= "2026" for r in rows) >= per_period:
            break
    pd.DataFrame(rows).to_csv(ROOT / "data/raw/wayback_display.csv", index=False)
    print(pd.DataFrame(rows).describe().T)


def main():
    if "--targets" in sys.argv:
        return targets()
    if "--display" in sys.argv:
        return display()
    done = {json.loads(l)["appid"] for l in open(OUT)} if OUT.exists() else set()
    tg = pd.read_csv(TGT)
    k, n = map(int, (sys.argv[1] if len(sys.argv) > 1 else "0/1").split("/"))
    with open(OUT, "a", encoding="utf-8") as f:
        for i, r in enumerate(tg.itertuples()):
            if r.appid in done or i % n != k:
                continue
            res = classify(int(r.appid), r.release_date)
            f.write(json.dumps({**res, "group": r.group}) + "\n")
            f.flush()
            print(i, res, flush=True)


if __name__ == "__main__":
    main()
