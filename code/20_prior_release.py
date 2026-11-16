import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
F = re.IGNORECASE
EDITION = re.compile(r"\b(remaster(ed)?|remake|hd|definitive|enhanced|anniversary|complete|deluxe|goty|game of the year|"
                     r"director'?s cut|redux|reloaded|special|ultimate|edition|version|steam|pc|re-?release|reborn|"
                     r"revisited|remastered collection)\b", F)
TITLE = re.compile(r"remaster|remake|definitive|enhanced|anniversary|\bHD\b|redux|director.?s cut|re-?release|reborn|"
                   r"revisited", F)
DESC = re.compile(r"|".join([
    r"\boriginally (released|published|launched|developed|created|made|came out)\b", r"\bfirst (released|published|launched) (in|on)\b",
    r"\bre-?release[sd]?\b", r"\bremaster(ed)?\b", r"\bremake of\b", r"\b(definitive|enhanced|anniversary) edition\b",
    r"\bdirector'?s cut\b", r"\bnow (also |finally )?(available )?on (steam|pc)\b",
    r"\b(comes|came|arrives|returns|is back|coming|brought) (back )?to (steam|pc)\b", r"\bfirst time on (steam|pc)\b",
    r"\bback on (steam|pc)\b", r"\bported from\b", r"\b(pc )?port of\b", r"\bfrom (19|20)\d\d\b",
    r"\b(official )?english (version|release|translation|localization) of\b", r"\btranslated (version|edition) of\b",
    r"\bfaithful (remake|recreation|port|remaster)\b", r"\b(classic|cult|beloved|acclaimed) \w+ (game|title|rpg) from\b",
]), F)


DISC = re.compile(r"\boriginal (game|release|version|edition)\b|\boriginally (released|developed|published|the game)\b|"
                  r"\bremaster|\bre-?release|\b(console|mobile) version\b|\bport(ed)? (of|from)\b", F)
YEAR = re.compile(r"\(\s*(19|20)\d\d\s*\)")


def base(t):
    t = YEAR.sub(" ", re.sub(r"[™®©]", "", (t or "").lower()))
    return re.sub(r"[^a-z0-9]+", " ", EDITION.sub(" ", t)).strip()


def series(t):
    t = re.split(r"\s*[:\-–—]\s+|:", YEAR.sub(" ", re.sub(r"[™®©]", "", (t or "").lower())))[0]
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    return re.sub(r"(\s+(\d+|[ivx]+))+$", "", t).strip()


def main():
    rows = []
    with open(ROOT / "data/raw/store_items.jsonl", encoding="utf-8") as f:
        for l in f:
            d = json.loads(l)
            bi, rel = d.get("basic_info") or {}, d.get("release") or {}
            rows.append({"appid": d.get("appid") or d.get("id"), "name": d.get("name") or "",
                         "ts": rel.get("steam_release_date"), "desc": bi.get("short_description") or "",
                         "makers": {x.get("name", "").lower().strip() for k in ("developers", "publishers")
                                    for x in bi.get(k) or [] if x.get("name")}})
    df = pd.DataFrame(rows).dropna(subset=["appid"])
    df["base"], df["series"] = df.name.map(base), df.name.map(series)
    cohort = set(int(a) for a in (ROOT / "data/processed/cohort_appids.txt").read_text().split())
    disc = pd.read_csv(ROOT / "data/processed/games.csv", usecols=["appid", "disclosure"]).set_index("appid").disclosure

    def earlier(r, key, idx):
        g = idx.get(getattr(r, key))
        return g is not None and bool(r.ts) and any(
            o.appid != r.appid and o.ts and o.ts <= r.ts - 30 * 86400 and r.makers & o.makers for o in g.itertuples())

    by_base = {b: g for b, g in df[df.base.str.len() >= 3].groupby("base")}
    by_series = {b: g for b, g in df[df.series.str.len() >= 4].groupby("series")}
    out = []
    for r in df[df.appid.isin(cohort)].itertuples():
        d = disc.get(int(r.appid))
        out.append({"appid": int(r.appid), "prior_title": int(earlier(r, "base", by_base)),
                    "prior_text": int(bool(TITLE.search(r.name) or DESC.search(r.desc)
                                           or (isinstance(d, str) and DISC.search(d)))),
                    "series": int(earlier(r, "series", by_series))})
    out = pd.DataFrame(out)
    out["prior_release"] = (out.prior_title | out.prior_text).astype(int)
    out.to_csv(ROOT / "data/processed/prior_release.csv", index=False)
    print(len(out), "cohort games;", out[["prior_title", "prior_text", "prior_release", "series"]].sum().to_dict())


if __name__ == "__main__":
    main()
