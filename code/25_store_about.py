import csv
import gzip
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from rule_coder import GR

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/store_about.jsonl"
HTML = ROOT / "data/raw/store_about_html"
OUT = ROOT / "annotation/store_about"
WINDOW_END = "2026-06-30"
AI_H2 = "AI Generated Content Disclosure"
UA = "Mozilla/5.0 (compatible; academic study of game store disclosures)"
COOKIES = {"birthtime": "0", "lastagecheckage": "1-0-1990", "wants_mature_content": "1"}
AI = re.compile(r"\bA\.?I\b|artificial intelligence|\bLLMs?\b|language model|\bGPT|chatbot|generative|neural|"
                r"machine learning|\bmodels?\b", re.I)
FIELDS = ("snippet", "about", "mature", "early_access")


def runtime_games():
    g = pd.read_csv(ROOT / "data/processed/games.csv", low_memory=False)
    g = g[(g.has_page == 1) & (g.release_date >= "2024-01-01") & (g.page_redirect != 1)]
    g = g[(g.release_date <= WINDOW_END) & (g.review_count >= 10) & (g.disclosed == 1) & g.code_src.notna()]
    g = g[(g.c_live_system == 1) | (g.c_live_player == 1)]
    n = json.load(open(ROOT / "results/headline.json"))["n_live_reception"]
    assert len(g) == n, (len(g), n)
    return g


def page_record(appid, html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style"]):
        t.decompose()
    sec = [d for d in soup.select("div.game_area_description")
           if d.find("h2") and AI_H2.lower() in d.find("h2").get_text(strip=True).lower()]
    sec_text = sec[0].get_text(" ", strip=True) if sec else ""
    disc = None
    if sec:
        i = sec[0].find("i")
        disc = i.get_text("\n", strip=True) if i else sec_text
    about = soup.select_one("#game_area_description")
    snippet = soup.select_one(".game_description_snippet")
    mature = [d for d in soup.select("div.game_area_description")
              if d.find("h2") and "mature content" in d.find("h2").get_text(strip=True).lower()]
    early = soup.select_one("#earlyAccessBody")
    summary = {}
    for row in soup.select(".user_reviews_summary_row"):
        label = (row.select_one(".subtitle") or row).get_text(" ", strip=True).lower()
        m = re.search(r"(\d+)% of the ([\d,]+) user reviews", row.get("data-tooltip-html", ""))
        for key in ("all", "english"):
            if m and label.startswith(key) and key not in summary:
                summary[key] = (int(m.group(1)), int(m.group(2).replace(",", "")))
    sub = [h for h in about.find_all(["h1", "h2", "h3", "h4"]) if AI.search(h.get_text(" ", strip=True))] if about else []
    other = [h for h in soup.find_all(["h1", "h2", "h3", "h4"])
             if not h.find_parent(id="game_area_description") and not h.find_parent(id="game_area_purchase")
             and AI.search(h.get_text(" ", strip=True)) and AI_H2.lower() not in h.get_text(" ", strip=True).lower()]
    visible = soup.get_text(" ", strip=True)
    n_page = len(re.findall(r"guard ?rails?", visible, re.I))
    n_sec = len(re.findall(r"guard ?rails?", sec_text, re.I))
    return {"appid": int(appid), "is_app_page": int(soup.select_one("div.apphub_AppName") is not None),
            "ai_headings": sum(AI_H2.lower() in h.get_text(" ", strip=True).lower() for h in soup.find_all("h2")),
            "ai_headings_other": len(other),
            "guardrail_term_outside_disclosure": n_page - n_sec, "disclosure": disc,
            "about": about.get_text("\n", strip=True) if about else "",
            "snippet": snippet.get_text(" ", strip=True) if snippet else "",
            "mature": mature[0].get_text("\n", strip=True) if mature else "",
            "early_access": early.get_text("\n", strip=True) if early else "",
            "ai_subheadings_about": len(sub),
            "page_all_pct": summary.get("all", (None, None))[0], "page_all_reviews": summary.get("all", (None, None))[1],
            "page_en_pct": summary.get("english", (None, None))[0], "page_en_reviews": summary.get("english", (None, None))[1]}


def fetch():
    import requests
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
    s.cookies.update(COOKIES)
    HTML.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["appid"] for l in open(RAW, encoding="utf-8")} if RAW.exists() else set()
    with open(RAW, "a", encoding="utf-8") as f:
        for appid in runtime_games().appid:
            if int(appid) in done:
                continue
            r = s.get(f"https://store.steampowered.com/app/{int(appid)}/", params={"cc": "us", "l": "english"}, timeout=30)
            (HTML / f"{int(appid)}.html.gz").write_bytes(gzip.compress(r.text.encode("utf-8")))
            rec = {**page_record(appid, r.text), "status": r.status_code,
                   "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            time.sleep(1.5)


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]


def candidates():
    OUT.mkdir(parents=True, exist_ok=True)
    crawl = runtime_games().set_index("appid").disclosure
    norm = lambda s: " ".join(str(s or "").split()).lower()
    rows, pages = [], []
    for rec in map(json.loads, open(RAW, encoding="utf-8")):
        html = HTML / f"{rec['appid']}.html.gz"
        rec.update(page_record(rec["appid"], gzip.decompress(html.read_bytes()).decode("utf-8")))
        pages.append({"appid": rec["appid"], "fetched_at": rec["fetched_at"][:10], "is_app_page": rec["is_app_page"],
                      "ai_headings": rec["ai_headings"], "ai_headings_other": rec["ai_headings_other"],
                      "ai_subheadings_about": rec["ai_subheadings_about"],
                      "has_early_access": int(bool(rec["early_access"])),
                      "guardrail_term_outside_disclosure": rec["guardrail_term_outside_disclosure"],
                      "disclosure_changed": int(norm(rec["disclosure"]) != norm(crawl[rec["appid"]])),
                      **{k: rec[k] for k in ("page_all_reviews", "page_all_pct", "page_en_reviews", "page_en_pct")}})
        for field in FIELDS:
            ss = sentences(rec[field])
            seen = set()
            for k, s in enumerate(ss):
                hit = [c for c, p in GR.items() if p.search(s)]
                by = ";".join(x for x, y in (("cue", hit), ("ai_term", AI.search(s))) if y)
                if by and s not in seen:
                    seen.add(s)
                    rows.append({"appid": rec["appid"], "field": field, "selected_by": by, "cue_codes": ";".join(hit),
                                 "previous": ss[k - 1][:300] if k else "", "sentence": s[:500]})
    with open(OUT / "candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["appid", "field", "selected_by", "cue_codes", "previous", "sentence"])
        w.writeheader()
        w.writerows(rows)
    pd.DataFrame(pages).to_csv(OUT / "pages.csv", index=False)
    cue = [r for r in rows if "cue" in r["selected_by"]]
    print(len(rows), "candidate sentences in", len({r["appid"] for r in rows}), "games;", len(cue), "match a cue")


if __name__ == "__main__":
    fetch() if "--fetch" in sys.argv else candidates()
