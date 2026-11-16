import html
import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

from steamhttp import SteamSession

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/search_index.jsonl"
DONE = ROOT / "data/raw/search_index.pages"
URL = "https://store.steampowered.com/search/results/"
PAGE = 100


def parse(results_html):
    soup = BeautifulSoup(results_html, "lxml")
    rows = []
    for a in soup.select("a.search_result_row"):
        appid = a.get("data-ds-appid")
        if not appid or "," in appid:
            continue
        rev = a.select_one(".search_review_summary")
        price = a.select_one(".search_price_discount_combined")
        rel = a.select_one(".search_released")
        rows.append({
            "appid": int(appid),
            "title": a.select_one(".title").get_text(strip=True) if a.select_one(".title") else None,
            "released_text": rel.get_text(strip=True) if rel else None,
            "tagids": json.loads(a.get("data-ds-tagids") or "[]"),
            "review_tooltip": html.unescape(rev.get("data-tooltip-html")) if rev else None,
            "price_final_cents": int(price.get("data-price-final")) if price and price.get("data-price-final") else None,
        })
    return rows


def main():
    sess = SteamSession(rps=float(sys.argv[1]) if len(sys.argv) > 1 else 1.5)
    done = set(int(x) for x in DONE.read_text().split()) if DONE.exists() else set()
    first = sess.get(URL, params={"query": "", "start": 0, "count": PAGE, "sort_by": "Released_DESC",
                                  "category1": 998, "infinite": 1, "cc": "us", "l": "english",
                                  "ignore_preferences": 1}).json()
    total = first["total_count"]
    print("total_count", total, flush=True)
    with OUT.open("a", encoding="utf-8") as fo, DONE.open("a") as fd:
        for start in range(0, total + PAGE, PAGE):
            if start in done:
                continue
            r = sess.get(URL, params={"query": "", "start": start, "count": PAGE, "sort_by": "Released_DESC",
                                      "category1": 998, "infinite": 1, "cc": "us", "l": "english",
                                      "ignore_preferences": 1})
            if r is None:
                print("failed", start, flush=True)
                continue
            rows = parse(r.json().get("results_html", ""))
            for row in rows:
                row["page_start"] = start
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            fo.flush()
            fd.write(f"{start}\n")
            fd.flush()
            if start % 5000 == 0:
                print(start, len(rows), rows[-1]["released_text"] if rows else None, flush=True)


if __name__ == "__main__":
    main()
