import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

from steamhttp import SteamSession

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/store_pages.jsonl"
AI_H2 = "AI Generated Content Disclosure"


def txt(el):
    return el.get_text(" ", strip=True) if el else None


def parse(appid, page, final_url):
    soup = BeautifulSoup(page, "lxml")
    row = {"appid": appid, "final_url": final_url,
           "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    row["is_app_page"] = soup.select_one("div.apphub_AppName") is not None
    disc = None
    for d in soup.select("div.game_area_description"):
        h2 = d.find("h2")
        if h2 and AI_H2.lower() in h2.get_text(strip=True).lower():
            i = d.find("i")
            disc = (i.get_text("\n", strip=True) if i else txt(d))
    row["ai_disclosure"] = disc
    row["name"] = txt(soup.select_one("div.apphub_AppName"))
    row["release_date"] = txt(soup.select_one("div.release_date div.date"))
    row["developers"] = [a.get_text(strip=True) for a in soup.select("#developers_list a")]
    pubs = []
    for dr in soup.select("div.dev_row"):
        if "Publisher" in (txt(dr.select_one(".subtitle")) or ""):
            pubs = [a.get_text(strip=True) for a in dr.select("a")]
    row["publishers"] = pubs
    row["tags"] = [a.get_text(strip=True) for a in soup.select("a.app_tag")]
    gm = soup.select_one("#genresAndManufacturer")
    row["genres"] = [a.get_text(strip=True) for a in gm.select('a[href*="/genre/"]')] if gm else []
    row["early_access"] = soup.select_one("div.early_access_header") is not None
    row["n_languages"] = len(soup.select("table.game_language_options tr")) - 1 if soup.select_one(
        "table.game_language_options") else None
    price = soup.select_one("div.game_purchase_action div.discount_original_price") or \
        soup.select_one("div.game_purchase_action div.game_purchase_price")
    row["price_text"] = txt(price)
    row["is_free"] = bool(re.search(r"free to play|^free$", (row["price_text"] or "").lower()))
    about = soup.select_one("#game_area_description")
    row["about_len"] = len(txt(about) or "")
    row["n_screenshots"] = len(soup.select("div.highlight_screenshot"))
    row["n_movies"] = len(soup.select("div.highlight_movie"))
    row["categories"] = [txt(a) for a in soup.select("div.game_area_features_list_ctn a.game_area_details_specs_ctn")]
    rv = soup.select("div.user_reviews_summary_row")
    row["review_tooltips"] = [r.get("data-tooltip-html") for r in rv]
    mature = None
    for d in soup.select("div.game_area_description"):
        h2 = d.find("h2")
        if h2 and "Mature Content" in h2.get_text():
            mature = txt(d)
    row["mature_descriptor"] = mature
    return row


def main():
    ids = [int(x) for x in Path(sys.argv[1]).read_text().split()]
    rps = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    done = set()
    if OUT.exists():
        with OUT.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["appid"])
                except json.JSONDecodeError:
                    pass
    todo = [a for a in ids if a not in done]
    print(f"{len(ids)} ids, {len(done)} done, {len(todo)} to fetch", flush=True)
    sess = SteamSession(rps=rps)
    lock = threading.Lock()
    n = [0]
    fo = OUT.open("a", encoding="utf-8")

    def work(appid):
        r = sess.get(f"https://store.steampowered.com/app/{appid}/", params={"l": "english", "cc": "us"})
        if r is None:
            row = {"appid": appid, "error": "no_response"}
        elif r.status_code != 200:
            row = {"appid": appid, "error": f"http_{r.status_code}"}
        else:
            try:
                row = parse(appid, r.text, r.url)
            except Exception as e:
                row = {"appid": appid, "error": f"parse:{e!r}"}
        with lock:
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            n[0] += 1
            fo.flush()
            if n[0] % 500 == 0:
                print(n[0], datetime.now().strftime("%H:%M:%S"), flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, todo))
    fo.close()


if __name__ == "__main__":
    main()
