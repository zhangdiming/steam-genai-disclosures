import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

from steamhttp import SteamSession

ROOT = Path(__file__).resolve().parents[1]
URL = "https://steamcommunity.com/app/{appid}/homecontent/"


def parse_cards(html):
    soup = BeautifulSoup(html, "lxml")
    out = []
    for c in soup.select("div.apphub_Card"):
        title = c.select_one("div.title")
        txt = c.select_one("div.apphub_CardTextContent")
        if txt is None:
            continue
        date = txt.select_one("div.date_posted")
        date_s = date.get_text(strip=True) if date else None
        if date:
            date.extract()
        ea = txt.select_one("div.early_access_review")
        is_ea = ea is not None
        if ea:
            ea.extract()
        recv = txt.select_one("div.received_compensation")
        recv_s = recv.get_text(" ", strip=True) if recv else None
        if recv:
            recv.extract()
        hours = c.select_one("div.hours")
        helpful = c.select_one("div.found_helpful")
        hs = helpful.get_text(" ", strip=True) if helpful else ""
        m = re.search(r"([\d,]+) (?:people|person) found this review helpful", hs)
        link = c.get("data-modal-content-url") or ""
        out.append({
            "recommended": int(title.get_text(strip=True) == "Recommended") if title else None,
            "hours": hours.get_text(strip=True) if hours else None,
            "date_posted": date_s, "early_access_review": int(is_ea), "compensation": recv_s,
            "helpful": int(m.group(1).replace(",", "")) if m else 0,
            "author": link.split("/recommended/")[0].rsplit("/", 1)[-1] if link else None,
            "text": txt.get_text("\n", strip=True)[:4000],
        })
    soup_inputs = {i.get("name"): i.get("value") for i in soup.select("input")}
    return out, soup_inputs


def main():
    ids = [int(x) for x in Path(sys.argv[1]).read_text().split()]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    bf = sys.argv[3] if len(sys.argv) > 3 else "toprated"
    rps = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    out_r = ROOT / f"data/raw/reviews_{bf}.jsonl"
    done_p = ROOT / f"data/raw/reviews_{bf}_done.txt"
    done = set(int(x) for x in done_p.read_text().split()) if done_p.exists() else set()
    todo = [a for a in ids if a not in done]
    print(f"{len(ids)} ids, {len(done)} done, {len(todo)} to fetch", flush=True)
    sess = SteamSession(rps=rps)
    sess.s.cookies.set("Steam_Language", "english", domain="steamcommunity.com")
    sess.s.cookies.set("wants_mature_content", "1", domain="steamcommunity.com")
    lock = threading.Lock()
    n = [0]
    fr, fd = out_r.open("a", encoding="utf-8"), done_p.open("a")

    def work(appid):
        got, seen = [], set()
        params = {"userreviewsoffset": 0, "p": 1, "numperpage": 10, "browsefilter": bf, "appid": appid,
                  "appHubSubSection": 10, "l": "english", "filterLanguage": "english", "searchText": "",
                  "maxInappropriateScore": 100, "forceanon": 1}
        status = "ok"
        for _ in range(cap // 10 + 2):
            r = sess.get(URL.format(appid=appid), params=params)
            if r is None or r.status_code != 200:
                status = f"http_{getattr(r, 'status_code', 'none')}"
                break
            cards, inputs = parse_cards(r.text)
            new = [c for c in cards if (c["author"], c["text"][:80]) not in seen]
            for c in new:
                seen.add((c["author"], c["text"][:80]))
                c["appid"] = appid
                c["rank"] = len(got)
                got.append(c)
            if not new or len(got) >= cap or "userreviewscursor" not in inputs:
                break
            params.update({"userreviewsoffset": inputs.get("userreviewsoffset"), "p": inputs.get("p"),
                           "userreviewscursor": inputs.get("userreviewscursor")})
        with lock:
            if status == "ok":
                for c in got[:cap]:
                    c.pop("author", None)
                    fr.write(json.dumps(c, ensure_ascii=False) + "\n")
                fr.flush()
                fd.write(f"{appid}\n"); fd.flush()
            n[0] += 1
            if n[0] % 200 == 0:
                print(n[0], datetime.now().strftime("%H:%M:%S"), flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, todo))
    fr.close(); fd.close()


if __name__ == "__main__":
    main()
