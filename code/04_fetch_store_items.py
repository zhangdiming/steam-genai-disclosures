import json
import sys
from pathlib import Path

from steamhttp import SteamSession

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/store_items.jsonl"
URL = "https://api.steampowered.com/IStoreBrowseService/GetItems/v1/"
REQ = {"include_assets": False, "include_release": True, "include_platforms": True,
       "include_all_purchase_options": False, "include_screenshots": False, "include_trailers": False,
       "include_ratings": True, "include_tag_count": 20, "include_reviews": True, "include_basic_info": True,
       "include_supported_languages": True, "include_full_description": False, "include_included_items": False,
       "apply_user_filters": False, "include_links": False}


def main():
    ids = [int(x) for x in Path(sys.argv[1]).read_text().split()]
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    sess = SteamSession(rps=float(sys.argv[3]) if len(sys.argv) > 3 else 1.0)
    done = set()
    if OUT.exists():
        with OUT.open(encoding="utf-8") as f:
            for line in f:
                done.add(json.loads(line).get("appid"))
    todo = [a for a in ids if a not in done]
    print(f"{len(ids)} ids, {len(done)} done, {len(todo)} to fetch", flush=True)
    with OUT.open("a", encoding="utf-8") as fo:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            inp = {"ids": [{"appid": a} for a in chunk],
                   "context": {"language": "english", "country_code": "US"}, "data_request": REQ}
            r = sess.get(URL, params={"input_json": json.dumps(inp)})
            if r is None or r.status_code != 200:
                print("failed batch", i, getattr(r, "status_code", None), flush=True)
                continue
            items = r.json().get("response", {}).get("store_items", [])
            got = set()
            for it in items:
                if "appid" in it:
                    got.add(it["appid"])
                    fo.write(json.dumps(it, ensure_ascii=False) + "\n")
            for a in set(chunk) - got:
                fo.write(json.dumps({"appid": a, "success": 0, "missing": True}) + "\n")
            fo.flush()
            if (i // batch) % 50 == 0:
                print(i, len(items), flush=True)


if __name__ == "__main__":
    main()
