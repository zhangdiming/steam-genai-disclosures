import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
END = datetime(2026, 6, 30, 23, 59, tzinfo=timezone.utc).timestamp()
START = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()


def lines(p):
    with open(p, encoding="utf-8") as f:
        for l in f:
            try:
                yield json.loads(l)
            except json.JSONDecodeError:
                continue


def control_draw(appid, pct=20):
    return int(hashlib.md5(f"ctrl-{appid}".encode()).hexdigest(), 16) % 100 < pct


def main():
    items = {it["appid"]: it for it in lines(ROOT / "data/raw/store_items.jsonl") if not it.get("missing")}
    disc, ctrl = [], []
    for p in lines(ROOT / "data/raw/store_pages.jsonl"):
        it = items.get(p["appid"])
        if "error" in p or it is None:
            continue
        ts = it.get("release", {}).get("steam_release_date") or 0
        n = it.get("reviews", {}).get("summary_filtered", {}).get("review_count", 0)
        if not (START <= ts <= END and n >= 10):
            continue
        if p.get("ai_disclosure"):
            disc.append(p["appid"])
        elif control_draw(p["appid"]):
            ctrl.append(p["appid"])
    q = disc + ctrl
    (ROOT / "data/processed/review_queue.txt").write_text("\n".join(map(str, q)) + "\n")
    print(f"review queue: {len(disc)} disclosing + {len(ctrl)} control = {len(q)}")


if __name__ == "__main__":
    main()
