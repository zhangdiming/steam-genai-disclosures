import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "annotation/bulk"
START = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
END = datetime(2026, 6, 30, 23, 59, tzinfo=timezone.utc).timestamp()
LIVE_CUE = re.compile(r"real[- ]?time|during (game ?play|play)|while (playing|the game|you play)|runtime|LLM|language model|"
                      r"dynamic|players? can (generate|create)|chat|conversation|talk|on[- ]device|locally|in response|prompt|"
                      r"live[- ]generat|on the fly|personali[sz]|restrict|filter|moderat|guardrail|safeguard|explicit|"
                      r"inappropriate|offensive|ai (system|model|engine|companion|npc|character|narrator|game ?master)", re.I)
BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 450


def norm(t):
    return re.sub(r"\s+", " ", t.strip().lower())


def lines(p):
    with open(p, encoding="utf-8") as f:
        for l in f:
            try:
                yield json.loads(l)
            except json.JSONDecodeError:
                continue


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = {it["appid"]: it for it in lines(ROOT / "data/raw/store_items.jsonl") if not it.get("missing")}
    gold_norm = set()
    gold_norm |= {norm(r["text"]) for r in lines(ROOT / "annotation/gold/disclosures_dev.jsonl")}
    map_p = OUT / "uid_map.csv"
    old = pd.read_csv(map_p) if map_p.exists() else pd.DataFrame(columns=["uid", "appid", "norm", "batch"])
    uid_of = {k: u for k, u in zip(old["norm"], old["uid"]) if u != "gold"} if len(old) else {}
    batch_of = dict(zip(old["uid"], old["batch"])) if len(old) else {}
    rows, new = [], {}
    for p in lines(ROOT / "data/raw/store_pages.jsonl"):
        t = p.get("ai_disclosure")
        if not t or "error" in p:
            continue
        it = items.get(p["appid"], {})
        ts = it.get("release", {}).get("steam_release_date") or 0
        n = it.get("reviews", {}).get("summary_filtered", {}).get("review_count", 0)
        if not ((START <= ts <= END and n >= 10) or LIVE_CUE.search(t)):
            continue
        k = norm(t)
        if k in gold_norm:
            rows.append({"uid": "gold", "appid": p["appid"], "norm": k, "batch": "gold"})
            continue
        if k not in uid_of:
            uid_of[k] = f"u{len(uid_of):05d}"
            new[uid_of[k]] = t
        rows.append({"uid": uid_of[k], "appid": p["appid"], "norm": k, "batch": batch_of.get(uid_of[k], None)})
    existing = sorted(OUT.glob("batch_*.jsonl"))
    nb = len(existing)
    new_items = list(new.items())
    for i in range(0, len(new_items), BATCH):
        chunk = new_items[i:i + BATCH]
        name = f"batch_{nb:02d}"
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as fo:
            for uid, t in chunk:
                fo.write(json.dumps({"uid": uid, "text": t}, ensure_ascii=False) + "\n")
                batch_of[uid] = name
        nb += 1
    df = pd.DataFrame(rows)
    df["batch"] = df["uid"].map(batch_of).fillna(df["batch"])
    df.drop_duplicates(["uid", "appid"]).to_csv(map_p, index=False)
    print(f"games mapped {df.appid.nunique()}; unique texts {len(uid_of)} (+{len(new)} new); gold reuse "
          f"{(df.uid == 'gold').sum()}; batches now {nb}")


if __name__ == "__main__":
    main()
