import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "annotation/gold"
LIVE_CUE = re.compile(r"real[- ]?time|during (game ?play|play)|while (playing|the game)|runtime|LLM|language model|"
                      r"dynamic|players? can (generate|create)|chat|conversation|on[- ]device|locally", re.I)


def norm(t):
    return re.sub(r"\s+", " ", t.strip().lower())


def main():
    split, n_total, n_live = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    OUT.mkdir(parents=True, exist_ok=True)
    meta_p = OUT / "disclosure_meta.csv"
    prev = pd.read_csv(meta_p) if meta_p.exists() else pd.DataFrame(columns=["gid", "appid", "live_cue", "split", "norm"])
    taken = set(prev["norm"]) if len(prev) else set()
    seen, recs = set(), []
    with open(ROOT / "data/raw/store_pages.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            t = p.get("ai_disclosure")
            if not t:
                continue
            k = norm(t)
            if k in seen or k in taken:
                continue
            seen.add(k)
            recs.append({"appid": p["appid"], "text": t, "norm": k, "live_cue": bool(LIVE_CUE.search(t))})
    rnd = random.Random(f"gold-{split}-20261024")
    live = [r for r in recs if r["live_cue"]]
    rest = [r for r in recs if not r["live_cue"]]
    s_live = rnd.sample(live, min(n_live, len(live)))
    s_rest = rnd.sample(rest, min(n_total - len(s_live), len(rest)))
    gold = s_live + s_rest
    rnd.shuffle(gold)
    start = len(prev)
    rows = []
    with open(OUT / f"disclosures_{split}.jsonl", "w", encoding="utf-8") as fo:
        for i, r in enumerate(gold):
            gid = f"d{start + i:04d}"
            fo.write(json.dumps({"gid": gid, "text": r["text"]}, ensure_ascii=False) + "\n")
            rows.append({"gid": gid, "appid": r["appid"], "live_cue": r["live_cue"], "split": split, "norm": r["norm"],
                         "stratum_pop_live": len(live), "stratum_pop_rest": len(rest)})
    pd.concat([prev, pd.DataFrame(rows)]).to_csv(meta_p, index=False)
    print(f"{split}: unique disclosures available {len(recs)} (live cue {len(live)}); sampled {len(gold)} = "
          f"{len(s_live)} live-cue + {len(s_rest)} other")


if __name__ == "__main__":
    main()
