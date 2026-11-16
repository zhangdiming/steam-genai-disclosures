import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "annotation/gold"
ROUND1 = [f"batch_{b:02d}.jsonl" for b in range(4)]
LIVE = re.compile(r"real[- ]?time|during (game ?play|play)|while (playing|the game|you play)|runtime|LLM|language model|"
                  r"dynamic|players? can (generate|create)|chat|conversation|talk|on[- ]device|locally|in response|prompt",
                  re.I)


def main():
    texts = {}
    for b in ROUND1:
        for l in open(ROOT / "annotation/bulk" / b, encoding="utf-8"):
            r = json.loads(l)
            texts[r["uid"]] = r["text"]
    live = [u for u, t in texts.items() if LIVE.search(t)]
    rest = [u for u in texts if u not in set(live)]
    rnd = random.Random("gold-test-20261024")
    s = rnd.sample(live, min(100, len(live))) + rnd.sample(rest, 400 - min(100, len(live)))
    rnd.shuffle(s)
    meta = pd.DataFrame({"gid": s, "live_cue": [u in set(live) for u in s], "split": "test",
                         "pop_live": len(live), "pop_rest": len(rest)})
    if "--write" in sys.argv:
        with open(G / "disclosures_test.jsonl", "w", encoding="utf-8") as f:
            for u in s:
                f.write(json.dumps({"gid": u, "text": texts[u]}, ensure_ascii=False) + "\n")
        meta.to_csv(G / "test_meta.csv", index=False)
        print("written", len(s))
    else:
        old = pd.read_csv(G / "test_meta.csv")
        same = old.gid.tolist() == meta.gid.tolist() and old.live_cue.tolist() == meta.live_cue.tolist() \
            and (old.pop_live == meta.pop_live).all() and (old.pop_rest == meta.pop_rest).all()
        print("round-1 texts", len(texts), "runtime cue", len(live), "reproduces test_meta.csv:", same)


if __name__ == "__main__":
    main()
