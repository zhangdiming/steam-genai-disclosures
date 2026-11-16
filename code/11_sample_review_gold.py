import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(r"\bai\b|\ba\.i\.|artificial intelligence|generative|\bslop\b|\bllms?\b|chat ?gpt|midjourney|"
                   r"stable diffusion|machine[- ]translat|\bgpt\b", re.I)


def main():
    n_tok, n_rand = int(sys.argv[1]), int(sys.argv[2])
    recs = []
    with open(ROOT / "data/raw/reviews_toprated.jsonl", encoding="utf-8") as f:
        for i, l in enumerate(f):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if r["text"].strip():
                recs.append({"line": i, "appid": r["appid"], "text": r["text"][:1500], "tok": bool(TOKEN.search(r["text"]))})
    rnd = random.Random(20261024)
    tok = [r for r in recs if r["tok"]]
    rest = [r for r in recs if not r["tok"]]
    s = [(r, "token") for r in rnd.sample(tok, min(n_tok, len(tok)))] + \
        [(r, "random_no_token") for r in rnd.sample(rest, min(n_rand, len(rest)))]
    rnd.shuffle(s)
    meta = []
    with open(ROOT / "annotation/gold/reviews_gold.jsonl", "w", encoding="utf-8") as fo:
        for i, (r, st) in enumerate(s):
            rid = f"r{i:04d}"
            fo.write(json.dumps({"rid": rid, "text": r["text"]}, ensure_ascii=False) + "\n")
            meta.append({"rid": rid, "appid": r["appid"], "line": r["line"], "stratum": st,
                         "pop_token": len(tok), "pop_no_token": len(rest)})
    pd.DataFrame(meta).to_csv(ROOT / "annotation/gold/review_meta.csv", index=False)
    print(f"reviews {len(recs)}; with token {len(tok)} ({len(tok)/len(recs):.3%}); sampled {len(s)}")


if __name__ == "__main__":
    main()
