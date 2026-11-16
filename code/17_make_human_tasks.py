import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "annotation/gold"
H = ROOT / "annotation/human"
SEED = 20261130
LIVE_CUE = re.compile(r"real[- ]?time|during (game ?play|play)|while (playing|the game)|runtime|LLM|language model|"
                      r"dynamic|players? can (generate|create)|chat|conversation|on[- ]device|locally", re.I)


def norm(t):
    return re.sub(r"\s+", " ", t.strip().lower())


def jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def draw(ids, n, rng):
    ids = sorted(ids)
    return ids if n >= len(ids) else sorted(rng.sample(ids, n))


HEAD = ('<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
        '<style>:root{color-scheme:light}body{margin:0;font:14px system-ui,sans-serif}'
        'img{max-width:100%}[hidden]{display:none!important}</style>\n</head><body>\n')


def build_page(payload):
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = (H / "annotate_template.html").read_text(encoding="utf-8")
    assert "/*__TASK_DATA__*/null" in page
    return page.replace("/*__TASK_DATA__*/null", data)


def recode(tasks, note):
    payload = json.load(open(H / "build/tasks.json", encoding="utf-8"))
    payload = {"version": f"{SEED}-v2-{''.join(tasks)}", "tasks": {t: payload["tasks"][t] for t in tasks}}
    if note:
        payload["note"] = note
    out = H / f"annotate_recode_{''.join(tasks)}.html"
    out.write_text(HEAD + build_page(payload) + "\n</body></html>\n", encoding="utf-8")
    print(f"{out.relative_to(ROOT)}: version {payload['version']}, " +
          ", ".join(f"{t} {len(v)} items" for t, v in payload["tasks"].items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recode", help="build a recoding page for these tasks only (e.g. G) from build/tasks.json")
    ap.add_argument("--note", help="instructions shown on the start page of a recoding page")
    ap.add_argument("--n-disc-live", type=int, default=50)
    ap.add_argument("--n-disc-rest", type=int, default=150)
    ap.add_argument("--n-rev-token", type=int, default=200)
    ap.add_argument("--n-rev-random", type=int, default=100)
    ap.add_argument("--n-decoy", type=int, default=40)
    ap.add_argument("--n-practice", type=int, default=12)
    a = ap.parse_args()
    if a.recode:
        return recode(list(a.recode), a.note)
    rng = random.Random(SEED)
    key, tasks = [], {}

    dev = {r["gid"]: r["text"] for r in jsonl(G / "disclosures_dev.jsonl")}
    dm = pd.read_csv(G / "disclosure_meta.csv")
    dm = dm[dm.split == "dev"]
    n_live = a.n_practice // 3
    pick = draw(dm[dm.live_cue].gid, n_live, rng) + draw(dm[~dm.live_cue].gid, a.n_practice - n_live, rng)
    tasks["P"] = [{"id": g, "text": dev[g]} for g in pick]
    key += [{"task": "P", "item": g, "source": g, "stratum": "live_cue" if l else "rest", "n_games": 1, "appids": ""}
            for g, l in zip(pick, [True] * n_live + [False] * (len(pick) - n_live))]

    test = {r["gid"]: r["text"] for r in jsonl(G / "disclosures_test.jsonl")}
    tm = pd.read_csv(G / "test_meta.csv")
    live = draw(tm[tm.live_cue].gid, a.n_disc_live, rng)
    rest = draw(tm[~tm.live_cue].gid, a.n_disc_rest, rng)
    tasks["D"] = [{"id": g, "text": test[g]} for g in live + rest]
    key += [{"task": "D", "item": g, "source": g, "stratum": s, "n_games": 1, "appids": ""}
            for s, ids in (("live_cue", live), ("rest", rest)) for g in ids]

    games = pd.read_csv(ROOT / "data/processed/games.csv", low_memory=False,
                        usecols=["appid", "disclosed", "disclosure", "c_live_system", "c_live_player"])
    games = games[(games.disclosed == 1) & games.disclosure.notna()].copy()
    games["norm"] = games.disclosure.map(norm)
    games["live"] = ((games.c_live_system == 1) | (games.c_live_player == 1)).astype(int)
    uid_of = {}
    for r in pd.read_csv(H.parent / "bulk/uid_map.csv").itertuples():
        if r.uid != "gold":
            uid_of.setdefault(r.norm, r.uid)
    grp = games.groupby("norm").agg(text=("disclosure", "first"), live=("live", "max"), n_games=("appid", "size"),
                                    appids=("appid", lambda s: ";".join(map(str, sorted(s))))).reset_index()
    rt = grp[grp.live == 1]
    pool = grp[(grp.live == 0) & grp.text.map(lambda t: bool(LIVE_CUE.search(t)))]
    dec = pool.loc[draw(pool.index, a.n_decoy, rng)]
    g_items = [(r, "runtime") for r in rt.itertuples()] + [(r, "decoy") for r in dec.itertuples()]
    g_items.sort(key=lambda x: hashlib.sha1(x[0].norm.encode()).hexdigest())
    tasks["G"] = []
    for i, (r, s) in enumerate(g_items, 1):
        iid = f"rt{i:03d}"
        tasks["G"].append({"id": iid, "text": r.text})
        key.append({"task": "G", "item": iid, "source": uid_of.get(r.norm, ""), "stratum": s,
                    "n_games": int(r.n_games), "appids": r.appids})

    revs = {r["rid"]: r["text"] for r in jsonl(G / "reviews_gold.jsonl")}
    rm = pd.read_csv(G / "review_meta.csv")
    tok = draw(rm[rm.stratum == "token"].rid, a.n_rev_token, rng)
    rnd = draw(rm[rm.stratum == "random_no_token"].rid, a.n_rev_random, rng)
    tasks["R"] = [{"id": r, "text": revs[r]} for r in tok + rnd]
    appid = dict(zip(rm.rid, rm.appid))
    key += [{"task": "R", "item": r, "source": r, "stratum": s, "n_games": 1, "appids": str(appid[r])}
            for s, ids in (("token", tok), ("random_no_token", rnd)) for r in ids]

    H.mkdir(parents=True, exist_ok=True)
    (H / "build").mkdir(exist_ok=True)
    pd.DataFrame(key).to_csv(H / "items_key.csv", index=False)
    payload = {"version": f"{SEED}-v1", "tasks": tasks}
    json.dump(payload, open(H / "build/tasks.json", "w", encoding="utf-8"), ensure_ascii=False)
    page = build_page(payload)
    (H / "build/annotate_artifact.html").write_text(page, encoding="utf-8")
    (H / "annotate.html").write_text(HEAD + page + "\n</body></html>\n", encoding="utf-8")
    for t, items in tasks.items():
        print(f"{t}: {len(items)} items")
    print("G strata:", pd.DataFrame(key).query("task == 'G'").stratum.value_counts().to_dict())
    print("page size: %.0f KB" % (len(page.encode()) / 1024))


if __name__ == "__main__":
    main()
