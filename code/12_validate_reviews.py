import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_ai import genai_critical, genai_mention

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "annotation/gold"


def load(p):
    with open(p, encoding="utf-8") as f:
        return {r["rid"]: r for r in map(json.loads, f)}


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def match_crawl(hashes, meta):
    want = {(int(meta.loc[r, "appid"]), h): r for r, h in hashes.items()}
    found = {}
    text = (ROOT / "data/raw/reviews_toprated.jsonl").read_text(encoding="utf-8")
    for line in text.split("\n"):
        if line:
            o = json.loads(line)
            r = want.get((o["appid"], sha(o["text"][:1500])))
            if r is not None and r not in found:
                found[r] = (o["text"][:1500], o.get("recommended"))
    return found


def main():
    A, B = load(G / "coderA_reviews.jsonl"), load(G / "coderB_reviews.jsonl")
    meta = pd.read_csv(G / "review_meta.csv").set_index("rid")
    if (G / "reviews_gold.jsonl").exists():
        T = load(G / "reviews_gold.jsonl")
        hashes = {r: sha(t["text"]) for r, t in T.items()}
    else:
        hashes = {r: v["sha256"] for r, v in load(G / "reviews_gold_ids.jsonl").items()}
    found = match_crawl(hashes, meta)
    if not (G / "reviews_gold.jsonl").exists():
        T = {r: {"rid": r, "text": t} for r, (t, _) in found.items()}
    print(f"validation reviews found in the crawl: {len(found)} of {len(hashes)}")
    adj = load(G / "adjudicated_reviews.jsonl") if (G / "adjudicated_reviews.jsonl").exists() else {}
    rids = [r for r in hashes if r in A and r in B]
    out = {"n": len(rids)}
    gold = {}
    for k in ("genai_mention", "genai_critical"):
        a = np.array([A[r][k] for r in rids]); b = np.array([B[r][k] for r in rids])
        out[f"kappa_{k}"] = cohen_kappa_score(a, b)
        out[f"disagree_{k}"] = int((a != b).sum())
        for r, va, vb in zip(rids, a, b):
            gold.setdefault(r, {})[k] = int(va) if va == vb else (int(adj[r][k]) if r in adj and k in adj[r] else None)
    with open(G / "gold_reviews.jsonl", "w", encoding="utf-8") as fo:
        for r in rids:
            fo.write(json.dumps({"rid": r, **gold[r]}) + "\n")
    out["crawl_unmatched"] = sum(1 for r in rids if r not in found)
    rids = [r for r in rids if r in found]
    rec = {r: found[r][1] for r in rids}
    pred = {r: {"genai_mention": genai_mention(T[r]["text"]), "genai_critical": genai_critical(T[r]["text"], rec[r]),
                "genai_critical_term": genai_critical(T[r]["text"])} for r in rids}
    for r in rids:
        gold[r]["genai_critical_term"] = gold[r]["genai_critical"]
    pop = {"token": meta["pop_token"].iloc[0], "random_no_token": meta["pop_no_token"].iloc[0]}
    nst = meta.loc[rids, "stratum"].value_counts().to_dict()
    for k in ("genai_mention", "genai_critical", "genai_critical_term"):
        idx = [r for r in rids if gold[r][k] is not None]
        y = np.array([gold[r][k] for r in idx]); p = np.array([pred[r][k] for r in idx])
        w = np.array([pop[meta.loc[r, "stratum"]] / nst[meta.loc[r, "stratum"]] for r in idx])
        tp, fp, fn = (w * (y == 1) * (p == 1)).sum(), (w * (y == 0) * (p == 1)).sum(), (w * (y == 1) * (p == 0)).sum()
        out[f"{k}_P_w"], out[f"{k}_R_w"] = tp / (tp + fp), tp / (tp + fn)
        out[f"{k}_F1_w"] = 2 * tp / (2 * tp + fp + fn)
        out[f"{k}_gold_prev_w"] = (w * y).sum() / w.sum()
        out[f"{k}_pred_prev_w"] = (w * p).sum() / w.sum()
        out[f"{k}_unresolved"] = sum(1 for r in rids if gold[r][k] is None)
    g = pd.read_csv(ROOT / "data/processed/games.csv",
                    usecols=["appid", "disclosed", "c_live_system", "c_live_player", "page_redirect"], low_memory=False)
    redir = set(g.appid[g.page_redirect == 1])
    out["redirect_n"] = sum(1 for r in rids if int(meta.loc[r, "appid"]) in redir)
    out["redirect_games"] = len({int(meta.loc[r, "appid"]) for r in rids} & redir)
    out["redirect_crit"] = sum(1 for r in rids if int(meta.loc[r, "appid"]) in redir and gold[r]["genai_critical"] == 1)
    g = g[g.page_redirect != 1]
    disc = dict(zip(g.appid, g.disclosed))
    live = dict(zip(g.appid, (g.c_live_system == 1) | (g.c_live_player == 1)))
    for grp, val in (("runtime", True), ("other", False)):
        idx = [r for r in rids if gold[r]["genai_mention"] == 1 and gold[r]["genai_critical"] is not None
               and disc.get(int(meta.loc[r, "appid"])) == 1 and live.get(int(meta.loc[r, "appid"])) == val]
        out[f"tone_n_{grp}"] = len(idx)
        out[f"tone_crit_{grp}"] = int(sum(gold[r]["genai_critical"] for r in idx))
    for grp, val in (("disc", 1), ("nondisc", 0)):
        for k in ("genai_mention", "genai_critical"):
            idx = [r for r in rids if gold[r][k] is not None and disc.get(int(meta.loc[r, "appid"])) == val]
            y = np.array([gold[r][k] for r in idx]); p = np.array([pred[r][k] for r in idx])
            w = np.array([pop[meta.loc[r, "stratum"]] / nst[meta.loc[r, "stratum"]] for r in idx])
            tp, fp, fn = (w * (y == 1) * (p == 1)).sum(), (w * (y == 0) * (p == 1)).sum(), (w * (y == 1) * (p == 0)).sum()
            out[f"{k}_P_{grp}"] = tp / (tp + fp) if tp + fp else np.nan
            out[f"{k}_R_{grp}"] = tp / (tp + fn) if tp + fn else np.nan
            out[f"{k}_n_{grp}"] = len(idx)
    for grp in ("runtime", "otherdisc", "nondisc"):
        idx = [r for r in rids if gold[r]["genai_critical"] is not None and
               {"runtime": disc.get(int(meta.loc[r, "appid"])) == 1 and live.get(int(meta.loc[r, "appid"])),
                "otherdisc": disc.get(int(meta.loc[r, "appid"])) == 1 and not live.get(int(meta.loc[r, "appid"])),
                "nondisc": disc.get(int(meta.loc[r, "appid"])) == 0}[grp]]
        y = np.array([gold[r]["genai_critical"] for r in idx])
        out[f"crit_n_{grp}"], out[f"crit_npos_{grp}"] = len(idx), int(y.sum())
        for k in ("genai_critical", "genai_critical_term"):
            p = np.array([pred[r][k] for r in idx])
            tp, fp, fn = ((y == 1) & (p == 1)).sum(), ((y == 0) & (p == 1)).sum(), ((y == 1) & (p == 0)).sum()
            out[f"{k}_P_u{grp}"] = tp / (tp + fp) if tp + fp else np.nan
            out[f"{k}_R_u{grp}"] = tp / (tp + fn) if tp + fn else np.nan
    nt = [r for r in rids if meta.loc[r, "stratum"] == "random_no_token"]
    out["no_token_n"] = len(nt)
    out["no_token_pos"] = int(sum(gold[r]["genai_mention"] or 0 for r in nt))
    out["no_token_upper95"] = 3 / len(nt) if out["no_token_pos"] == 0 else np.nan
    out["pop_token"], out["pop_no_token"] = float(pop["token"]), float(pop["random_no_token"])
    for k, v in out.items():
        print(f"{k:28s} {v:.3f}" if isinstance(v, float) else f"{k:28s} {v}")
    json.dump({k: float(v) for k, v in out.items()}, open(G / "review_validation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
