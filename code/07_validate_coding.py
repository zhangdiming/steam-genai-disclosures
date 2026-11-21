import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rule_coder import code_disclosure

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "annotation/gold"
BIN = ["pre", "live_system", "live_player", "marketing", "marketing_only", "explicit_no_live", "non_genai",
       "ai_on_human", "mixed_oversight", "d_visual", "d_music", "d_sfx", "d_voice", "d_text", "d_localization",
       "d_code", "d_other", "scope_minor", "scope_extensive", "placeholder", "tools_named", "provenance_claim",
       "small_team", "gr_filter", "gr_constrained", "gr_local", "gr_human", "gr_report", "gr_optout",
       "gr_provider", "gr_none"]


def load(p):
    with open(p, encoding="utf-8") as f:
        return {r["gid"]: r for r in map(json.loads, f)}


def main():
    split = sys.argv[1]
    A, B = load(G / f"coderA_disclosures_{split}.jsonl"), load(G / f"coderB_disclosures_{split}.jsonl")
    texts = load(G / f"disclosures_{split}.jsonl")
    gids = [g for g in texts if g in A and g in B]
    print(f"{split}: {len(gids)} items coded by both coders (of {len(texts)})")
    rows = []
    for k in BIN + ["authority"]:
        a = np.array([int(A[g].get(k, 0)) for g in gids])
        b = np.array([int(B[g].get(k, 0)) for g in gids])
        w = "quadratic" if k == "authority" else None
        kap = cohen_kappa_score(a, b, weights=w) if len(set(a) | set(b)) > 1 else np.nan
        rows.append({"code": k, "prev_A": a.mean(), "prev_B": b.mean(), "agree": (a == b).mean(), "kappa": kap})
    agree = pd.DataFrame(rows)

    adj_p = G / f"adjudicated_{split}.jsonl"
    adj = load(adj_p) if adj_p.exists() else {}
    gold = {}
    for g in gids:
        rec = {"gid": g}
        for k in BIN + ["authority"]:
            va, vb = int(A[g].get(k, 0)), int(B[g].get(k, 0))
            if va == vb:
                rec[k] = va
            elif g in adj and k in adj[g]:
                rec[k] = int(adj[g][k])
            else:
                rec[k] = None
        gold[g] = rec
    unresolved = sum(1 for r in gold.values() for k in BIN + ["authority"] if r[k] is None)
    with open(G / f"gold_{split}.jsonl", "w", encoding="utf-8") as fo:
        for r in gold.values():
            fo.write(json.dumps(r) + "\n")
    print("unresolved cells:", unresolved)

    w = {g: 1.0 for g in gids}
    if split == "test":
        meta = pd.read_csv(G / "test_meta.csv").set_index("gid")
        n_live = int(meta.live_cue.sum()); n_rest = len(meta) - n_live
        w = {g: (meta.pop_live.iloc[0] / n_live if meta.loc[g, "live_cue"] else meta.pop_rest.iloc[0] / n_rest)
             for g in gids}
    coders = {"rule": {g: code_disclosure(texts[g]["text"]) for g in gids}}
    if split == "test":
        bulk = {}
        for bp in sorted((ROOT / "annotation/bulk").glob("coded_batch_[0-9][0-9].jsonl")):
            bulk.update({r["uid"]: r for r in map(json.loads, open(bp, encoding="utf-8"))})
        coders["bulk"] = {g: bulk[g] for g in gids if g in bulk}
        if (ROOT / "annotation/bulk/runtime_pass.jsonl").exists():
            spec = importlib.util.spec_from_file_location("bd", Path(__file__).parent / "05_build_dataset.py")
            bd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)
            final = bd.bulk_codes()
            coders["final"] = {g: final[g] for g in gids if g in final}
    for name, pred in coders.items():
        val = score(gids, gold, pred, w)
        print(f"--- {name} coder vs gold ({split}) ---")
        print(val.round(3).to_string(index=False))
        val.to_csv(G / f"{name}coder_validation_{split}.csv", index=False)
    pd.set_option("display.width", 200)
    print(agree.round(3).to_string(index=False))
    agree.to_csv(G / f"agreement_{split}.csv", index=False)


def score(gids, gold, pred, w):
    vrows = []
    live = lambda r: int(bool(int(r.get("live_system", 0)) or int(r.get("live_player", 0))))
    gold = {g: {**r, "live_any": None if r["live_system"] is None or r["live_player"] is None else live(r)}
            for g, r in gold.items()}
    pred = {g: {**r, "live_any": live(r)} for g, r in pred.items()}
    for k in BIN[:3] + ["live_any"] + BIN[3:]:
        idx = [g for g in gids if g in pred and gold[g][k] is not None]
        if k.startswith("gr_"):
            idx = [g for g in idx if gold[g]["live_system"] or gold[g]["live_player"]]
        if not idx:
            continue
        y = np.array([gold[g][k] for g in idx]); p = np.array([int(pred[g].get(k, 0)) for g in idx])
        ww = np.array([w[g] for g in idx])
        tp, fp, fn = (ww * (y == 1) * (p == 1)).sum(), (ww * (y == 0) * (p == 1)).sum(), (ww * (y == 1) * (p == 0)).sum()
        vrows.append({"code": k, "n": len(y), "n_pos": int(y.sum()), "gold_prev": (ww * y).sum() / ww.sum(),
                      "pred_prev": (ww * p).sum() / ww.sum(),
                      "P": tp / (tp + fp) if tp + fp else np.nan, "R": tp / (tp + fn) if tp + fn else np.nan,
                      "F1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan,
                      "acc": (ww * (y == p)).sum() / ww.sum()})
    idx = [g for g in gids if g in pred and gold[g]["authority"] is not None]
    y = np.array([gold[g]["authority"] for g in idx]); p = np.array([int(pred[g]["authority"]) for g in idx])
    ww = np.array([w[g] for g in idx])
    vrows.append({"code": "authority (exact)", "n": len(y), "n_pos": np.nan, "gold_prev": np.nan, "pred_prev": np.nan,
                  "P": np.nan, "R": np.nan, "F1": f1_score(y, p, average="macro"), "acc": (ww * (y == p)).sum() / ww.sum()})
    vrows.append({"code": "authority (qw-kappa vs gold)", "n": len(y), "n_pos": np.nan, "gold_prev": np.nan,
                  "pred_prev": np.nan, "P": np.nan, "R": np.nan,
                  "F1": cohen_kappa_score(y, p, weights="quadratic", sample_weight=ww), "acc": np.nan})
    return pd.DataFrame(vrows)


if __name__ == "__main__":
    main()
